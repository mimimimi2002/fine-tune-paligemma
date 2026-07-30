import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from configs import object_detection_config

from paligemma_ft.data_utis import collate_fn
from functools import partial
import matplotlib
from matplotlib import pyplot as plt, patches
import os
import json
import argparse
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)

# the COCO IoU sweep, used for mAP@[.5:.95]
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
# thresholds precision / recall / F1 are reported at
PR_IOU_THRESHOLDS = (0.5, 0.75)

def box_iou(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)"""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)

    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def match_boxes(pred_boxes, gt_boxes):
    """
    one-to-one matching maximising the total IoU -> (pred idx, gt idx, iou)
    return ndarrays of the same length, which is the number of matched pairs
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, float)

    iou = box_iou(pred_boxes, gt_boxes)
    pred_idx, gt_idx = linear_sum_assignment(iou, maximize=True)
    return pred_idx, gt_idx, iou[pred_idx, gt_idx]


def average_precision(predictions, ground_truths, iou_threshold):
    """COCO style AP: rank every detection by confidence, greedily match it to a
    free ground truth box of the same image, then average the 101-point
    interpolated precision-recall curve."""
    if not ground_truths:
        return float("nan")
    if not predictions:
        return 0.0

    gt_boxes_per_image = defaultdict(list)
    for gt in ground_truths:
        gt_boxes_per_image[gt["image_id"]].append(gt["xyxy"])
    claimed = {
        image_id: np.zeros(len(boxes), bool)
        for image_id, boxes in gt_boxes_per_image.items()
    }

    ranked = sorted(predictions, key=lambda p: p["score"], reverse=True)
    true_positive = np.zeros(len(ranked))
    false_positive = np.zeros(len(ranked))

    for rank, prediction in enumerate(ranked):
        gt_boxes = gt_boxes_per_image.get(prediction["image_id"])
        if not gt_boxes:
            false_positive[rank] = 1
            continue

        ious = box_iou([prediction["xyxy"]], gt_boxes)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not claimed[prediction["image_id"]][best]:
            claimed[prediction["image_id"]][best] = True
            true_positive[rank] = 1
        else:
            false_positive[rank] = 1

    cum_tp = np.cumsum(true_positive)
    cum_fp = np.cumsum(false_positive)
    recall = cum_tp / len(ground_truths)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # make the precision curve monotonically decreasing, then sample it at 101
    # evenly spaced recall levels (precision is 0 for recall levels never reached)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_levels = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, recall_levels, side="left")
    sampled = np.zeros_like(recall_levels)
    reached = indices < len(precision)
    sampled[reached] = precision[indices[reached]]
    return float(sampled.mean())


def sequence_confidence(model, generated, prompt_length, pad_token_id):
    """mean probability of the generated tokens, one score per sample.

    PaliGemma emits no per-box confidence, so the sequence level probability is
    used as the confidence of every box decoded from that sequence. It is only
    needed to rank detections for AP; a constant fallback keeps AP well defined
    when the scores are unavailable."""
    batch_size = generated.sequences.shape[0]
    if getattr(generated, "scores", None) is None:
        return [1.0] * batch_size

    transition_scores = model.compute_transition_scores(
        generated.sequences, generated.scores, normalize_logits=True
    ).to(torch.float32)
    transition_scores = torch.nan_to_num(transition_scores, neginf=0.0)

    new_tokens = generated.sequences[:, prompt_length:]
    if pad_token_id is None:
        mask = torch.ones_like(new_tokens, dtype=torch.bool)
    else:
        mask = new_tokens != pad_token_id
    probabilities = transition_scores.exp() * mask
    return (probabilities.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().tolist()


def normalize_detection_string(detection_string):
    return " ".join(detection_string.split()).strip(" ;")


def extract_objects(detection_string, image_width, image_height, unique_labels=False):
    objects = []
    seen_labels = set()

    while detection_string:
        match = DETECT_RE.match(detection_string)
        if not match:
            break

        prefix, locations, label = match.groups()
        location_values = [int(loc) for loc in re.findall(r"\d{4}", locations)]
        y1, x1, y2, x2 = [value / 1024 for value in location_values]
        y1, x1, y2, x2 = map(
            round,
            (y1 * image_height, x1 * image_width, y2 * image_height, x2 * image_width),
        )

        label = label.strip()  # Remove trailing spaces from label

        if unique_labels and label in seen_labels:
            label = (label or "") + "'"
        seen_labels.add(label)

        objects.append(dict(xyxy=(x1, y1, x2, y2), name=label))

        detection_string = detection_string[len(match.group()) :]

    return objects


def unnormalize_image(pixel_value, processor):
    # help from : https://discuss.huggingface.co/t/vitimageprocessor-output-visualization/76335/6
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    pixel_value = pixel_value.cpu().to(torch.float32)
    image = (pixel_value.numpy() * np.array(std)[:, None, None]) + np.array(mean)[
        :, None, None
    ]
    image = (image * 255).astype(np.uint8)
    return np.moveaxis(image, 0, -1)


def get_detection_string(decoded_output):
    # the decoded output is "<prompt>\n<answer>"; an empty answer means no bbox
    parts = decoded_output.split("\n", 1)
    return parts[1] if len(parts) > 1 else ""


def draw_bbox(image, objects, save_path=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for obj in objects:
        bbox = obj["xyxy"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 10, obj["name"], color="red", fontsize=12, weight="bold"
        )

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, bbox_inches="tight")

    # close explicitly, otherwise the figures pile up over the whole test set
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./predictions",
        help="directory the annotated images are written to",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many images"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display each image instead of writing it to --output-dir",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="write the summary metrics to this json file",
    )
    args = parser.parse_args()

    if not args.show:
        # no display is needed when we only write files
        matplotlib.use("Agg")
        os.makedirs(args.output_dir, exist_ok=True)

    # get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load the test dataset
    print(f"[INFO] loading {object_detection_config.DATASET_ID} from hub...")
    test_dataset = load_dataset(object_detection_config.DATASET_ID, split="test")

    # get the processor
    print(f"[INFO] loading processor from {args.checkpoint}...")
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    # build the data loader
    print("[INFO] building the test data loader...")
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="Detect license plate.",
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=False,
        ),
        batch_size=object_detection_config.BATCH_SIZE,
        shuffle=False,
    )

    # load the fine tuned model
    print(f"[INFO] loading model from {args.checkpoint}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=object_detection_config.MODEL_DTYPE,
        device_map=device,
    )
    model.eval()

    print("[INFO] running inference on the test set...")
    image_index = 0
    reached_limit = False

    total_iou = 0.0
    total_pairs = 0
    total_predictions = 0
    total_ground_truths = 0
    exact_matches = 0
    correct_labels = 0
    # tp / fp / fn per IoU threshold, for precision, recall and F1
    counts = {
        threshold: dict(tp=0, fp=0, fn=0) for threshold in set(PR_IOU_THRESHOLDS)
    }
    # every detection and ground truth box of the run, for the AP sweep
    all_predictions = []
    all_ground_truths = []

    for batch in test_dataloader:
        gt_detection_strings = batch.pop("label_for_paligemma")
        batch.pop("labels", None)
        prompt_length = batch["input_ids"].shape[1]
        with torch.inference_mode():
            generated_outputs = model.generate(
                **batch,
                max_new_tokens=100,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        confidences = sequence_confidence(
            model,
            generated_outputs,
            prompt_length,
            processor.tokenizer.pad_token_id,
        )
        generated_outputs = processor.batch_decode(
            generated_outputs.sequences, skip_special_tokens=True
        )

        for element, pixel_values, gt_detection_string, confidence in zip(
            generated_outputs,
            batch["pixel_values"],
            gt_detection_strings,
            confidences,
            strict=True,
        ):
            image = unnormalize_image(pixel_values, processor)
            image_height, image_width = image.shape[:2]

            detection_string = get_detection_string(element)
            objects = extract_objects(
                detection_string, image_width, image_height, unique_labels=False
            )

            gt_objects = extract_objects(
                gt_detection_string, image_width, image_height, unique_labels=False
            )

            pred_boxes = np.array([o["xyxy"] for o in objects], float).reshape(-1, 4)
            gt_boxes = np.array([o["xyxy"] for o in gt_objects], float).reshape(-1, 4)

            total_predictions += len(pred_boxes)
            total_ground_truths += len(gt_boxes)
            exact_matches += normalize_detection_string(
                detection_string
            ) == normalize_detection_string(gt_detection_string)

            for box in pred_boxes:
                all_predictions.append(
                    dict(image_id=image_index, xyxy=box, score=confidence)
                )
            for box in gt_boxes:
                all_ground_truths.append(dict(image_id=image_index, xyxy=box))

            pred_idx, gt_idx, matched = match_boxes(pred_boxes, gt_boxes)
            total_iou += matched.sum()
            total_pairs += len(matched)
            correct_labels += sum(
                objects[p]["name"] == gt_objects[g]["name"]
                for p, g in zip(pred_idx, gt_idx)
            )

            for threshold, count in counts.items():
                true_positives = int((matched >= threshold).sum())
                count["tp"] += true_positives
                count["fp"] += len(pred_boxes) - true_positives
                count["fn"] += len(gt_boxes) - true_positives

            if objects:
                print(f"[{image_index:05d}] Predicted    {detection_string}")
                print(f"[{image_index:05d}] Ground Truth {gt_detection_string}")
                if len(gt_boxes):
                    print(
                        f"[{image_index:05d}] IoU {np.round(matched, 3).tolist()}  "
                        f"pred={len(pred_boxes)} gt={len(gt_boxes)} "
                        f"conf={confidence:.3f}"
                    )
                else:
                    print(
                        f"[{image_index:05d}] no ground truth: "
                        f"all {len(pred_boxes)} predictions are false positives"
                    )
            else:
                print(f"[{image_index:05d}] No bbox found (gt={len(gt_boxes)})")

            save_path = (
                None
                if args.show
                else os.path.join(args.output_dir, f"{image_index:05d}.png")
            )
            draw_bbox(image, objects, save_path=save_path)

            image_index += 1
            if args.limit is not None and image_index >= args.limit:
                reached_limit = True
                break

        if reached_limit:
            break

    if args.show:
        print(f"[INFO] done, {image_index} images")
    else:
        print(f"[INFO] done, {image_index} images written to {args.output_dir}")

    metrics = {
        "images": image_index,
        "predictions": total_predictions,
        "ground_truths": total_ground_truths,
        "matched_pairs": total_pairs,
        "mean_iou_matched": float(total_iou / total_pairs)
        if total_pairs
        else float("nan"),
        "label_accuracy": correct_labels / total_pairs if total_pairs else float("nan"),
        "exact_match_rate": exact_matches / image_index if image_index else float("nan"),
    }

    for threshold in sorted(counts):
        true_positives = counts[threshold]["tp"]
        false_positives = counts[threshold]["fp"]
        false_negatives = counts[threshold]["fn"]
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else float("nan")
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else float("nan")
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        key = f"{threshold:.2f}".rstrip("0").rstrip(".")
        metrics[f"precision@{key}"] = precision
        metrics[f"recall@{key}"] = recall
        metrics[f"f1@{key}"] = f1
        metrics[f"tp@{key}"] = true_positives
        metrics[f"fp@{key}"] = false_positives
        metrics[f"fn@{key}"] = false_negatives

    average_precisions = {
        threshold: average_precision(all_predictions, all_ground_truths, threshold)
        for threshold in COCO_IOU_THRESHOLDS
    }
    metrics["ap@0.5"] = average_precisions[0.5]
    metrics["ap@0.75"] = average_precisions[0.75]
    metrics["map@[0.5:0.95]"] = float(np.mean(list(average_precisions.values())))

    print("\n[INFO] ===== evaluation summary =====")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"[INFO] {name:>18}: {value:.3f}")
        else:
            print(f"[INFO] {name:>18}: {value}")

    if args.metrics_json is not None:
        with open(args.metrics_json, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        print(f"[INFO] metrics written to {args.metrics_json}")