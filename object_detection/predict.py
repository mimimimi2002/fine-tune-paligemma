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
import argparse
from scipy.optimize import linear_sum_assignment

DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)

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
    for batch in test_dataloader:
        gt_detection_strings = batch.pop("label_for_paligemma")
        batch.pop("labels", None)
        with torch.inference_mode():
            generated_outputs = model.generate(
                **batch, max_new_tokens=100, do_sample=False
            )
        generated_outputs = processor.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

        for element, pixel_values, gt_detection_string in zip(
            generated_outputs, batch["pixel_values"], gt_detection_strings, strict=True
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

            if objects:
                print(f"[{image_index:05d}] Predicted    {detection_string}")
                print(f"[{image_index:05d}] Ground Truth {gt_detection_string}")

                pred_boxes = np.array([o["xyxy"] for o in objects], float).reshape(-1, 4)
                gt_boxes = np.array([o["xyxy"] for o in gt_objects], float).reshape(-1, 4)

                if len(gt_boxes):
                    iou = box_iou(pred_boxes, gt_boxes)
                    r, c = linear_sum_assignment(iou, maximize=True)
                    matched = iou[r, c]                     # ← 別名にする

                    total_iou += matched.sum()
                    total_pairs += len(matched)

                    print(f"[{image_index:05d}] IoU {np.round(matched, 3).tolist()}  "
                          f"pred={len(pred_boxes)} gt={len(gt_boxes)}")
                else:
                    print(f"[{image_index:05d}] GT なし: {len(pred_boxes)} 件すべて FP")
            else:
                print(f"[{image_index:05d}] No bbox found (gt={len(gt_objects)})")

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

    if total_pairs > 0:
        mean_iou = total_iou / total_pairs
        print(f"[INFO] mean IoU over {total_pairs} pairs: {mean_iou:.3f}")