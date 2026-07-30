import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from configs import object_detection_config
from PIL import Image

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
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image file",
        required=True,
    )
    parser.add_argument("--output_path", type=str, required=True,
                            help="Path to the output file")
    args = parser.parse_args()

    # get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    image_input = Image.open(args.image_path).convert("RGB")

    # get the processor
    print(f"[INFO] loading processor from {args.checkpoint}...")
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    # load the fine tuned model
    print(f"[INFO] loading model from {args.checkpoint}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=object_detection_config.MODEL_DTYPE,
        device_map=device,
    )
    model.eval()

    print(f"[INFO] running inference on the image {args.image_path}...")

    # Create a simple batch for the single image
    inputs = processor(
        text="<image> " + "Detect license plate.",
        images=image_input,
        return_tensors="pt"
    )
    
    inputs = inputs.to(torch.bfloat16).to(device)
    
    with torch.inference_mode():
        generated_outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )
    generated_outputs = processor.batch_decode(
        generated_outputs, skip_special_tokens=True
    )

    for element, pixel_values in zip(
        generated_outputs, inputs["pixel_values"], strict=True
    ):
        image = unnormalize_image(pixel_values, processor)
        image_height, image_width = image.shape[:2]

        detection_string = get_detection_string(element)
        objects = extract_objects(
            detection_string, image_width, image_height, unique_labels=False
        )

        if objects:
            print(f"Predicted {detection_string}")

            pred_boxes = np.array([o["xyxy"] for o in objects], float).reshape(-1, 4)

        save_path = args.output_path
        draw_bbox(image, objects, save_path=save_path)