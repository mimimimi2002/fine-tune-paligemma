import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from configs import object_detection_config

from paligemma_ft.data_utis import collate_fn
from paligemma_ft.model_utils import freeze_layers
from functools import partial
from matplotlib import pyplot as plt, patches
import os
import argparse

DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)


def model_inputs(batch, keep_labels=True):
    """collate_fn keeps the raw label strings on the batch so that the evaluation can
    read them back, but the model only takes its own tensor inputs. `generate` also
    rejects `labels`, so it is dropped for the inference call."""
    drop = {"label_for_paligemma"}
    if not keep_labels:
        drop.add("labels")
    return {key: value for key, value in batch.items() if key not in drop}


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


def draw_bbox(image, objects):
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
        plt.text(
            bbox[0], bbox[1] - 10, "plate", color="red", fontsize=12, weight="bold"
        )
    plt.show()


def infer_on_model(model, test_batch, before_pt=True):
    # hardcoding the index to get same before and after results
    index = 0

    # help from : https://discuss.huggingface.co/t/vitimageprocessor-output-visualization/76335/6
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std

    pixel_value = test_batch["pixel_values"][index].cpu().to(torch.float32)

    unnormalized_image = (
        pixel_value.numpy() * np.array(std)[:, None, None]
    ) + np.array(mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

    with torch.inference_mode():
        generated_outputs = model.generate(
            **model_inputs(test_batch, keep_labels=False),
            max_new_tokens=100,
            do_sample=False,
        )
        generated_outputs = processor.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

    if before_pt:
        # generation of the pre trained model
        for element in generated_outputs:
            location = element.split("\n")[1]
            if location == "":
                print("No bbox found")
            else:
                print(location)
    else:
        # generation of the fine tuned model
        element = generated_outputs[index]
        detection_string = element.split("\n")[1]
        objects = extract_objects(detection_string, 224, 224, unique_labels=False)
        draw_bbox(unnormalized_image, objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    # get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load the dataset
    print(f"[INFO] loading {object_detection_config.DATASET_ID} from hub...")
    train_dataset = load_dataset(object_detection_config.DATASET_ID, split="train")
    test_dataset = load_dataset(object_detection_config.DATASET_ID, split="test")
    print(f"[INFO] {len(train_dataset)=}")
    print(f"[INFO] {len(test_dataset)=}")

    # get the processor
    print(f"[INFO] loading {object_detection_config.MODEL_ID} processor from hub...")
    if args.resume:
        processor = AutoProcessor.from_pretrained(args.resume)
    else:
        processor = AutoProcessor.from_pretrained(
            object_detection_config.MODEL_ID
        )

    # build the data loaders
    print("[INFO] building the data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="Detect license plate.",
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=object_detection_config.BATCH_SIZE,
        shuffle=True,
    )
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

    # load the pre trained model
    print(f"[INFO] loading {object_detection_config.MODEL_ID} model...")
    if args.resume:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.resume,
            torch_dtype=object_detection_config.MODEL_DTYPE,
            device_map=device,
        )
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            object_detection_config.MODEL_ID,
            torch_dtype=object_detection_config.MODEL_DTYPE,
            device_map=device,
            revision=object_detection_config.MODEL_REVISION,
        )

    # freeze the weights
    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, not_to_freeze="attn")

    # fine tune the model
    print("[INFO] fine tuning the model...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=object_detection_config.LEARNING_RATE
    )
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(
            os.path.join(args.resume, "training_state.pt"),
            map_location="cpu"
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"]
        print(f"Resume from epoch {start_epoch}")

    # run model generation before fine tuning only if not resuming from a checkpoint
    test_batch = next(iter(test_dataloader))
    if not args.resume:
        # run model generation before fine tuning
        infer_on_model(model, test_batch)

    if args.resume:
        # restore the RNG state as the very last step, so that nothing above
        # (model loading, the sample generation) consumes the restored stream
        cpu_rng_state = checkpoint.get("cpu_rng_state")
        if cpu_rng_state is None:
            print("[WARN] no RNG state in the checkpoint, the shuffling order will differ")
        else:
            torch.set_rng_state(cpu_rng_state)

            cuda_rng_state = checkpoint.get("cuda_rng_state")
            if cuda_rng_state is not None and torch.cuda.is_available():
                if len(cuda_rng_state) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_rng_state)
                else:
                    print("[WARN] skipping the CUDA RNG state, the device count changed")

    for epoch in range(start_epoch, object_detection_config.EPOCHS):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**model_inputs(batch))
            loss = outputs.loss
            if idx % 500 == 0:
                print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % object_detection_config.SAVE_EPOCH == 0:
            save_dir = f"./checkpoints/epoch-{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)

            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                os.path.join(save_dir, "training_state.pt")
            )

    # run model generation after fine tuning
    infer_on_model(model, test_batch, before_pt=False)
