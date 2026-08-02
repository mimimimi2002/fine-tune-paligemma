import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset
from configs import object_detection_ocr_config

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


def set_seed(seed):
    """seed every RNG the run draws from. `DataLoader(shuffle=True)` seeds its own
    sampler from the global torch RNG, so seeding that stream is enough to make the
    shuffling order reproducible (and it is what --resume restores)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, dataloader):
    """mean loss over the dataloader, weighted by the number of samples per batch"""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for batch in dataloader:
            outputs = model(**model_inputs(batch))
            batch_size = batch["input_ids"].shape[0]
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

    model.train(was_training)
    return total_loss / total_samples if total_samples else float("nan")


def save_checkpoint(save_dir, model, processor, optimizer, epoch, early_stopping_state):
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            **early_stopping_state,
        },
        os.path.join(save_dir, "training_state.pt"),
    )
    print(f"[INFO] checkpoint written to {save_dir}")


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

    # seed before anything random happens; when resuming, the RNG state saved in the
    # checkpoint is restored further down and takes over from here
    print(f"[INFO] seeding with {object_detection_ocr_config.SEED}...")
    set_seed(object_detection_ocr_config.SEED)

    # load the dataset
    print(f"[INFO] loading {object_detection_ocr_config.DATASET_ID} from hub...")
    train_dataset = load_dataset(object_detection_ocr_config.DATASET_ID, split="train")
    # the validation split drives early stopping, the test split is only used for the
    # before / after sample generation
    validation_dataset = load_dataset(
        object_detection_ocr_config.DATASET_ID, split="validation"
    )
    test_dataset = load_dataset(object_detection_ocr_config.DATASET_ID, split="test")
    print(f"[INFO] {len(train_dataset)=}")
    print(f"[INFO] {len(validation_dataset)=}")
    print(f"[INFO] {len(test_dataset)=}")

    # get the processor
    print(f"[INFO] loading {object_detection_ocr_config.MODEL_ID} processor from hub...")
    if args.resume:
        processor = AutoProcessor.from_pretrained(args.resume)
    else:
        processor = AutoProcessor.from_pretrained(
            object_detection_ocr_config.MODEL_ID
        )

    # build the data loaders
    print("[INFO] building the data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt=object_detection_ocr_config.PROMPT,
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=object_detection_ocr_config.BATCH_SIZE,
        shuffle=True,
    )
    # train=True so that the suffix is tokenized into `labels` and the validation
    # loss can be computed the same way as the training loss
    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt=object_detection_ocr_config.PROMPT,
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=object_detection_ocr_config.BATCH_SIZE,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt=object_detection_ocr_config.PROMPT,
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=False,
        ),
        batch_size=object_detection_ocr_config.BATCH_SIZE,
        shuffle=False,
    )

    # load the pre trained model
    print(f"[INFO] loading {object_detection_ocr_config.MODEL_ID} model...")
    if args.resume:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.resume,
            torch_dtype=object_detection_ocr_config.MODEL_DTYPE,
            device_map=device,
        )
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            object_detection_ocr_config.MODEL_ID,
            torch_dtype=object_detection_ocr_config.MODEL_DTYPE,
            device_map=device,
            revision=object_detection_ocr_config.MODEL_REVISION,
        )

    # freeze the weights
    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, not_to_freeze="attn")

    # fine tune the model
    print("[INFO] fine tuning the model...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=object_detection_ocr_config.LEARNING_RATE
    )
    start_epoch = 0
    best_validation_loss = float("inf")
    evaluations_without_improvement = 0

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
        # a checkpoint written before early stopping existed carries neither key,
        # in that case the patience counter simply starts over
        best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
        evaluations_without_improvement = checkpoint.get(
            "evaluations_without_improvement", 0
        )
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

    model.train()
    stop_early = False

    for epoch in range(start_epoch, object_detection_ocr_config.EPOCHS):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**model_inputs(batch))
            loss = outputs.loss
            if idx % 500 == 0:
                print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        early_stopping_state = dict(
            best_validation_loss=best_validation_loss,
            evaluations_without_improvement=evaluations_without_improvement,
        )

        if (epoch + 1) % object_detection_ocr_config.EVAL_EPOCH == 0:
            validation_loss = evaluate_loss(model, validation_dataloader)
            improved = (
                validation_loss
                < best_validation_loss
                - object_detection_ocr_config.EARLY_STOPPING_MIN_DELTA
            )

            if improved:
                best_validation_loss = validation_loss
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

            early_stopping_state = dict(
                best_validation_loss=best_validation_loss,
                evaluations_without_improvement=evaluations_without_improvement,
            )
            print(
                f"Epoch: {epoch} Validation loss: {validation_loss:.4f} "
                f"(best {best_validation_loss:.4f}, "
                f"{evaluations_without_improvement}/"
                f"{object_detection_ocr_config.EARLY_STOPPING_PATIENCE} "
                f"evaluations without improvement)"
            )

            # the best weights are kept separately, the periodic checkpoints below
            # are overwritten by later, possibly worse, epochs
            if improved:
                save_checkpoint(
                    "./checkpoints/ocr/best",
                    model,
                    processor,
                    optimizer,
                    epoch + 1,
                    early_stopping_state,
                )

            if (
                evaluations_without_improvement
                >= object_detection_ocr_config.EARLY_STOPPING_PATIENCE
            ):
                print(
                    f"[INFO] early stopping at epoch {epoch}: the validation loss did "
                    f"not improve for {evaluations_without_improvement} evaluations"
                )
                stop_early = True

        # always keep the last state when stopping early, otherwise the run would
        # end on an epoch that was never written to disk
        if (epoch + 1) % object_detection_ocr_config.SAVE_EPOCH == 0 or stop_early:
            save_checkpoint(
                f"./checkpoints/ocr/epoch-{epoch+1}",
                model,
                processor,
                optimizer,
                epoch + 1,
                early_stopping_state,
            )

        if stop_early:
            break

    print(f"[INFO] best validation loss: {best_validation_loss:.4f}")
    print("[INFO] the best weights are in ./checkpoints/ocr/best")

    # run model generation after fine tuning
    infer_on_model(model, test_batch, before_pt=False)
