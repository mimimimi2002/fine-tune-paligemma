# Object Detection with PaliGemma

This folder contains code for creating dataset and fine-tuning the PaliGemma vision-language model for license plate detection.

## Overview

The folder consists of two main components:
1. Dataset preparation `create_od_dataset.py`
2. Model fine-tuning and inference `object_detection_ft.py`, and also an interactive notebook `license_plate_detection_paligemma_ft.ipynb`

## Dataset

The project uses the license plate detection dataset from Hugging Face Hub:
- Source dataset: [`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
- Processed dataset: [`mimimimi2002/license-detection-paligemma`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma)

`create_od_dataset.py` converts the COCO format annotations into the `<locYYYY>` detection string
format PaliGemma is trained on, and pushes the result to the Hub.

## Usage

1. First, prepare the dataset:
```bash
python -m object_detection.create_od_dataset
```

2. Run the fine-tuning:
```bash
python -m object_detection.object_detection_ft
```

## Checkpoints and resuming

Every `SAVE_EPOCH` epochs the training script writes a checkpoint to `./checkpoints/epoch-<N>/`
(relative to the working directory), containing:

- the model weights (`model.save_pretrained`)
- the processor (`processor.save_pretrained`)
- `training_state.pt` — the number of completed epochs, the AdamW optimizer state, and the CPU and
  CUDA RNG states

To continue an interrupted run, pass the checkpoint directory to `--resume`:

```bash
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-3
```

The model and the processor are then loaded from that directory instead of the Hub, the optimizer
state is restored (and moved onto the training device), and training continues from the next epoch.
The "before fine tuning" sample inference is skipped.

## Configuration

Key parameters can be modified in `configs/object_detection_config.py`:
- BATCH_SIZE
- LEARNING_RATE
- EPOCHS
- SAVE_EPOCH — how often a checkpoint is written, in epochs
- MODEL_ID
- DATASET_ID
