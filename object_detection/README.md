# Object Detection with PaliGemma

This folder contains code for creating dataset and fine-tuning the PaliGemma vision-language model for license plate detection.

## Overview

The folder consists of three main components:
1. Dataset preparation `create_od_dataset.py`
2. Model fine-tuning and inference `object_detection_ft.py`, and also an interactive notebook `license_plate_detection_paligemma_ft.ipynb`
3. Evaluation on the test split `evaluation.py`

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

3. Evaluate a checkpoint on the test split:
```bash
python -m object_detection.evaluation --checkpoint ./checkpoints/epoch-100
```

## Evaluation

`evaluation.py` runs greedy generation over the whole `test` split, parses the `<locYYYY>` detection
strings back into pixel boxes, writes an annotated image per sample and prints a metric summary.

```bash
python -m object_detection.evaluation \
    --checkpoint ./checkpoints/epoch-100 \
    --output-dir ./predictions \
    --metrics-json ./predictions/metrics.json
```

| flag | meaning |
| --- | --- |
| `--checkpoint` | directory the model and the processor are loaded from (required) |
| `--output-dir` | where the annotated images are written, one PNG per sample (default `./predictions`) |
| `--limit` | stop after this many images, useful for a quick smoke test |
| `--show` | display each image instead of writing it to `--output-dir` |
| `--metrics-json` | also dump the summary metrics to this JSON file |

### Metrics

Predictions and ground truth boxes are paired per image with a one-to-one assignment that maximises
the total IoU (`scipy.optimize.linear_sum_assignment`), so a prediction can never be credited to two
ground truth boxes. `min(#pred, #gt)` pairs are produced per image; leftover predictions count as
false positives and leftover ground truth boxes as false negatives.

| metric | how it is computed | what it measures |
| --- | --- | --- |
| `mean_iou_matched` | mean IoU over all matched pairs | how tight the boxes are **when** something was found; ignores misses and extra boxes |
| `precision@t` | `TP / (TP + FP)` where a pair counts as TP if its IoU ≥ `t` | of the boxes the model emitted, how many are real |
| `recall@t` | `TP / (TP + FN)` | of the real plates, how many were found |
| `f1@t` | harmonic mean of the two | single number balancing misses against false alarms |
| `ap@t` | rank every detection of the run by confidence, greedily match each one to a still free ground truth box of the same image, then average the precision-recall curve at 101 recall levels (COCO protocol) | performance across all confidence thresholds, not just one operating point |
| `map@[0.5:0.95]` | mean of `ap@t` for `t` = 0.50, 0.55, …, 0.95 | the standard COCO headline number, comparable with published detectors |
| `label_accuracy` | fraction of matched pairs whose label string is identical | whether the generated class name is right, not only the box |
| `exact_match_rate` | fraction of images whose whole detection string equals the ground truth string | strictest possible check — every one of the four `<locYYYY>` tokens must match exactly |

PaliGemma emits no per-box confidence, so `ap@t` uses the mean probability of the generated tokens as
a sequence level confidence and assigns it to every box decoded from that sequence. Boxes from the
same image therefore tie, and their relative order inside the ranking is arbitrary — the AP numbers
are slightly pessimistic because of that.

### Results

One full pass over the `test` split (882 images, 902 ground truth boxes) of the fine tuned model:

| metric | value |
| --- | --- |
| images / predictions / ground truths | 882 / 878 / 902 |
| matched pairs | 876 |
| `mean_iou_matched` | 0.797 |
| `precision@0.5` / `recall@0.5` / `f1@0.5` | 0.962 / 0.937 / 0.949 |
| `tp@0.5` / `fp@0.5` / `fn@0.5` | 845 / 33 / 57 |
| `precision@0.75` / `recall@0.75` / `f1@0.75` | 0.744 / 0.724 / 0.734 |
| `tp@0.75` / `fp@0.75` / `fn@0.75` | 653 / 225 / 249 |
| `ap@0.5` | 0.922 |
| `ap@0.75` | 0.575 |
| `map@[0.5:0.95]` | 0.539 |
| `label_accuracy` | 1.000 |
| `exact_match_rate` | 0.002 |

Reading the numbers:

- **Finding the plate is solved, localising it precisely is not.** At the loose IoU 0.5 threshold the
  model is at 0.949 F1, but at IoU 0.75 it drops to 0.734 — the same 192 boxes move from TP to FP/FN.
  `mean_iou_matched` of 0.797 says the average box is close but not tight, which is exactly the gap
  `map@[0.5:0.95]` = 0.539 reflects.
- **Almost every image gets exactly one prediction** (878 predictions over 882 images), while the
  ground truth has 902 boxes. The model rarely hallucinates a second plate, but it also misses the
  extra plate on multi-plate images, which is most of the 26 unmatched ground truth boxes.
- **`label_accuracy` of 1.000** means the generated label is always `license plate`; the format never
  degrades. It is not an informative metric on this single class dataset.
- **`exact_match_rate` of 0.002 is expected, not a bug.** Coordinates are quantised into 1024 buckets,
  so all four location tokens landing on the ground truth value at once is near impossible. It is
  useful only as a sanity check that the output format is being reproduced.
- `ap@0.75` (0.575) sits below `f1@0.75` (0.734) partly for the confidence tie reason above: with no
  per-box score the ranking cannot put the accurate boxes first, which is what AP rewards.

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
