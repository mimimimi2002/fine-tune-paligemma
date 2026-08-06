# Object Detection with PaliGemma

This folder contains code for creating dataset and fine-tuning the PaliGemma vision-language model for license plate detection.

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
  - [OCR dataset](#ocr-dataset)
- [Usage](#usage)
- [Evaluation](#evaluation)
  - [Metrics](#metrics)
    - [On the OCR dataset](#on-the-ocr-dataset)
  - [Results](#results)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Early stopping and best checkpoint](#early-stopping-and-best-checkpoint)
- [Seeding](#seeding)
- [Configuration](#configuration)

## Overview

Two variants of the task are implemented, each with its own dataset builder, training script and
evaluation script:

| step | detection only | detection + OCR |
| --- | --- | --- |
| dataset | `create_od_dataset.py` | `create_od_ocr_dataset.py` |
| fine tuning | `object_detection_ft.py` | `object_detection_ocr_ft.py` |
| evaluation | `evaluation.py` | `evaluation_ocr.py` |
| config | `configs/object_detection_config.py` | `configs/object_detection_ocr_config.py` |
| checkpoints | `./checkpoints/epoch-<N>/`, `./checkpoints/best/` | `./checkpoints/ocr/epoch-<N>/`, `./checkpoints/ocr/best/` |

The two training scripts are the same code on two configs: checkpointing, resuming, early stopping
and seeding all apply to both.

Shared on top of that:
- `predict.py` — run a checkpoint on a single image file
- `license_plate_detection_paligemma_ft.ipynb` — interactive version of the detection only flow

The detection only model outputs a box labelled `plate`; the OCR model outputs the box together with
the characters on the plate, e.g. `<loc0512><loc0330><loc0570><loc0450> plate ABC1234`.

## Dataset

The project uses the license plate detection dataset from Hugging Face Hub:
- Source dataset: [`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
- Processed dataset: [`mimimimi2002/license-detection-paligemma`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma)
- Processed dataset with plate numbers: [`mimimimi2002/license-detection-paligemma-ocr`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma-ocr)

`create_od_dataset.py` converts the COCO format annotations into the `<locYYYY>` detection string
format PaliGemma is trained on, and pushes the result to the Hub.

The source is the Vehicle Registration Plates Dataset, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Both processed datasets are derivatives of
it — the images are unchanged, only the annotations are reformatted (and extended with plate numbers
for the OCR variant) — so the same attribution applies to them.

### OCR dataset

`create_od_ocr_dataset.py` does the same conversion and additionally reads the plate number, so that
the label trains detection and recognition at once.

For every ground truth box the image is cropped to that box and passed to
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (`use_angle_cls=True, lang="en"`). Text lines
scoring below `MIN_OCR_CONFIDENCE` (0.5) are discarded, the rest are joined with a space and appended
to the label:

```
<loc0512><loc0330><loc0570><loc0450> plate ABC1234 ; <loc0700><loc0120><loc0740><loc0210> plate
```

The second box in that example is what happens when the OCR reads nothing above the threshold: the
label falls back to the bare `plate`, so the box still works as a detection example.

Building this dataset needs `paddleocr` and `paddlepaddle` (both in `requirements.txt`); training and
evaluation do not.

**Caveat:** the plate numbers are auto labelled and never checked by a human, so PaddleOCR errors end
up in the training data and in the evaluation ground truth.

## Usage

Detection only:

```bash
# 1. prepare the dataset
python -m object_detection.create_od_dataset

# 2. fine tune, with early stopping on the validation split
python -m object_detection.object_detection_ft

# 3. evaluate the best checkpoint on the test split
python -m object_detection.evaluation --checkpoint ./checkpoints/best
```

Detection + OCR:

```bash
# 1. prepare the dataset (labels the plate numbers with PaddleOCR)
python -m object_detection.create_od_ocr_dataset

# 2. fine tune, with early stopping on the validation split
python -m object_detection.object_detection_ocr_ft

# 3. evaluate the best checkpoint on the test split
python -m object_detection.evaluation_ocr --checkpoint ./checkpoints/ocr/best
```

Single image:

```bash
python -m object_detection.predict \
    --checkpoint ./checkpoints/best \
    --image_path ./sample.jpg \
    --output_path ./sample_pred.png
```

`predict.py` loads one image, runs greedy generation, draws the predicted boxes and writes the result
to `--output_path`. It also prints the total time of the run, which includes loading the processor
and the model.

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

`evaluation_ocr.py` takes the same flags; only the config it reads (dataset, prompt) and the default
`--output-dir` (`./predictions/ocr`) differ.

Both scripts time every `generate()` call and print the wall clock cost of the batch and of a single
image, with `torch.cuda.synchronize()` on both sides of the timer when running on GPU:

```
Batch inference: 6.214 sec
Per image: 1.554 sec/image
```

Measured over the `test` split, in seconds per image:

| variant | CPU | GPU |
| --- | --- | --- |
| detection only | 10.343 | 0.094 |
| detection + OCR | 12.298 | 0.592 |

The OCR model has to generate the plate characters after the four location tokens, and generation is
sequential, so it pays for the longer output — a 19% overhead on CPU but 6.3x on GPU, where the per
token compute no longer hides it.

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

#### On the OCR dataset

The metric code is identical, but the label is now `plate <number>` instead of `plate`, which changes
what two of the metrics mean:

- `label_accuracy` becomes the **plate number recognition accuracy** over the matched pairs — a pair
  only counts when the model read the same characters as the ground truth. On the detection only
  dataset it is 1.000 by construction and carries no information.
- `exact_match_rate` now requires the four location tokens *and* the plate number to be identical, so
  it stays a format sanity check rather than a performance number.

The box metrics (`mean_iou_matched`, precision / recall / F1, AP, mAP) do not look at the label at
all and stay directly comparable between the two runs.

Because the ground truth plate numbers come from PaddleOCR, `label_accuracy` measures agreement with
PaddleOCR, not with the true plate.

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
- **`label_accuracy` of 1.000** means the generated label is always `plate`; the format never
  degrades. It is not an informative metric on this single class dataset — it becomes one on the OCR
  dataset, where the label carries the plate number.
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

The early stopping state is stored in the same `training_state.pt`. `object_detection_ocr_ft.py`
writes to `./checkpoints/ocr/epoch-<N>/` instead.

## Early stopping and best checkpoint

Both training scripts also load the `validation` split, with `train=True` in the collate function
so that the suffix is tokenized into `labels` and the validation loss is computed exactly like the
training loss. Every `EVAL_EPOCH` epochs `evaluate_loss()` runs the model over that split under
`torch.inference_mode()` and returns the mean loss weighted by the number of samples per batch (the
model is put back into whichever mode it was in afterwards).

An evaluation counts as an improvement when the loss falls more than `EARLY_STOPPING_MIN_DELTA` below
the best loss so far. If it does not, a counter is incremented, and the run stops once it reaches
`EARLY_STOPPING_PATIENCE`:

```
Epoch: 12 Validation loss: 0.1842 (best 0.1791, 3/5 evaluations without improvement)
```

Every time the loss improves the weights are saved to `./checkpoints/best/` (`./checkpoints/ocr/best/`
for the OCR run), separately from the periodic `epoch-<N>` checkpoints — those hold the latest state,
which after the model starts overfitting is worse than the best one. `best/` is therefore what the
evaluation scripts should be pointed at.

When the run stops early a checkpoint is written unconditionally as well, so the final state is not
lost if it falls between two `SAVE_EPOCH` boundaries.

`best_validation_loss` and `evaluations_without_improvement` are part of `training_state.pt`, so
`--resume` picks the patience counter back up instead of restarting it. Checkpoints written before
this existed carry neither key and start the counter over.

## Seeding

`set_seed(SEED)` seeds `random`, `numpy` and `torch` (CPU and all CUDA devices) before anything
random happens, so weight initialisation, dropout and the shuffling order repeat across runs of the
same config.

`DataLoader(shuffle=True)` draws its sampler seed from the global torch RNG, so seeding that single
stream is enough to fix the epoch order — and it is the same stream `--resume` restores from the
checkpoint. The seeding call runs before the resume path, so when a checkpoint is given its RNG state
takes over.

## Configuration

Key parameters can be modified in `configs/object_detection_config.py`:
- BATCH_SIZE
- LEARNING_RATE
- EPOCHS
- SAVE_EPOCH — how often a checkpoint is written, in epochs
- MODEL_ID
- DATASET_ID
- PROMPT
- SEED — 42
- EVAL_EPOCH — how often the validation loss is computed, in epochs
- EARLY_STOPPING_PATIENCE — evaluations without improvement before the run stops
- EARLY_STOPPING_MIN_DELTA — how much the loss has to drop to count as an improvement

With early stopping enabled, `EPOCHS` acts as an upper bound rather than the length of the run.

`configs/object_detection_ocr_config.py` holds the same keys for the OCR run, with the OCR dataset
and the prompt `Detect license plate and read its number.`
