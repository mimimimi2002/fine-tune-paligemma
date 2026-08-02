# Fine Tuning PaliGemma — Object Detection

This repository is a fork of [ariG23498/fine-tune-paligemma](https://github.com/ariG23498/fine-tune-paligemma).

Work in this fork is limited to the **object detection** task on license plates. Two variants are
implemented:

- **detection only** — the model outputs a bounding box labelled `plate`
- **detection + OCR** — the model outputs a bounding box *and* the characters written on the plate,
  e.g. `<loc0512><loc0330><loc0570><loc0450> plate ABC1234`

Around both of them the dataset was rebuilt, checkpointing / resumable training, early stopping,
seeding and a full evaluation suite were added.

PaliGemma (PG) is a family of Vision Language Models from Google. It uses
SigLIP as the vision encoder, and the Gemma family of models as its language counterpart.

## Contents

- [TODO](#todo)
- [Pretrained Model](#pretrained-model)
- [Usage](#usage)
  - [Set up env](#set-up-env)
  - [Run script](#run-script)
- [Changes in this fork](#changes-in-this-fork)
  - [Common to both variants](#common-to-both-variants)
    - [Dataset](#dataset)
    - [`<image>` token in the prompt](#image-token-in-the-prompt)
    - [Checkpoint saving](#checkpoint-saving)
    - [Resuming training](#resuming-training)
    - [Early stopping and best checkpoint](#early-stopping-and-best-checkpoint)
    - [Fixed seed](#fixed-seed)
    - [Inference time](#inference-time)
  - [OCR only](#ocr-only)
    - [OCR dataset](#ocr-dataset)
    - [OCR fine tuning](#ocr-fine-tuning)
- [Training](#training)
  - [Configuration](#configuration)
  - [Training process](#training-process)
  - [Evaluation](#evaluation)
  - [OCR evaluation](#ocr-evaluation)
  - [Test image](#test-image)
- [Citation](#citation)

## TODO

- [x] **Dataset** — rebuild the license plate dataset and push it to our own Hugging Face account
- [x] **`<image>` token** — prepend it to the prompt, so that recent `transformers` versions work
- [x] **Checkpoint saving** — write the model, the processor, the optimizer state and the RNG state
  every `SAVE_EPOCH` epochs
- [x] **Resumable training** — continue an interrupted run from a checkpoint with `--resume`
- [x] **Evaluation metrics** — `evaluation.py` reports mean IoU, precision / recall / F1 and
  COCO style AP over the test split, and draws the ground truth next to every prediction
- [x] **Single image inference** — `predict.py` runs the fine tuned model on one image file and
  writes the annotated result
- [x] **Fine tuning: attention layers only** — freeze every weight except the attention ones, with
  `freeze_layers(model, not_to_freeze="attn")`
- [x] **License plate number recognition (OCR)** — the plate characters are auto labelled with
  PaddleOCR (`create_od_ocr_dataset.py`), and the model is fine tuned to output the plate number
  together with its bounding box (`object_detection_ocr_ft.py`, `evaluation_ocr.py`)
- [x] **Early stopping and best checkpoint** — the validation loss is evaluated every `EVAL_EPOCH`
  epochs, the best weights are kept separately and the run stops after
  `EARLY_STOPPING_PATIENCE` evaluations without improvement
- [x] **Fixed seed** — `SEED` seeds `random`, `numpy` and `torch` (CPU and CUDA) so a run is
  reproducible
- [x] **Inference time measurement** — the evaluation scripts report the wall clock time per batch
  and per image

## Pretrained Model

The pretrained model fine tuned in this project is
[`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224), loaded from its
`bfloat16` revision.

- Parameters: 3 billion
- Input image size: 224 x 224 pixels

## Usage
### Set up env
```
conda create -n paligemma python=3.10
conda activate paligemma
pip install -r requirements.txt
```

PaliGemma is a gated model, so accept the license on the model page and log in before running the
scripts:

```
huggingface-cli login
hf auth login
```

### Run script

Detection only:

```
# 1. convert the dataset and push it to the Hub
python -m object_detection.create_od_dataset

# 2. fine tune (early stopping on the validation split)
python -m object_detection.object_detection_ft

# 3. resume from a checkpoint
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-10

# 4. evaluate the best checkpoint on the test split
python -m object_detection.evaluation --checkpoint ./checkpoints/best

# 5. run the model on a single image
python -m object_detection.predict \
    --checkpoint ./checkpoints/best \
    --image_path ./sample.jpg \
    --output_path ./sample_pred.png
```

Detection + OCR — same flow, with the `_ocr` scripts:

```
# 1. label the plate numbers with PaddleOCR and push the dataset to the Hub
python -m object_detection.create_od_ocr_dataset

# 2. fine tune (early stopping on the validation split)
python -m object_detection.object_detection_ocr_ft

# 3. resume from a checkpoint
python -m object_detection.object_detection_ocr_ft --resume ./checkpoints/ocr/epoch-10

# 4. evaluate the best checkpoint on the test split
python -m object_detection.evaluation_ocr --checkpoint ./checkpoints/ocr/best
```

See the [object detection readme](../object_detection/README.md) for more details.

## Changes in this fork

The changes split into two groups: those that apply to both variants of the task, and those that
only exist for the detection + OCR variant.

### Common to both variants

#### Dataset

We use the license plate dataset from Hugging Face
([`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)),
which consists of `image_id`, `image`, `width`, `height` and `objects` (`id`, `area`, `bbox`,
`category`). `object_detection/create_od_dataset.py` converts the COCO style bounding boxes into
the `<locYYYY>` detection string format PaliGemma is trained on, and pushes the result to the Hub.

The converted dataset is pushed to our own account, so `DATASET_ID` points at
[`mimimimi2002/license-detection-paligemma`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma)
instead of the upstream copy.

#### `<image>` token in the prompt

`paligemma_ft/data_utis.py` now prepends the `<image>` token to the prompt:

```python
prompt = ["<image> " + prompt for _ in examples]
```

Recent versions of `transformers` expect the image token to be present in the text explicitly
rather than inserting it inside the processor.

#### Checkpoint saving

`object_detection/object_detection_ft.py` writes a checkpoint every `SAVE_EPOCH` epochs to
`./checkpoints/epoch-<N>/` (relative to the working directory), containing:

- the model weights (`model.save_pretrained`)
- the processor (`processor.save_pretrained`)
- `training_state.pt` — the number of completed epochs, the AdamW optimizer state, and the CPU and
  CUDA RNG states

The early stopping state (`best_validation_loss` and `evaluations_without_improvement`) is stored in
the same file. The OCR script writes to `./checkpoints/ocr/` instead, so that its checkpoints do not
collide with the detection only run.

#### Resuming training

```bash
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-10
```

When `--resume` is given, the model and the processor are loaded from the checkpoint directory
instead of the Hub, the optimizer state is restored (and moved onto the training device), and
training continues from the next epoch.

The RNG state is restored as well, so the data shuffling order of the remaining epochs matches an
uninterrupted run.

#### Early stopping and best checkpoint

Both training scripts load the `validation` split as well, with `train=True` in the collate function
so that the suffix is tokenized into `labels` and the validation loss is computed exactly like the
training loss.

Every `EVAL_EPOCH` epochs `evaluate_loss()` runs the model over that split under
`torch.inference_mode()` and returns the sample weighted mean loss. An evaluation counts as an
improvement when the loss drops by more than `EARLY_STOPPING_MIN_DELTA` below the best loss seen so
far. Otherwise a patience counter is incremented, and the run stops once it reaches
`EARLY_STOPPING_PATIENCE`.

| Setting | Value | Meaning |
|---|---|---|
| `EVAL_EPOCH` | 1 | how often the validation loss is computed, in epochs |
| `EARLY_STOPPING_PATIENCE` | 5 | evaluations without improvement before the run stops |
| `EARLY_STOPPING_MIN_DELTA` | 1e-4 | how much the loss has to drop to count as an improvement |

Whenever the loss improves the weights are written to `./checkpoints/best/`, or
`./checkpoints/ocr/best/` for the OCR run. This is separate from the periodic `epoch-<N>` checkpoints
on purpose: those are simply the latest state, which after overfitting is *worse* than the best one.
When the run stops early a checkpoint is also written unconditionally, so the final state is never
lost between two `SAVE_EPOCH` boundaries.

`training_state.pt` carries `best_validation_loss` and `evaluations_without_improvement` as well, so
`--resume` continues with the same patience counter instead of restarting it. Checkpoints written
before early stopping existed have neither key and simply start the counter over.

#### Fixed seed

`set_seed()` seeds `random`, `numpy` and `torch` (CPU and all CUDA devices) from `SEED` (42) before
anything random happens, so two runs of the same config see the same weight initialisation, dropout
draws and shuffling order.

`DataLoader(shuffle=True)` seeds its own sampler from the global torch RNG, so seeding that one
stream is enough to make the epoch order reproducible — and it is the same stream that `--resume`
restores from the checkpoint. Seeding runs *before* the resume path, so when both apply the restored
RNG state wins.

#### Inference time

Both evaluation scripts time the `generate()` call of every batch and print the per batch and per
image wall clock time. `torch.cuda.synchronize()` is called on both sides of the timer when running
on GPU, otherwise the measurement would only cover the kernel launches:

```
Batch inference: 6.214 sec
Per image: 1.554 sec/image
```

`predict.py` prints one number for the whole single image run, which includes loading the processor
and the model — useful as a cold start figure, not comparable with the per image numbers above.

### OCR only

#### OCR dataset

`object_detection/create_od_ocr_dataset.py` builds a second dataset in which the label carries the
plate number as well, so that the model can be trained on detection **and** recognition at once.

For every ground truth box the image is cropped to that box and read with
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (`use_angle_cls=True, lang="en"`). Text lines
below `MIN_OCR_CONFIDENCE` (0.5) are dropped, the surviving ones are joined with a space, and the
result is appended to the label:

```
<loc0512><loc0330><loc0570><loc0450> plate ABC1234
```

When the OCR reads nothing above the threshold the label falls back to the bare `plate`, so the box
is still usable as a detection example. Multiple boxes are joined with ` ; ` exactly as before.

The result is pushed to
[`mimimimi2002/license-detection-paligemma-ocr`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma-ocr).
Note that the plate numbers are **auto labelled, not human verified** — PaddleOCR mistakes become
training noise, which is the main known limitation of this dataset.

`paddleocr` and `paddlepaddle` were added to `requirements.txt` for this step. They are only needed
to build the dataset, not to train or to evaluate.

#### OCR fine tuning

`object_detection/object_detection_ocr_ft.py` is the training script for the detection + OCR
variant. It is the same script as `object_detection_ft.py` — checkpointing, resuming, early stopping
and seeding included — with two differences:

- it reads `configs/object_detection_ocr_config.py` instead of the detection config, so it trains on
  the OCR dataset with the prompt `"Detect license plate and read its number."`
- checkpoints are written under `./checkpoints/ocr/` so that they do not collide with the detection
  only run

## Training
### Configuration

`configs/object_detection_config.py`:

| Setting | Upstream | This fork |
|---|---|---|
| `DATASET_ID` | `ariG23498/license-detection-paligemma` | `mimimimi2002/license-detection-paligemma` |
| `BATCH_SIZE` | 8 | 4 |
| `EPOCHS` | 1 | 100 |
| `SAVE_EPOCH` | – | 10 |
| `SEED` | – | 42 |
| `EVAL_EPOCH` | – | 1 |
| `EARLY_STOPPING_PATIENCE` | – | 5 |
| `EARLY_STOPPING_MIN_DELTA` | – | 1e-4 |

`BATCH_SIZE` was lowered to fit the GPU used for these runs, and `EPOCHS` was raised so that the
checkpoint and resume flow is actually exercised. With early stopping in place `EPOCHS` is an upper
bound rather than the actual length of the run.

`configs/object_detection_ocr_config.py` holds the same keys, with the OCR dataset and prompt:

| Setting | Value |
|---|---|
| `DATASET_ID` | `mimimimi2002/license-detection-paligemma-ocr` |
| `PROMPT` | `Detect license plate and read its number.` |

### Training process
<img width="1500" height="auto" alt="finetune_loss (1)" src="https://github.com/user-attachments/assets/445b90fd-4303-4895-a46b-d455dc7dbf57" />

### Evaluation

```bash
python -m object_detection.evaluation --checkpoint ./checkpoints/epoch-100
```

`object_detection/evaluation.py` runs greedy generation over the whole `test` split (882 images, 902
ground truth boxes), pairs the predicted boxes with the ground truth ones by maximising the total IoU,
and writes an annotated image per sample (prediction in red, ground truth in dashed green).

| Metric | Value |
|---|---|
| images / predictions / ground truths | 882 / 878 / 902 |
| matched pairs | 876 |
| mean IoU (matched pairs) | 0.797 |
| Precision / Recall / F1 @ IoU 0.5 | 0.962 / 0.937 / 0.949 |
| TP / FP / FN @ IoU 0.5 | 845 / 33 / 57 |
| Precision / Recall / F1 @ IoU 0.75 | 0.744 / 0.724 / 0.734 |
| TP / FP / FN @ IoU 0.75 | 653 / 225 / 249 |
| AP @ IoU 0.5 | 0.922 |
| AP @ IoU 0.75 | 0.575 |
| mAP @ IoU [0.5:0.95] | 0.539 |
| label accuracy | 1.000 |
| exact match rate | 0.002 |

Finding the plate is essentially solved, localising it precisely is not: F1 is 0.949 at the loose IoU
0.5 threshold but 0.734 at IoU 0.75, and the average matched box sits at IoU 0.797. The model emits
almost exactly one box per image (878 boxes over 882 images), so it rarely hallucinates a plate but
does miss the second plate on multi-plate images.

`label accuracy` (the generated label is always `plate`) and `exact match rate` (all four
`<locYYYY>` tokens identical to the ground truth) are format sanity checks rather than performance
numbers — with coordinates quantised into 1024 buckets an exact string match is near impossible.

See the [object detection readme](../object_detection/README.md#metrics) for how each metric is
computed.

### OCR evaluation

```bash
python -m object_detection.evaluation_ocr --checkpoint ./checkpoints/ocr/best
```

`object_detection/evaluation_ocr.py` is the same evaluation, pointed at the OCR dataset and the OCR
prompt. The metrics are computed identically, but two of them change meaning because the label now
contains the plate number:

- **`label_accuracy`** stops being a formality. On the detection dataset the label is always `plate`,
  so it is 1.000 by construction; on the OCR dataset it is `plate <number>`, so a matched pair only
  counts when the model read the characters correctly. It is effectively the recognition accuracy
  over the boxes that were found.
- **`exact_match_rate`** now requires the four `<locYYYY>` tokens *and* the plate number to match, so
  it stays near zero for the same quantisation reason as before.

The box metrics (IoU, precision / recall / F1, AP) are unaffected by the plate number and stay
directly comparable with the detection only run above.

Ground truth caveat: the reference plate numbers come from PaddleOCR, not from a human, so
`label_accuracy` measures agreement with PaddleOCR rather than with the true plate.

### Test image
<img width="425" height="431" alt="00001" src="https://github.com/user-attachments/assets/5029a0ee-4322-4dec-9aa1-934a05b8536b" />

## Citation

This is a fork. If you like the original work and would use it please cite the upstream authors:

```
@misc{github_repository,
  author = {Aritra Roy Gosthipaty, Ritwik Raha}, 
  title = {ft-pali-gemma}, 
  publisher = {{GitHub}(https://github.com)},
  howpublished = {\url{https://github.com/ariG23498/ft-pali-gemma/edit/main/README.md}},
  year = {2024}  
}
```
