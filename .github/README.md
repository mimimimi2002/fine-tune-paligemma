# Fine Tuning PaliGemma — Object Detection

This repository is a fork of [ariG23498/fine-tune-paligemma](https://github.com/ariG23498/fine-tune-paligemma).

Work in this fork is limited to the **object detection** task (license plate detection): the dataset
was rebuilt, and checkpoint saving and
resumable training were added.

PaliGemma (PG) is a family of Vision Language Models from Google. It uses
SigLIP as the vision encoder, and the Gemma family of models as its language counterpart.

## TODO

- [x] **Dataset** — rebuild the license plate dataset and push it to our own Hugging Face account
- [x] **`<image>` token** — prepend it to the prompt, so that recent `transformers` versions work
- [x] **Checkpoint saving** — write the model, the processor, the optimizer state and the RNG state
  every `SAVE_EPOCH` epochs
- [x] **Resumable training** — continue an interrupted run from a checkpoint with `--resume`
- [x] **Inference script** — `predict.py` runs the fine tuned model over the test set and writes the
  annotated images
- [x] **Evaluation metrics** — `evaluation.py` reports mean IoU, precision / recall / F1 and
  COCO style AP over the test split, and draws the ground truth next to every prediction
- [x] **Fine tuning: attention layers only** — freeze every weight except the attention ones, with
  `freeze_layers(model, not_to_freeze="attn")`
- [ ] **Fine tuning: LoRA** — train low rank adapters instead, and compare the result against the
  attention only run
- [ ] **License plate number recognition (OCR)** — the goal is to extend the task so that the model
  reads the characters on the plate as well, and outputs the plate number together with its
  bounding box

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
```

### Run script
```
# 1. convert the dataset and push it to the Hub
python -m object_detection.create_od_dataset

# 2. fine tune
python -m object_detection.object_detection_ft

# 3. resume from a checkpoint
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-10

# 4. evaluate a checkpoint on the test split
python -m object_detection.evaluation --checkpoint ./checkpoints/epoch-100
```

See the [object detection readme](../object_detection/README.md) for more details.

## Changes in this fork

### Dataset

We use the license plate dataset from Hugging Face
([`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)),
which consists of `image_id`, `image`, `width`, `height` and `objects` (`id`, `area`, `bbox`,
`category`). `object_detection/create_od_dataset.py` converts the COCO style bounding boxes into
the `<locYYYY>` detection string format PaliGemma is trained on, and pushes the result to the Hub.

The converted dataset is pushed to our own account, so `DATASET_ID` points at
[`mimimimi2002/license-detection-paligemma`](https://huggingface.co/datasets/mimimimi2002/license-detection-paligemma)
instead of the upstream copy.

### `<image>` token in the prompt

`paligemma_ft/data_utis.py` now prepends the `<image>` token to the prompt:

```python
prompt = ["<image> " + prompt for _ in examples]
```

Recent versions of `transformers` expect the image token to be present in the text explicitly
rather than inserting it inside the processor.

### Checkpoint saving

`object_detection/object_detection_ft.py` writes a checkpoint every `SAVE_EPOCH` epochs to
`./checkpoints/epoch-<N>/` (relative to the working directory), containing:

- the model weights (`model.save_pretrained`)
- the processor (`processor.save_pretrained`)
- `training_state.pt` — the number of completed epochs, the AdamW optimizer state, and the CPU and
  CUDA RNG states

### Resuming training

```bash
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-10
```

When `--resume` is given, the model and the processor are loaded from the checkpoint directory
instead of the Hub, the optimizer state is restored (and moved onto the training device), and
training continues from the next epoch.

The RNG state is restored as well, so the data shuffling order of the remaining epochs matches an
uninterrupted run.

## Training
### Configuration

`configs/object_detection_config.py`:

| Setting | Upstream | This fork |
|---|---|---|
| `DATASET_ID` | `ariG23498/license-detection-paligemma` | `mimimimi2002/license-detection-paligemma` |
| `BATCH_SIZE` | 8 | 4 |
| `EPOCHS` | 1 | 100 |
| `SAVE_EPOCH` | – | 10 |

`BATCH_SIZE` was lowered to fit the GPU used for these runs, and `EPOCHS` was raised so that the
checkpoint and resume flow is actually exercised.

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

`label accuracy` (the generated label is always `license plate`) and `exact match rate` (all four
`<locYYYY>` tokens identical to the ground truth) are format sanity checks rather than performance
numbers — with coordinates quantised into 1024 buckets an exact string match is near impossible.

See the [object detection readme](../object_detection/README.md#metrics) for how each metric is
computed.

### Test image



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
