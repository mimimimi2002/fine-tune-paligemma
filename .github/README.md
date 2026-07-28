# Fine Tuning PaliGemma — Object Detection

This repository is a fork of [ariG23498/fine-tune-paligemma](https://github.com/ariG23498/fine-tune-paligemma).

Work in this fork is limited to the **object detection** task (license plate detection): the dataset
was rebuilt, and checkpoint saving and
resumable training were added.

PaliGemma (PG) is a family of Vision Language Models from Google. It uses
SigLIP as the vision encoder, and the Gemma family of models as its language counterpart.

> [!Note]
The upstream authors have covered SigLIP in depth in their blog
[Choosing Between SigLIP and CLIP for Language Image Pretraining](https://blog.ritwikraha.dev/choosing-between-siglip-and-clip-for-language-image-pretraining) if you wanted to read
about it.

Models hosted on Hugging Face Hub:

1. [PaliGemma](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda)
2. [PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)


## Usage
### Set up env
```
conda create -n paligemma python=3.10
conda activate paligemma
pip install -r requirements.txt
```

### Run script
```
# 1. convert the dataset and push it to the Hub
python -m object_detection.create_od_dataset

# 2. fine tune
python -m object_detection.object_detection_ft

# 3. resume from a checkpoint
python -m object_detection.object_detection_ft --resume ./checkpoints/epoch-10
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
