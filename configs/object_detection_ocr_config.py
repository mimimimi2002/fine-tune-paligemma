import torch

DATASET_ID = "mimimimi2002/license-detection-paligemma-ocr"
MODEL_ID = "google/paligemma-3b-pt-224"
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MODEL_DTYPE = torch.bfloat16
MODEL_REVISION = "bfloat16"
EPOCHS = 100
SAVE_EPOCH = 10
PROMPT = "Detect license plate and read its number."
