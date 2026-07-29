import torch


def collate_fn(examples, image_title, prompt, suffix_title, processor, device, train):
    images = [example[image_title].convert("RGB") for example in examples]

    # add <image> token to the prompt
    prompt = ["<image> " + prompt for _ in examples]
    if train:
        suffix = [example[suffix_title] for example in examples]
    # if not training, not allowed to cheat by using the ground truth labels
    else:
        suffix = None

    # Help from: https://github.com/huggingface/transformers/issues/30987
    # Processor put the images and text together, and returns a dictionary of tensors
    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
        padding="longest",
    )

    inputs = inputs.to(torch.bfloat16).to(device)
    inputs[suffix_title] = [example[suffix_title] for example in examples]
    return inputs
