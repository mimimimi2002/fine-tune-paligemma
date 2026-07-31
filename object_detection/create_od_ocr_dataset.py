from datasets import load_dataset
from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR

def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]


def convert_to_detection_string(bboxs, image_width, image_height):
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for bbox in bboxs:
        x1, y1, x2, y2 = coco_to_xyxy(bbox)
        name = "plate"
        locs = [
            format_location(y1, image_height),
            format_location(x1, image_width),
            format_location(y2, image_height),
            format_location(x2, image_width),
        ]
        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)


def format_objects(example):
    height = example["height"]
    width = example["width"]
    bboxs = example["objects"]["bbox"]
    formatted_objects = convert_to_detection_string(bboxs, width, height)
    return {"label_for_paligemma": formatted_objects}

def format_objects_with_ocr(example):
    height = example["height"]
    width = example["width"]
    bboxs = example["objects"]["bbox"]
    formatted_objects = convert_to_detection_string(bboxs, width, height)

    # Perform OCR on the image
    image_path = example['image']
    image = np.array(image_path)
    ocr_results = ocr.ocr(image)

    # Extract text from OCR results
    ocr_texts = []
    if ocr_results is not None:
        for line in ocr_results:
            if line is not None:  # Check if the line has text
                for word_info in line:
                    text = word_info[1][0]  # The recognized text
                    ocr_texts.append(text)
            
    # Combine the formatted objects and OCR texts into a single string
    combined_label = formatted_objects + " ; " + " ; ".join(ocr_texts)
    
    return {"label_for_paligemma": combined_label}


if __name__ == "__main__":
    # load the dataset
    dataset_id = "keremberke/license-plate-object-detection"
    print(f"[INFO] loading {dataset_id} from hub...")
    dataset = load_dataset("keremberke/license-plate-object-detection", "full")

    # modify the coco bbox format
    dataset["train"] = dataset["train"].map(format_objects_with_ocr)
    dataset["validation"] = dataset["validation"].map(format_objects_with_ocr)
    dataset["test"] = dataset["test"].map(format_objects_with_ocr)

    # push to hub
    dataset.push_to_hub("license-detection-paligemma")
