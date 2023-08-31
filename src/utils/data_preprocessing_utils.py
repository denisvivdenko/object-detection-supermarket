from PIL import Image
import albumentations as A
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import os
import logging


def load_dataset(params) -> pd.DataFrame:
    columns = ["image_name", "x1", "y1", "x2", "y2", "class_name", "image_width", "image_height"]
    train_annotations = pd.read_csv(f'{params["yolo_preprocessing"]["unprocessed_dataset_path"]}/annotations/annotations_train.csv', header=None)
    train_annotations.columns = columns
    test_annotations = pd.read_csv(f'{params["yolo_preprocessing"]["unprocessed_dataset_path"]}/annotations/annotations_test.csv', header=None)
    test_annotations.columns = columns
    val_annotations = pd.read_csv(f'{params["yolo_preprocessing"]["unprocessed_dataset_path"]}/annotations/annotations_val.csv', header=None)
    val_annotations.columns = columns

    return {
        "train": train_annotations,
        "test": test_annotations,
        "val": val_annotations
    }


def clear_directory(path: str) -> None:
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Remove files and symbolic links
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove subdirectories


def get_augmentations() -> A.ReplayCompose:
    return A.ReplayCompose([
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.1),
        A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.1),
        A.Resize(height=416, width=416, p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1), p=1)


def convert_boxes_to_yolo_format(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> str:
    """
        Converts the bounding box coordinates to yolo format

        returns: str "x_center_n y_center_n bbox_width_n bbox_height_n"
    """

    x2 = min(x2, image_width)
    y2 = min(y2, image_height)

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    x_center_n = x_center / image_width
    y_center_n = y_center / image_height
    width_n = width / image_width
    height_n = height / image_height

    x_center_n = min(max(x_center_n, 0), 1)
    y_center_n = min(max(y_center_n, 0), 1)
    width_n = min(max(width_n, 0), 1)
    height_n = min(max(height_n, 0), 1)
    
    return {
        "x_center_n": x_center_n,
        "y_center_n": y_center_n,
        "width_n": width_n,
        "height_n": height_n
    } 


def process_image(params, image_metadata: pd.DataFrame, split: str, classes_dict: dict) -> None:
    image_name =  image_metadata["image_name"].values[0]
    image_fpath = Path(params["yolo_preprocessing"]["images_dir"]) / image_name

    new_image_dir = Path(params["yolo_preprocessing"]["data_path"]) / "images" / split
    new_label_dir = Path(params["yolo_preprocessing"]["data_path"]) / "labels" / split

    image = Image.open(image_fpath)
    image = np.array(image)
    bboxes = [
        convert_boxes_to_yolo_format(
            box.x1, 
            box.y1, 
            box.x2, 
            box.y2, 
            box.image_width, 
            box.image_height
        ).values()
        for _, box in image_metadata.iterrows()
    ]
    classes = [classes_dict[box["class_name"]] for _, box in image_metadata.iterrows()]
    base_random_state = params["base"]["random_state"]

    augmentations = [{
        "image": image,
        "bboxes": bboxes,
        "classes": classes,
        f"fname": f"{image_name.split('.')[0]}_{0}"
    }]
    
    for i in range(1, params["yolo_preprocessing"]["n_augmentations"] + 1):
        new_seed = base_random_state + i
        np.random.seed(new_seed)
        random.seed(new_seed)
        augmenter = get_augmentations()
        augmentation = augmenter(image=image, bboxes=bboxes, labels=classes)
        augmented_image = augmentation["image"]
        augmented_bboxes = augmentation["bboxes"]

        augmentations.append({
            "image": augmented_image, 
            "bboxes": augmented_bboxes,
            "classes": classes,
            f"fname": f"{image_name.split('.')[0]}_{i}"
        })
        
    np.random.seed(base_random_state)
    random.seed(base_random_state)
    
    for augmented_image_metadata in augmentations:
        augmented_image = Image.fromarray(augmented_image_metadata["image"])
        augmented_image.save(new_image_dir / f'{augmented_image_metadata["fname"]}.jpg')

        with open(new_label_dir / f'{augmented_image_metadata["fname"]}.txt', "a") as f:
            for class_name, bbox in zip(augmented_image_metadata["classes"], augmented_image_metadata["bboxes"]):
                f.write(f"{class_name} " + ' '.join([str(v) for v in list(bbox)]) + "\n")


def prepare_dataset(
        params, 
        datasets: pd.DataFrame
    ) -> None:
    """
        datasets: dict
            {
                "train": ...,
                "test": ...,
                "val": ...
            }

        Creates this data structure

        dataset
            - images
                - train
                    - 001.jpg
                - test
                    - 001.jpg
                - val
                    - 001.jpg
            - labels
                - train
                    - 001.txt
                - test
                    - 001.txt
                - val
                    - 001.txt

        YOLO expects this format x_center_n, y_center_n bbox_width_n, bbox_height_n
    """
    data_dir = Path(params["yolo_preprocessing"]["data_path"])

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    for split in datasets:
        os.makedirs(data_dir / f"images/{split}", exist_ok=True)
        os.makedirs(data_dir / f"labels/{split}", exist_ok=True)

    classes_dict = {class_name: i for i, class_name in enumerate(pd.concat([d for d in datasets.values()])["class_name"].unique())}
    for split in datasets:
        for i, (_, image_metadata) in enumerate(tqdm(datasets[split].groupby("image_name"))):
            if params["yolo_preprocessing"]["max_batch_size"] is not None \
                and i > params["yolo_preprocessing"]["max_batch_size"]:
                break

            try:
                process_image(
                    params=params,
                    image_metadata=image_metadata,
                    split=split,
                    classes_dict=classes_dict
                )
            except Exception as e:
                logging.info(f"Error processing image: {image_metadata['image_name'].values[0]} {e}")
                continue