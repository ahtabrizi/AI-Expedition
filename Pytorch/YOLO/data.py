import os
import json
import tqdm

import torch
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, "r") as file:
            return json.load(file)
    return {}


def save_class_dict(obj):
    with open(config.CLASSES_PATH, "w") as file:
        json.dump(obj, file, indent=2)


def get_bounding_boxes(label):
    size = label["annotation"]["size"]
    width, height = int(size["width"]), int(size["height"])

    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label["annotation"]["object"]
    for obj in objects:
        box = obj["bndbox"]
        coords = (
            int(int(box["xmin"]) * x_scale),
            int(int(box["xmax"]) * x_scale),
            int(int(box["ymin"]) * y_scale),
            int(int(box["ymax"]) * y_scale),
        )
        name = obj["name"]
        boxes.append((name, coords))
    return boxes


class YoloPascalVocDataset(Dataset):
    def __init__(self, type, normalize=False, augment=False) -> None:

        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year="2012",
            image_set=("train" if type == "train" else "val"),
            download=True,
            transform=T.Compose([T.ToTensor(), T.Resize(config.IMAGE_SIZE)]),
        )

        self.normalize = normalize
        self.augment = augment
        self.classes = load_class_dict()

        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f"Generating class dict")):
                data, label = data_pair
                for j, bbox_pair in enumerate(get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            save_class_dict(self.classes)
