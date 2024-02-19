import os
import json

import config

## @brief Loads class dictionary from a JSON file.
#  @return A dictionary with class names and their corresponding indices.
def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, "r") as file:
            return json.load(file)
    return {}

## @brief Saves class dictionary to a JSON file.
#  @param obj The dictionary object to save.
def save_class_dict(obj):
    with open(config.CLASSES_PATH, "w") as file:
        json.dump(obj, file, indent=2)

## @brief Extracts bounding boxes from label data.
#  @param label The label data containing bounding box information.
#  @return A list of tuples containing class names and their bounding box coordinates.
def get_bounding_boxes(label):
    size = label["annotation"]["size"]
    width, height = int(size["width"]), int(size["height"])

    # Calculate scaling factors for bounding box coordinates
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label["annotation"]["object"]
    for obj in objects:
        box = obj["bndbox"]
        # Scale bounding box coordinates
        coords = (
            int(int(box["xmin"]) * x_scale),
            int(int(box["xmax"]) * x_scale),
            int(int(box["ymin"]) * y_scale),
            int(int(box["ymax"]) * y_scale),
        )
        name = obj["name"]
        boxes.append((name, coords))
    return boxes

## @brief Scales a bounding box coordinate.
#  @param coord The original coordinate to scale.
#  @param center The center point for scaling.
#  @param scale The scaling factor.
#  @return The scaled coordinate.
def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center
