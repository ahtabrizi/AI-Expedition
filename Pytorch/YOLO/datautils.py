import os
import json

import config


def load_class_dict():
    """
    Load class dictionary from the classes file.

    Returns:
    - dict: The loaded class dictionary.
    """
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, "r") as file:
            return json.load(file)
    return {}


def load_class_array():
    """
    Load class array from the class dictionary.

    Returns:
    - list: The loaded class array.
    """
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    """
    Save class dictionary to the classes file.

    Parameters:
    - obj (dict): The class dictionary to save.
    """
    with open(config.CLASSES_PATH, "w") as file:
        json.dump(obj, file, indent=2)


def get_bounding_boxes(label):
    """
    Get bounding boxes from the label data.

    Parameters:
    - label (dict): The label data containing annotations.

    Returns:
    - list: List of bounding boxes with class names and coordinates.
    """
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


def scale_bbox_coord(coord, center, scale):
    """
    Scale bounding box coordinate.

    Parameters:
    - coord (int): The coordinate value to scale.
    - center (int): The center value for scaling.
    - scale (float): The scaling factor.

    Returns:
    - int: The scaled coordinate value.
    """
    return ((coord - center) * scale) + center
