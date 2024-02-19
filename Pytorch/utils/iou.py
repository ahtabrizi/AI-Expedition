import torch
import numpy as np


import config


def clamp(x, limit):
    if x > limit:
        return x
    return limit


def get_iou(box_preds, box_labels):
    # points of each box
    box1_x1 = box_preds[..., 0] - box_preds[..., 2] / 2
    box1_x2 = box_preds[..., 0] + box_preds[..., 2] / 2
    box1_y1 = box_preds[..., 1] - box_preds[..., 3] / 2
    box1_y2 = box_preds[..., 1] + box_preds[..., 3] / 2

    box2_x1 = box_labels[..., 0] - box_labels[..., 2] / 2
    box2_x2 = box_labels[..., 0] + box_labels[..., 2] / 2
    box2_y1 = box_labels[..., 1] - box_labels[..., 3] / 2
    box2_y2 = box_labels[..., 1] + box_labels[..., 3] / 2

    # intersection points
    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate the intersetion area
    # Clamp by 0 when there is no intersetion
    intersetion = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the union area
    area_box1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    area_box2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = area_box2 + area_box1 - intersetion + config.EPSILON

    return intersetion / union
