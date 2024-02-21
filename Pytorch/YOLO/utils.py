import torch
import config


def get_areas(box_preds, box_labels, box_format="mid"):
    """
    Calculate the areas of bounding boxes.

    Parameters:
    - box_preds (torch.Tensor): Predicted bounding boxes.
    - box_labels (torch.Tensor): Ground truth bounding boxes.
    - box_format (str): Format of the bounding boxes. Default is "mid".

    Returns:
    - torch.Tensor: Intersection area.
    - torch.Tensor: Area of box 1.
    - torch.Tensor: Area of box 2.
    """
    # Calculate points of each box
    if box_format == "mid":
        box1_x1 = box_preds[..., 0] - box_preds[..., 2] / 2
        box1_x2 = box_preds[..., 0] + box_preds[..., 2] / 2
        box1_y1 = box_preds[..., 1] - box_preds[..., 3] / 2
        box1_y2 = box_preds[..., 1] + box_preds[..., 3] / 2

        box2_x1 = box_labels[..., 0] - box_labels[..., 2] / 2
        box2_x2 = box_labels[..., 0] + box_labels[..., 2] / 2
        box2_y1 = box_labels[..., 1] - box_labels[..., 3] / 2
        box2_y2 = box_labels[..., 1] + box_labels[..., 3] / 2
    elif box_format == "corner":
        box1_x1 = box_preds[..., 0]
        box1_x2 = box_preds[..., 2]
        box1_y1 = box_preds[..., 1]
        box1_y2 = box_preds[..., 3]

        box2_x1 = box_labels[..., 0]
        box2_x2 = box_labels[..., 2]
        box2_y1 = box_labels[..., 1]
        box2_y2 = box_labels[..., 3]
    else:
        raise Exception("Unsupported box_format")

    # Calculate intersection points
    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate the intersection area
    # Clamp by 0 when there is no intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the union area
    area_box1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    area_box2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection, area_box1, area_box2


def get_iou(box_preds, box_labels, box_format="mid"):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box_preds (torch.Tensor): Predicted bounding boxes.
    - box_labels (torch.Tensor): Ground truth bounding boxes.
    - box_format (str): Format of the bounding boxes. Default is "mid".

    Returns:
    - torch.Tensor: IoU value.
    """
    intersection, area_box1, area_box2 = get_areas(box_preds, box_labels, box_format)
    union = area_box2 + area_box1 - intersection + config.EPSILON

    return intersection / union


def get_overlap(box_preds, box_labels, box_format="mid"):
    """
    Calculate the overlap between two bounding boxes.

    Parameters:
    - box_preds (torch.Tensor): Predicted bounding boxes.
    - box_labels (torch.Tensor): Ground truth bounding boxes.
    - box_format (str): Format of the bounding boxes. Default is "mid".

    Returns:
    - torch.Tensor: Overlap value.
    """
    intersection, a_area, b_area = get_areas(box_preds, box_labels, box_format)

    intersection[b_area == 0] = 0
    b_area[b_area == 0] = config.EPSILON
    return intersection / b_area


# non-maximum suppression
def nms(bboxes, iou_threshold):
    """
    Perform non-maximum suppression on a list of bounding boxes.

    Parameters:
    - bboxes (List[List[float]]): List of bounding boxes.
    - iou_threshold (float): IoU threshold for suppression.

    Returns:
    - List[List[float]]: Suppressed bounding boxes.
    """
    result = []
    num_boxes = len(bboxes)

    # Sort based on confidence
    bboxes.sort(key=lambda x: x[4], reverse=True)

    bboxes = torch.Tensor(bboxes).unsqueeze(1)
    suppressed = set()
    # Overlap Check
    for i in range(num_boxes):
        if i in suppressed:
            continue

        curbox = bboxes[i]
        ious = get_overlap(curbox[..., 0:4], bboxes[..., 0:4], "corner")

        for j in range(i + 1, num_boxes):
            if i != j and curbox[0, -1] == bboxes[j, 0, -1] and ious[j].item() > iou_threshold:
                suppressed.add(j)

    for i in range(num_boxes):
        if i not in suppressed:
            result.append(bboxes[i][0])
    return result


def gen_bboxes(pred, min_confidence, grid_size):
    """
    Generate bounding boxes based on model predictions.

    Parameters:
    - pred (torch.Tensor): Model predictions.
    - min_confidence (float): Minimum confidence threshold.
    - grid_size (Tuple[int, int]): Grid size for bounding boxes.

    Returns:
    - List[List[float]]: List of generated bounding boxes.
    """
    bboxes = []
    for i in range(config.S):
        for j in range(config.S):
            for k in range((pred.size(dim=2) - config.C) // 5):
                start_index = k * 5 + config.C
                end_index = start_index + 5
                class_index = torch.argmax(pred[i, j, : config.C]).item()
                bbox = pred[i, j, start_index:end_index]

                # class confidene * object confidence(iou)
                confidence = pred[i, j, class_index].item() * bbox[0].item()

                if confidence > min_confidence:
                    width = bbox[3] * config.IMAGE_SIZE[0]
                    height = bbox[4] * config.IMAGE_SIZE[1]
                    x1 = bbox[1] * config.IMAGE_SIZE[0] + j * grid_size[0] - width / 2
                    y1 = bbox[2] * config.IMAGE_SIZE[1] + i * grid_size[1] - height / 2
                    x2 = x1 + width
                    y2 = y1 + height

                    bboxes.append([x1, y1, x2, y2, confidence, class_index])

    return bboxes


if __name__ == "__main__":
    a = torch.Tensor([0, 0, 1, 1]).unsqueeze(0)
    b = torch.Tensor([0.5, 0, 4, 1]).unsqueeze(0)

    iou = get_iou(a, a, "corner")
    print(iou)
