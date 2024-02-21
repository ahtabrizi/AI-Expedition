import os
import argparse

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

import config
import datautils
import utils
from model import YOLOv1
from data import YoloPascalVocDataset


def load_model(model_path):
    """
    Loads the YOLOv1 model from a given file path.

    Parameters:
    - model_path (str): The path to the model file.

    Returns:
    - model (YOLOv1): The loaded YOLOv1 model in evaluation mode.
    """
    # Initialize and load the model
    model = YOLOv1()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Switch to evaluation mode
    return model


def load_and_transform_image(image_path):
    """
    Loads an image from a given path and applies transformations to it.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - Tuple[Image, torch.Tensor]: A tuple containing the original image and the transformed image tensor.
    """
    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    transform = T.Compose(
        [
            T.PILToTensor(),
            T.Resize(config.IMAGE_SIZE),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    image = transform(image)
    transformed_image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension

    # return resized image as well for show purposes
    return image, transformed_image


def plot_bbox(image, pred, classes, min_confidence=0.4, iou_threshold=0.5, file=None, show=False):
    """
    Plots bounding boxes on the image based on the prediction.

    Parameters:
    - image (torch.Tensor): The image tensor.
    - pred (torch.Tensor): The prediction tensor.
    - classes (List[str]): List of class names.
    - min_confidence (float): Minimum confidence threshold.
    - iou_threshold (float): IOU threshold for non-max suppression.
    - file (str, optional): Path to save the annotated image.
    - show (bool, optional): Whether to display the image.
    """
    print(image.size)
    # Calculate grid sizes
    grid_size_x = image.size(dim=2) / config.S
    grid_size_y = image.size(dim=1) / config.S
    # Generate and suppress bounding boxes
    bboxes = utils.gen_bboxes(pred, min_confidence, (grid_size_x, grid_size_y))
    bboxes = utils.nms(bboxes, iou_threshold)
    # Convert tensor image to PIL image for drawing
    pimage = TF.to_pil_image(image)
    draw = ImageDraw.Draw(pimage)
    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, confidence, class_index = bbox.tolist()
        draw.rectangle(((x1, y1), (x2, y2)), outline="orange")
        text_pos = (max(0, x1), max(0, y1 - 11))
        text = f"{classes[int(class_index)]} {round(confidence * 100, 1)}%"
        text_bbox = draw.textbbox(text_pos, text)
        draw.rectangle(text_bbox, fill="orange")
        draw.text(text_pos, text)
    # Save or show image if specified
    if file:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pimage.save(file)
    if show:
        pimage.show()


def detect_all_dataloader(loader, model):
    """
    Detects objects in all images from a DataLoader using the given model.

    Parameters:
    - loader (DataLoader): The DataLoader containing the dataset.
    - model (YOLOv1): The YOLOv1 model for detection.
    """
    classes = datautils.load_class_array()
    for image, labels, original, index in loader:
        for i in range(image.size(dim=0)):
            with torch.no_grad():  # Inference mode, no gradient tracking
                preds = model(image)
                plot_bbox(original[i, ...], preds[i, ...], classes, file=f"./results/result-{index[i]}.png")


def detect_from_dataset(dataset, model, index):
    """
    Detects objects in a specific image from a dataset using the given model.

    Parameters:
    - dataset (Dataset): The dataset containing the images.
    - model (YOLOv1): The YOLOv1 model for detection.
    - index (int): The index of the image in the dataset.
    """
    classes = datautils.load_class_array()
    image, labels, original, _ = dataset[index]
    with torch.no_grad():  # Inference mode, no gradient tracking
        preds = model(image.unsqueeze(0))
        plot_bbox(original, preds[0, ...], classes, file=f"./results/result-{index}.png", show=True)


def detect_from_path(image_path, model):
    """
    Detects objects in an image from a given path using the given model.

    Parameters:
    - image_path (str): The path to the image file.
    - model (YOLOv1): The YOLOv1 model for detection.
    """
    classes = datautils.load_class_array()
    image_name = image_path.split("/")[-1].split(".")[0]
    original, trans_image = load_and_transform_image(image_path)
    with torch.no_grad():  # Inference mode, no gradient tracking
        preds = model(trans_image)
        plot_bbox(original, preds[0, ...], classes, file=f"./results/result-{image_name}.png", show=True)


def main():
    """
    Main function to perform object detection using YOLOv1 model.
    Parses command line arguments and performs detection accordingly.
    """
    parser = argparse.ArgumentParser(description="Inference with PyTorch model on an image")
    parser.add_argument("model", type=str, help="Path to YOLO model")
    parser.add_argument("--image_path", type=str, help="Path to the input image", default=None)
    parser.add_argument("--index", type=int, help="Specific index to test from test dataset", default=None)
    args = parser.parse_args()

    # yolo_v1_model_8_epoch_old.pth
    model = load_model(args.model)
    dataset = YoloPascalVocDataset("test", normalize=True, augment=False, return_index=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    if args.image_path:
        detect_from_path(args.image_path, model)
    elif args.index is not None:
        detect_from_dataset(dataset, model, args.index)
    else:
        detect_all_dataloader(loader, model)


if __name__ == "__main__":
    main()
