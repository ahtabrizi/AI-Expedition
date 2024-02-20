import random

from tqdm import tqdm
import torch
from torchvision.datasets.voc import VOCDetection, ET_parse, Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF


import config
from datautils import *


class YoloPascalVocDataset(Dataset):
    """
    @brief A dataset class for YOLO model training and validation on Pascal VOC dataset.

    @param type A string indicating the dataset type, either 'train' or 'val'.
    @param normalize A boolean indicating whether to normalize the images.
    @param augment A boolean indicating whether to augment the images.
    """

    def __init__(self, type, normalize=False, augment=False) -> None:
        download = False
        if not os.path.exists(os.path.join(config.DATA_PATH, "VOCdevkit")):
            download = True
        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year="2012",
            image_set=("train" if type == "train" else "val"),
            download=download,
            transform=T.Compose(
                [
                    T.PILToTensor(),
                    T.Resize(config.IMAGE_SIZE),
                    T.ToDtype(torch.float32, scale=True),
                ]
            ),
        )

        self.normalize = normalize
        self.augment = augment
        self.classes = load_class_dict()

        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            print("Generating class index")
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f"Generating class dict")):
                data, label = data_pair
                for j, bbox_pair in enumerate(get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            save_class_dict(self.classes)

    def __getitem__(self, i):
        """
        @brief Retrieves an item from the dataset at the specified index.

        @param i An integer index of the item to retrieve.

        @return A tuple containing the augmented/normalized image, the ground truth tensor, and the original image.
        """
        data, label = self.dataset[i]
        original_data = data
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        # Normalize
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Grid in YOLO
        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}  # Track what class each grid cell has been assigned to
        depth = 5 * config.B + config.C  # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, : config.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            1.0,  # Confidence
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],  # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],  # Y coord relative to grid square
                            (x_max - x_min) / config.IMAGE_SIZE[0],  # Width
                            (y_max - y_min) / config.IMAGE_SIZE[1],  # Height
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        """
        @brief Returns the total number of items in the dataset.
        """
        return len(self.dataset)


if __name__ == "__main__":
    # Display data
    # obj_classes = load_class_array()
    train_set = YoloPascalVocDataset("train", normalize=True, augment=True)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label, _ in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        # utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
    print("num_negatives", negative_labels)
    print("dist", smallest, largest)
