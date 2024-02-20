import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary

import config


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


class Resnet_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)  # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        # renet_to_yolo_adapter =

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

    def forward(self, x):
        return self.model.forward(x)


class YOLO_from_scratch_backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = [
            # Conv 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        # Conv 4
        for _ in range(4):
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        # Conv 5
        for _ in range(2):  # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]

        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        ]
        # Conv 6
        for _ in range(2):
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)


class Head_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        layers = []
        layers += [
            nn.Flatten(),
            # FC 1
            nn.Linear(config.S * config.S * 1024, 4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            # FC 2
            nn.Linear(4096, config.S * config.S * self.depth),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(self.model.forward(x), (x.size(dim=0), config.S, config.S, self.depth))


class YOLOv1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.model = nn.Sequential(Resnet_Backbone(), Head_Module())
        else:
            self.model = nn.Sequential(YOLO_from_scratch_backbone(), Head_Module())

    def forward(self, x):
        return self.model.forward(x)


if __name__ == "__main__":
    model = YOLOv1()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (3, 448, 448))
