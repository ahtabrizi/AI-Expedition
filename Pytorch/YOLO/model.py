from torch import nn
import torch
import config


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

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
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        # Conv 6
        for _ in range(2):
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]

        layers += [
            nn.Flatten(),
            # FC 1
            nn.Linear(config.S * config.S * 1024, 4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            # FC 2
            nn.Linear(4096, config.S * config.S * self.depth),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(self.model.forward(x), (x.size(dim=0), config.S, config.S, self.depth))
