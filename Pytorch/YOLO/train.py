import os
from datetime import datetime
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import config
from model import YOLOv1
from data import YoloPascalVocDataset
from loss import Yolov1Loss
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available else "cpu"
    torch.autograd.set_detect_anomaly(True)  # Check for nan loss
    model = YOLOv1().to(device)
    loss_fn = Yolov1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_data = YoloPascalVocDataset("train", normalize=True, augment=True)
    test_data = YoloPascalVocDataset("test", normalize=True, augment=False)

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    # Create folders
    now = datetime.now()
    root = os.path.join("models", "yolo_v1", now.strftime("%m_%d_%Y"), now.strftime("%H_%M_%S"))
    weight_dir = os.path.join(root, "weights")
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))

    def save_metrics():
        np.save(os.path.join(root, "train_losses"), train_losses)
        np.save(os.path.join(root, "test_losses"), test_losses)
        np.save(os.path.join(root, "train_errors"), train_errors)
        np.save(os.path.join(root, "test_errors"), test_errors)

    writer = SummaryWriter()
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0

        tqdm_dataloader = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", dynamic_ncols=True)
        for data, labels, _ in tqdm_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            # update progress bar
            tqdm_dataloader.set_postfix(loss=loss.item())

        tqdm_dataloader.close()
        epoch_loss = epoch_loss / len(train_loader)

        train_losses = np.append(train_losses, [[epoch], [epoch_loss]], axis=1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"\nTrain Loss at epoch {epoch}: {epoch_loss}\n")

        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in test_loader:
                    data = data.to(device)
                    labels = labels.to(device)

                    outputs = model.forward(data)
                    loss = loss_fn(outputs, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.adds_scalar("Loss/test", test_loss, epoch)
            print(f"Test Loss at epoch {epoch}: {test_loss}")
            save_metrics()
            torch.save(model.state_dict(), f"./yolo_v1_model_{epoch}_epoch.pth")
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, "final"))


if __name__ == "__main__":
    main()
