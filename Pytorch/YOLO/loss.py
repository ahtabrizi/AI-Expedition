import os
import torch
from torch import nn

import config
from utils import get_iou

DEBUG = os.getenv("DEBUG", 0) != 0


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]


class Yolov1Loss(nn.Module):
    def __init__(self) -> None:
        super(Yolov1Loss, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

        self.S = config.S
        self.B = config.B
        self.C = config.C

    def mse_loss(self, a, b):
        flattened_a = torch.flatten(a, end_dim=-2)
        flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
        return self.mse(flattened_a, flattened_b)

    # y = [classes[20],C1, x1,y1,w1,h1, C2, x2,y2,w2,h2]
    def forward(self, preds, targets):
        # TODO: handle different B values other than default 2
        # Calculate the iou's for each bbox in each cell - (batch, S, S)
        iou_b1 = get_iou(preds[..., 21:25], targets[..., 21:25])
        iou_b2 = get_iou(preds[..., 26:30], targets[..., 26:30])

        # The first confidence has to be higher and dominant object so we only check that
        # obj_i shows that if a cell has detected at least one object - (batch, S, S, 1)
        # Since we need to perform matric multiplication later we add a dim at the end
        obj_i = targets[..., 20].unsqueeze(3) > 0.0  # 1 if ith grid has any object

        # Calculate which prediction (size determined by B) has the max iou in each cell
        _, bestbox_indices = torch.max(torch.stack([iou_b1, iou_b2]), dim=0)  # (batch, S, S)

        # bestbox_indices has 0 where first pred vector has highest iou and 1 for the second
        # In another words it represents which pred vector is responsible.
        # We create responsible tensor to seperate each pred vector's responsibility
        responsible = torch.stack([1 - bestbox_indices, bestbox_indices], dim=-1)

        # exists * responsible
        obj_ij = obj_i * responsible
        noobj_ij = 1 - obj_ij

        # XY losses
        x_loss = self.mse_loss(obj_ij * bbox_attr(preds, 1), obj_ij * bbox_attr(targets, 1))
        y_loss = self.mse_loss(obj_ij * bbox_attr(preds, 2), obj_ij * bbox_attr(targets, 2))
        xy_loss = x_loss + y_loss

        # WH losses
        p_width = bbox_attr(preds, 3)  # (batch, S, S, B)
        t_width = bbox_attr(targets, 3)  # (batch, S, S, B)

        # since predicted width can be negative we use abs. To accomodate for gradient going to zero,
        # we add epsilon. To preserve gradient direction we add sign
        width_loss = self.mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + config.EPSILON),
            obj_ij * torch.sqrt(t_width),
        )
        p_height = bbox_attr(preds, 4)  # (batch, S, S, B)
        t_height = bbox_attr(targets, 4)  # (batch, S, S, B)
        height_loss = self.mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + config.EPSILON),
            obj_ij * torch.sqrt(t_height),
        )
        wh_loss = width_loss + height_loss

        # Condifdence losses
        obj_confidence_loss = self.mse_loss(obj_ij * bbox_attr(preds, 0), obj_ij * bbox_attr(targets, 0))
        noobj_confidence_loss = self.mse_loss(noobj_ij * bbox_attr(preds, 0), noobj_ij * bbox_attr(targets, 0))

        # Class losses
        class_loss = self.mse_loss(obj_i * preds[..., : config.C], obj_i * targets[..., : config.C])
        total_loss = (
            self.lambda_coord * (xy_loss + wh_loss)
            + obj_confidence_loss
            + self.lambda_noobj * noobj_confidence_loss
            + class_loss
        )

        if DEBUG:
            print("X Loss:", x_loss.item())
            print("Y Loss:", y_loss.item())
            print("XY Loss:", xy_loss.item())

            print("Width Loss:", width_loss.item())
            print("Height Loss:", height_loss.item())
            print("WH Loss:", wh_loss.item())

            print("Object Confidence Loss:", obj_confidence_loss.item())
            print("No Object Confidence Loss:", noobj_confidence_loss.item())

            print("Class Loss:", class_loss.item())

            print("Total Loss:", total_loss.item())

        return total_loss


if __name__ == "__main__":
    loss = Yolov1Loss()
    size = (3, config.S, config.S, config.B * 5 + config.C)
    preds = torch.randn(size)
    targets = torch.abs(torch.randn(size))
    loss(preds, targets)
