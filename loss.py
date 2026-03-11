import torch
from icecream import ic
from torch import nn


class MyYOLOV1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.debug = True
        self.save_first = True

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        debug = False
        # if self.debug and self.save_first:
        #     torch.save(predictions, "predictions.pt")
        #     torch.save(targets, "targets.pt")
        #     self.save_first = False
        #     ic("saved predictions.pt, targets.pt")

        # predictions 形状: (batch, S*S*(B*5+C)) -> reshape 为 (batch, S, S, 30)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        if debug:
            ic(predictions.shape, targets.shape)

        # 1. 计算每个网格中两个 BBox 与真实框的 IOU，选出负责预测的那个 (Responsible Box)
        # target[..., 21:25] 是真实框坐标，predictions[..., 21:25] 和 [..., 26:30] 是两个预测框
        iou_b1 = self.intersection_over_union(
            predictions[..., 21:25], targets[..., 21:25]
        )
        iou_b2 = self.intersection_over_union(
            predictions[..., 26:30], targets[..., 21:25]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        if debug:
            ic(iou_b1.shape, iou_b2.shape, ious.shape)
            # ic(iou_b1[0], iou_b2[0])

        # 找到 IOU 最大的那个框的索引 (exists_box 相当于公式中的 1_obj)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        bestbox = bestbox.unsqueeze(-1)
        exists_box = targets[..., 20].unsqueeze(3)  # 1 if object in cell

        if debug:
            ic(iou_maxes.shape, bestbox.shape)
            ic(exists_box.shape)

        # ======================== #
        #   1. 坐标损失 (Coord Loss) #
        # ======================== #
        # 只计算 Responsible Box 的损失
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * targets[..., 21:25]

        if debug:
            ic(box_predictions.shape, box_targets.shape)

        # w, h 开根号处理 (注意数值稳定性，加个 epsilon)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        coord_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        if debug:
            ic(coord_loss)

        # ======================== #
        #  2. 置信度损失 (Obj Loss)  #
        # ======================== #
        pred_conf = exists_box * (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        target_conf = exists_box * targets[..., 20:21]

        conf_loss_obj = self.mse(
            torch.flatten(pred_conf),
            torch.flatten(target_conf),
        )
        if debug:
            ic(pred_conf.shape, target_conf.shape)
            ic(conf_loss_obj)

        # ======================== #
        # 3. 置信度损失 (No Obj Loss) #
        # ======================== #
        # 两个预测框都要计算无物体的损失
        no_obj_mask_b1 = (1 - exists_box) + (exists_box * (1 - bestbox))
        no_obj_loss = self.mse(
            torch.flatten(no_obj_mask_b1 * predictions[..., 20:21]),
            torch.flatten(no_obj_mask_b1 * targets[..., 20:21]),  # 此时 target 为 0
        )
        # no_obj_loss = self.mse(
        #     torch.flatten((1 - exists_box) * predictions[..., 20:21]),
        #     torch.flatten((1 - exists_box) * targets[..., 20:21]),
        # )

        # no_obj_loss += self.mse(
        #     torch.flatten((1 - exists_box) * predictions[..., 25:26]),
        #     torch.flatten((1 - exists_box) * targets[..., 20:21]),
        # )

        no_obj_mask_b2 = (1 - exists_box) + (exists_box * bestbox)
        no_obj_loss += self.mse(
            torch.flatten(no_obj_mask_b2 * predictions[..., 25:26]),
            torch.flatten(no_obj_mask_b2 * targets[..., 20:21]),
        )

        # ======================== #
        #   4. 分类损失 (Class Loss) #
        # ======================== #
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2),
        )
        # 总损失加权求和
        loss = (
            self.lambda_coord * coord_loss
            + conf_loss_obj
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss, (
            coord_loss.item(),
            conf_loss_obj.item(),
            no_obj_loss.item(),
            class_loss.item(),
        )

    def intersection_over_union(self, boxes_preds, boxes_lables, box_format="midpoint"):
        """
        计算两个框的 IOU。

        参数:
            boxes_preds (tensor): 预测框的坐标 (BATCH_SIZE, S, S, 4)
            boxes_labels (tensor): 真实框的坐标 (BATCH_SIZE, S, S, 4)
            box_format (str): midpoint (x,y,w,h) 或 corners (x1,y1,x2,y2)
        """
        debug = False
        if debug:
            ic(boxes_preds.shape, boxes_lables.shape)
        if box_format == "midpoint":
            # 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
            # 注意：这里的 x,y 是相对于网格的偏移，w,h 是相对于整张图的比例

            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

            box2_x1 = boxes_lables[..., 0:1] - boxes_lables[..., 2:3] / 2
            box2_y1 = boxes_lables[..., 1:2] - boxes_lables[..., 3:4] / 2
            box2_x2 = boxes_lables[..., 0:1] + boxes_lables[..., 2:3] / 2
            box2_y2 = boxes_lables[..., 1:2] + boxes_lables[..., 3:4] / 2

        if box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]

            box2_x1 = boxes_lables[..., 0:1]
            box2_y1 = boxes_lables[..., 1:2]
            box2_x2 = boxes_lables[..., 2:3]
            box2_y2 = boxes_lables[..., 3:4]

        if debug:
            ic(box1_x1.shape, box1_y1.shape, box1_x2.shape, box1_y2.shape)
            ic(box2_x1.shape, box2_y1.shape, box2_x2.shape, box2_y2.shape)
        # 1. 计算交集矩形的左上角和右下角坐标
        intersection_box_x1 = torch.max(box1_x1, box2_x1)
        intersection_box_y1 = torch.max(box1_y1, box2_y1)
        intersection_box_x2 = torch.min(box1_x2, box2_x2)
        intersection_box_y2 = torch.min(box1_y2, box2_y2)

        if debug:
            ic(
                intersection_box_x1.shape,
                intersection_box_y1.shape,
                intersection_box_x2.shape,
                intersection_box_y2.shape,
            )

        # 2. 计算交集面积
        # .clamp(0) 是为了处理两个矩形完全不重叠的情况（此时 x2-x1 < 0）
        intersection = (intersection_box_x2 - intersection_box_x1).clamp(0) * (
            intersection_box_y2 - intersection_box_y1
        ).clamp(0)

        if debug:
            ic(intersection.shape)

        # 3. 计算并集面积 (Union = Area1 + Area2 - Intersection)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        union = box1_area + box2_area - intersection + 1e-6  # 加 epsilon 防止除以 0

        if debug:
            ic(box1_area.shape, box2_area.shape, union.shape)
        return intersection / union


if __name__ == "__main__":
    loss = MyYOLOV1Loss()
    predictions = torch.load("predictions.pt")
    # ic(predictions.shape)
    targets = torch.load("targets.pt")
    # ic(targets.shape)
    # ic(predictions[0])
    # ic(targets[0])
    loss(predictions, targets)
