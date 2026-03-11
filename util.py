from icecream import ic
import torch


def intersection_over_union_single(box1, box2):
    # 简化的单框IOU计算 (x,y,w,h)
    box1_x1, box1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    box1_x2, box1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    box2_x2, box2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    intersection_x1, intersection_y1 = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    intersection_x2, intersection_y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)
    # intersection = (intersection_x2 - intersection_x1).clamp(0) * (
    #     intersection_y2 - intersection_y1
    # ).clamp(0)
    intersection = intersection_w * intersection_h
    w1, h1 = box1[2], box1[3]
    w2, h2 = box2[2], box2[3]
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / (union + 1e-6)


def intersection_over_union(boxes_preds, boxes_lables, box_format="midpoint"):
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


def cellboxes_to_boxes(out, S=7):
    """
    将模型输出 (batch, S, S, 30) 转换为 [batch, S*S, [class_id, score, x, y, w, h]]
    """
    batch_size = out.shape[0]
    out = out.to("cpu")

    # 提取预测信息
    # 假设格式：[0:20]类别, [20]置信度1, [21:25]框1, [25]置信度2, [26:30]框2
    classes = torch.argmax(out[..., :20], dim=-1).unsqueeze(-1)  # (batch, S, S, 1)

    conf1 = out[..., 20:21]     # (batch, S, S, 1)
    conf2 = out[..., 25:26]     # (batch, S, S, 1)

    # 比较两个框的置信度，取较大的那个作为该网格的代表
    best_conf, best_box_idx = torch.max(
        torch.cat([conf1, conf2], dim=-1), dim=-1, keepdim=True
    )

    # 获取对应的框坐标 (x, y, w, h)
    # 如果 best_box_idx 是 0，取 21:25；如果是 1，取 26:30
    box1 = out[..., 21:25]
    box2 = out[..., 26:30]
    best_boxes = box1 * (
        1 - best_box_idx.repeat(1, 1, 1, 4)
    ) + box2 * best_box_idx.repeat(1, 1, 1, 4)

    # 转换 x, y 坐标：从网格相对偏移 -> 全图相对比例
    # cell_indices 生成每个网格的左上角坐标 (0,0), (0,1)...
    x_cell = torch.arange(S).repeat(S, 1).view(1, S, S, 1).expand(batch_size, S, S, 1)
    y_cell = x_cell.transpose(1, 2)

    # x = (x_offset + col_index) / S
    x = (best_boxes[..., 0:1] + x_cell) / S
    # y = (y_offset + row_index) / S (注意转置以匹配行索引)
    y = (best_boxes[..., 1:2] + y_cell) / S
    w = best_boxes[..., 2:3]
    h = best_boxes[..., 3:4]

    converted_bboxes = torch.cat([classes, best_conf, x, y, w, h], dim=-1)
    # 展平 S*S 维度
    return converted_bboxes.reshape(batch_size, S * S, -1).tolist()


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    bboxes: [[class_id, score, x, y, w, h], ...]
    """
    # 1. 过滤掉置信度低于阈值的框
    bboxes = [box for box in bboxes if box[1] > threshold]

    # 2. 按置信度降序排列
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # 3. 过滤掉与已选框 IOU 过大且类别相同的框
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union_single(
                chosen_box[2:],
                box[2:],
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


if __name__ == "__main__":
    input = torch.randn(16, 7, 7, 30)
    # ic(input.shape)
    cellboxes_to_boxes(input)
    ic(111)
