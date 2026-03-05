import torch
from model import MyYoloV1
from torch.utils.data import DataLoader
from collections import Counter
from util import (
    cellboxes_to_boxes,
    non_max_suppression,
    intersection_over_union_single,
)


def get_bboxes(
    loader: DataLoader, model: MyYoloV1, iou_threshold, threshold, device="cuda"
):
    """从数据集中获取所有预测框和真实框"""
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0

    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        # 这里需要实现一个 nms_boxes 函数，将 7x7x30 转为 [train_idx, class_id, conf, x1, y1, x2, y2]
        # 简化起见，这里假设你已经有处理好的转换逻辑
        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx], iou_threshold=iou_threshold, threshold=threshold
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1

    return all_pred_boxes, all_true_boxes


def calculate_precision_recall(
    all_pred_boxes, all_true_boxes, iou_threshold=0.5, num_classes=20
):
    """
    计算每个类别的 Precision 和 Recall
    all_pred_boxes: [[train_idx, class_prediction, prob_score, x, y, w, h], ...]
    """
    stats = {}

    for c in range(num_classes):
        # 筛选当前类别的检测结果
        detections = [d for d in all_pred_boxes if d[1] == c]
        ground_truths = [gt for gt in all_true_boxes if gt[1] == c]

        # 如果该类在数据集中完全没出现
        if len(ground_truths) == 0:
            continue

        # 按置信度降序排列
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # 统计每个图片中该类真实框的数量
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        for detection_idx, detection in enumerate(detections):
            img_idx = detection[0]
            best_iou = 0
            best_gt_idx = -1

            # 找到同一张图中相同类别的GT
            image_gts = [i for i, gt in enumerate(ground_truths) if gt[0] == img_idx]

            for i in image_gts:
                iou = intersection_over_union_single(
                    torch.tensor(detection[3:]), torch.tensor(ground_truths[i][3:])
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou > iou_threshold:
                if amount_bboxes[img_idx][image_gts.index(best_gt_idx)] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[img_idx][image_gts.index(best_gt_idx)] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        tp_sum = torch.sum(TP).item()
        fp_sum = torch.sum(FP).item()

        precision = tp_sum / (tp_sum + fp_sum + 1e-6)
        recall = tp_sum / (total_true_bboxes + 1e-6)

        stats[c] = {"precision": precision, "recall": recall}
    return stats
