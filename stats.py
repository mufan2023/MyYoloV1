import torch
from model import MyYoloV1
from torch.utils.data import DataLoader
from dataset import test_voc_datasets,train_voc_datasets
from tqdm import tqdm
from util import cellboxes_to_boxes, non_max_suppression, intersection_over_union_single

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "output/checkpoint/checkpoint.pth"
S = 7
B = 2
C = 20

# VOC 类别映射 (用于显示标签)
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def get_category_accuracy(
    loader, model, iou_threshold=0.5, conf_threshold=0.1, device="cpu"
):
    model.eval()

    # 统计字典：{class_id: [correct_count, total_gt_count]}
    stats = {i: [0, 0] for i in range(20)}

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]

        # 转换为列表格式 [class_id, conf, x, y, w, h]
        batch_pred_boxes = cellboxes_to_boxes(predictions)
        batch_true_boxes = cellboxes_to_boxes(y)

        for idx in range(batch_size):
            # 1. 获取 NMS 后的预测结果
            preds = non_max_suppression(
                batch_pred_boxes[idx],
                iou_threshold=iou_threshold,
                threshold=conf_threshold,
            )

            # 2. 获取该图片中的真实框 (过滤掉置信度为0的填充框)
            gts = [box for box in batch_true_boxes[idx] if box[1] > 0.5]

            # 统计 GT 数量
            for gt in gts:
                stats[int(gt[0])][1] += 1

            # 3. 匹配预测框与真实框
            # 记录哪些 GT 已经被匹配过了，防止一个 GT 被多次匹配
            gt_matched = [False] * len(gts)

            for pred in preds:
                pred_cls = int(pred[0])
                best_iou = 0
                best_gt_idx = -1

                for i, gt in enumerate(gts):
                    if pred_cls == int(gt[0]) and not gt_matched[i]:
                        iou = intersection_over_union_single(pred[2:], gt[2:])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                # 如果最大 IOU 超过阈值，视为预测正确 (TP)
                if best_iou > iou_threshold and best_gt_idx != -1:
                    stats[pred_cls][0] += 1
                    gt_matched[best_gt_idx] = True

    # 打印结果
    print("\n" + "=" * 45)
    print(f"{'Category':<15} | {'Correct':<8} | {'Total GT':<8} | {'Accuracy':<8}")
    print("-" * 45)
    for i, name in enumerate(CLASSES):
        correct, total = stats[i]
        acc = correct / total if total > 0 else 0
        print(f"{name:<15} | {correct:<8} | {total:<8} | {acc:.2%}")

    print("=" * 45)


def main():
    model = MyYoloV1(S=S, B=B, C=C).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model from {CHECKPOINT_PATH}")

    test_loader = DataLoader(
        dataset=train_voc_datasets,
        batch_size=16,
        num_workers=4,
        shuffle=False,
    )

    get_category_accuracy(test_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
