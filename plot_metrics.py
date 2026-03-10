import torch
import matplotlib.pyplot as plt
from icecream import ic


def plot_recall_curves(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch_and_precision_recall = checkpoint.get("epoch_and_precision_recall", {})

    if not epoch_and_precision_recall:
        print("未在 checkpoint 中找到评估记录。")
        return

    # YOLO V1 类别映射（为了让图例显示名称而不是 ID）
    class_names = [
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

    # 2. 整理绘图数据
    # 假设 history 格式: {"10": {0: {"recall": 0.1, ...}, 1: {...}}, "20": ...}
    epochs = sorted([int(e) for e in epoch_and_precision_recall.keys()])

    plt.figure(figsize=(12, 8))

    # 遍历每一个类别 (0-19)
    for class_id in range(20):
        recalls = []
        for epoch in epochs:
            # 获取该 epoch 下该类别的 recall，如果不存在则记为 0
            epoch_stats = epoch_and_precision_recall[str(epoch)]
            ic(class_id, epoch, epoch_stats)
            recall_val = epoch_stats.get(class_id, {}).get("recall", 0)
            recalls.append(recall_val)

        # 只绘制在数据集中出现过（Recall不全为0）的类别，防止图例爆炸
        # if sum(recalls) > 0:
        #     ic(class_id)
        #     plt.plot(epochs, recalls, marker="o", label=class_names[class_id])
        plt.plot(epochs, recalls, marker="o", label=class_names[class_id])

    # 3. 图形美化
    plt.title("Recall vs Epoch per Class (YOLO v1)", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    # 保存并显示
    plt.savefig("recall_curve.png")
    plt.show()


if __name__ == "__main__":
    plot_recall_curves("output/checkpoint/checkpoint.pth")
    ic(111)
