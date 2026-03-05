import torch
from icecream import ic
from torch import nn, optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from model import MyYoloV1
from loss import MyYOLOV1Loss
from dataset import train_voc_datasets
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
from eval import calculate_precision_recall, get_bboxes

# --- 超参数配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5  # YOLO训练初期建议用较小的学习率防止梯度爆炸
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
EPOCHS = 100
NUM_WORKERS = 4
SAVE_MODEL_PATH = "output/checkpoint/checkpoint.pth"


def train_fn(train_loader, model, optimizer, loss_fn, epoch, writer=None):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 前向传播
        out = model(x)
        loss, loss_parts = loss_fn(out, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止训练初期梯度爆炸

        optimizer.step()

        # 更新进度条
        losses.append(loss.item())
        # loop.set_postfix(loss=sum(mean_loss) / len(mean_loss))
        loop.set_postfix(loss=loss.item())

    # 记录到 TensorBoard
    avg_loss = sum(losses) / len(losses)
    # if writer is not None:
    #     writer.add_scalar("Loss/Total", avg_loss, epoch)
    #     writer.add_scalar("Loss/Coord", loss_parts[0], epoch)
    #     writer.add_scalar("Loss/Obj", loss_parts[1], epoch)
    #     writer.add_scalar("Loss/NoObj", loss_parts[2], epoch)
    #     writer.add_scalar("Loss/Class", loss_parts[3], epoch)

    print(f"Mean loss was {sum(losses)/len(losses)}")


def main():
    # 1. 初始化模型
    model = MyYoloV1(S=7, B=2, C=20).to(DEVICE)

    # 2. 优化器与损失函数
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = MyYOLOV1Loss()

    # --- 新增：加载断点逻辑 ---
    start_epoch = 0
    epoch_and_precision_recall = {}
    if os.path.exists(SAVE_MODEL_PATH):
        print(f"--- Loading checkpoint: {SAVE_MODEL_PATH} ---")

        check_point = torch.load(SAVE_MODEL_PATH, map_location=DEVICE)

        # 恢复模型权重
        model.load_state_dict(check_point["state_dict"])

        # 恢复优化器参数（非常重要！）
        optimizer.load_state_dict(check_point["optimizer"])

        # 恢复轮数（从下一轮开始）
        start_epoch = check_point["epoch"]

        epoch_and_precision_recall = check_point["epoch_and_precision_recall"]

        print(f"--- Resuming from epoch {start_epoch} ---")
    # -----------------------

    # 3. 数据加载器
    train_loader = DataLoader(
        dataset=train_voc_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=True,
    )

    # writer = SummaryWriter(log_dir="logs")
    # 4. 训练循环
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_fn(train_loader, model, optimizer, loss_fn, epoch)

        # if (epoch + 1) % 10 == 0:
        #     TEN_SAVE_MODEL_PATH = "output/checkpoint/checkpoint-" + str(epoch + 1) + ".pt"
        #     torch.save(check_point, TEN_SAVE_MODEL_PATH)
        #     print(f"--> Checkpoint saved to {TEN_SAVE_MODEL_PATH}")
        if (epoch + 1) % 10 == 0:

            pred_boxes, true_boxes = get_bboxes(
                train_loader,
                model,
                iou_threshold=0.5,
                threshold=0.1,
                device=DEVICE,
            )
            stats = calculate_precision_recall(
                pred_boxes, true_boxes, iou_threshold=0.5
            )
            epoch_and_precision_recall[str(epoch + 1)] = stats

        # 保存模型
        check_point = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,  # 存入下一轮的起点
            "epoch_and_precision_recall": epoch_and_precision_recall,
        }
        torch.save(check_point, SAVE_MODEL_PATH)
        print(f"--> Checkpoint saved to {SAVE_MODEL_PATH}")

        if (epoch + 1) % 10 == 0:
            TEN_SAVE_MODEL_PATH = (
                "output/checkpoint/" + str(epoch + 1) + "-checkpoint.pth"
            )
            torch.save(check_point, TEN_SAVE_MODEL_PATH)
            print(f"--> Checkpoint saved to {TEN_SAVE_MODEL_PATH}")

    ic("111")


if __name__ == "__main__":
    main()
