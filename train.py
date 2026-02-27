import torch
from icecream import ic
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from model import MyYoloV1
from loss import MyYOLOV1Loss
from dataset import train_voc_datasets
from tqdm import tqdm

# --- 超参数配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5  # YOLO训练初期建议用较小的学习率防止梯度爆炸
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
EPOCHS = 100
NUM_WORKERS = 4
SAVE_MODEL_PATH = "checkpoint.pth"


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 前向传播
        out = model(x)
        loss = loss_fn(out, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        mean_loss.append(loss.item())
        loop.set_postfix(loss=sum(mean_loss) / len(mean_loss))


    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    # 1. 初始化模型
    model = MyYoloV1(S=7, B=2, C=20).to(DEVICE)

    # 2. 优化器与损失函数
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = MyYOLOV1Loss()

    # 3. 数据加载器
    train_loader = DataLoader(
        dataset=train_voc_datasets,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=True,
    )

    # 4. 训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_fn(train_loader, model, optimizer, loss_fn)
        # 保存模型
        check_point = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(check_point, SAVE_MODEL_PATH)
        print(f"--> Checkpoint saved to {SAVE_MODEL_PATH}")

    ic("111")


if __name__ == "__main__":
    main()
