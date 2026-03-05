import torch
from model import MyYoloV1
from dataset import test_voc_datasets


# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "output/checkpoint/checkpoint.pth"


def test_and_visualize(num_samples=5):
    # 1. 加载模型
    model = MyYoloV1(S=7, B=2, C=20).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # 2. 遍历测试集
    for i in range(num_samples):
        img_tensor, _ = test_voc_datasets[i]
        # 获取原始图(用于画框)和Tensor图(用于推理)

        # 推理

        # 解码所有框 (1, 98, 6)

        # 应用 NMS (针对第0张图)

        # 3. 可视化
        # 将归一化 Tensor 转回 uint8 图片格式

        # 保存或显示


if __name__ == "__main__":
    test_and_visualize()
