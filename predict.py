import torch
from model import MyYoloV1
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from util import cellboxes_to_boxes, non_max_suppression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S = 7
B = 2
C = 20

CHECKPOINT_PATH = "output/checkpoint/checkpoint.pth"
OUTPUT_DIR = "output/inference_results"

inference_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def predict_and_draw(model, image_path, threshhold=0.2, iou_threshhold=0.45):
    """
    加载单张图片，预测并画框，保存新文件
    """
    # 1. 加载并转换图片
    original_img = Image.open(image_path).convert("RGB")
    width, height = original_img.size
    img_input = inference_transform(original_img).unsqueeze(0).to(DEVICE)

    # 2. 模型推理
    model.eval()
    with torch.no_grad():
        prediction = model(img_input)

    # 3. 将 Tensor 转换为 Box 列表 [class_id, score, x, y, w, h]
    # 调用你 util.py 里的逻辑
    bboxes = cellboxes_to_boxes(prediction, S=S)[0]

    # 4. NMS 非极大值抑制
    bboxes = non_max_suppression(
        bboxes, iou_threshold=iou_threshhold, threshold=threshhold
    )

    # 5. 绘图
    draw = ImageDraw.Draw(original_img)
    try:
        font = ImageFont.truetype("Arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    for box in bboxes:
        class_id, score, x, y, w, h = box

        # 将相对比例坐标还原回像素坐标
        # x, y 是中心点坐标 (0~1)
        left = (x - w / 2) * width
        top = (y - h / 2) * height
        right = (x + w / 2) * width
        bottom = (y + h / 2) * height

        # 绘制矩形框
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # 绘制标签文本
        label = f"{CLASSES[int(class_id)]} {score:.2}"
        draw.text((left, top - 20), text=label, fill="red", font=font)

    # 6. 保存文件
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_filename = os.path.join(OUTPUT_DIR, "pred_" + os.path.basename(image_path))
    original_img.save(save_filename)
    print(f"Successfully saved prediction to: {save_filename}")


def main():
    model = MyYoloV1(S=S, B=B, C=C).to(DEVICE)
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    test_images = [
        # "data/voc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
        # "data/voc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/009943.jpg",
        "data/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000228.jpg",
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            predict_and_draw(model, img_path, threshhold=0.2, iou_threshhold=0.45)
        else:
            print(f"File not found: {img_path}")


if __name__ == "__main__":
    main()
