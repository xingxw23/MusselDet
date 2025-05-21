
from ultralytics import YOLO
import os

# ✅ 路径设置
dataset_dir = "yolo_mussel_dataset"  # 解压后的数据集目录
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
model_output_dir = "runs/segment/train"

# ✅ 加载模型（你也可以选择 yolov8n-seg.pt）
model = YOLO("yolov8s-seg.pt")

# ✅ 开始训练
model.train(
    data=data_yaml_path,
    epochs=100,
    imgsz=640,
    batch=4
)

# ✅ 验证模型
metrics = model.val()

# ✅ 使用训练好的模型进行推理
# 替换为你自己的图像路径
predict_results = model.predict(
    source=os.path.join(dataset_dir, "images/val"),  # 或单张图像路径
    save=True,
    conf=0.25
)

print("✅ 推理完成，结果保存在 runs/segment/predict/")
