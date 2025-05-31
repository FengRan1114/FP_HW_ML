import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载ResNet50模型
try:
    resnet50 = models.resnet50(pretrained=True).to(device)
    resnet50.eval()
    resnet50.fc = torch.nn.Identity()
except Exception as e:
    print(f"加载 ResNet50 模型失败: {e}")
    exit(1)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据路径
data_dir = "E:\\python project\\Believe\\Data"
output_dir = "E:\\python project\\Believe"
devices = ["Alpha", "Beta", "Delta", "Gamma", "Epsilon"] + [f"Device{i + 1}" for i in range(10)]
expected_images_per_device = 192

# 预检查数据目录
print("检查数据目录...")
log_records = []
total_images = 0
for device in devices:
    device_path = os.path.join(data_dir, device)
    img_files = []
    try:
        img_files = [f for f in os.listdir(device_path) if f.lower().endswith(".png")]
        status = "正常" if len(img_files) == expected_images_per_device else f"图像数量异常（{len(img_files)}）"
    except FileNotFoundError:
        print(f"错误：设备路径 {device_path} 不存在")
        status = "路径不存在"
    except Exception as e:
        print(f"错误：访问 {device_path} 失败: {e}")
        status = "访问失败"
    total_images += len(img_files)
    log_records.append({
        "Device": device,
        "Status": status,
        "Image Count": len(img_files),
        "Failed Images": ""
    })
    print(f"{device}: {len(img_files)} 张图像（预期 {expected_images_per_device} 张）")
print(f"总计图像数量: {total_images} 张（预期 2880 张）")

# 提取特征
print("\n提取ResNet50特征...")
start_time = time.time()
features = []
labels = []
image_paths = []

for label, device in enumerate(devices):
    device_path = os.path.join(data_dir, device)
    try:
        img_files = [f for f in os.listdir(device_path) if f.lower().endswith(".png")]
    except FileNotFoundError:
        continue
    failed_images = []
    print(f"处理设备 {device}，找到 {len(img_files)} 张图像")
    for img_file in img_files:
        img_path = os.path.join(device_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = resnet50(img_tensor)
            feature_np = feature.squeeze(0).cpu().numpy()
            if feature_np.shape[0] != 2048:
                raise ValueError(f"特征维度异常: {feature_np.shape[0]}")
            features.append(feature_np)
            labels.append(label)
            image_paths.append(img_path)
        except Exception as e:
            print(f"处理图像 {img_path} 失败: {str(e)}")
            failed_images.append(f"{img_path}: {str(e)}")
    for record in log_records:
        if record["Device"] == device:
            record["Failed Images"] = ";".join(failed_images)
            record["Status"] = "部分成功" if features and failed_images else "成功" if features else "无有效特征"

features = np.array(features)
labels = np.array(labels)
elapsed_time = time.time() - start_time
print(f"特征提取完成！特征形状: {features.shape}, 标签形状: {labels.shape}, 用时: {elapsed_time:.2f}秒")

if features.shape[0] != 2880:
    print(f"错误：提取特征数量为 {features.shape[0]}，预期 2880")
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(os.path.join(output_dir, "resnet50_individual_feature_extraction_log.csv"), index=False)
    exit(1)

# 保存特征
np.save(os.path.join(output_dir, "features_resnet.npy"), features)
np.save(os.path.join(output_dir, "labels_resnet.npy"), labels)
np.save(os.path.join(output_dir, "image_paths_resnet.npy"), np.array(image_paths))

# 记录日志
log_df = pd.DataFrame(log_records)
log_df.to_csv(os.path.join(output_dir, "resnet50_individual_feature_extraction_log.csv"), index=False)

print("\nResNet50个体认证特征提取完成！")
print(f"特征保存至: features_resnet.npy, labels_resnet.npy")
print(f"日志保存至: resnet50_individual_feature_extraction_log.csv")