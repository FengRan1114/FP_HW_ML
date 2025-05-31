import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# 加载修改后的 VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
# 移除最后一层分类器，保留 4096 维特征
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])

# 图像预处理（VGG16 需要 224x224 RGB 输入）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 灰度转 RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据路径
data_dir = "Data"  # 替换为你的实际路径，例如 "E:\\python project\\Believe\\Data"
devices = ["Alpha", "Beta", "Delta", "Gamma", "Epsilon"]+ [f"Device{i + 1}" for i in range(10)]

# 提取特征
features = []
labels = []
for label, device in enumerate(devices):
    device_path = os.path.join(data_dir, device)
    for img_file in os.listdir(device_path):
        img_path = os.path.join(device_path, img_file)
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            feature = vgg16(img_tensor)  # [1, 4096]
        features.append(feature.squeeze(0).numpy())
        labels.append(label)  # Alpha=0, Beta=1, etc.

# 转换为 NumPy 数组
features = np.array(features)  # [960, 4096] (5 设备 × 192 张)
labels = np.array(labels)      # [960]

# 保存
np.save("fuxian_features.npy", features)
np.save("fuxian_labels.npy", labels)
print("特征提取完成！")
print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")