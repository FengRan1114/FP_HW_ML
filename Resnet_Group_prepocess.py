import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import KFold

# 加载 VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])  # 4096 维

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据路径
data_dir = "E:\\python project\\Believe\\Data"
devices = ["Alpha", "Beta", "Delta", "Gamma", "Epsilon"] + [f"Device{i + 1}" for i in range(10)]

# 收集所有图片路径和标签
all_images = []
all_labels = []
for label, device in enumerate(devices):
    device_path = os.path.join(data_dir, device)
    if not os.path.exists(device_path):
        print(f"Error: Device path {device_path} does not exist")
        continue
    for img_file in os.listdir(device_path):
        if img_file.endswith(".png"):
            all_images.append(os.path.join(device_path, img_file))
            all_labels.append(label)

print(f"Total images found: {len(all_images)}")

# K=5 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_features = []

for fold, (train_idx, test_idx) in enumerate(kf.split(all_images)):
    print(f"Processing Fold {fold + 1}...")
    train_images = [all_images[i] for i in train_idx]
    test_images = [all_images[i] for i in test_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_labels = [all_labels[i] for i in test_idx]

    # 提取训练特征
    train_features = {d: [] for d in devices}
    for img_path, label in zip(train_images, train_labels):
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = vgg16(img_tensor)
        train_features[devices[label]].append(feature.squeeze(0).numpy())

    # 提取测试特征
    test_features = {d: [] for d in devices}
    for img_path, label in zip(test_images, test_labels):
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = vgg16(img_tensor)
        test_features[devices[label]].append(feature.squeeze(0).numpy())

    # 聚合特征（平均）
    train_fold = {d: np.mean(fs, axis=0) if fs else np.zeros(4096) for d, fs in train_features.items()}
    test_fold = {d: np.mean(fs, axis=0) if fs else np.zeros(4096) for d, fs in test_features.items()}

    fold_features.append((train_fold, test_fold))

# 保存每折特征
for i, (train_f, test_f) in enumerate(fold_features):
    np.save(f"E:\\python project\\Believe\\train_features_fold_{i}.npy", train_f)
    np.save(f"E:\\python project\\Believe\\test_features_fold_{i}.npy", test_f)
print("K=5 特征提取完成！")