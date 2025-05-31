import os
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time

# 设置随机种子
np.random.seed(42)

# 数据路径
data_dir = "E:\\python project\\Believe\\Data"
output_dir = "E:\\python project\\Believe"
devices = ["Alpha", "Beta", "Delta", "Gamma", "Epsilon"] + [f"Device{i + 1}" for i in range(10)]
expected_images_per_device = 192

# 加载已有特征
try:
    features = np.load(os.path.join(output_dir, "fuxian_features.npy"))
    labels = np.load(os.path.join(output_dir, "fuxian_labels.npy"))
    image_paths = np.load(os.path.join(output_dir, "fuxian_image_paths.npy"))
    print(f"加载特征形状: {features.shape}, 标签形状: {labels.shape}")
except FileNotFoundError:
    print("错误：未找到 fuxian_features.npy 或 fuxian_labels.npy，请先运行 extract_individual_features.py")
    exit(1)

if features.shape[0] != 2880:
    print(f"错误：特征数量为 {features.shape[0]}，预期 2880")
    exit(1)

# 预检查数据目录（仅为日志）
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

# K=5 交叉验证
print("\n为GNN预处理特征（K=5 交叉验证）...")
start_time = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_features = []

num_features = features.shape[0]
all_images = list(range(num_features))
all_labels = labels.tolist()

for fold, (train_idx, test_idx) in enumerate(kf.split(all_images)):
    print(f"处理第 {fold + 1} 折...")
    train_features = {d: [] for d in devices}
    test_features = {d: [] for d in devices}
    for idx in train_idx:
        device_idx = all_labels[idx]
        train_features[devices[device_idx]].append(features[idx])
    for idx in test_idx:
        device_idx = all_labels[idx]
        test_features[devices[device_idx]].append(features[idx])
    train_fold = {d: np.mean(fs, axis=0) if fs else np.zeros(4096) for d, fs in train_features.items()}
    test_fold = {d: np.mean(fs, axis=0) if fs else np.zeros(4096) for d, fs in test_features.items()}
    fold_features.append((train_fold, test_fold))

# 保存每折特征
for i, (train_f, test_f) in enumerate(fold_features):
    np.save(os.path.join(output_dir, f"vgg16_train_features_fold_{i}.npy"), train_f)
    np.save(os.path.join(output_dir, f"vgg16_test_features_fold_{i}.npy"), test_f)

# 记录日志
log_df = pd.DataFrame(log_records)
log_df.to_csv(os.path.join(output_dir, "gnn_feature_extraction_log.csv"), index=False)

elapsed_time = time.time() - start_time
print(f"\nGNN预处理特征处理完成！用时: {elapsed_time:.2f}秒")
print(f"折特征保存至: vgg16_train/test_features_fold_*.npy")
print(f"日志保存至: gnn_feature_extraction_log.csv")