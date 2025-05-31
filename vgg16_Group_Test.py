import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# GNN 模型
class PUF_GNN(nn.Module):
    def __init__(self, in_dim=4098):
        super(PUF_GNN, self).__init__()
        self.conv1 = GATConv(in_dim, 512, heads=4)
        self.conv2 = GATConv(512 * 4, 512, heads=2)
        self.conv3 = GATConv(512 * 2, 512, heads=1)
        self.group_classifier = nn.Linear(512, 3)
        self.nongroup_classifier = nn.Linear(512, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=0.6, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = self.conv3(h, edge_index, edge_weight)
        group_logits = self.group_classifier(h)
        nongroup_logits = self.nongroup_classifier(h)
        return group_logits, nongroup_logits

# 图结构
edge_index = torch.tensor([
    [0, 1], [1, 0], [0, 5], [5, 0], [0, 6], [6, 0],
    [1, 5], [5, 1], [1, 6], [6, 1], [5, 6], [6, 5],
    [2, 3], [3, 2], [2, 4], [4, 2], [2, 7], [7, 2], [2, 8], [8, 2],
    [3, 4], [4, 3], [3, 7], [7, 3], [3, 8], [8, 3],
    [4, 7], [7, 4], [4, 8], [8, 4], [7, 8], [8, 7],
    [9, 10], [10, 9], [9, 11], [11, 9], [9, 12], [12, 9], [9, 13], [13, 9], [9, 14], [14, 9],
    [10, 11], [11, 10], [10, 12], [12, 10], [10, 13], [13, 10], [10, 14], [14, 10],
    [11, 12], [12, 11], [11, 13], [13, 11], [11, 14], [14, 11],
    [12, 13], [13, 12], [12, 14], [14, 12], [13, 14], [14, 13]
], dtype=torch.long).t()

# 标签
group_labels = torch.tensor([
    0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2
], dtype=torch.long)
nongroup_labels = torch.tensor([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=torch.float)

# 设备和群组
devices = ["Alpha", "Beta", "Delta", "Gamma", "Epsilon"] + [f"Device{i + 1}" for i in range(10)]
group_map = {
    "Alpha": "A", "Beta": "A", "Device1": "A", "Device2": "A",
    "Delta": "B", "Gamma": "B", "Epsilon": "B", "Device3": "B", "Device4": "B",
    "Device5": "C", "Device6": "C", "Device7": "C", "Device8": "C", "Device9": "C", "Device10": "C"
}

# 生成通信频率和跳数
freqs = {}
hops = {}
np.random.seed(42)
for device in devices:
    group = group_map[device]
    if group == "A":
        freqs[device] = np.random.uniform(8, 10)
        hops[device] = np.random.uniform(1, 2)
    elif group == "B":
        freqs[device] = np.random.uniform(6, 8)
        hops[device] = np.random.uniform(2, 3)
    else:  # Group C
        freqs[device] = np.random.uniform(3, 6)
        hops[device] = np.random.uniform(3, 5)

# 运行 50 次
num_runs = 50
num_folds = 5
all_group_accuracies = []
all_nongroup_accuracies = []
all_group_predictions = []
all_nongroup_predictions = []
all_group_cm = []
all_nongroup_cm = []
all_nongroup_roc = []
last_dynamic_edge_index = None
last_edge_weight = None
last_nongroup_indices = None

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}...")
    run_group_accuracies = []
    run_nongroup_accuracies = []
    run_group_predictions = []
    run_nongroup_predictions = []

    for fold in range(num_folds):
        # 加载特征
        train_features = np.load(f"E:\\python project\\Believe\\train_features_fold_{fold}.npy",
                                 allow_pickle=True).item()
        test_features = np.load(f"E:\\python project\\Believe\\test_features_fold_{fold}.npy", allow_pickle=True).item()

        # 合并特征
        for device in devices:
            img_feature = train_features[device]
            freq = freqs[device]
            hop = hops[device]
            img_feature_norm = img_feature / np.linalg.norm(img_feature)
            freq_norm = (freq - 3) / (10 - 3)
            hop_norm = (hop - 1) / (5 - 1)
            train_features[device] = np.concatenate([img_feature_norm * 0.6, [freq_norm * 0.2], [hop_norm * 0.2]])
            test_features[device] = np.concatenate(
                [test_features[device] / np.linalg.norm(test_features[device]) * 0.6, [freq_norm * 0.2],
                 [hop_norm * 0.2]])

        train_x = torch.from_numpy(np.array([train_features[d] for d in devices])).float()
        test_x = torch.from_numpy(np.array([test_features[d] for d in devices])).float()

        # 模拟非群组
        np.random.seed(run * num_folds + fold)
        nongroup_indices = np.random.choice(15, size=np.random.randint(1, 4), replace=False)
        nongroup_labels = torch.tensor([1 if i in nongroup_indices else 0 for i in range(15)], dtype=torch.float)

        # 动态调整边
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        for idx in nongroup_indices:
            edge_mask &= (edge_index[0] != idx) & (edge_index[1] != idx)
        dynamic_edge_index = edge_index[:, edge_mask]

        # 边权重
        edge_weight = torch.ones(dynamic_edge_index.shape[1], dtype=torch.float)
        for i in range(dynamic_edge_index.shape[1]):
            src, dst = dynamic_edge_index[0, i].item(), dynamic_edge_index[1, i].item()
            src_group = group_map[devices[src]]
            dst_group = group_map[devices[dst]]
            if src_group != dst_group:
                edge_weight[i] = 0.1

        # 保存最后一次的边信息
        if run == num_runs - 1 and fold == num_folds - 1:
            last_dynamic_edge_index = dynamic_edge_index
            last_edge_weight = edge_weight
            last_nongroup_indices = nongroup_indices

        # 创建数据
        train_data = Data(x=train_x, edge_index=dynamic_edge_index, edge_weight=edge_weight, y_group=group_labels,
                          y_nongroup=nongroup_labels)
        test_data = Data(x=test_x, edge_index=dynamic_edge_index, edge_weight=edge_weight, y_group=group_labels,
                         y_nongroup=nongroup_labels)

        # 初始化模型
        model = PUF_GNN(in_dim=4098)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion_group = nn.CrossEntropyLoss()
        criterion_nongroup = nn.BCEWithLogitsLoss()

        # 训练
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            group_logits, nongroup_logits = model(train_data.x, train_data.edge_index, train_data.edge_weight)
            loss_group = criterion_group(group_logits, train_data.y_group)
            loss_nongroup = criterion_nongroup(nongroup_logits.squeeze(), train_data.y_nongroup)
            loss = loss_group + loss_nongroup
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        with torch.no_grad():
            group_logits, nongroup_logits = model(test_data.x, test_data.edge_index, test_data.edge_weight)
            group_pred = torch.argmax(group_logits, dim=1)
            nongroup_pred = (nongroup_logits.squeeze() > 0).float()

            group_acc = accuracy_score(test_data.y_group.numpy(), group_pred.numpy())
            group_f1 = f1_score(test_data.y_group.numpy(), group_pred.numpy(), average="weighted")
            nongroup_acc = accuracy_score(test_data.y_nongroup.numpy(), nongroup_pred.numpy())
            nongroup_f1 = f1_score(test_data.y_nongroup.numpy(), nongroup_pred.numpy(), average="macro")

            group_cm = confusion_matrix(test_data.y_group.numpy(), group_pred.numpy())
            nongroup_cm = confusion_matrix(test_data.y_nongroup.numpy(), nongroup_pred.numpy())
            nongroup_probs = torch.sigmoid(nongroup_logits.squeeze()).numpy()
            fpr, tpr, _ = roc_curve(test_data.y_nongroup.numpy(), nongroup_probs)
            roc_auc = auc(fpr, tpr)

            run_group_accuracies.append(group_acc)
            run_nongroup_accuracies.append(nongroup_acc)
            all_group_cm.append(group_cm)
            all_nongroup_cm.append(nongroup_cm)
            all_nongroup_roc.append((fpr, tpr, roc_auc))

            run_group_predictions.append({
                "Run": run + 1,
                "Fold": fold + 1,
                "True": test_data.y_group.numpy(),
                "Pred": group_pred.numpy()
            })
            run_nongroup_predictions.append({
                "Run": run + 1,
                "Fold": fold + 1,
                "True": test_data.y_nongroup.numpy(),
                "Pred": nongroup_pred.numpy()
            })

            print(f"Fold {fold + 1}: Group Acc = {group_acc:.4f}, Group F1 = {group_f1:.4f}, Non-Group Acc = {nongroup_acc:.4f}, Non-Group F1 = {nongroup_f1:.4f}")

    all_group_accuracies.append(run_group_accuracies)
    all_nongroup_accuracies.append(run_nongroup_accuracies)
    all_group_predictions.extend(run_group_predictions)
    all_nongroup_predictions.extend(run_nongroup_predictions)

# 计算平均准确率
mean_group_acc = np.mean(all_group_accuracies)
std_group_acc = np.std(all_group_accuracies)
mean_nongroup_acc = np.mean(all_nongroup_accuracies)
std_nongroup_acc = np.std(all_nongroup_accuracies)

# 保存准确率
acc_df = pd.DataFrame({
    "Run": [f"Run {i + 1}" for i in range(num_runs)],
    "Group Accuracy": [np.mean(acc) for acc in all_group_accuracies],
    "Non-Group Accuracy": [np.mean(acc) for acc in all_nongroup_accuracies]
})
acc_df.to_csv("E:\\python project\\Believe\\vgg16_accuracies.csv")

# 保存预测
group_pred_df = pd.DataFrame([
    {"Run": pred["Run"], "Fold": pred["Fold"], "Device": devices[i], "True": pred["True"][i], "Pred": pred["Pred"][i]}
    for pred in all_group_predictions for i in range(len(pred["True"]))
])
group_pred_df.to_csv("E:\\python project\\Believe\\vgg16_group_predictions.csv")
nongroup_pred_df = pd.DataFrame([
    {"Run": pred["Run"], "Fold": pred["Fold"], "Device": devices[i], "True": pred["True"][i], "Pred": pred["Pred"][i]}
    for pred in all_nongroup_predictions for i in range(len(pred["True"]))
])
nongroup_pred_df.to_csv("E:\\python project\\Believe\\vgg16_nongroup_predictions.csv")

# 保存混淆矩阵
avg_group_cm = np.mean(all_group_cm, axis=0)
avg_nongroup_cm = np.mean(all_nongroup_cm, axis=0)
pd.DataFrame(avg_group_cm, index=["A", "B", "C"], columns=["A", "B", "C"]).to_csv("E:\\python project\\Believe\\vgg16_group_cm.csv")
pd.DataFrame(avg_nongroup_cm, index=["Normal", "Malicious"], columns=["Normal", "Malicious"]).to_csv("E:\\python project\\Believe\\vgg16_nongroup_cm.csv")

print("\n最终结果（VGG16, 50 runs）：")
print(f"Group 平均准确率: {mean_group_acc:.4f}, 标准差: {std_group_acc:.4f}")
print(f"Non-Group 平均准确率: {mean_nongroup_acc:.4f}, 标准差: {std_nongroup_acc:.4f}")

# 绘制群组混淆矩阵
plt.style.use("default")
plt.figure(figsize=(8, 6))
sns.heatmap(avg_group_cm, annot=True, fmt=".1f", cmap="Blues", square=True,
            xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])
plt.title("VGG16 Group Confusion Matrix", fontsize=14, fontfamily="Arial")
plt.xlabel("Predicted Group", fontsize=12, fontfamily="Arial")
plt.ylabel("True Group", fontsize=12, fontfamily="Arial")
plt.savefig("E:\\python project\\Believe\\vgg16_group_cm.png", dpi=300, bbox_inches="tight")
plt.close()

# 绘制恶意节点 ROC 曲线
plt.figure(figsize=(8, 6))
for fpr, tpr, roc_auc in all_nongroup_roc:
    plt.plot(fpr, tpr, alpha=0.1, color="blue")
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12, fontfamily="Arial")
plt.ylabel("True Positive Rate", fontsize=12, fontfamily="Arial")
plt.title(f"VGG16 Malicious Node ROC (AUC = {np.mean([r[2] for r in all_nongroup_roc]):.2f})", fontsize=14, fontfamily="Arial")
plt.savefig("E:\\python project\\Believe\\vgg16_nongroup_roc.png", dpi=300, bbox_inches="tight")
plt.close()

# 绘制群组与恶意节点网络图
G = nx.Graph()
for i, device in enumerate(devices):
    G.add_node(i, label=device, group=group_map[device])
if last_dynamic_edge_index is not None and last_edge_weight is not None:
    for i in range(last_dynamic_edge_index.shape[1]):
        src, dst = last_dynamic_edge_index[0, i].item(), last_dynamic_edge_index[1, i].item()
        weight = last_edge_weight[i].item()
        G.add_edge(src, dst, weight=weight)

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))
colors = ["blue" if group_map[devices[i]] == "A" else "green" if group_map[devices[i]] == "B" else "red" for i in range(15)]
if last_nongroup_indices is not None:
    for idx in last_nongroup_indices:
        colors[idx] = "yellow"  # 恶意节点
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800)
nx.draw_networkx_edges(G, pos, width=[1 if G[u][v]["weight"] == 1.0 else 0.5 for u, v in G.edges()])
nx.draw_networkx_labels(G, pos, labels={i: devices[i] for i in range(15)}, font_size=10, font_family="Arial")
plt.title("VGG16 Group and Malicious Node Network", fontsize=14, fontfamily="Arial")
plt.savefig("E:\\python project\\Believe\\vgg16_group_network.png", dpi=300, bbox_inches="tight")
plt.close()