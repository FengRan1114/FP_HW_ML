import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle

# 加载 ResNet 提取的特征和标签
features = np.load("features_resnet.npy")
labels = np.load("labels_resnet.npy")

# 检查标签分布
unique_labels, counts = np.unique(labels, return_counts=True)
print("标签分布：")
for label, count in zip(unique_labels, counts):
    print(f"设备 {label}: {count} 张图像")

# PCA 降维
pca = PCA(n_components=512)
features = pca.fit_transform(features)

# 标准化
scaler = RobustScaler()
features = scaler.fit_transform(features)

# 优化后的分类器
classifiers = {
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
    "LR": LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0),
    "DT": DecisionTreeClassifier(max_depth=5, min_samples_split=10),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    # "Voting": VotingClassifier(
    #     estimators=[
    #         ("SVM", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    #         ("RF", RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)),
    #         ("XGB", xgb.XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=300, eval_metric="mlogloss", tree_method="gpu_hist"))
    #     ],
    #     voting="soft"
    # )
}

# 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
num_runs = 50
results = {name: [] for name in classifiers.keys()}
f1_results = {name: [] for name in classifiers.keys()}
all_confusion_matrices = {name: [] for name in classifiers.keys()}
all_roc_data = {name: [] for name in classifiers.keys()}

for run in range(num_runs):
    print(f"\n运行次数：{run+1}")
    for name, clf in classifiers.items():
        print(f"\n训练和测试 {name}...")
        fold_accuracies = []
        fold_f1_scores = []
        fold_cm = []
        fold_roc = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            cm = confusion_matrix(y_test, y_pred)
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_cm.append(cm)

            # ROC 数据
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)
                n_classes = len(np.unique(labels))
                fpr = {}
                tpr = {}
                roc_auc = {}
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                fold_roc.append((fpr, tpr, roc_auc))

            print(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_f1 = np.mean(fold_f1_scores)
        results[name].append(mean_accuracy)
        f1_results[name].append(mean_f1)
        all_confusion_matrices[name].append(np.mean(fold_cm, axis=0))
        if fold_roc:
            all_roc_data[name].append(fold_roc)
        print(f"{name} 平均准确率: {mean_accuracy:.4f}, 标准差: {std_accuracy:.4f}, 平均 F1: {mean_f1:.4f}")

print("\n最终结果：")
for name, acc_list in results.items():
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    mean_f1 = np.mean(f1_results[name])
    print(f"{name}: 平均准确率 {mean_acc:.4f}, 标准差: {std_acc:.4f}, 平均 F1: {mean_f1:.4f}")

# 保存混淆矩阵
for name, cms in all_confusion_matrices.items():
    avg_cm = np.mean(cms, axis=0)
    cm_df = pd.DataFrame(avg_cm, index=range(15), columns=range(15))
    cm_df.to_csv(f"E:\\python project\\Believe\\resnet_confusion_matrix_{name}.csv")

# 绘制混淆矩阵热图
plt.style.use("default")
for name, cms in all_confusion_matrices.items():
    avg_cm = np.mean(cms, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues", square=True,
                xticklabels=range(15), yticklabels=range(15))
    plt.title(f"ResNet50 {name} Confusion Matrix", fontsize=14, fontfamily="Arial")
    plt.xlabel("Predicted Label", fontsize=12, fontfamily="Arial")
    plt.ylabel("True Label", fontsize=12, fontfamily="Arial")
    plt.savefig(f"E:\\python project\\Believe\\resnet_confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# 绘制 ROC 曲线
for name, roc_data in all_roc_data.items():
    if roc_data:
        plt.figure(figsize=(10, 8))
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red"] * 3)
        n_classes = 15
        for run_idx, folds in enumerate(roc_data):
            for fold_idx, (fpr, tpr, roc_auc) in enumerate(folds):
                for i in range(n_classes):
                    plt.plot(fpr[i], tpr[i], color=next(colors), alpha=0.1,
                             label=f"Class {i} (AUC = {roc_auc[i]:.2f})" if run_idx == 0 and fold_idx == 0 else "")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12, fontfamily="Arial")
        plt.ylabel("True Positive Rate", fontsize=12, fontfamily="Arial")
        plt.title(f"ResNet50 {name} ROC Curve", fontsize=14, fontfamily="Arial")
        plt.legend(loc="lower right", fontsize=10)
        plt.savefig(f"E:\\python project\\Believe\\resnet_roc_curve_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()