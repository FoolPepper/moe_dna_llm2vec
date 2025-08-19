#!/usr/bin/env python3

import argparse
import importlib
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
from genomic_benchmarks.loc2seq import download_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
import torch.optim as optim
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 设置随机数种子，保证实验可复现


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For numpy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
# -------------------------------
# 所有评估的数据集名称
# -------------------------------
DATASETS = [
    'demo_coding_vs_intergenomic_seqs',
    # 'demo_human_or_worm',
    'human_enhancers_cohn',
    'human_enhancers_ensembl',
    'human_nontata_promoters',
    'human_ocr_ensembl',
]

# -------------------------------
# MLP 分类器
# -------------------------------


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        hidden_dim1 = input_dim//2
        hidden_dim2 = input_dim//4
        hidden_dim3 = input_dim//8
        if hidden_dim3 > 128:
            hidden_dim3 = 128
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# -------------------------------
# MLP 训练函数
# -------------------------------


def train_mlp_classifier(X_train, y_train, X_val, y_val, device,
                         hidden_dim=128, dropout=0.2, batch_size=64,
                         lr=1e-3, epochs=50, random_state=42, wandb_project="mlp"):

    # 设置随机种子
    torch.manual_seed(random_state)

    # 数据转换
    X_train_tensor = X_train.to(device)
    y_train_tensor = y_train.to(device).float().view(-1, 1)
    X_val_tensor = X_val.to(device)
    y_val_tensor = y_val.to(device).float().view(-1, 1)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim, hidden_dim, dropout).to(device)
    wandb.watch(model)  # 跟踪模型参数

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_val_acc = 0.0
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            epoch_loss += loss.item()

        # 计算平均损失和准确率
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 在验证集上评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_predicted = (val_outputs > 0.5).float()
            val_correct = (val_predicted == y_val_tensor).sum().item()
            val_acc = 100 * val_correct / y_val_tensor.size(0)

        model.train()

        # 记录到WandB
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        # print(
        #     f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    # 最终预测
    model.eval()
    with torch.no_grad():
        probs = model(X_val_tensor).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)

    # 记录预测结果到WandB
    # wandb.log({
    #     "confusion_matrix": wandb.plot.confusion_matrix(
    #         probs=None,
    #         y_true=y_val.cpu().numpy(),
    #         preds=preds,
    #         class_names=["Class 0", "Class 1"]
    #     ),
    #     "roc_curve": wandb.plot.roc_curve(
    #         y_val.cpu().numpy(),
    #         probs,
    #         classes_to_plot=[1]
    #     )
    # })

    return preds, probs

# -------------------------------
# 随机森林训练与预测函数
# -------------------------------


def train_rf_classifier(X_train, y_train, X_val, y_val, device, n_estimators=100, random_state=42):
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    print("train_rf_classifier")
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train_np, y_train_np)
    print("train_rf_classifier done")
    probs = rf.predict_proba(X_val_np)[:, 1]
    preds = (probs > 0.5).astype(int)

    return preds, probs

# -------------------------------
# 评估某个数据集
# -------------------------------


def evaluate_model_on_dataset(dataset_name, device, embedding_dir, layer):

    data_train_path = f"{embedding_dir}-all-layers/{dataset_name}-{layer}layer_train.pt"
    data_test_path = f"{embedding_dir}-all-layers/{dataset_name}-{layer}layer_test.pt"
    if os.path.exists(data_train_path) and os.path.exists(data_test_path):
        pass
    else:
        print(f"[INFO] data not found: {dataset_name}-{layer}layer")
        return None, None, None, None, None, None
    X_train = torch.load(data_train_path)["embeddings"]
    y_train = torch.load(data_train_path)["labels"]
    X_test = torch.load(data_test_path)["embeddings"]
    y_test = torch.load(data_test_path)["labels"]

    y_pred, y_score = train_mlp_classifier(
        X_train, y_train, X_test, y_test, device, hidden_dim=1024, epochs=100, dropout=0.05, lr=1e-4)
    y_true = y_test.numpy()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, auc, precision, recall, f1, mcc

# -------------------------------
# 主程序入口
# -------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="HuggingFace model path")
    parser.add_argument("--embedding_base_path", required=True,
                    help="HuggingFace model path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model from {args.model_path}")

    # 支持多GPU自动分配模型权重
    model = AutoModel.from_pretrained(
        args.model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    # model_name = args.model_path.split("/")[-1]
    uni_p = Path(args.model_path).resolve()
    model_name = f"{uni_p.parent.name}-{uni_p.name}" # 取 模型文件夹的母文件夹-模型文件夹名 作为模型最终名称
    print(f"the model name is {model_name}")
    
    # embedding_dir = f"embedding/{model_name}"
    embedding_dir = os.path.join(args.embedding_base_path, f"embedding/{model_name}")
    
    # 结果文件夹
    eval_base_dir = os.path.join(args.embedding_base_path, "eval_result")
    os.makedirs(eval_base_dir, exist_ok=True)
    
    eval_model_dir = os.path.join(eval_base_dir, model_name)
    os.makedirs(eval_model_dir, exist_ok=True)
    
    
    
    num_layers = model.config.num_hidden_layers
    print(f"[IMPORTANT] the layers number is {num_layers}")
    # 删除model
    del model

    results = []

    # 建立pd df,行为DATASETS名，列为range(num_layers+1)
    acc_df = pd.DataFrame(
        index=DATASETS,
        columns=[f"{layer}layer" for layer in range(num_layers+1)]
    )

    for dataset in DATASETS:
        # 初始化WandB
        wandb.init(project="mlp", entity="foolpepper-huazhong-university-of-science-and-technology", name=f"{model_name}_{dataset}_multi_layer", config={
            "model_type": "MLP", })

        for layer in range(num_layers+1):
            print(f"[INFO] Evaluating on dataset: {dataset}-{layer}layer")

            task = f"{dataset}-{layer}layer"
            try:
                acc, auc, precision, recall, f1, mcc = evaluate_model_on_dataset(
                    dataset, device, embedding_dir, layer)
                if acc:
                    results.append({
                        "task": task,
                        "accuracy": acc,
                        "roc_auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "mcc": mcc
                    })
                    print(
                        f"[INFO] {task}: {acc:.4f} {auc:.4f} {precision:.4f} {recall:.4f} {f1:.4f} {mcc:.4f}")
                    df = pd.DataFrame(results)
                    acc_df.loc[dataset, f"{layer}layer"] = acc
                    df.to_csv(f"{eval_model_dir}/all_result.tsv",
                              sep='\t', index=False)
                    acc_df.to_csv(f"{eval_model_dir}/acc.tsv",
                                  sep='\t', index=True)

            except Exception as e:
                print(f"[ERROR] {task}: {e}")
        wandb.finish()
    df.to_csv(f"{eval_model_dir}/all_result.tsv",
              sep='\t', index=False)
    acc_df.to_csv(f"{eval_model_dir}/acc.tsv",
                  sep='\t', index=True)
    print(f"\n[SUMMARY] Results saved to {eval_model_dir}")
    print(df.round(4))

    # 将acc_df的值转换为float类型，防止绘图报错
    acc_df_float = acc_df.astype(float)

    plt.figure(figsize=(14, 6))
    sns.heatmap(acc_df_float, annot=True, fmt=".4f", cmap="viridis", cbar=True)
    plt.title(f"Accuracy for {model_name}")
    plt.ylabel("Dataset")
    plt.xlabel("Layer")
    plt.tight_layout()
    plt.savefig(f"{eval_model_dir}/acc_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
