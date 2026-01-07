#!/usr/bin/env python3
"""
MHC-I 模型评估与可视化脚本

功能：
1. 加载训练好的模型
2. 在测试集上进行预测
3. 生成 ROC 曲线和混淆矩阵
4. 计算全面的评估指标

Author: [Your Name]
Date: 2025-01-07
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# 导入模型
import importlib.util
spec = importlib.util.spec_from_file_location("model", "03_model.py")
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

MHCIBindingPredictor = model_module.MHCIBindingPredictor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===========================
# 数据集类
# ===========================

class MHCIDataset(Dataset):
    """MHC-I 数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化数据集

        Args:
            X: 特征数据
            y: 标签数据
        """
        self.X = torch.LongTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ===========================
# 评估器类
# ===========================

class MHCEvaluator:
    """MHC-I 模型评估器"""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        threshold: float = 0.5
    ):
        """
        初始化评估器

        Args:
            model: 训练好的模型
            device: 设备（'cuda' 或 'cpu'）
            threshold: 分类阈值
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()

        logger.info(f"评估器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"分类阈值: {self.threshold}")

    def predict(
        self,
        test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在测试集上进行预测

        Args:
            test_loader: 测试数据加载器

        Returns:
            (预测概率, 预测标签, 真实标签)
        """
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 将数据移动到设备
                X_batch = X_batch.to(self.device)

                # 前向传播
                predictions = self.model(X_batch)
                probs = predictions.cpu().numpy().flatten()

                # 记录结果
                all_probs.extend(probs)
                all_preds.extend((probs > self.threshold).astype(int))
                all_labels.extend(y_batch.numpy())

        return np.array(all_probs), np.array(all_preds), np.array(all_labels)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }

        return metrics

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        auc_score: float,
        save_path: str = "results/roc_curve.png"
    ):
        """
        绘制 ROC 曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            auc_score: AUC 分数
            save_path: 保存路径
        """
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 绘制 ROC 曲线
        plt.plot(
            fpr,
            tpr,
            color='#1f77b4',
            linewidth=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})'
        )

        # 绘制对角线（随机猜测）
        plt.plot(
            [0, 1],
            [0, 1],
            color='gray',
            linewidth=1,
            linestyle='--',
            label='Random Guess (AUC = 0.5000)'
        )

        # 设置图形属性
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve - MHC-I Binding Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().set_aspect('equal', adjustable='box')

        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC 曲线已保存至: {save_path}")
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "results/confusion_matrix.png"
    ):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制热图
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            xticklabels=['Non-Binder', 'Binder'],
            yticklabels=['Non-Binder', 'Binder'],
            annot_kws={'size': 14}
        )

        # 设置图形属性
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - MHC-I Binding Prediction', fontsize=14, fontweight='bold')

        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存至: {save_path}")
        plt.close()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str = "results/pr_curve.png"
    ):
        """
        绘制 Precision-Recall 曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        # 计算 PR 曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 绘制 PR 曲线
        plt.plot(
            recall,
            precision,
            color='#1f77b4',
            linewidth=2,
            label=f'PR Curve (AP = {ap_score:.4f})'
        )

        # 设置图形属性
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
        plt.title('Precision-Recall Curve - MHC-I Binding Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR 曲线已保存至: {save_path}")
        plt.close()

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict,
        save_path: str = "results/evaluation_report.txt"
    ):
        """
        生成评估报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            metrics: 指标字典
            save_path: 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 计算额外指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # 生成报告
        report = f"""
{'=' * 70}
MHC-I Binding Prediction Model - Evaluation Report
{'=' * 70}

1. Overall Performance
{'-' * 70}
  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  ROC AUC:           {metrics['roc_auc']:.4f}
  Precision:         {metrics['precision']:.4f}
  Recall (Sensitivity): {metrics['recall']:.4f}
  F1 Score:          {metrics['f1_score']:.4f}

2. Confusion Matrix
{'-' * 70}
  True Negatives:    {tn:5d}  (Correctly predicted Non-Binders)
  False Positives:   {fp:5d}  (Incorrectly predicted as Binders)
  False Negatives:   {fn:5d}  (Incorrectly predicted as Non-Binders)
  True Positives:    {tp:5d}  (Correctly predicted Binders)

3. Classification Metrics
{'-' * 70}
  Sensitivity (Recall):     {sensitivity:.4f}
  Specificity:              {specificity:.4f}
  Positive Predictive Value (Precision): {ppv:.4f}
  Negative Predictive Value: {npv:.4f}

4. Classification Report
{'-' * 70}
{classification_report(y_true, y_pred, target_names=['Non-Binder', 'Binder'], digits=4)}

{'=' * 70}
"""

        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"评估报告已保存至: {save_path}")
        print(report)


# ===========================
# 主函数
# ===========================

def main():
    """主函数"""
    print("=" * 70)
    print("MHC-I 模型评估与可视化程序")
    print("=" * 70)

    # ===========================
    # 1. GPU/CUDA 检查
    # ===========================
    print("\n正在检查计算环境...")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("✗ 使用 CPU")

    # ===========================
    # 2. 加载数据
    # ===========================
    print("\n正在加载测试数据...")
    data_file = Path("data/processed_data.pkl")

    if not data_file.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_file}\n"
            f"请先运行 02_data_processing.py 处理数据"
        )

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X_test = data['X_test']
    y_test = data['y_test']

    print(f"测试集样本数: {X_test.shape[0]}")

    # ===========================
    # 3. 加载模型
    # ===========================
    print("\n正在加载模型...")
    model_file = Path("models/best_model.pth")

    if not model_file.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_file}\n"
            f"请先运行 04_train.py 训练模型"
        )

    # 创建模型
    model = MHCIBindingPredictor(
        vocab_size=data['vocab_size'],
        embed_dim=64,
        max_length=data['max_length'],
        conv_channels=[128, 256],
        kernel_sizes=[3, 5],
        dropout=0.3,
        hidden_dims=[256, 128]
    )

    # 加载模型权重
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功 (Epoch {checkpoint['epoch'] + 1})")

    # ===========================
    # 4. 创建评估器和数据加载器
    # ===========================
    print("\n正在创建评估器...")

    evaluator = MHCEvaluator(
        model=model,
        device=device,
        threshold=0.5
    )

    test_dataset = MHCIDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    # ===========================
    # 5. 预测
    # ===========================
    print("\n正在进行预测...")
    y_prob, y_pred, y_true = evaluator.predict(test_loader)

    print(f"预测完成: {len(y_pred)} 样本")
    print(f"正样本: {y_pred.sum()} ({y_pred.mean():.2%})")
    print(f"负样本: {(y_pred == 0).sum()} ({(y_pred == 0).mean():.2%})")

    # ===========================
    # 6. 计算指标
    # ===========================
    print("\n正在计算评估指标...")
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "=" * 70)
    print("评估指标")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    # ===========================
    # 7. 生成可视化
    # ===========================
    print("\n正在生成可视化图表...")

    # 创建结果目录
    Path("results").mkdir(parents=True, exist_ok=True)

    # ROC 曲线
    evaluator.plot_roc_curve(
        y_true,
        y_prob,
        metrics['roc_auc'],
        save_path="results/roc_curve.png"
    )

    # 混淆矩阵
    evaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path="results/confusion_matrix.png"
    )

    # Precision-Recall 曲线
    evaluator.plot_precision_recall_curve(
        y_true,
        y_prob,
        save_path="results/pr_curve.png"
    )

    # ===========================
    # 8. 生成报告
    # ===========================
    evaluator.generate_report(
        y_true,
        y_pred,
        metrics,
        save_path="results/evaluation_report.txt"
    )

    # ===========================
    # 9. 综合可视化
    # ===========================
    print("\n正在生成综合可视化图表...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["roc_auc"]:.4f}')
    axes[0].plot([0, 1], [0, 1], 'gray', linestyle='--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Non-Binder', 'Binder'],
                yticklabels=['Non-Binder', 'Binder'])
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title('Confusion Matrix')

    # Precision-Recall 曲线
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    axes[2].plot(recall, precision, linewidth=2, label=f'AP = {ap_score:.4f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/model_performance.png", dpi=300, bbox_inches='tight')
    logger.info("综合可视化图表已保存至: results/model_performance.png")
    plt.close()

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)
    print(f"所有结果已保存至: {Path('results').absolute()}")


if __name__ == "__main__":
    main()
