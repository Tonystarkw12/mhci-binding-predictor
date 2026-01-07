#!/usr/bin/env python3
"""
MHC-I 模型训练脚本

功能：
1. 加载处理后的数据
2. 配置 GPU/CUDA 训练环境
3. 实现训练循环和验证
4. 保存最佳模型

Author: [Your Name]
Date: 2025-01-07
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple
import logging
from sklearn.metrics import accuracy_score, roc_auc_score

# 导入模型
import importlib.util
spec = importlib.util.spec_from_file_location("model", "03_model.py")
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

MHCIBindingPredictor = model_module.MHCIBindingPredictor
AttentionMHCIModel = model_module.AttentionMHCIModel
count_parameters = model_module.count_parameters

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
# 训练器类
# ===========================

class MHCTrainer:
    """MHC-I 模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        num_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """
        初始化训练器

        Args:
            model: 模型
            device: 设备（'cuda' 或 'cpu'）
            learning_rate: 学习率
            batch_size: 批次大小
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # 损失函数（二元交叉熵）
        self.criterion = nn.BCELoss()

        # 优化器（Adam）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # L2 正则化
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'lr': []
        }

        # 最佳模型指标
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        logger.info(f"训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"模型参数数量: {count_parameters(self.model):,}")

    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        训练一个 Epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            (平均损失, 平均准确率)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # 将数据移动到设备
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 记录指标
            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().numpy().flatten())
            all_labels.extend(y_batch.detach().cpu().numpy().flatten())

        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(
            all_labels,
            (np.array(all_preds) > 0.5).astype(int)
        )

        return avg_loss, avg_acc

    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            (平均损失, 平均准确率, AUC)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # 将数据移动到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                # 前向传播
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                # 记录指标
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        # 计算平均指标
        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(
            all_labels,
            (np.array(all_preds) > 0.5).astype(int)
        )
        avg_auc = roc_auc_score(all_labels, all_preds)

        return avg_loss, avg_acc, avg_auc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "models"
    ):
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 模型保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc, val_auc = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(current_lr)

            # 记录最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0

                # 保存最佳模型
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_auc': val_auc,
                        'history': self.history
                    },
                    save_dir / "best_model.pth"
                )

                logger.info(f"✓ 保存最佳模型 (Epoch {epoch + 1})")
            else:
                self.patience_counter += 1

            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch + 1}/{self.num_epochs}] "
                f"{epoch_time:.1f}s - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\n早停触发！连续 {self.early_stopping_patience} 个 Epoch 无改善")
                break

        total_time = time.time() - start_time

        # 训练完成
        logger.info("\n" + "=" * 60)
        logger.info("训练完成")
        logger.info("=" * 60)
        logger.info(f"总训练时间: {total_time / 60:.1f} 分钟")
        logger.info(f"最佳 Epoch: {self.best_epoch + 1}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        logger.info(f"最佳验证准确率: {self.best_val_acc:.4f}")
        logger.info(f"最佳验证 AUC: {self.best_val_auc:.4f}")
        logger.info(f"模型保存位置: {save_dir / 'best_model.pth'}")


# ===========================
# 主函数
# ===========================

def main():
    """主函数"""
    print("=" * 60)
    print("MHC-I 模型训练程序")
    print("=" * 60)

    # ===========================
    # 1. GPU/CUDA 检查
    # ===========================
    print("\n正在检查计算环境...")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA 可用")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        print(f"  当前 GPU: {torch.cuda.current_device()}")
        print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("✗ CUDA 不可用，使用 CPU 训练")

    print(f"\n使用设备: {device}")

    # ===========================
    # 2. 加载数据
    # ===========================
    print("\n正在加载数据...")
    data_file = Path("data/processed_data.pkl")

    if not data_file.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_file}\n"
            f"请先运行 02_data_processing.py 处理数据"
        )

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")

    # ===========================
    # 3. 创建数据加载器
    # ===========================
    print("\n正在创建数据加载器...")

    train_dataset = MHCIDataset(X_train, y_train)
    val_dataset = MHCIDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # ===========================
    # 4. 创建模型
    # ===========================
    print("\n正在创建模型...")

    model = MHCIBindingPredictor(
        vocab_size=data['vocab_size'],
        embed_dim=64,
        max_length=data['max_length'],
        conv_channels=[128, 256],
        kernel_sizes=[3, 5],
        dropout=0.3,
        hidden_dims=[256, 128]
    )

    print(f"模型参数数量: {count_parameters(model):,}")

    # ===========================
    # 5. 训练
    # ===========================
    trainer = MHCTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=50,
        early_stopping_patience=10
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir="models"
    )

    # ===========================
    # 6. 保存训练历史
    # ===========================
    history_df = pd.DataFrame(trainer.history)
    history_file = Path("results/training_history.csv")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_file, index=False)

    print(f"\n训练历史已保存至: {history_file}")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
