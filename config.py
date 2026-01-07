#!/usr/bin/env python3
"""
项目配置文件

集中管理所有超参数和路径配置

Author: [Your Name]
Date: 2025-01-07
"""

from pathlib import Path


class Config:
    """项目配置类"""

    # ===========================
    # 路径配置
    # ===========================
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # ===========================
    # 数据配置
    # ===========================
    # HLA 等位基因
    ALLELE = "HLA-A*02:01"

    # 多肽长度范围
    MIN_PEPTIDE_LENGTH = 8
    MAX_PEPTIDE_LENGTH = 14

    # 数据集划分比例
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    TRAIN_SIZE = 1 - TEST_SIZE - VAL_SIZE

    # 随机种子
    RANDOM_STATE = 42

    # ===========================
    # 模型配置
    # ===========================
    # 词汇表大小（20种氨基酸 + 1个填充符）
    VOCAB_SIZE = 21

    # Embedding 配置
    EMBED_DIM = 64

    # 卷积层配置
    CONV_CHANNELS = [128, 256]
    KERNEL_SIZES = [3, 5]

    # 全连接层配置
    HIDDEN_DIMS = [256, 128]

    # Dropout 比例
    DROPOUT = 0.3

    # ===========================
    # 训练配置
    # ===========================
    # 训练轮数
    NUM_EPOCHS = 50

    # 批次大小
    BATCH_SIZE = 128

    # 学习率
    LEARNING_RATE = 0.001

    # 早停耐心值
    EARLY_STOPPING_PATIENCE = 10

    # 学习率调度器耐心值
    LR_SCHEDULER_PATIENCE = 5

    # 权重衰减（L2 正则化）
    WEIGHT_DECAY = 1e-5

    # 梯度裁剪阈值
    GRAD_CLIP_MAX_NORM = 1.0

    # ===========================
    # 评估配置
    # ===========================
    # 分类阈值
    CLASSIFICATION_THRESHOLD = 0.5

    # ===========================
    # IEDB API 配置
    # ===========================
    IEDB_BASE_URL = "http://tools.iedb.org/tools_api/mhci/"
    IEDB_TIMEOUT = 30  # 秒

    # ===========================
    # 日志配置
    # ===========================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """将配置转换为字典"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


# 如果直接运行此脚本，打印所有配置
if __name__ == "__main__":
    print("=" * 60)
    print("项目配置")
    print("=" * 60)

    config_dict = Config.to_dict()

    for key, value in sorted(config_dict.items()):
        if not key.startswith('_'):
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("创建项目目录...")
    print("=" * 60)

    Config.create_directories()

    print(f"✓ 数据目录: {Config.DATA_DIR}")
    print(f"✓ 模型目录: {Config.MODELS_DIR}")
    print(f"✓ 结果目录: {Config.RESULTS_DIR}")
    print(f"✓ 日志目录: {Config.LOGS_DIR}")

    print("\n配置完成！")
