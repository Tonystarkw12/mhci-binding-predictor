#!/usr/bin/env python3
"""
MHC-I 数据处理脚本

功能：
1. 筛选 HLA-A*02:01 等位基因数据
2. 多肽序列编码（One-hot 和 Index Encoding）
3. 数据集划分（训练集、验证集、测试集）

Author: [Your Name]
Date: 2025-01-07
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PeptideEncoder:
    """多肽序列编码器"""

    # 20种标准氨基酸
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
    AA_TO_IDX['X'] = 0  # 未知氨基酸或填充位

    MAX_LENGTH = 14  # MHC-I 结合肽段通常为 8-14 个氨基酸

    @classmethod
    def get_vocab_size(cls) -> int:
        """获取词汇表大小（包含填充符）"""
        return len(cls.AMINO_ACIDS) + 1

    @staticmethod
    def pad_sequence(sequence: str, max_length: int) -> str:
        """
        将序列填充到固定长度

        Args:
            sequence: 氨基酸序列
            max_length: 目标长度

        Returns:
            填充后的序列
        """
        if len(sequence) >= max_length:
            return sequence[:max_length]
        return sequence + 'X' * (max_length - len(sequence))

    @classmethod
    def one_hot_encode(cls, sequence: str) -> np.ndarray:
        """
        One-hot 编码

        Args:
            sequence: 氨基酸序列

        Returns:
            One-hot 编码矩阵 (max_length, vocab_size)
        """
        padded_seq = cls.pad_sequence(sequence, cls.MAX_LENGTH)

        encoding = np.zeros((cls.MAX_LENGTH, cls.get_vocab_size()), dtype=np.float32)

        for i, aa in enumerate(padded_seq):
            idx = cls.AA_TO_IDX.get(aa, 0)
            encoding[i, idx] = 1.0

        return encoding

    @classmethod
    def index_encode(cls, sequence: str) -> np.ndarray:
        """
        Index 编码

        Args:
            sequence: 氨基酸序列

        Returns:
            Index 编码数组 (max_length,)
        """
        padded_seq = cls.pad_sequence(sequence, cls.MAX_LENGTH)

        encoding = np.array([
            cls.AA_TO_IDX.get(aa, 0)
            for aa in padded_seq
        ], dtype=np.int64)

        return encoding

    @classmethod
    def batch_one_hot_encode(cls, sequences: list) -> np.ndarray:
        """
        批量 One-hot 编码

        Args:
            sequences: 氨基酸序列列表

        Returns:
            One-hot 编码矩阵 (batch_size, max_length, vocab_size)
        """
        return np.array([
            cls.one_hot_encode(seq)
            for seq in sequences
        ])

    @classmethod
    def batch_index_encode(cls, sequences: list) -> np.ndarray:
        """
        批量 Index 编码

        Args:
            sequences: 氨基酸序列列表

        Returns:
            Index 编码矩阵 (batch_size, max_length)
        """
        return np.array([
            cls.index_encode(seq)
            for seq in sequences
        ])


class MHCIDataProcessor:
    """MHC-I 数据处理器"""

    def __init__(
        self,
        data_dir: str = "data",
        encoding_type: str = "index",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        初始化数据处理器

        Args:
            data_dir: 数据目录
            encoding_type: 编码类型 ('onehot' 或 'index')
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.encoding_type = encoding_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # 数据存储
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self, filename: str = "iedb_mock_data.csv") -> pd.DataFrame:
        """
        加载原始数据

        Args:
            filename: 数据文件名

        Returns:
            数据 DataFrame
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        logger.info(f"正在加载数据: {filepath}")
        df = pd.read_csv(filepath)

        logger.info(f"成功加载 {len(df)} 条数据")
        return df

    def filter_hla_a0201(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        筛选 HLA-A*02:01 数据

        Args:
            df: 原始数据

        Returns:
            筛选后的数据
        """
        logger.info("正在筛选 HLA-A*02:01 数据...")

        # 筛选 Allele 列包含 HLA-A*02:01 的数据
        filtered_df = df[df['Allele'].str.contains('HLA-A.*02:01', na=False)].copy()

        logger.info(f"筛选后剩余 {len(filtered_df)} 条数据")
        return filtered_df

    def encode_peptides(
        self,
        sequences: list,
        encoding_type: str = None
    ) -> np.ndarray:
        """
        编码多肽序列

        Args:
            sequences: 多肽序列列表
            encoding_type: 编码类型（None 则使用默认）

        Returns:
            编码后的数据
        """
        if encoding_type is None:
            encoding_type = self.encoding_type

        logger.info(f"使用 {encoding_type} 编码...")

        if encoding_type == "onehot":
            return PeptideEncoder.batch_one_hot_encode(sequences)
        elif encoding_type == "index":
            return PeptideEncoder.batch_index_encode(sequences)
        else:
            raise ValueError(f"不支持的编码类型: {encoding_type}")

    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分数据集

        Args:
            df: 输入数据

        Returns:
            (训练集, 验证集, 测试集)
        """
        logger.info("正在划分数据集...")

        # 首先分离测试集
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['Label'],
            random_state=self.random_state
        )

        # 从训练验证集中分离验证集
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df['Label'],
            random_state=self.random_state
        )

        logger.info(f"训练集: {len(train_df)} 样本")
        logger.info(f"验证集: {len(val_df)} 样本")
        logger.info(f"测试集: {len(test_df)} 样本")

        return train_df, val_df, test_df

    def process_pipeline(
        self,
        input_file: str = "hla_a0201_real_data.csv"
    ) -> Dict:
        """
        完整数据处理流程

        Args:
            input_file: 输入文件名

        Returns:
            处理后的数据字典
        """
        logger.info("=" * 60)
        logger.info("开始数据处理流程")
        logger.info("=" * 60)

        # 1. 加载数据
        df = self.load_data(input_file)

        # 2. 筛选 HLA-A*02:01
        df_filtered = self.filter_hla_a0201(df)

        # 3. 划分数据集
        train_df, val_df, test_df = self.split_data(df_filtered)

        # 4. 编码序列
        X_train = self.encode_peptides(train_df['Peptide'].tolist())
        X_val = self.encode_peptides(val_df['Peptide'].tolist())
        X_test = self.encode_peptides(test_df['Peptide'].tolist())

        y_train = train_df['Label'].values
        y_val = val_df['Label'].values
        y_test = test_df['Label'].values

        # 5. 保存处理后的数据
        processed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'encoding_type': self.encoding_type,
            'max_length': PeptideEncoder.MAX_LENGTH,
            'vocab_size': PeptideEncoder.get_vocab_size()
        }

        output_file = self.data_dir / "processed_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)

        logger.info(f"处理后的数据已保存至: {output_file}")
        logger.info(f"数据形状: X_train={X_train.shape}, y_train={y_train.shape}")

        return processed_data

    def print_statistics(self, processed_data: Dict):
        """打印数据统计信息"""
        print("\n" + "=" * 60)
        print("数据统计")
        print("=" * 60)

        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            y_key = f'y_{split}'

            X = processed_data[X_key]
            y = processed_data[y_key]

            print(f"\n{split.upper()} 集合:")
            print(f"  样本数: {X.shape[0]}")
            print(f"  特征维度: {X.shape}")
            print(f"  正样本: {y.sum()} ({y.mean():.2%})")
            print(f"  负样本: {(y == 0).sum()} ({(y == 0).mean():.2%})")


def main():
    """主函数"""
    print("=" * 60)
    print("MHC-I 数据处理程序")
    print("=" * 60)

    # 初始化数据处理器
    processor = MHCIDataProcessor(
        data_dir="data",
        encoding_type="index",  # 使用 index encoding，更适合 Embedding 层
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # 执行处理流程
    processed_data = processor.process_pipeline(
        input_file="netmhcpan_hla_a0201.csv"
    )

    # 打印统计信息
    processor.print_statistics(processed_data)

    print("\n数据处理完成！")


if __name__ == "__main__":
    main()
