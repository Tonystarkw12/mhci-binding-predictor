#!/usr/bin/env python3
"""
解析 NetMHCpan 标准数据集

数据来源：DTU Health Tech (Nature Protocols/NAR 论文配套数据)
数据格式：Peptide Score Allele
- Score 通常为 -log50000(IC50) 转换后的值
- Score 越低表示结合越强

Author: [Your Name]
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_netmhcpan_file(filepath: str) -> pd.DataFrame:
    """
    解析单个 NetMHCpan 数据文件

    Args:
        filepath: 文件路径

    Returns:
        DataFrame with columns: Peptide, Score, Allele
    """
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                peptide = parts[0]
                score = float(parts[1])
                allele = parts[2]

                data.append({
                    'Peptide': peptide,
                    'Score': score,
                    'Allele': allele
                })

    return pd.DataFrame(data)


def load_netmhcpan_data(data_dir: str = "NetMHCpan_train") -> pd.DataFrame:
    """
    加载所有 NetMHCpan 数据文件

    Args:
        data_dir: 数据目录

    Returns:
        合并后的 DataFrame
    """
    data_dir = Path(data_dir)

    # 读取所有 Binding Affinity 文件
    ba_files = [
        'c000_ba',
        'c001_ba',
        'c002_ba',
        'c003_ba',
        'c004_ba'
    ]

    all_data = []

    for filename in ba_files:
        filepath = data_dir / filename
        if filepath.exists():
            logger.info(f"正在读取: {filepath}")
            df = parse_netmhcpan_file(str(filepath))
            all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"总数据量: {len(combined_df)} 条")

    return combined_df


def filter_hla_a0201(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选 HLA-A*02:01 数据

    Args:
        df: 输入数据

    Returns:
        筛选后的数据
    """
    # 筛选包含 HLA-A02:01 的等位基因（支持多种写法）
    pattern = r'HLA-A0?2:01'

    filtered_df = df[df['Allele'].str.contains(pattern, case=False, na=False)].copy()

    logger.info(f"筛选出 HLA-A*02:01 数据: {len(filtered_df)} 条")

    return filtered_df


def convert_score_to_label(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    将结合分数转换为二分类标签

    NetMHCpan 的分数说明：
    - 分数越低表示结合越强
    - 通常 < 0.5 表示强结合（IC50 < 500 nM）
    - 0.5-1.0 表示弱结合
    - > 1.0 表示不结合

    Args:
        df: 输入数据
        threshold: 分类阈值

    Returns:
        添加 Label 列的数据
    """
    # 添加标签列
    # Score < threshold: 结合阳性 (1)
    # Score >= threshold: 结合阴性 (0)
    df['Label'] = (df['Score'] < threshold).astype(int)

    # 添加多肽长度
    df['Length'] = df['Peptide'].str.len()

    logger.info(f"阳性样本: {df['Label'].sum()} ({df['Label'].mean():.2%})")
    logger.info(f"阴性样本: {(df['Label'] == 0).sum()} ({(df['Label'] == 0).mean():.2%})")

    return df


def main():
    """主函数"""
    print("=" * 70)
    print("解析 NetMHCpan 标准数据集")
    print("=" * 70)

    # 1. 加载数据
    print("\n步骤 1: 加载 NetMHCpan 数据...")
    df = load_netmhcpan_data(data_dir="NetMHCpan_train")

    print(f"  ✓ 总数据量: {len(df):,} 条")

    # 2. 筛选 HLA-A*02:01
    print("\n步骤 2: 筛选 HLA-A*02:01 数据...")
    df_filtered = filter_hla_a0201(df)

    print(f"  ✓ HLA-A*02:01 数据: {len(df_filtered):,} 条")

    # 3. 转换为标签
    print("\n步骤 3: 转换结合分数为标签...")
    df_labeled = convert_score_to_label(df_filtered, threshold=0.5)

    # 4. 统计信息
    print("\n" + "=" * 70)
    print("数据统计")
    print("=" * 70)

    print(f"\n多肽长度分布:")
    print(df_labeled['Length'].value_counts().sort_index())

    print(f"\n结合分数分布:")
    print(f"  最小值: {df_labeled['Score'].min():.4f}")
    print(f"  最大值: {df_labeled['Score'].max():.4f}")
    print(f"  平均值: {df_labeled['Score'].mean():.4f}")
    print(f"  中位数: {df_labeled['Score'].median():.4f}")

    print(f"\n分数分布（按标签）:")
    print(df_labeled.groupby('Label')['Score'].describe())

    # 5. 保存数据
    print("\n步骤 4: 保存处理后的数据...")

    output_file = Path("data/netmhcpan_hla_a0201.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 选择需要的列
    output_df = df_labeled[['Peptide', 'Allele', 'Label', 'Length', 'Score']]
    output_df.to_csv(output_file, index=False)

    print(f"  ✓ 数据已保存至: {output_file}")
    print(f"  ✓ 文件大小: {output_file.stat().st_size / 1024:.1f} KB")

    # 6. 显示示例数据
    print("\n" + "=" * 70)
    print("示例数据")
    print("=" * 70)

    print("\n阳性样本示例（前 10 条）:")
    print(output_df[output_df['Label'] == 1][['Peptide', 'Score', 'Allele', 'Length']].head(10).to_string(index=False))

    print("\n阴性样本示例（前 10 条）:")
    print(output_df[output_df['Label'] == 0][['Peptide', 'Score', 'Allele', 'Length']].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("数据解析完成！")
    print("=" * 70)

    print(f"\n输出文件: {output_file.absolute()}")
    print(f"总样本数: {len(output_df):,}")
    print(f"阳性样本: {output_df['Label'].sum():,} ({output_df['Label'].mean():.2%})")
    print(f"阴性样本: {(output_df['Label'] == 0).sum():,} ({(output_df['Label'] == 0).mean():.2%})")

    return output_df


if __name__ == "__main__":
    main()
