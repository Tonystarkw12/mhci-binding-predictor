#!/usr/bin/env python3
"""
MHC-I 数据下载脚本

功能：
1. 从 IEDB 数据库下载 MHC-I 结合亲和力数据
2. 提供模拟数据生成器作为备选方案

Author: [Your Name]
Date: 2025-01-07
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IEDBDataDownloader:
    """IEDB 数据下载器"""

    def __init__(self, save_dir: str = "data"):
        """
        初始化下载器

        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # IEDB MHC-I 数据下载端点
        self.iedb_url = (
            "http://tools.iedb.org/tools_api/mhci/"
            "?method=download_data&allele=HLA-A*02:01&length=9"
        )

    def download_from_iedb(self) -> Optional[pd.DataFrame]:
        """
        从 IEDB 下载数据

        Returns:
            包含 MHC-I 结合数据的 DataFrame，失败返回 None
        """
        logger.info("正在从 IEDB 下载数据...")

        try:
            # 使用 IEDB MHC-I 数据库的 HTTP POST API
            url = "http://tools.iedb.org/tools_api/mhci/"

            # 设置请求参数
            params = {
                'method': 'mhc_i_scaling',
                'sequence_text': '',  # 空字符串表示下载所有数据
                'allele': 'HLA-A*02:01',
                'length': '9',  # 9mer 肽段
            }

            logger.info("正在请求 IEDB API...")
            response = requests.post(url, data=params, timeout=60)

            if response.status_code == 200:
                # 解析响应
                lines = response.text.strip().split('\n')

                # 解析 CSV 格式的数据
                data = []
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 3:
                        peptide = parts[0].strip()
                        allele = parts[1].strip() if len(parts) > 1 else 'HLA-A*02:01'
                        # 根据 IC50 值判断是否结合（IC50 < 500 nM 为阳性）
                        try:
                            ic50 = float(parts[2]) if len(parts) > 2 else 0
                            label = 1 if ic50 < 500 else 0
                        except:
                            label = 0

                        data.append({
                            'Peptide': peptide,
                            'Allele': allele,
                            'Label': label,
                            'Length': len(peptide)
                        })

                if data:
                    df = pd.DataFrame(data)
                    raw_file = self.save_dir / "iedb_raw_data.csv"
                    df.to_csv(raw_file, index=False)
                    logger.info(f"成功下载 {len(df)} 条真实数据")
                    logger.info(f"数据已保存至: {raw_file}")
                    return df
                else:
                    logger.warning("IEDB 返回数据为空")
                    return None
            else:
                logger.warning(f"IEDB 返回状态码: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"从 IEDB 下载数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_mock_data(
        self,
        n_samples: int = 5000,
        peptide_length_range: tuple = (8, 11),
        positive_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        生成模拟数据（用于演示和测试）

        模拟数据生成逻辑：
        1. 随机生成多肽序列（8-14个氨基酸）
        2. 根据序列特征生成结合标签
        3. 添加一定噪声以模拟真实数据

        Args:
            n_samples: 样本数量
            peptide_length_range: 多肽长度范围
            positive_ratio: 正样本比例

        Returns:
            包含模拟数据的 DataFrame
        """
        logger.info(f"正在生成 {n_samples} 条模拟数据...")

        # 20种标准氨基酸
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

        data = []

        for i in range(n_samples):
            # 随机生成长度
            length = np.random.randint(
                peptide_length_range[0],
                peptide_length_range[1] + 1
            )

            # 生成随机多肽序列
            peptide = ''.join(np.random.choice(amino_acids, size=length))

            # 生成标签（基于简单的启发式规则）
            # HLA-A*02:01 倾向于结合 9mer 肽段
            # 位置 2 偏好 L, M, V, I 等疏水氨基酸
            # 位置 9 偏好 L, V, I 等疏水氨基酸

            is_positive = False

            if length == 9:
                # 9mer 肽段有更高概率结合
                position_2 = peptide[1] if len(peptide) > 1 else ''
                position_9 = peptide[8] if len(peptide) > 8 else ''

                preferred_anchor = ('L', 'M', 'V', 'I')

                if position_2 in preferred_anchor and position_9 in preferred_anchor:
                    is_positive = True

                # 添加随机噪声
                if np.random.random() < 0.2:
                    is_positive = not is_positive

            # 确保正样本比例
            if len(data) < int(n_samples * positive_ratio):
                label = 1 if is_positive or np.random.random() < 0.3 else 0
            else:
                label = 0 if not is_positive or np.random.random() < 0.3 else 1

            data.append({
                'Peptide': peptide,
                'Allele': 'HLA-A*02:01',
                'Label': label,
                'Length': length
            })

        df = pd.DataFrame(data)

        # 保存模拟数据
        mock_file = self.save_dir / "iedb_mock_data.csv"
        df.to_csv(mock_file, index=False, encoding='utf-8')

        logger.info(f"模拟数据已保存至: {mock_file}")
        logger.info(f"正样本数: {df['Label'].sum()}, 负样本数: {(df['Label'] == 0).sum()}")

        return df


def main():
    """主函数"""
    print("=" * 60)
    print("MHC-I 数据下载程序")
    print("=" * 60)

    # 初始化下载器
    downloader = IEDBDataDownloader(save_dir="data")

    # 尝试从 IEDB 下载数据
    df = downloader.download_from_iedb()

    # 如果下载失败，使用模拟数据
    if df is None:
        logger.warning("使用模拟数据作为替代方案")
        df = downloader.generate_mock_data(
            n_samples=5000,
            peptide_length_range=(8, 11),
            positive_ratio=0.5
        )

    # 数据筛选和初步统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"总样本数: {len(df)}")
    print(f"正样本数: {df['Label'].sum()}")
    print(f"负样本数: {(df['Label'] == 0).sum()}")
    print(f"正样本比例: {df['Label'].mean():.2%}")

    if 'Length' in df.columns:
        print(f"\n多肽长度分布:")
        print(df['Length'].value_counts().sort_index())

    print("\n数据下载完成！")
    print(f"数据保存位置: {downloader.save_dir.absolute()}")


if __name__ == "__main__":
    main()
