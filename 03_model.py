#!/usr/bin/env python3
"""
MHC-I 结合亲和力预测模型

深度学习模型架构设计：
1. Embedding 层：将氨基酸索引编码为稠密向量，学习氨基酸的语义表示
2. Conv1d 层：捕捉序列中的局部 Motif 模式（如锚定位点）
3. GlobalMaxPooling：提取最显著的特征
4. Dense 层：整合特征并预测结合概率

生物学意义：
- HLA-A*02:01 倾向于结合 9mer 肽段
- 位置 2 和位置 9 是锚定位点（Anchor positions）
- Conv1d 可以学习这些位置的模式

Author: [Your Name]
Date: 2025-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MHCIBindingPredictor(nn.Module):
    """
    MHC-I 多肽结合亲和力预测模型

    架构：
        Embedding -> Conv1d -> GlobalMaxPooling -> Dense -> Sigmoid
    """

    def __init__(
        self,
        vocab_size: int = 21,  # 20种氨基酸 + 1个填充符
        embed_dim: int = 64,
        max_length: int = 14,
        conv_channels: list = [128, 256],
        kernel_sizes: list = [3, 5],
        dropout: float = 0.3,
        hidden_dims: list = [256, 128]
    ):
        """
        初始化模型

        Args:
            vocab_size: 词汇表大小（20种氨基酸 + 填充符）
            embed_dim: Embedding 维度
            max_length: 最大序列长度
            conv_channels: 卷积层通道数列表
            kernel_sizes: 卷积核大小列表
            dropout: Dropout 比例
            hidden_dims: 全连接层隐藏维度列表
        """
        super(MHCIBindingPredictor, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        # 1. Embedding 层
        # 生物学意义：学习每个氨基酸的语义表示
        # 相似的氨基酸（如疏水性氨基酸 L, I, V）会有相近的嵌入向量
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0  # 索引 0 用于填充
        )

        # 2. 多尺度 Conv1d 层
        # 生物学意义：捕捉不同长度的 Motif 模式
        # 例如：3-mer 可以捕捉单个锚定位点，5-mer 可以捕捉更长的模式
        self.conv_layers = nn.ModuleList()

        in_channels = embed_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # 保持序列长度
                )
            )
            in_channels = out_channels

        # 3. Batch Normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(channels) for channels in conv_channels
        ])

        # 4. Dropout
        self.dropout_conv = nn.Dropout(dropout)

        # 计算全连接层输入维度
        fc_input_dim = conv_channels[-1]

        # 5. 全连接层
        # 生物学意义：整合 Motif 特征，预测结合概率
        self.fc_layers = nn.ModuleList()

        prev_dim = fc_input_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 6. 输出层
        self.output_layer = nn.Linear(prev_dim, 1)

        # 7. Sigmoid 激活（将输出映射到 [0, 1]）
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, max_length)
            return_features: 是否返回中间特征

        Returns:
            预测概率 (batch_size, 1)
        """
        # 1. Embedding
        # 输入: (batch_size, max_length)
        # 输出: (batch_size, max_length, embed_dim)
        x = self.embedding(x)
        x = self.dropout_conv(x)

        # 转换为 (batch_size, embed_dim, max_length) 以适应 Conv1d
        x = x.transpose(1, 2)

        # 2. 卷积层
        # 生物学意义：提取局部 Motif 特征
        features = []
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            features.append(x)

        # 3. Global Max Pooling
        # 生物学意义：提取每个特征图中最强的 Motif 信号
        # 这使得模型能够捕捉到序列中任何位置出现的强结合信号
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch_size, channels)

        # 保存池化后的特征
        pooled_features = x

        # 4. 全连接层
        for fc in self.fc_layers:
            x = fc(x)

        # 5. 输出层
        x = self.output_layer(x)  # (batch_size, 1)
        x = self.sigmoid(x)

        if return_features:
            return x, pooled_features
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取嵌入向量

        Args:
            x: 输入张量 (batch_size, max_length)

        Returns:
            嵌入向量 (batch_size, max_length, embed_dim)
        """
        with torch.no_grad():
            return self.embedding(x)

    def get_conv_features(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        获取卷积层特征

        Args:
            x: 输入张量 (batch_size, max_length)
            layer_idx: 层索引（-1 表示最后一层）

        Returns:
            卷积特征 (batch_size, channels, length)
        """
        with torch.no_grad():
            x = self.embedding(x).transpose(1, 2)

            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x = conv(x)
                x = bn(x)
                x = F.relu(x)

                if i == layer_idx or (layer_idx == -1 and i == len(self.conv_layers) - 1):
                    return x

        return x


class AttentionMHCIModel(nn.Module):
    """
    带注意力机制的 MHC-I 预测模型

    使用自注意力机制捕捉长距离依赖关系
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 14,
        dropout: float = 0.3,
        hidden_dim: int = 256
    ):
        """
        初始化模型

        Args:
            vocab_size: 词汇表大小
            embed_dim: Embedding 维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            max_length: 最大序列长度
            dropout: Dropout 比例
            hidden_dim: 隐藏层维度
        """
        super(AttentionMHCIModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, embed_dim))

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 全连接层
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, max_length)

        Returns:
            预测概率 (batch_size, 1)
        """
        # 1. Embedding + Position Encoding
        x = self.embedding(x) + self.pos_encoding.unsqueeze(0)

        # 2. Transformer
        x = self.transformer(x)  # (batch_size, max_length, embed_dim)

        # 3. Global Average Pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # 4. 全连接层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 5. 输出
        x = self.output(x)
        x = self.sigmoid(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """测试模型"""
    print("=" * 60)
    print("模型测试")
    print("=" * 60)

    # 测试 Conv1d 模型
    model_conv = MHCIBindingPredictor(
        vocab_size=21,
        embed_dim=64,
        max_length=14,
        conv_channels=[128, 256],
        kernel_sizes=[3, 5],
        dropout=0.3,
        hidden_dims=[256, 128]
    )

    print(f"\nConv1d 模型参数数量: {count_parameters(model_conv):,}")

    # 测试前向传播
    batch_size = 32
    max_length = 14

    x = torch.randint(0, 21, (batch_size, max_length))
    output = model_conv(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试 Attention 模型
    model_attn = AttentionMHCIModel(
        vocab_size=21,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_length=14,
        dropout=0.3
    )

    print(f"\nAttention 模型参数数量: {count_parameters(model_attn):,}")

    output_attn = model_attn(x)
    print(f"Attention 模型输出形状: {output_attn.shape}")

    print("\n模型测试完成！")


if __name__ == "__main__":
    test_model()
