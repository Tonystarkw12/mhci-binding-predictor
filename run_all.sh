#!/bin/bash
# MHC-I 项目完整运行脚本

echo "========================================"
echo "MHC-I Binding Predictor - 完整流程"
echo "========================================"

# 激活 Python 环境（如果需要）
# source /path/to/your/venv/bin/activate

echo ""
echo "Step 1: 下载数据..."
python 01_data_download.py

if [ $? -ne 0 ]; then
    echo "错误：数据下载失败"
    exit 1
fi

echo ""
echo "Step 2: 处理数据..."
python 02_data_processing.py

if [ $? -ne 0 ]; then
    echo "错误：数据处理失败"
    exit 1
fi

echo ""
echo "Step 3: 测试模型架构..."
python 03_model.py

if [ $? -ne 0 ]; then
    echo "错误：模型测试失败"
    exit 1
fi

echo ""
echo "Step 4: 训练模型..."
python 04_train.py

if [ $? -ne 0 ]; then
    echo "错误：模型训练失败"
    exit 1
fi

echo ""
echo "Step 5: 评估模型..."
python 05_evaluate.py

if [ $? -ne 0 ]; then
    echo "错误：模型评估失败"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 所有步骤完成！"
echo "========================================"
echo ""
echo "结果文件："
echo "  - 模型权重: models/best_model.pth"
echo "  - ROC 曲线: results/roc_curve.png"
echo "  - 混淆矩阵: results/confusion_matrix.png"
echo "  - 综合可视化: results/model_performance.png"
echo "  - 评估报告: results/evaluation_report.txt"
echo ""
