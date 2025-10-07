#!/bin/bash

# 简化的训练损失可视化脚本
# 专注于总损失对比，自动使用logs目录中的3个配置

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 训练损失曲线可视化工具"
echo "📊 专注于总损失对比分析"
echo ""

# 默认使用logs目录中的3个配置
CONFIGS=(
    "base:logs/trial_base/20250822_211127.log"
    "no_ffn:logs/trial_no_ffn/20250830_204119.log" 
    "base fuse:logs/trial_base_fuse_012/20250824_114219.log" 
) 

# 检查配置文件是否存在
echo "🔍 检查日志文件..."
for config in "${CONFIGS[@]}"; do
    config_name="${config%%:*}"
    log_path="${config#*:}"
    
    if [ ! -f "$log_path" ]; then
        echo "❌ 缺少日志: $log_path"
        exit 1
    fi
    echo "✅ $config_name: $log_path"
done

echo ""
echo "🎯 开始绘制损失曲线..."

# 运行Python脚本 - 使用简化的参数
python3 vis_train.py \
    --configs "${CONFIGS[@]}" \
    --output "loss_comparison.png" \
    --figsize 15 8 \
    --smooth \
    --no-val

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 完成! 输出文件: loss_comparison.png" 
    echo "📈 已生成总损失对比图"
    
    # 可选：自动打开图片
    if command -v xdg-open &> /dev/null; then
        echo "💡 运行 'xdg-open loss_comparison.png' 查看结果"
    fi
else
    echo "❌ 生成失败!"
    exit 1
fi 
 