#!/usr/bin/env python3
"""
简化的损失曲线绘制工具 - 专注于总损失对比
支持多个配置文件，自动解析训练和验证的总损失
可扩展支持任意数量的损失类型
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches

class LossParser:
    """简化的损失日志解析器 - 专注于总损失，可扩展支持多种损失类型"""
    
    def __init__(self):
        # 主要模式：专注于总损失的训练和验证
        self.train_pattern = re.compile(
            r'\[TRAIN\]\s+Epoch\s+(\d+)\s+Iter\s+(\d+)/(\d+).*?Loss:\s+([\d.]+)\s+\('
        )
        self.val_pattern = re.compile(
            r'\[EVAL\].*?Loss:\s+([\d.]+)\s+\('
        )
        
        # 可扩展的损失类型检测（自动发现）
        self.loss_type_pattern = re.compile(r'(\w+Loss):\s+([\d.]+)\s+\(')
    
    def parse_log_file(self, log_path: str) -> Dict:
        """
        解析日志文件，主要提取总损失，同时支持其他损失类型
        
        Returns:
            dict: {
                'train': {
                    'total_iter': [...],
                    'Loss': [...],           # 主要关注
                    'other_losses': {...}    # 其他损失类型（可扩展）
                },
                'val': {
                    'total_iter': [...], 
                    'Loss': [...],
                    'other_losses': {...}
                },
                'epochs': int,
                'iters_per_epoch': int
            }
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"日志文件不存在: {log_path}")
        
        print(f"处理日志文件: {os.path.basename(log_path)}")
        
        # 初始化数据结构
        data = {
            'train': {
                'total_iter': [],
                'Loss': [],
                'other_losses': {}  # 动态发现的其他损失
            },
            'val': {
                'total_iter': [],
                'Loss': [],
                'other_losses': {}
            },
            'epochs': 0,
            'iters_per_epoch': 0
        }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"总行数: {len(lines)}")
            
            discovered_loss_types = set()
            current_epoch = 0
            max_epoch = 0
            iters_per_epoch = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 训练损失匹配
                train_match = self.train_pattern.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    iter_in_epoch = int(train_match.group(2))
                    max_iter = int(train_match.group(3))
                    total_loss = float(train_match.group(4))
                    
                    # 更新基本信息
                    max_epoch = max(max_epoch, epoch)
                    iters_per_epoch = max_iter
                    
                    # 计算总迭代数
                    total_iter = (epoch - 1) * iters_per_epoch + iter_in_epoch + 1
                    
                    # 存储主要损失
                    data['train']['total_iter'].append(total_iter)
                    data['train']['Loss'].append(total_loss)
                    
                    # 动态发现并提取其他损失类型
                    other_losses = self.loss_type_pattern.findall(line)
                    for loss_name, loss_value in other_losses:
                        if loss_name != 'Loss':  # 排除主损失
                            discovered_loss_types.add(loss_name)
                            if loss_name not in data['train']['other_losses']:
                                data['train']['other_losses'][loss_name] = []
                            data['train']['other_losses'][loss_name].append(float(loss_value))
                    
                    continue
                
                # 验证损失匹配
                if '[EVAL]' in line:
                    val_match = self.val_pattern.search(line)
                    if val_match:
                        total_loss = float(val_match.group(1))
                        
                        # 使用当前epoch的总迭代数作为验证点
                        val_total_iter = max_epoch * iters_per_epoch
                        
                        data['val']['total_iter'].append(val_total_iter)
                        data['val']['Loss'].append(total_loss)
                        
                        # 提取验证时的其他损失
                        other_losses = self.loss_type_pattern.findall(line)
                        for loss_name, loss_value in other_losses:
                            if loss_name != 'Loss':
                                if loss_name not in data['val']['other_losses']:
                                    data['val']['other_losses'][loss_name] = []
                                data['val']['other_losses'][loss_name].append(float(loss_value))
        
        except Exception as e:
            print(f"解析错误: {e}")
            import traceback
            traceback.print_exc()
        
        # 设置最终信息
        data['epochs'] = max_epoch
        data['iters_per_epoch'] = iters_per_epoch
        
        # 统计结果
        train_points = len(data['train']['Loss'])
        val_points = len(data['val']['Loss'])
        discovered_types = len(discovered_loss_types)
        
        print(f"解析完成:")
        print(f"  - 训练步数: {train_points}")
        print(f"  - 验证步数: {val_points}")
        print(f"  - 总Epochs: {max_epoch}")
        print(f"  - 每epoch迭代数: {iters_per_epoch}")
        if discovered_types > 0:
            print(f"  - 发现的损失类型: {sorted(discovered_loss_types)}")
        
        return data

class LossCurvePlotter:
    """简化的损失曲线绘制器 - 专注于总损失"""
    
    def __init__(self):
        # 简化的颜色方案
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_multiple_configs(self, config_data: Dict[str, Dict], 
                            save_path: str = 'loss_curves.png',
                            figsize: Tuple[int, int] = (15, 8),
                            dpi: int = 300,
                            smooth: bool = True,
                            show_val: bool = True):
        """
        绘制多个配置的损失曲线 - 专注于总损失
        
        Args:
            config_data: {config_name: parsed_data, ...}
            save_path: 保存路径
            figsize: 图片大小
            dpi: 图片分辨率
            smooth: 是否平滑曲线
            show_val: 是否显示验证损失
        """
        # 单图布局：专注于总损失对比
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 绘制每个配置的总损失
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            
            # 训练损失
            if data['train']['Loss']:
                iters = data['train']['total_iter']
                losses = data['train']['Loss']
                
                # 平滑处理
                if smooth and len(losses) > 10:
                    from scipy.ndimage import uniform_filter1d
                    window_size = max(1, len(losses) // 100)
                    losses = uniform_filter1d(losses, size=window_size)
                
                ax.plot(iters, losses, color=color, linewidth=2.0, 
                       label=f"{config_name} (Train)", alpha=0.8)
            
            # 验证损失
            if show_val and data['val']['Loss']:
                val_iters = data['val']['total_iter']
                val_losses = data['val']['Loss']
                
                ax.plot(val_iters, val_losses, color=color, linewidth=2.0, 
                       linestyle='--', marker='o', markersize=6,
                       label=f"{config_name} (Val)", alpha=0.7)
            
            # 添加epoch边界线（可选）
            if data.get('epochs', 0) > 1:
                iters_per_epoch = data.get('iters_per_epoch', 1)
                for epoch in range(1, data['epochs']):
                    epoch_iter = epoch * iters_per_epoch
                    ax.axvline(x=epoch_iter, color=color, linestyle=':', alpha=0.3)
        
        # 设置图表属性
        ax.set_xlabel('Total Iterations', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 使用对数坐标（如果损失范围很大）
        if self._should_use_log_scale(config_data):
            ax.set_yscale('log')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"损失曲线已保存至: {save_path}")
        plt.close()
    
    def _should_use_log_scale(self, config_data: Dict) -> bool:
        """判断是否应该使用对数坐标"""
        all_losses = []
        for data in config_data.values():
            if data['train']['Loss']:
                all_losses.extend(data['train']['Loss'])
            if data['val']['Loss']:
                all_losses.extend(data['val']['Loss'])
        
        if not all_losses:
            return False
        
        min_loss, max_loss = min(all_losses), max(all_losses)
        return max_loss > 0 and min_loss > 0 and max_loss / min_loss > 10

def main():
    parser = argparse.ArgumentParser(description='训练损失曲线可视化 - 专注于总损失对比')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='配置文件路径列表，格式: name1:path1 name2:path2')
    parser.add_argument('--output', default='loss_curves.png',
                       help='输出图片路径 (default: loss_curves.png)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 8],
                       help='图片大小 (default: 15 8)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率 (default: 300)')
    parser.add_argument('--smooth', action='store_true', default=True,
                       help='是否平滑曲线 (default: True)')
    parser.add_argument('--no-val', action='store_true', default=False,
                       help='不显示验证损失 (default: False)')
    
    args = parser.parse_args()
    
    # 解析配置参数
    config_paths = {}
    for config_str in args.configs:
        if ':' not in config_str:
            raise ValueError(f"配置格式错误: {config_str}. 应该是 name:path")
        name, path = config_str.split(':', 1)
        config_paths[name] = path
    
    # 解析所有日志文件
    parser = LossParser()
    config_data = {}
    
    print("解析训练日志...")
    print("=" * 50)
    
    for config_name, log_path in config_paths.items():
        try:
            data = parser.parse_log_file(log_path)
            config_data[config_name] = data
            
            train_points = len(data['train']['Loss'])
            val_points = len(data['val']['Loss'])
            
            print(f"✓ {config_name}:")
            print(f"    训练点数: {train_points}")
            print(f"    验证点数: {val_points}")
            print(f"    总Epochs: {data['epochs']}")
            print(f"    每epoch迭代: {data['iters_per_epoch']}")
            
            if train_points > 0:
                final_loss = data['train']['Loss'][-1]
                print(f"    最终训练损失: {final_loss:.4f}")
            
            if val_points > 0:
                final_val_loss = data['val']['Loss'][-1]
                print(f"    最终验证损失: {final_val_loss:.4f}")
                
            print("-" * 50)
            
        except Exception as e:
            print(f"✗ 解析 {config_name} 失败: {e}")
            continue
    
    if not config_data:
        print("没有成功解析任何日志文件！")
        return
    
    # 绘制损失曲线
    print(f"绘制损失曲线...")
    plotter = LossCurvePlotter()
    plotter.plot_multiple_configs(
        config_data, 
        save_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        smooth=args.smooth, 
        show_val=not args.no_val
    )
    
    print(f"✓ 完成! 输出文件: {args.output}")

if __name__ == "__main__":
    main()

