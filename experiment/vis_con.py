import os
import torch 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.mixture import GaussianMixture
from plyfile import PlyData, PlyElement
from pathlib import Path
import colorsys
import scipy.stats

def gaussian_mixture_filter(entropies: np.ndarray, 
                            n_components: int = 2,
                            remove_component: int = -1):
        # 拟合高斯混合模型  
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(entropies)
        
        # 获取每个分量的统计信息
        component_means = []
        for i in range(n_components):
            component_mask = labels == i
            if component_mask.sum() > 0:
                component_mean = entropies[component_mask].mean()
                component_means.append((i, component_mean, component_mask.sum()))
        
        # 按均值排序，移除指定分量
        component_means.sort(key=lambda x: x[1])  # 按均值排序

        if remove_component == -1:
            # 移除最高熵的分量
            remove_idx = component_means[-1][0]
        else:
            remove_idx = remove_component
        
        keep_mask = labels != remove_idx
        
        removed_count = (~keep_mask).sum()
        print(f"GMM筛选: 识别到{n_components}个分量，移除分量{remove_idx}")
        print(f"分量统计: {[(f'分量{i}: mean={mean:.3f}, count={count}') for i, mean, count in component_means]}")
        print(f"移除: {removed_count} ({removed_count/len(entropies)*100:.1f}%)")
        
        return keep_mask


def fast_kmeans_entropy_filter(entropies, n_clusters=2, max_iters=30, tolerance=1e-6):
    """使用优化的K-Means进行熵过滤，更快更稳定"""
    
    # 1. 预处理：移除异常值，提升稳定性
    entropy_tensor = torch.tensor(entropies.flatten(), dtype=torch.float32, device="cuda:0")
    
    # 移除极端异常值（可选）
    q1, q99 = torch.quantile(entropy_tensor, torch.tensor([0.01, 0.99], device="cuda:0"))
    valid_mask = (entropy_tensor >= q1) & (entropy_tensor <= q99)
    if valid_mask.sum() < len(entropy_tensor) * 0.9:  # 如果异常值太多，不过滤
        valid_mask = torch.ones_like(entropy_tensor, dtype=torch.bool)
    
    valid_entropies = entropy_tensor[valid_mask]
    n_valid = len(valid_entropies)
    
    if n_valid < n_clusters:
        print(f"Warning: 有效数据点({n_valid})少于聚类数({n_clusters})，使用简单阈值过滤")
        threshold = torch.median(entropy_tensor)
        keep_mask = entropy_tensor <= threshold
        return keep_mask.cpu().numpy()
    
    # 2. 智能初始化：使用K-means++思想
    centers = torch.zeros(n_clusters, device="cuda:0", dtype=torch.float32)
    
    if n_clusters == 2:
        # 对于二分类，使用更稳定的初始化
        centers[0] = torch.quantile(valid_entropies, 0.2)
        centers[1] = torch.quantile(valid_entropies, 0.8)
    else:
        # K-means++风格初始化
        centers[0] = valid_entropies[torch.randint(0, n_valid, (1,))]
        for i in range(1, n_clusters):
            # 计算到最近中心的距离
            distances = torch.min(torch.abs(valid_entropies.unsqueeze(1) - centers[:i].unsqueeze(0)), dim=1)[0]
            # 按距离加权选择下一个中心
            probs = distances / distances.sum()
            centers[i] = valid_entropies[torch.multinomial(probs, 1)]
    
    # 3. 优化的K-means主循环
    prev_labels = torch.zeros(n_valid, dtype=torch.long, device="cuda:0")
    
    for iteration in range(max_iters):
        # 向量化距离计算 (n_points, n_clusters)
        distances = torch.abs(valid_entropies.unsqueeze(1) - centers.unsqueeze(0))
        labels = torch.argmin(distances, dim=1)
        
        # 早期停止检查
        if iteration > 0 and torch.equal(labels, prev_labels):
            break
        
        prev_labels = labels.clone()
        
        # 批量更新所有中心
        old_centers = centers.clone()
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = valid_entropies[mask].mean()
            # 如果某个聚类为空，重新随机初始化
            else:
                centers[k] = valid_entropies[torch.randint(0, n_valid, (1,))]
        
        # 收敛检查
        center_shift = torch.norm(centers - old_centers)
        if center_shift < tolerance:
            break
    
    # 4. 选择最优聚类（低熵）
    cluster_stats = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_mean = centers[k].item()
            cluster_size = mask.sum().item()
            cluster_std = valid_entropies[mask].std().item() if mask.sum() > 1 else 0.0
            cluster_stats.append((k, cluster_mean, cluster_size, cluster_std))
    
    # 按均值排序，选择低熵聚类
    cluster_stats.sort(key=lambda x: x[1])
    keep_component = cluster_stats[0][0]
    
    # 5. 生成完整的keep_mask
    valid_keep_mask = labels == keep_component
    
    # 将结果映射回原始数据
    full_keep_mask = torch.zeros_like(entropy_tensor, dtype=torch.bool)
    full_keep_mask[valid_mask] = valid_keep_mask
    
    # 对于被过滤掉的异常值，保守处理（保留低熵的）
    if not torch.all(valid_mask):
        invalid_entropies = entropy_tensor[~valid_mask]
        threshold = centers[keep_component]
        full_keep_mask[~valid_mask] = invalid_entropies <= threshold
    
    # 统计信息
    filter_rate = (~full_keep_mask).sum().item() / len(full_keep_mask) * 100
    kept_mean = entropy_tensor[full_keep_mask].mean().item()
    
    print(f"优化K-Means过滤: 保留聚类{keep_component} (中心:{cluster_stats[0][1]:.4f})")
    print(f"  聚类统计: {[(f'C{i}:μ={mean:.3f},n={size},σ={std:.3f}') for i,mean,size,std in cluster_stats]}")
    print(f"  过滤率: {filter_rate:.1f}%, 保留均值: {kept_mean:.4f}, 迭代: {iteration + 1}")
    
    return full_keep_mask.cpu().numpy()


# 进一步优化：如果数据量很大，可以使用采样版本
def fast_kmeans_entropy_filter_sampled(entropies, n_clusters=2, sample_ratio=0.1, max_iters=20):
    """使用采样的K-Means进行熵过滤，适用于大数据集"""
    
    entropy_tensor = torch.tensor(entropies.flatten(), dtype=torch.float32, device="cuda:0")
    n_total = len(entropy_tensor)
    
    # 如果数据量小，直接使用完整版本
    if n_total < 10000:
        return fast_kmeans_entropy_filter(entropies, n_clusters, max_iters)
    
    # 分层采样：保证各个熵值范围都有代表性
    sorted_indices = torch.argsort(entropy_tensor)
    n_sample = max(1000, int(n_total * sample_ratio))
    
    # 均匀采样
    sample_indices = sorted_indices[torch.linspace(0, n_total-1, n_sample, dtype=torch.long)]
    sample_entropies = entropy_tensor[sample_indices]
    
    # 在采样数据上运行K-means
    sample_keep_mask = fast_kmeans_entropy_filter(
        sample_entropies.cpu().numpy(), n_clusters, max_iters
    )
    
    # 找到决策边界
    kept_samples = sample_entropies[torch.tensor(sample_keep_mask, device="cuda:0")]
    threshold = kept_samples.max().item()
    
    # 应用到全数据集
    full_keep_mask = entropy_tensor <= threshold
    
    filter_rate = (~full_keep_mask).sum().item() / len(full_keep_mask) * 100
    print(f"采样K-Means过滤: 采样{n_sample}/{n_total}, 阈值:{threshold:.4f}, 过滤率:{filter_rate:.1f}%")
    
    return full_keep_mask.cpu().numpy()


def calculate_entropy(semantic_probs):
    """
    Calculate entropy for semantic probabilities
    Args:
        semantic_probs: (N, 11) semantic probability distribution (already normalized)
    Returns:
        entropy: (N,) entropy values in bits
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    semantic_probs = semantic_probs + epsilon 
    
    # Calculate entropy: -sum(p * log2(p)) 
    entropy = -(semantic_probs * np.log2(semantic_probs)).sum(axis=1) 

    return entropy 
 

def compute_confidence_methods(entropy_values, opacity=None): 
    """
    计算三种不同的置信度方法
    Args:
        entropy_values: numpy array of entropy values
    Returns:
        dict: 包含三种方法的置信度结果
    """  
    # 转换为torch tensor  
    entropy_tensor = torch.tensor(entropy_values, dtype=torch.float32)
    
    # 方法1: 简单指数变换
    conf_exp = torch.exp(-entropy_tensor)
    
    # 方法2: 锐化sigmoid变换
    def sharp_sigmoid_transform(entropy, midpoint=1.0, steepness=3.0): 
        """
        Inverse sigmoid with adjustable steepness
        steepness越大越锐利 
        """ 
        return torch.sigmoid(-steepness * (entropy - midpoint)) 
    
    conf_sigmoid = sharp_sigmoid_transform(entropy_tensor) 
     
    # 方法3: 幂变换
    def power_transform(entropy, max_entropy=3.0, power=3.0): 
        """ 
        Power-based transformation 
        power越大越锐利 
        """ 
        normalized_entropy = torch.clamp(entropy / max_entropy, 0, 1)
        return (1 - normalized_entropy) ** power
    
    conf_power = power_transform(entropy_tensor, power=3.0)  
    
    if opacity is not None: 
        opacity_tensor = torch.tensor(opacity, dtype=torch.float32) 
        conf_exp *= opacity_tensor 
        conf_sigmoid *= opacity_tensor 
        conf_power *= opacity_tensor 
    
    return {
        'entropy': entropy_values,
        'exp': conf_exp.numpy(),
        'sigmoid': conf_sigmoid.numpy(), 
        'power': conf_power.numpy()
    }


def plot_confidence_comparison(entropy_values, opacity=None, output_path=None, figsize=(15, 10)):
    """
    绘制三种置信度计算方法的对比图
    Args:
        entropy_values: numpy array of entropy values
        output_path: 输出文件路径
        figsize: 图片尺寸
    """
    # 计算三种置信度
    conf_results = compute_confidence_methods(entropy_values, opacity)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Confidence Computation Methods Comparison', fontsize=16)
    
    # 1. 熵分布直方图
    axes[0, 0].hist(conf_results['entropy'], bins=50, alpha=0.7, color='gray', edgecolor='black')
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Original Entropy Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 方法1: exp(-entropy)
    axes[0, 1].hist(conf_results['exp'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Method 1: exp(-entropy)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 方法2: sigmoid
    axes[0, 2].hist(conf_results['sigmoid'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_xlabel('Confidence')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Method 2: Sharp Sigmoid')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 方法3: power
    axes[1, 0].hist(conf_results['power'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Method 3: Power Transform')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 变换函数曲线对比
    entropy_range = np.linspace(0, entropy_values.max(), 1000)
    conf_curves = compute_confidence_methods(entropy_range)
    
    axes[1, 1].plot(entropy_range, conf_curves['exp'], 'r-', label='exp(-entropy)', linewidth=2)
    axes[1, 1].plot(entropy_range, conf_curves['sigmoid'], 'b-', label='sharp sigmoid', linewidth=2)
    axes[1, 1].plot(entropy_range, conf_curves['power'], 'g-', label='power transform', linewidth=2)
    axes[1, 1].set_xlabel('Entropy')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Transformation Functions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 散点图对比 (entropy vs confidence)
    sample_indices = np.random.choice(len(entropy_values), min(5000, len(entropy_values)), replace=False)
    sample_entropy = conf_results['entropy'][sample_indices]
    
    axes[1, 2].scatter(sample_entropy, conf_results['exp'][sample_indices], 
                      alpha=0.3, s=1, color='red', label='exp(-entropy)')
    axes[1, 2].scatter(sample_entropy, conf_results['sigmoid'][sample_indices], 
                      alpha=0.3, s=1, color='blue', label='sharp sigmoid')
    axes[1, 2].scatter(sample_entropy, conf_results['power'][sample_indices], 
                      alpha=0.3, s=1, color='green', label='power transform')
    axes[1, 2].set_xlabel('Entropy')
    axes[1, 2].set_ylabel('Confidence')
    axes[1, 2].set_title('Entropy vs Confidence (Sampled)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 打印统计信息
    print("\n=== Confidence Methods Statistics ===")
    for method in ['exp', 'sigmoid', 'power']:
        conf_vals = conf_results[method]
        print(f"{method.upper():>8}: mean={conf_vals.mean():.4f}, std={conf_vals.std():.4f}, "
              f"min={conf_vals.min():.4f}, max={conf_vals.max():.4f}")
    
    # 相关性分析
    print("\n=== Method Correlations ===")
    methods = ['exp', 'sigmoid', 'power']
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            corr, _ = scipy.stats.pearsonr(conf_results[methods[i]], conf_results[methods[j]])
            print(f"{methods[i]} vs {methods[j]}: correlation = {corr:.4f}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nConfidence comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return conf_results


def confidence_to_color(confidence_values, colormap='plasma'):
    """ 
    Convert confidence values to colors 
    Args: 
        confidence_values: (N,) confidence values 
        colormap: matplotlib colormap name 
    Returns: 
        colors: (N, 3) RGB colors 
    """
    # Normalize confidence to [0, 1]
    conf_norm = (confidence_values - confidence_values.min()) / (confidence_values.max() - confidence_values.min() + 1e-8)
     
    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colors = cmap(conf_norm)[:, :3]  # Take only RGB, ignore alpha
    
    return colors

def entropy_to_color(entropy_values, colormap='viridis'):
    """
    Convert entropy values to colors
    Args:
        entropy_values: (N,) entropy values
        colormap: matplotlib colormap name
    Returns:
        colors: (N, 3) RGB colors
    """
    # Normalize entropy to [0, 1]
    entropy_norm = (entropy_values - entropy_values.min()) / (entropy_values.max() - entropy_values.min() + 1e-8)
     
    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colors = cmap(entropy_norm)[:, :3]  # Take only RGB, ignore alpha
    
    return colors

def semantic_to_color(semantic_probs):
    """
    Convert semantic probabilities to RGB colors using dominant class
    Args:
        semantic_probs: (N, 11) semantic probability distribution
    Returns:
        colors: (N, 3) RGB colors
    """
    # Define colors for 11 semantic classes (you can adjust these) 
    class_colors = np.array([
        # [0.0, 0.0, 0.0],      # 0: Background
        [0.84, 0.48, 0.48],   # 1: Ceiling
        [0.48, 0.84, 0.48],   # 2: Floor
        [0.48, 0.48, 0.84],   # 3: Wall 
        [0.84, 0.84, 0.48],   # 4: Window
        [0.84, 0.48, 0.84],   # 5: Chair
        [0.63, 0.87, 0.96],   # 6: Bed
        [0.79, 0.68, 0.85],   # 7: Sofa
        [0.96, 0.72, 0.48],   # 8: Table
        [0.6, 0.72, 0.48],    # 9: TVs
        [0.48, 0.72, 0.72],   # 10: Furniture
        [0.32, 0.54, 0.78],   # 11: Object
        # [0.96, 0.96, 0.96],   # 12: Empty
    ]) 
    
    # Get dominant class for each gaussian
    dominant_class = np.argmax(semantic_probs, axis=1)
    
    # Map to colors
    colors = class_colors[dominant_class]
    
    return colors

class GaussianSemanticVisualizer:
    """Gaussian visualizer with semantic information and entropy filtering"""
    
    def __init__(self, ply_file_path):
        """
        Initialize the visualizer
        Args:
            ply_file_path: Path to the .ply file
        """
        self.ply_file_path = ply_file_path
        self.load_ply_data()
        self.confidence = None
    
    def load_ply_data(self):
        """Load Gaussian data with semantic information from .ply file"""
        print(f"Loading PLY file: {self.ply_file_path}")
        plydata = PlyData.read(self.ply_file_path)
        vertex = plydata['vertex']

        # Extract basic attributes
        xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        self.xyz = xyz[:, [0, 2, 1]] * -1  # Apply same coordinate transformation
        # self.xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
 
        self.opacity = vertex['opacity']
        self.scales = np.column_stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']])
        self.rotations = np.column_stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']])
        
        # Extract semantic information (11 dimensions)
        self.semantics = np.column_stack([
            vertex[f'sem_{i}'] for i in range(11)
        ])
        
        # Convert log scale back to original scale
        self.scales = np.exp(self.scales)
        
        # Convert logit opacity back to original opacity
        self.opacity = 1 / (1 + np.exp(-self.opacity))
        
        # Apply softmax to semantic probabilities
        # self.semantics = np.exp(self.semantics) / np.sum(np.exp(self.semantics), axis=1, keepdims=True)
        
        # Calculate entropy
        self.entropy = calculate_entropy(self.semantics) 
        
        # Calculate confidence using all methods
        self.conf_results = compute_confidence_methods(self.entropy, self.opacity)
        self.confidence = self.conf_results['sigmoid']  # Default to sigmoid method
        
        # Generate colors from semantics, entropy, and confidence
        self.semantic_colors = semantic_to_color(self.semantics)
        self.entropy_colors = entropy_to_color(self.entropy)
        self.confidence_colors = confidence_to_color(self.confidence)
        
        print(f"Successfully loaded {len(self.xyz)} Gaussian ellipsoids")
        print(f"Position range: X[{self.xyz[:,0].min():.2f}, {self.xyz[:,0].max():.2f}], "
              f"Y[{self.xyz[:,1].min():.2f}, {self.xyz[:,1].max():.2f}], "
              f"Z[{self.xyz[:,2].min():.2f}, {self.xyz[:,2].max():.2f}]")
        print(f"Entropy range: [{self.entropy.min():.3f}, {self.entropy.max():.3f}]")
        print(f"Confidence range: [{self.confidence.min():.4f}, {self.confidence.max():.4f}]")
        print(f"Semantic shape: {self.semantics.shape}")
    
    def add_confidence(self, method='sigmoid'):
        """
        添加置信度计算
        Args:
            method: 'exp', 'sigmoid', or 'power'
        """
        self.confidence = self.conf_results[method]
        self.confidence_colors = confidence_to_color(self.confidence)
        print(f"Updated confidence using {method} method: range [{self.confidence.min():.4f}, {self.confidence.max():.4f}]")
    
    def filter_by_entropy(self, entropy_threshold_low=0.5, entropy_threshold_high=2.0, 
                         opacity_threshold=0.01):
        """
        Filter gaussians by entropy range and opacity
        Args:
            entropy_threshold_low: Minimum entropy threshold
            entropy_threshold_high: Maximum entropy threshold
            opacity_threshold: Minimum opacity threshold
        Returns: 
            Filtered data 
        """ 

        # Create filter mask 
        entropy = np.expand_dims(self.entropy, axis=-1) 
        # keep_mask = gaussian_mixture_filter(entropy, 2)  
        # 根据数据量选择版本
        if len(entropy) > 50000: 
            keep_mask = fast_kmeans_entropy_filter_sampled(entropy, n_clusters=2)
        else:
            keep_mask = fast_kmeans_entropy_filter(entropy, n_clusters=2, max_iters=30)
    
        entropy_mask = (self.entropy >= entropy_threshold_low) & (self.entropy <= entropy_threshold_high) 
        opacity_mask = self.opacity.flatten() > opacity_threshold 

        # combined_mask = entropy_mask & opacity_mask
        combined_mask = keep_mask & opacity_mask
        print(f"keep_mask: {keep_mask.shape}, entropy_mask: {entropy_mask.shape}, opacity_mask: {opacity_mask.shape}, combined_mask: {combined_mask.shape}")

        # Apply filter
        filtered_xyz = self.xyz[combined_mask]
        filtered_opacity = self.opacity[combined_mask]
        filtered_scales = self.scales[combined_mask]
        filtered_rotations = self.rotations[combined_mask]
        filtered_semantics = self.semantics[combined_mask]
        filtered_entropy = self.entropy[combined_mask]
        filtered_semantic_colors = self.semantic_colors[combined_mask]
        filtered_entropy_colors = self.entropy_colors[combined_mask]
        filtered_confidence = self.confidence[combined_mask]
        filtered_confidence_colors = self.confidence_colors[combined_mask]
        
        print(f"After entropy filtering [{entropy_threshold_low:.2f}, {entropy_threshold_high:.2f}]: "
              f"{len(filtered_xyz)} ellipsoids remaining "
              f"({len(filtered_xyz)/len(self.xyz)*100:.1f}% of original)")
        
        return (filtered_xyz, filtered_opacity, filtered_scales, filtered_rotations,
                filtered_semantics, filtered_entropy, filtered_semantic_colors, filtered_entropy_colors,
                filtered_confidence, filtered_confidence_colors)
    
    def save_filtered_gaussians(self, filtered_data, output_path):
        """
        Save filtered gaussians to PLY file 
        Args:
            filtered_data: Tuple of filtered data
            output_path: Output PLY file path
        """
        (xyz, opacity, scales, rotations, semantics, entropy, sem_colors, ent_colors, confidence, conf_colors) = filtered_data
 
        # Convert back to original format 
        xyz_original = xyz.copy() 
        xyz_original = xyz_original * -1 
        xyz_original = xyz_original[:, [0, 2, 1]]  # Reverse coordinate transformation
        
        # Convert back to log scale and logit opacity
        scales_log = np.log(scales)
        opacity_logit = np.log(opacity / (1 - opacity + 1e-8))
        
        # Convert semantics back to logits (inverse softmax)
        semantics_logits = np.log(semantics + 1e-8)
        
        # Create vertex data 
        vertex_data = []
        for i in range(len(xyz_original)):
            vertex_tuple = (
                xyz_original[i, 0], xyz_original[i, 1], xyz_original[i, 2],  # x, y, z
                opacity_logit[i],  # opacity
                sem_colors[i, 0], sem_colors[i, 1], sem_colors[i, 2],  # f_dc_0, f_dc_1, f_dc_2 
                scales_log[i, 0], scales_log[i, 1], scales_log[i, 2],  # scale_0, scale_1, scale_2
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],  # rot_0-3
                # *semantics_logits[i],  # sem_0 to sem_10  
                # entropy[i]  # Add entropy as additional attribute
            ) 
            vertex_data.append(vertex_tuple)
        
        # Define vertex properties
        vertex_properties = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4'), 
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), 
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), 
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'), 
        ] 
        
        # Add semantic properties
        # for i in range(11):
        #     vertex_properties.append((f'sem_{i}', 'f4'))
        
        # Add entropy property
        # vertex_properties.append(('entropy', 'f4'))
        
        # Create vertex element
        vertex_array = np.array(vertex_data, dtype=vertex_properties)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        
        # Save PLY file
        PlyData([vertex_element]).write(output_path)
        print(f"Filtered gaussians saved to: {output_path}")
    
    def save_confidence_colored_gaussians(self, filtered_data, output_path):
        """
        Save confidence-colored gaussians to PLY file 
        Args:
            filtered_data: Tuple of filtered data
            output_path: Output PLY file path
        """
        (xyz, opacity, scales, rotations, semantics, entropy, sem_colors, ent_colors, confidence, conf_colors) = filtered_data
 
        # Convert back to original format 
        xyz_original = xyz.copy() 
        xyz_original = xyz_original * -1 
        xyz_original = xyz_original[:, [0, 2, 1]]  # Reverse coordinate transformation
        
        # Convert back to log scale and logit opacity
        scales_log = np.log(scales)
        opacity_logit = np.log(opacity / (1 - opacity + 1e-8))
        
        # Create vertex data with confidence colors in f_dc_*
        vertex_data = []
        for i in range(len(xyz_original)):
            vertex_tuple = (
                xyz_original[i, 0], xyz_original[i, 1], xyz_original[i, 2],  # x, y, z
                opacity_logit[i],  # opacity
                conf_colors[i, 0], conf_colors[i, 1], conf_colors[i, 2],  # f_dc_0, f_dc_1, f_dc_2 (confidence colors)
                scales_log[i, 0], scales_log[i, 1], scales_log[i, 2],  # scale_0, scale_1, scale_2
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],  # rot_0-3
            )
            vertex_data.append(vertex_tuple)
        
        # Define vertex properties
        vertex_properties = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4'), 
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), 
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), 
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'), 
        ] 
        
        # Create vertex element
        vertex_array = np.array(vertex_data, dtype=vertex_properties)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        
        # Save PLY file
        PlyData([vertex_element]).write(output_path)
        print(f"Confidence-colored gaussians saved to: {output_path}")
        print(f"Confidence range in saved file: [{confidence.min():.4f}, {confidence.max():.4f}]")
    
    def visualize_entropy_distribution(self, output_path=None, figsize=(12, 8)):
        """
        Visualize entropy distribution
        Args: 
            output_path: Output file path
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Entropy histogram
        ax1.hist(self.entropy, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Entropy')
        ax1.set_ylabel('Count')
        ax1.set_title('Entropy Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Entropy vs Opacity scatter plot
        scatter = ax2.scatter(self.opacity, self.entropy, c=self.entropy, 
                            cmap='viridis', alpha=0.6, s=1)
        ax2.set_xlabel('Opacity')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy vs Opacity')
        plt.colorbar(scatter, ax=ax2, label='Entropy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Entropy distribution plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Gaussian Semantic Visualization and Filtering Tool') 
    parser.add_argument('--ply_dir', type=str, required=True, help='PLY file directory')
    parser.add_argument('--ply_name', type=str, required=True, help='PLY file name (without extension)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--entropy_low', type=float, default=0.5, help='Minimum entropy threshold')
    parser.add_argument('--entropy_high', type=float, default=2.0, help='Maximum entropy threshold')
    parser.add_argument('--opacity_threshold', type=float, default=0.01, help='Opacity threshold')
    parser.add_argument('--save_filtered', action='store_true', help='Save filtered gaussians to PLY file')
    parser.add_argument('--save_confidence_colored', action='store_true', help='Save confidence-colored gaussians to PLY file')
    parser.add_argument('--confidence_method', type=str, default='sigmoid', choices=['exp', 'sigmoid', 'power'],
                       help='Confidence computation method') 
     
    args = parser.parse_args() 
    
    # Construct full PLY file path
    ply_file_path = os.path.join(args.ply_dir, f"{args.ply_name}.ply")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create visualizer
        visualizer = GaussianSemanticVisualizer(ply_file_path)
        
        # Generate entropy distribution plot
        entropy_plot_path = output_dir / f"{args.ply_name}_entropy_dist.png"
        visualizer.visualize_entropy_distribution(output_path=str(entropy_plot_path))
        
        # Generate confidence comparison plot
        confidence_plot_path = output_dir / f"{args.ply_name}_confidence_comparison.png"
        conf_results = plot_confidence_comparison(visualizer.entropy, visualizer.opacity, output_path=str(confidence_plot_path))
        # conf_results = plot_confidence_comparison(visualizer.entropy, None, output_path=str(confidence_plot_path))
        
        # Add confidence to visualizer
        visualizer.add_confidence(method=args.confidence_method)
        
        # Determine if we need to filter
        need_filtering = args.save_filtered
        
        if need_filtering:
            # Filter by entropy
            filtered_data = visualizer.filter_by_entropy(
                entropy_threshold_low=args.entropy_low,
                entropy_threshold_high=args.entropy_high,
                opacity_threshold=args.opacity_threshold
            )
            
            # Save filtered gaussians if requested
            if args.save_filtered:
                filtered_ply_path = output_dir / f"{args.ply_name}_filtered.ply"
                visualizer.save_filtered_gaussians(filtered_data, str(filtered_ply_path))
            
            # Save confidence-colored filtered gaussians if requested
            if args.save_confidence_colored:
                confidence_ply_path = output_dir / f"{args.ply_name}_confidence_{args.confidence_method}.ply"
                visualizer.save_confidence_colored_gaussians(filtered_data, str(confidence_ply_path))
        else: 
            # No filtering needed, work with all data
            if args.save_confidence_colored:
                # Create "all data" tuple in the same format as filtered_data
                all_data = (visualizer.xyz, visualizer.opacity, visualizer.scales, visualizer.rotations,
                           visualizer.semantics, visualizer.entropy, visualizer.semantic_colors, 
                           visualizer.entropy_colors, visualizer.confidence, visualizer.confidence_colors)
                
                confidence_ply_path = output_dir / f"{args.ply_name}_all_confidence_{args.confidence_method}.ply"
                visualizer.save_confidence_colored_gaussians(all_data, str(confidence_ply_path)) 
        
        print(f"\nProcessing completed! Output files saved in: {output_dir}")
        print(f"Entropy range: [{args.entropy_low}, {args.entropy_high}]")
        print(f"Opacity threshold: {args.opacity_threshold}") 
        print(f"Confidence method: {args.confidence_method}")
        print(f"Filtering applied: {need_filtering}")
        
        if args.save_filtered: 
            print(f"Filtered PLY saved: {args.ply_name}_filtered.ply")
        if args.save_confidence_colored:
            if need_filtering:
                print(f"Confidence-colored (filtered) PLY saved: {args.ply_name}_confidence_{args.confidence_method}.ply")
            else:
                print(f"Confidence-colored (all data) PLY saved: {args.ply_name}_all_confidence_{args.confidence_method}.ply")
        
    except FileNotFoundError:
        print(f"File not found: {ply_file_path}")
        print("Please ensure the file path is correct!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

