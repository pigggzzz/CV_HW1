"""
可视化与分析功能
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams


def plot_training_curves(train_losses, val_losses, val_accuracies, save_path='./figures'):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
        save_path: 保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Loss 曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(epochs, val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到 {save_path}/training_curves.png")
    plt.close()


def visualize_first_layer_weights(model, save_path='./figures', input_shape=(28, 28)):
    """
    可视化第一层权重（作为图像）
    
    Args:
        model: MLP 模型
        save_path: 保存路径
        input_shape: 输入图像形状
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 获取第一层权重
    weights = model.get_first_layer_weights()  # shape: (784, hidden_dim)
    
    # 取前 100 个神经元的权重
    num_neurons = min(100, weights.shape[1])
    weights_subset = weights[:, :num_neurons]  # shape: (784, 100)
    
    # 计算最小最大值用于归一化
    w_min = weights_subset.min()
    w_max = weights_subset.max()
    weights_normalized = (weights_subset - w_min) / (w_max - w_min + 1e-8)
    
    # 创建网格显示
    grid_size = int(np.sqrt(num_neurons))
    if grid_size * grid_size < num_neurons:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(grid_size * grid_size):
        ax = axes[idx]
        
        if idx < num_neurons:
            # 获取该神经元的权重并重塑为图像
            neuron_weights = weights_normalized[:, idx].reshape(input_shape)
            
            # 显示图像
            im = ax.imshow(neuron_weights, cmap='gray')
            ax.set_title(f'Neuron {idx}', fontsize=8)
        
        ax.axis('off')
    
    plt.suptitle(f'First Hidden Layer Weights ({num_neurons} neurons)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/first_layer_weights.png', dpi=300, bbox_inches='tight')
    print(f"第一层权重可视化已保存到 {save_path}/first_layer_weights.png")
    plt.close()


def visualize_confusion_matrix(confusion_matrix, class_names=None, save_path='./figures'):
    """
    可视化混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 添加数值
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到 {save_path}/confusion_matrix.png")
    plt.close()


def visualize_misclassified_samples(misclassified_samples, class_names=None, 
                                   save_path='./figures', num_samples=5):
    """
    可视化分类错误的样本
    
    Args:
        misclassified_samples: 分类错误的样本列表
        class_names: 类别名称
        save_path: 保存路径
        num_samples: 显示的样本数
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    num_display = min(num_samples, len(misclassified_samples))
    
    fig, axes = plt.subplots(2, num_display, figsize=(14, 6))
    if num_display == 1:
        axes = axes.reshape(2, 1)
    
    for idx, sample in enumerate(misclassified_samples[:num_display]):
        # 获取图像
        image = sample['image'].reshape(28, 28)
        
        # 显示图像
        ax = axes[0, idx]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {class_names[sample["true_label"]]}', fontsize=10)
        ax.axis('off')
        
        # 显示预测结果
        ax = axes[1, idx]
        logits = sample['logits']
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        ax.barh(range(10), probabilities)
        ax.set_yticks(range(10))
        ax.set_yticklabels([name.split('/')[0] for name in class_names], fontsize=8)
        ax.set_xlabel('Probability', fontsize=9)
        ax.set_title(f'Pred: {class_names[sample["predicted_label"]]}', fontsize=10)
        
        # 高亮真实标签和预测标签
        true_label = sample['true_label']
        pred_label = sample['predicted_label']
        ax.get_children()[true_label].set_color('green')
        ax.get_children()[pred_label].set_color('red')
    
    plt.suptitle('Misclassified Samples and Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/misclassified_samples.png', dpi=300, bbox_inches='tight')
    print(f"错误分类样本已保存到 {save_path}/misclassified_samples.png")
    plt.close()


def analyze_misclassified_samples(misclassified_samples, class_names=None):
    """
    分析错误分类的原因
    
    Args:
        misclassified_samples: 分类错误的样本列表
        class_names: 类别名称
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    print("\n" + "="*70)
    print("错误分类样本分析")
    print("="*70)
    
    for idx, sample in enumerate(misclassified_samples):
        true_label = sample['true_label']
        pred_label = sample['predicted_label']
        logits = sample['logits']
        
        # 计算 softmax 概率
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # 获取置信度
        pred_confidence = probs[pred_label]
        true_confidence = probs[true_label]
        
        print(f"\n样本 {idx+1}:")
        print(f"  真实标签: {class_names[true_label]}")
        print(f"  预测标签: {class_names[pred_label]} (置信度: {pred_confidence:.4f})")
        print(f"  真实类别置信度: {true_confidence:.4f}")
        print(f"  置信度差: {pred_confidence - true_confidence:.4f}")
        
        # 获取前3个预测
        top_3_indices = np.argsort(probs)[-3:][::-1]
        print(f"  Top-3 预测:")
        for rank, pred_idx in enumerate(top_3_indices):
            print(f"    {rank+1}. {class_names[pred_idx]}: {probs[pred_idx]:.4f}")


def _hparam_run_label(r, max_len=52):
    """将单条超参记录压缩为短标签，用于条形图。"""
    parts = []
    for k in ("learning_rate", "hidden_dim", "l2_lambda", "batch_size", "activation"):
        if k in r:
            v = r[k]
            if k == "learning_rate" or k == "l2_lambda":
                parts.append(f"{k[:3]}={v:.4g}")
            else:
                parts.append(f"{k[:3]}={v}")
    s = " ".join(parts)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def plot_hyperparameter_search_results(results, save_path="./figures", metric="best_val_accuracy", top_n=16):
    """
    可视化超参数搜索结果：按 run_id 的指标变化 + 最优若干组的横向对比。

    Args:
        results: hyperparameter_search 中记录的 run 列表（含 run_id 与各超参）
        save_path: 输出目录
        metric: 用于排序与绘图的字段，默认 best_val_accuracy
        top_n: 条形图展示前多少组
    """
    import os

    if not results:
        print("plot_hyperparameter_search_results: 无数据，跳过。")
        return
    os.makedirs(save_path, exist_ok=True)
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    by_id = sorted(results, key=lambda x: x.get("run_id", 0))
    xs = [r.get("run_id", i) for i, r in enumerate(by_id)]
    ys = [float(r.get(metric, 0.0)) for r in by_id]

    sorted_desc = sorted(results, key=lambda x: float(x.get(metric, 0.0)), reverse=True)
    top = sorted_desc[: min(top_n, len(sorted_desc))]
    labels = [_hparam_run_label(r) for r in top]
    vals = [float(r.get(metric, 0.0)) for r in top]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(xs, ys, "o-", color="#2c7fb8", linewidth=1.5, markersize=5, label=metric)
    axes[0].set_xlabel("run_id / 搜索顺序", fontsize=11)
    axes[0].set_ylabel(metric, fontsize=11)
    axes[0].set_title("各次超参组合的验证指标（按搜索顺序）", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    y_pos = np.arange(len(top))
    axes[1].barh(y_pos, vals, color="#7fcdbb", edgecolor="#2c7fb8", linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"#{t.get('run_id', '?')} " + _hparam_run_label(t, 40) for t in top], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(metric, fontsize=11)
    axes[1].set_title(f"Top-{len(top)} 超参组合（按 {metric}）", fontsize=13, fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.35)

    plt.tight_layout()
    out = os.path.join(save_path, "hyperparameter_search_summary.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"超参数搜索汇总图已保存: {out}")
