"""
主训练脚本：完整的训练流程
"""
import numpy as np
import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import FashionMNISTLoader
from src.model import MLP
from src.optim import SGD, LearningRateDecay
from src.train import Trainer
from src.evaluate import Evaluator
from src.hyperparameter_search import GridSearchCV, RandomSearchCV


def download_and_prepare_data():
    """下载并准备 Fashion-MNIST 数据集"""
    print("="*60)
    print("准备数据集...")
    print("="*60)
    
    # 创建数据加载器
    loader = FashionMNISTLoader('./data')
    
    # 下载数据
    try:
        loader.download_data()
    except Exception as e:
        print(f"下载失败: {e}")
        print("请确保有网络连接或手动下载数据集")
    
    # 加载数据
    loader.load_data()
    
    # 获取训练集、验证集、测试集
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        loader.get_train_test_split(val_ratio=0.1, normalize=True, flatten=True)
    
    print(f"训练集大小: {train_images.shape}")
    print(f"验证集大小: {val_images.shape}")
    print(f"测试集大小: {test_images.shape}")
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def train_model(train_images, train_labels, val_images, val_labels,
                hidden_dim=256, learning_rate=0.01, l2_lambda=0.0001,
                batch_size=32, epochs=100, activation='relu'):
    """
    训练模型
    
    Args:
        train_images: 训练集图像
        train_labels: 训练集标签
        val_images: 验证集图像
        val_labels: 验证集标签
        hidden_dim: 隐藏层维度
        learning_rate: 学习率
        l2_lambda: L2 正则化强度
        batch_size: 批大小
        epochs: 训练轮数
        activation: 激活函数
        
    Returns:
        (trained_model, trainer)
    """
    print("\n" + "="*60)
    print("模型训练...")
    print("="*60)
    print(f"参数:")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - L2正则化: {l2_lambda}")
    print(f"  - 批大小: {batch_size}")
    print(f"  - 激活函数: {activation}")
    print(f"  - 训练轮数: {epochs}")
    
    # 创建模型
    input_dim = 784  # 28x28
    num_classes = 10
    model = MLP(input_dim, hidden_dim, num_classes, activation=activation)
    
    # 创建优化器
    optimizer = SGD(learning_rate=learning_rate)
    
    # 创建学习率衰减调度器
    lr_decay = LearningRateDecay(optimizer, epochs, decay_type='step', decay_rate=0.95)
    
    # 创建训练器
    trainer = Trainer(model, optimizer, None, checkpoint_dir='./checkpoints')
    
    # 训练
    trainer.train(
        train_images, train_labels,
        val_images, val_labels,
        epochs=epochs,
        batch_size=batch_size,
        l2_lambda=l2_lambda,
        learning_rate_decay=lr_decay,
        patience=20,
        verbose=True
    )
    
    print(f"\n训练完成!")
    print(f"最优验证准确率: {trainer.best_val_accuracy:.4f}")
    
    return model, trainer


def evaluate_model(model, test_images, test_labels):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        test_images: 测试集图像
        test_labels: 测试集标签
        
    Returns:
        评估结果
    """
    print("\n" + "="*60)
    print("模型评估...")
    print("="*60)
    
    evaluator = Evaluator(model)
    results = evaluator.evaluate(test_images, test_labels, batch_size=32)
    
    evaluator.print_results(results)
    evaluator.print_confusion_matrix_readable(results, FashionMNISTLoader.CLASS_NAMES)
    
    return results


def hyperparameter_search(train_images, train_labels, val_images, val_labels,
                         search_type='grid', n_iter=10):
    """
    超参数搜索
    
    Args:
        train_images: 训练集图像
        train_labels: 训练集标签
        val_images: 验证集图像
        val_labels: 验证集标签
        search_type: 搜索类型 ('grid' 或 'random')
        n_iter: 搜索次数 (仅用于随机搜索)
    """
    print("\n" + "="*60)
    print(f"超参数搜索 ({search_type})...")
    print("="*60)
    
    input_dim = 784
    num_classes = 10
    
    if search_type == 'grid':
        # 定义参数网格
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [128, 256, 512],
            'l2_lambda': [0.0, 0.0001, 0.001],
            'batch_size': [16, 32, 64],
            'activation': ['relu', 'sigmoid']
        }
        
        search = GridSearchCV(param_grid)
        search.search(train_images, train_labels, val_images, val_labels,
                     input_dim, num_classes, epochs=50, verbose=True)
    
    else:  # random search
        param_distributions = {
            'learning_rate': (0.0001, 0.1),
            'hidden_dim': [128, 256, 512, 768],
            'l2_lambda': (0.0, 0.01),
            'batch_size': [16, 32, 64],
            'activation': ['relu', 'sigmoid', 'tanh']
        }
        
        search = RandomSearchCV(param_distributions, n_iter=n_iter)
        search.search(train_images, train_labels, val_images, val_labels,
                     input_dim, num_classes, epochs=50, verbose=True)
    
    # 打印最优参数
    print("\n" + "="*60)
    print("搜索完成!")
    print("="*60)
    print(f"最优参数: {search.best_params}")
    print(f"最优验证准确率: {search.best_score:.4f}")
    
    # 打印前10个结果
    print("\n前10个结果:")
    for i, result in enumerate(search.get_results_sorted()[:10]):
        print(f"{i+1}. {result}")
    
    return search.best_params


def main():
    """主函数"""
    print("\n" + "*"*60)
    print("Fashion-MNIST 三层神经网络分类器")
    print("*"*60)
    
    # 1. 准备数据
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        download_and_prepare_data()
    
    # 2. 模型训练（使用默认参数）
    model, trainer = train_model(
        train_images, train_labels, val_images, val_labels,
        hidden_dim=256,
        learning_rate=0.01,
        l2_lambda=0.0001,
        batch_size=32,
        epochs=100,
        activation='relu'
    )
    
    # 3. 模型评估
    results = evaluate_model(model, test_images, test_labels)
    
    # 4. 保存训练历史
    np.savez('./checkpoints/training_history.npz',
             train_losses=np.array(trainer.train_losses),
             val_losses=np.array(trainer.val_losses),
             val_accuracies=np.array(trainer.val_accuracies))
    
    print("\n训练历史已保存到 ./checkpoints/training_history.npz")
    
    # 5. 可选：超参数搜索
    # 取消下面的注释以进行超参数搜索
    # best_params = hyperparameter_search(train_images, train_labels,
    #                                     val_images, val_labels,
    #                                     search_type='grid')


if __name__ == '__main__':
    main()
