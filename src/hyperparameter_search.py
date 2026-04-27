"""
超参数搜索与调调
"""
import numpy as np
import itertools
from .model import MLP
from .optim import SGD, LearningRateDecay
from .train import Trainer
from .data_loader import DataBatchIterator


class GridSearchCV:
    """网格搜索超参数"""
    
    def __init__(self, param_grid):
        """
        初始化网格搜索
        
        Args:
            param_grid: 参数网格字典
                {
                    'learning_rate': [...],
                    'hidden_dim': [...],
                    'l2_lambda': [...],
                    'batch_size': [...],
                    'activation': [...]
                }
        """
        self.param_grid = param_grid
        self.results = []
        self.best_params = None
        self.best_score = 0.0
    
    def search(self, train_images, train_labels, val_images, val_labels,
               input_dim, num_classes, epochs=50, verbose=True):
        """
        执行网格搜索
        
        Args:
            train_images: 训练集图像
            train_labels: 训练集标签
            val_images: 验证集图像
            val_labels: 验证集标签
            input_dim: 输入维度
            num_classes: 分类数
            epochs: 训练轮数
            verbose: 是否打印日志
        """
        # 生成所有参数组合
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        
        for idx, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            if verbose:
                print(f"\n[{idx+1}/{total_combinations}] Testing parameters: {params}")
            
            try:
                # 训练模型
                val_acc = self._train_model(
                    train_images, train_labels,
                    val_images, val_labels,
                    input_dim, num_classes,
                    params, epochs, verbose=False
                )
                
                # 记录结果
                result = {**params, 'val_accuracy': val_acc}
                self.results.append(result)
                
                if verbose:
                    print(f"  -> Val Accuracy: {val_acc:.4f}")
                
                # 更新最优参数
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_params = params.copy()
                    if verbose:
                        print(f"  ** New best! **")
            
            except Exception as e:
                print(f"  Error: {e}")
    
    def _train_model(self, train_images, train_labels, val_images, val_labels,
                    input_dim, num_classes, params, epochs, verbose=True):
        """
        训练单个模型
        
        Args:
            train_images: 训练集图像
            train_labels: 训练集标签
            val_images: 验证集图像
            val_labels: 验证集标签
            input_dim: 输入维度
            num_classes: 分类数
            params: 参数字典
            epochs: 训练轮数
            verbose: 是否打印日志
            
        Returns:
            验证集准确率
        """
        # 提取参数
        learning_rate = params.get('learning_rate', 0.01)
        hidden_dim = params.get('hidden_dim', 128)
        l2_lambda = params.get('l2_lambda', 0.0)
        batch_size = params.get('batch_size', 32)
        activation = params.get('activation', 'relu')
        
        # 创建模型
        model = MLP(input_dim, hidden_dim, num_classes, activation=activation)
        
        # 创建优化器
        optimizer = SGD(learning_rate=learning_rate)
        
        # 创建学习率衰减调度器
        lr_decay = LearningRateDecay(optimizer, epochs, decay_type='step', decay_rate=0.95)
        
        # 创建训练器
        trainer = Trainer(model, optimizer, None)
        
        # 训练
        trainer.train(
            train_images, train_labels,
            val_images, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            l2_lambda=l2_lambda,
            learning_rate_decay=lr_decay,
            patience=20,
            verbose=False
        )
        
        return trainer.best_val_accuracy
    
    def get_results_sorted(self):
        """获取排序后的结果"""
        return sorted(self.results, key=lambda x: x['val_accuracy'], reverse=True)


class RandomSearchCV:
    """随机搜索超参数"""
    
    def __init__(self, param_distributions, n_iter=10):
        """
        初始化随机搜索
        
        Args:
            param_distributions: 参数分布字典
                {
                    'learning_rate': (min, max),  # log uniform
                    'hidden_dim': [values],       # 选择
                    'l2_lambda': (min, max),      # log uniform
                    ...
                }
            n_iter: 搜索次数
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.results = []
        self.best_params = None
        self.best_score = 0.0
    
    def search(self, train_images, train_labels, val_images, val_labels,
               input_dim, num_classes, epochs=50, verbose=True):
        """
        执行随机搜索
        
        Args:
            train_images: 训练集图像
            train_labels: 训练集标签
            val_images: 验证集图像
            val_labels: 验证集标签
            input_dim: 输入维度
            num_classes: 分类数
            epochs: 训练轮数
            verbose: 是否打印日志
        """
        for idx in range(self.n_iter):
            # 采样参数
            params = self._sample_params()
            
            if verbose:
                print(f"\n[{idx+1}/{self.n_iter}] Testing parameters: {params}")
            
            try:
                # 训练模型
                val_acc = self._train_model(
                    train_images, train_labels,
                    val_images, val_labels,
                    input_dim, num_classes,
                    params, epochs, verbose=False
                )
                
                # 记录结果
                result = {**params, 'val_accuracy': val_acc}
                self.results.append(result)
                
                if verbose:
                    print(f"  -> Val Accuracy: {val_acc:.4f}")
                
                # 更新最优参数
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_params = params.copy()
                    if verbose:
                        print(f"  ** New best! **")
            
            except Exception as e:
                print(f"  Error: {e}")
    
    def _sample_params(self):
        """采样参数"""
        params = {}
        
        for name, distribution in self.param_distributions.items():
            if isinstance(distribution, (list, tuple)) and len(distribution) > 0:
                if isinstance(distribution[0], (int, float)):
                    # 已有的值列表
                    params[name] = np.random.choice(distribution)
                else:
                    # (min, max) 元组
                    if name in ['learning_rate', 'l2_lambda']:
                        # 对数均匀分布
                        log_min = np.log10(distribution[0])
                        log_max = np.log10(distribution[1])
                        params[name] = 10 ** np.random.uniform(log_min, log_max)
                    else:
                        # 线性均匀分布
                        params[name] = np.random.uniform(distribution[0], distribution[1])
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # (min, max) 元组
                if name in ['learning_rate', 'l2_lambda']:
                    # 对数均匀分布
                    log_min = np.log10(distribution[0])
                    log_max = np.log10(distribution[1])
                    params[name] = 10 ** np.random.uniform(log_min, log_max)
                else:
                    # 线性均匀分布
                    params[name] = np.random.uniform(distribution[0], distribution[1])
        
        return params
    
    def _train_model(self, train_images, train_labels, val_images, val_labels,
                    input_dim, num_classes, params, epochs, verbose=True):
        """训练单个模型"""
        # 提取参数
        learning_rate = params.get('learning_rate', 0.01)
        hidden_dim = int(params.get('hidden_dim', 128))
        l2_lambda = params.get('l2_lambda', 0.0)
        batch_size = int(params.get('batch_size', 32))
        activation = params.get('activation', 'relu')
        
        # 创建模型
        model = MLP(input_dim, hidden_dim, num_classes, activation=activation)
        
        # 创建优化器
        optimizer = SGD(learning_rate=learning_rate)
        
        # 创建学习率衰减调度器
        lr_decay = LearningRateDecay(optimizer, epochs, decay_type='step', decay_rate=0.95)
        
        # 创建训练器
        trainer = Trainer(model, optimizer, None)
        
        # 训练
        trainer.train(
            train_images, train_labels,
            val_images, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            l2_lambda=l2_lambda,
            learning_rate_decay=lr_decay,
            patience=20,
            verbose=False
        )
        
        return trainer.best_val_accuracy
    
    def get_results_sorted(self):
        """获取排序后的结果"""
        return sorted(self.results, key=lambda x: x['val_accuracy'], reverse=True)
