"""
优化器的实现
"""
import numpy as np


class SGD:
    """随机梯度下降优化器"""
    
    def __init__(self, learning_rate=0.01):
        """
        初始化 SGD 优化器
        
        Args:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
    
    def update(self, params, gradients):
        """
        参数更新
        
        Args:
            params: 参数列表 [(weight, bias), ...]
            gradients: 梯度列表 [(grad_weight, grad_bias), ...]
        """
        for i, (param, grad) in enumerate(zip(params, gradients)):
            weight, bias = param
            grad_weight, grad_bias = grad
            
            # 权重更新
            weight -= self.learning_rate * grad_weight
            # 偏置更新
            bias -= self.learning_rate * grad_bias
            
            params[i] = (weight, bias)


class SGDWithMomentum:
    """SGD 加动量"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        初始化 SGD with Momentum
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, gradients):
        """
        参数更新
        
        Args:
            params: 参数列表 [(weight, bias), ...]
            gradients: 梯度列表 [(grad_weight, grad_bias), ...]
        """
        # 初始化速度
        if self.velocity is None:
            self.velocity = []
            for param in params:
                weight, bias = param
                self.velocity.append((np.zeros_like(weight), np.zeros_like(bias)))
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            weight, bias = param
            grad_weight, grad_bias = grad
            vel_weight, vel_bias = self.velocity[i]
            
            # 更新速度
            vel_weight = self.momentum * vel_weight - self.learning_rate * grad_weight
            vel_bias = self.momentum * vel_bias - self.learning_rate * grad_bias
            
            # 更新参数
            weight += vel_weight
            bias += vel_bias
            
            params[i] = (weight, bias)
            self.velocity[i] = (vel_weight, vel_bias)


class LearningRateDecay:
    """学习率衰减调度器"""
    
    def __init__(self, optimizer, total_epochs, decay_type='step', decay_rate=0.95):
        """
        初始化学习率衰减
        
        Args:
            optimizer: 优化器对象
            total_epochs: 总的训练轮数
            decay_type: 衰减类型 ('step', 'exponential', 'linear')
            decay_rate: 衰减率
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.initial_lr = optimizer.learning_rate
    
    def step(self, epoch):
        """
        更新学习率
        
        Args:
            epoch: 当前训练轮数 (0-indexed)
        """
        if self.decay_type == 'step':
            # 每30个epoch衰减一次
            decay_steps = max(self.total_epochs // 10, 1)
            lr = self.initial_lr * (self.decay_rate ** (epoch // decay_steps))
        elif self.decay_type == 'exponential':
            lr = self.initial_lr * np.exp(-self.decay_rate * epoch / self.total_epochs)
        elif self.decay_type == 'linear':
            lr = self.initial_lr * (1 - epoch / self.total_epochs)
        else:
            lr = self.initial_lr
        
        self.optimizer.learning_rate = max(lr, 1e-6)
