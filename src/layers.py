"""
神经网络层的实现 - 包含全连接层和激活函数
"""
import numpy as np


class Linear:
    """全连接层的实现"""
    
    def __init__(self, in_features, out_features):
        """
        初始化全连接层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
        """
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重初始化 (Xavier 初始化)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.bias = np.zeros((1, out_features))
        
        # 梯度存储
        self.grad_weight = None
        self.grad_bias = None
        
        # 缓存用于反向传播
        self.input_cache = None
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，shape (batch_size, in_features)
            
        Returns:
            输出张量，shape (batch_size, out_features)
        """
        self.input_cache = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, dout):
        """
        反向传播
        
        Args:
            dout: 输出梯度，shape (batch_size, out_features)
            
        Returns:
            输入梯度，shape (batch_size, in_features)
        """
        batch_size = self.input_cache.shape[0]
        
        # 计算权重和偏置的梯度
        self.grad_weight = np.dot(self.input_cache.T, dout) / batch_size
        self.grad_bias = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        # 计算输入的梯度
        dinput = np.dot(dout, self.weight.T)
        
        return dinput
    
    def get_params(self):
        """返回权重和偏置"""
        return self.weight, self.bias
    
    def set_params(self, weight, bias):
        """设置权重和偏置"""
        self.weight = weight.copy()
        self.bias = bias.copy()


class ReLU:
    """ReLU 激活函数"""
    
    def __init__(self):
        self.input_cache = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            max(0, x)
        """
        self.input_cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        """
        反向传播
        
        Args:
            dout: 输出梯度
            
        Returns:
            输入梯度
        """
        dinput = dout.copy()
        dinput[self.input_cache <= 0] = 0
        return dinput


class Sigmoid:
    """Sigmoid 激活函数"""
    
    def __init__(self):
        self.output_cache = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            1 / (1 + exp(-x))
        """
        # 为了数值稳定性
        x = np.clip(x, -500, 500)
        self.output_cache = 1.0 / (1.0 + np.exp(-x))
        return self.output_cache
    
    def backward(self, dout):
        """
        反向传播
        
        Args:
            dout: 输出梯度
            
        Returns:
            输入梯度
        """
        dinput = dout * self.output_cache * (1.0 - self.output_cache)
        return dinput


class Tanh:
    """Tanh 激活函数"""
    
    def __init__(self):
        self.output_cache = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            tanh(x)
        """
        self.output_cache = np.tanh(x)
        return self.output_cache
    
    def backward(self, dout):
        """
        反向传播
        
        Args:
            dout: 输出梯度
            
        Returns:
            输入梯度
        """
        dinput = dout * (1.0 - self.output_cache ** 2)
        return dinput
