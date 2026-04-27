"""
损失函数的实现
"""
import numpy as np


class CrossEntropyLoss:
    """交叉熵损失函数"""
    
    def __init__(self):
        self.input_cache = None
        self.target_cache = None
    
    def forward(self, logits, targets):
        """
        前向传播: 计算交叉熵损失
        
        Args:
            logits: 模型输出，shape (batch_size, num_classes)
            targets: 目标标签（one-hot编码），shape (batch_size, num_classes)
            
        Returns:
            标量损失值
        """
        self.input_cache = logits
        self.target_cache = targets
        
        batch_size = logits.shape[0]
        
        # 数值稳定性：减去最大值
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        
        # Softmax
        exp_logits = np.exp(logits_stable)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 交叉熵损失
        loss = -np.sum(targets * np.log(softmax + 1e-8)) / batch_size
        
        return loss
    
    def backward(self):
        """
        反向传播
        
        Returns:
            梯度，shape (batch_size, num_classes)
        """
        logits = self.input_cache
        targets = self.target_cache
        batch_size = logits.shape[0]
        
        # 数值稳定性：减去最大值
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        
        # Softmax
        exp_logits = np.exp(logits_stable)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 梯度 = softmax - targets
        dout = (softmax - targets) / batch_size
        
        return dout
