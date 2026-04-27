"""
三层 MLP 模型的实现
"""
import numpy as np
from .layers import Linear, ReLU, Sigmoid, Tanh
from .loss import CrossEntropyLoss


class MLP:
    """三层多层感知机分类器"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, activation='relu'):
        """
        初始化 MLP 模型
        
        Args:
            input_dim: 输入维度 (784 for 28x28 images)
            hidden_dim: 隐藏层维度
            num_classes: 分类数 (10 for Fashion-MNIST)
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation = activation
        
        # 第一层: input_dim -> hidden_dim
        self.fc1 = Linear(input_dim, hidden_dim)
        
        # 激活函数
        if activation == 'relu':
            self.act1 = ReLU()
        elif activation == 'sigmoid':
            self.act1 = Sigmoid()
        elif activation == 'tanh':
            self.act1 = Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 第二层: hidden_dim -> hidden_dim
        self.fc2 = Linear(hidden_dim, hidden_dim)
        
        # 激活函数
        if activation == 'relu':
            self.act2 = ReLU()
        elif activation == 'sigmoid':
            self.act2 = Sigmoid()
        elif activation == 'tanh':
            self.act2 = Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 第三层: hidden_dim -> num_classes
        self.fc3 = Linear(hidden_dim, num_classes)
        
        # 损失函数
        self.loss_fn = CrossEntropyLoss()
        
        # 缓存用于反向传播
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，shape (batch_size, input_dim)
            
        Returns:
            输出张量，shape (batch_size, num_classes)
        """
        # 第一层
        h1 = self.fc1.forward(x)
        a1 = self.act1.forward(h1)
        
        # 第二层
        h2 = self.fc2.forward(a1)
        a2 = self.act2.forward(h2)
        
        # 第三层（输出层）
        output = self.fc3.forward(a2)
        
        # 缓存用于反向传播
        self.cache = {'h1': h1, 'a1': a1, 'h2': h2, 'a2': a2}
        
        return output
    
    def backward(self, dout):
        """
        反向传播
        
        Args:
            dout: 输出梯度
        """
        # 第三层反向传播
        dh2 = self.fc3.backward(dout)
        
        # 第二层激活函数反向传播
        dh2 = self.act2.backward(dh2)
        
        # 第二层反向传播
        da1 = self.fc2.backward(dh2)
        
        # 第一层激活函数反向传播
        da1 = self.act1.backward(da1)
        
        # 第一层反向传播
        _ = self.fc1.backward(da1)
    
    def compute_loss(self, logits, targets, l2_lambda=0.0):
        """
        计算损失（包括正则化项）
        
        Args:
            logits: 模型输出
            targets: 目标标签（one-hot）
            l2_lambda: L2 正则化强度
            
        Returns:
            总损失
        """
        # 交叉熵损失
        ce_loss = self.loss_fn.forward(logits, targets)
        
        # L2 正则化
        l2_loss = 0.0
        if l2_lambda > 0:
            w1, _ = self.fc1.get_params()
            w2, _ = self.fc2.get_params()
            w3, _ = self.fc3.get_params()
            l2_loss = l2_lambda * (np.sum(w1**2) + np.sum(w2**2) + np.sum(w3**2)) / 2.0
        
        return ce_loss + l2_loss, ce_loss
    
    def get_gradients_with_l2(self, l2_lambda=0.0):
        """
        获取梯度（包括 L2 正则化项的梯度）
        
        Args:
            l2_lambda: L2 正则化强度
            
        Returns:
            梯度列表 [(grad_w1, grad_b1), (grad_w2, grad_b2), (grad_w3, grad_b3)]
        """
        gradients = []
        
        for layer in [self.fc1, self.fc2, self.fc3]:
            w, b = layer.get_params()
            grad_w, grad_b = layer.grad_weight, layer.grad_bias
            
            # 添加 L2 正则化项的梯度
            if l2_lambda > 0:
                grad_w = grad_w + l2_lambda * w
            
            gradients.append((grad_w, grad_b))
        
        return gradients
    
    def predict(self, x):
        """
        预测
        
        Args:
            x: 输入张量，shape (batch_size, input_dim)
            
        Returns:
            预测标签，shape (batch_size,)
        """
        logits = self.forward(x)
        predictions = np.argmax(logits, axis=1)
        return predictions
    
    def get_params(self):
        """获取所有参数"""
        params = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            w, b = layer.get_params()
            params.append((w.copy(), b.copy()))
        return params
    
    def set_params(self, params):
        """设置所有参数"""
        for layer, (w, b) in zip([self.fc1, self.fc2, self.fc3], params):
            layer.set_params(w, b)
    
    def get_first_layer_weights(self):
        """获取第一层权重用于可视化"""
        w, _ = self.fc1.get_params()
        return w
