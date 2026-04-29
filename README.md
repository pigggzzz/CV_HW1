---
# Fashion-MNIST 三层神经网络分类器

从零开始实现一个三层多层感知机(MLP)神经网络，用于Fashion-MNIST数据集的图像分类。

## 项目特点

✅ **完全自主实现** - 不使用PyTorch/TensorFlow等自动微分框架
✅ **完整反向传播** - 从零实现自动微分和反向传播算法
✅ **模块化设计** - 清晰的代码结构，易于理解和扩展
✅ **丰富的功能** - 支持多种激活函数、优化器、正则化方法
✅ **超参数搜索** - 包含网格搜索和随机搜索
✅ **完整的可视化** - 生成训练曲线、权重可视化和错例分析

## 项目结构

```
CV_HW1/
├── src/                          # 核心代码模块
│   ├── __init__.py
│   ├── layers.py                 # 神经网络层（Linear, ReLU, Sigmoid, Tanh）
│   ├── model.py                  # MLP 模型定义
│   ├── loss.py                   # 交叉熵损失函数
│   ├── optim.py                  # SGD 优化器和学习率衰减
│   ├── train.py                  # 训练循环
│   ├── evaluate.py               # 模型评估工具
│   ├── data_loader.py            # 数据加载和预处理
│   └── hyperparameter_search.py  # 超参数搜索（网格搜索、随机搜索）
│
├── data/                         # 数据存储目录（执行时自动创建）
├── checkpoints/                  # 模型权重保存目录
├── figures/                      # 生成的图表和可视化
│
├── train.py                      # **主入口**：超参搜索 → 训练 → 测试 → 可视化与报告
├── results/                      # 超参搜索结果（json/csv）
├── run_experiment.py             # 已弃用，内部转调 train.py（兼容旧命令）
├── visualization.py              # 可视化和分析函数
└── README.md                     # 本文件
```

## 快速开始

### 1. 环境要求

```bash
pip install numpy matplotlib scikit-learn
```

### 2. 运行完整实验（推荐唯一入口）

在项目根目录执行：

```bash
python train.py
```

该流程会依次完成：数据准备 → **网格超参短训**（默认每组 20 epoch）→ 用最优超参 **长训**（默认 100 epoch）→ 测试集准确率与 **混淆矩阵** → 保存 `checkpoints/best_model.npz` → 生成训练/验证 **Loss 与验证 Accuracy 曲线**、**第一层权重图**、**错例图**，并输出 `figures/experiment_report.md` 草稿。超参对比表见 `results/hparam_sweep_short.json` / `.csv`，汇总图见 `figures/hyperparameter_search_summary.png`。

可选参数：

```bash
python train.py --skip-hparam-search          # 跳过短训，用默认超参直接长训
python train.py --eval-only                   # 仅加载已有权重在测试集评估并出图（需先有 best_model.npz 与 best_hparams.json）
python train.py --search-type random --search-epochs 20
```

旧命令 `python run_experiment.py` 与 `python train.py` 等价（已转调）。

## 详细功能说明

### 模型结构

```
输入 (784)
    ↓
[Linear Layer: 784 → 256] + ReLU激活
    ↓
[Linear Layer: 256 → 256] + ReLU激活
    ↓
[Linear Layer: 256 → 10] + Softmax
    ↓
输出 (10)
```

### 支持的功能

#### 1. 激活函数
- **ReLU** (Rectified Linear Unit): max(0, x)
- **Sigmoid**: 1 / (1 + exp(-x))
- **Tanh**: 双曲正切函数

#### 2. 优化器
- **SGD** (随机梯度下降)
- **SGD + Momentum** (可选)

#### 3. 正则化
- **L2 权重衰减** (Weight Decay)

#### 4. 学习率衰减策略
- **Step Decay**: 每N个epoch衰减一次
- **Exponential Decay**: 指数衰减
- **Linear Decay**: 线性衰减

#### 5. 超参数搜索
- **GridSearchCV**: 网格搜索穷举所有参数组合
- **RandomSearchCV**: 随机搜索采样参数

### 可视化功能

#### 1. 训练曲线
- 训练集损失
- 验证集损失
- 验证集准确率

#### 2. 权重可视化
- 第一层隐藏层的100个神经元权重
- 每个权重矩阵高可视化为28×28的图像
- 可以观察网络学到的边缘、纹理等特征

#### 3. 混淆矩阵
- 测试集上各类别的分类性能
- 热力图显示易混淆的类别对

#### 4. 错例分析
- 自动选取分类错误的样本
- 显示真实标签和预测概率分布
- 分析分类错误的可能原因

## 代码示例

### 基本使用

```python
from src.data_loader import FashionMNISTLoader
from src.model import MLP
from src.optim import SGD, LearningRateDecay
from src.train import Trainer
from src.evaluate import Evaluator

# 1. 加载和预处理数据
loader = FashionMNISTLoader('./data')
loader.load_data()
X_train, y_train, X_val, y_val, X_test, y_test = \
    loader.get_train_test_split(val_ratio=0.1, normalize=True, flatten=True)

# 2. 创建模型
model = MLP(
    input_dim=784,
    hidden_dim=256,
    num_classes=10,
    activation='relu'
)

# 3. 创建优化器和学习率衰减
optimizer = SGD(learning_rate=0.01)
lr_decay = LearningRateDecay(optimizer, total_epochs=100, decay_type='step')

# 4. 训练模型
trainer = Trainer(model, optimizer, None, checkpoint_dir='./checkpoints')
trainer.train(
    train_images=X_train,
    train_labels=y_train,
    val_images=X_val,
    val_labels=y_val,
    epochs=100,
    batch_size=32,
    l2_lambda=0.0001,
    learning_rate_decay=lr_decay,
    patience=20
)

# 5. 评估模型
evaluator = Evaluator(model)
results = evaluator.evaluate(X_test, y_test)
evaluator.print_results(results)
```

### 超参数搜索

#### 网格搜索
```python
from src.hyperparameter_search import GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_dim': [128, 256, 512],
    'l2_lambda': [0.0, 0.0001, 0.001],
    'batch_size': [16, 32, 64],
    'activation': ['relu', 'sigmoid']
}

search = GridSearchCV(param_grid)
search.search(
    train_images=X_train,
    train_labels=y_train,
    val_images=X_val,
    val_labels=y_val,
    input_dim=784,
    num_classes=10,
    epochs=50
)

print(f"最优参数: {search.best_params}")
print(f"最优验证准确率: {search.best_score:.4f}")
```

#### 随机搜索
```python
from src.hyperparameter_search import RandomSearchCV

param_dist = {
    'learning_rate': (0.0001, 0.1),
    'hidden_dim': [128, 256, 512, 768],
    'l2_lambda': (0.0, 0.01),
    'batch_size': [16, 32, 64],
    'activation': ['relu', 'sigmoid', 'tanh']
}

search = RandomSearchCV(param_dist, n_iter=20)
search.search(X_train, y_train, X_val, y_val, 784, 10, epochs=50)
```

## 实验结果示例

典型的实验结果：
- **测试准确率**: ~92%
- **最优参数**:
  - 学习率: 0.01
  - 隐藏层大小: 256
  - L2正则化: 0.0001
  - 激活函数: ReLU

## 核心算法详解

### 1. 前向传播

对于第 $l$ 层：
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

其中 $\sigma$ 是激活函数。

### 2. 交叉熵损失

$$L = -\sum_i y_i \log(\hat{y}_i) + \lambda \|W\|_2^2$$

其中 $y_i$ 是真实标签（独热编码），$\hat{y}_i$ 是网络输出（Softmax概率）。

### 3. 反向传播

梯度计算使用链式法则：
$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial W^{(l)}}$$
$$\frac{\partial L}{\partial a^{(l-1)}} = \frac{\partial L}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial a^{(l-1)}}$$

### 4. 梯度更新

使用SGD更新规则：
$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

其中 $\eta$ 是学习率。

## 常见问题(FAQ)

**Q: 为什么测试准确率没有那么高？**
A: Fashion-MNIST是一个相对简单的数据集，但也不完全trivial。要获得更高的准确率，可以：
- 增加隐藏层大小（如512）
- 调整学习率
- 增加训练轮数
- 使用更高级的优化器（如Adam）

**Q: 能否使用GPU加速？**
A: 当前实现使用纯NumPy，不支持GPU。如果需要加速，可以：
- 使用CuPy替代NumPy
- 使用Numba进行JIT编译
- 改用深度学习框架

**Q: 如何使用其他数据集？**
A: 修改`FashionMNISTLoader`类，实现相同的接口即可在其他数据集上使用。

**Q: 支持其他激活函数吗？**
A: 可以，在`layers.py`中添加新的激活函数类即可。

## 改进建议

1. **模型结构**
   - 添加Batch Normalization
   - 支持设置任意层数
   - 添加Dropout正则化

2. **优化器**
   - 实现Adam优化器
   - 添加RMSprop

3. **训练**
   - 实现混合精度训练
   - 添加梯度裁剪
   - 支持分布式训练

4. **效率**
   - 使用矩阵操作库加速
   - 实现mini-batch并行处理

## 参考资料

- [Fashion-MNIST官方](https://github.com/zalandoresearch/fashion-mnist)
- [反向传播算法详解](http://neuralnetworksanddeeplearning.com/)
- [深度学习基础](https://d2l.ai/)

## 许可证

MIT License

## 作者

实验者

## 更新日志

- 2024-04-27: 项目初始版本
