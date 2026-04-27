"""
数据加载与预处理
"""
import numpy as np
import os
from urllib import request
import gzip
import pickle


class FashionMNISTLoader:
    """Fashion-MNIST 数据集加载器"""
    
    # Fashion-MNIST 类别名称
    CLASS_NAMES = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]
    
    def __init__(self, data_dir='./data'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
    
    def download_data(self):
        """下载 Fashion-MNIST 数据集"""
        base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        files = {
            'train-images-idx3-ubyte.gz': 'train_images',
            'train-labels-idx1-ubyte.gz': 'train_labels',
            't10k-images-idx3-ubyte.gz': 'test_images',
            't10k-labels-idx1-ubyte.gz': 'test_labels'
        }
        
        for filename, name in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                request.urlretrieve(base_url + filename, filepath)
    
    def load_data(self):
        """加载数据集"""
        # 首先尝试从本地加载
        try:
            with open(os.path.join(self.data_dir, 'fashion_mnist.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.train_images = data['train_images']
                self.train_labels = data['train_labels']
                self.test_images = data['test_images']
                self.test_labels = data['test_labels']
                print("Data loaded from local cache")
                return
        except FileNotFoundError:
            pass
        
        # 从 GZ 文件加载
        print("Loading Fashion-MNIST dataset...")
        
        # 加载训练数据
        train_images = self._load_images(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz'))
        train_labels = self._load_labels(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz'))
        
        # 加载测试数据
        test_images = self._load_images(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz'))
        test_labels = self._load_labels(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz'))
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        # 保存到本地缓存
        with open(os.path.join(self.data_dir, 'fashion_mnist.pkl'), 'wb') as f:
            pickle.dump({
                'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels
            }, f)
    
    def _load_images(self, filepath):
        """加载图像数据"""
        with gzip.open(filepath, 'rb') as f:
            # 跳过魔数和维度信息 (16 字节)
            f.read(16)
            # 读取所有像素值
            data = np.frombuffer(f.read(), dtype=np.uint8)
            # 重塑为 (num_samples, 28, 28)
            data = data.reshape(-1, 28, 28)
        return data
    
    def _load_labels(self, filepath):
        """加载标签数据"""
        with gzip.open(filepath, 'rb') as f:
            # 跳过魔数和样本数信息 (8 字节)
            f.read(8)
            # 读取标签
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
    
    def preprocess(self, images, labels=None, normalize=True, flatten=True):
        """
        预处理数据
        
        Args:
            images: 图像数据，shape (num_samples, 28, 28)
            labels: 标签数据，可选
            normalize: 是否进行归一化
            flatten: 是否展平为向量
            
        Returns:
            处理后的图像和标签（独热编码）
        """
        # 归一化
        if normalize:
            images = images.astype(np.float32) / 255.0
        else:
            images = images.astype(np.float32)
        
        # 展平
        if flatten:
            images = images.reshape(images.shape[0], -1)
        
        # 标签转换为独热编码
        if labels is not None:
            num_samples = labels.shape[0]
            one_hot_labels = np.zeros((num_samples, 10))
            one_hot_labels[np.arange(num_samples), labels] = 1
            return images, one_hot_labels
        
        return images
    
    def get_train_test_split(self, val_ratio=0.1, normalize=True, flatten=True):
        """
        获取训练集、验证集、测试集
        
        Args:
            val_ratio: 验证集比例
            normalize: 是否归一化
            flatten: 是否展平
            
        Returns:
            (train_images, train_labels, val_images, val_labels, test_images, test_labels)
        """
        if self.train_images is None:
            self.load_data()
        
        # 处理训练数据
        train_images, train_labels = self.preprocess(
            self.train_images, self.train_labels, normalize, flatten
        )
        
        # 分割训练集和验证集
        num_train = train_images.shape[0]
        num_val = int(num_train * val_ratio)
        
        # 随机打乱
        indices = np.random.permutation(num_train)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        val_images = train_images[val_indices]
        val_labels = train_labels[val_indices]
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]
        
        # 处理测试数据
        test_images, test_labels = self.preprocess(
            self.test_images, self.test_labels, normalize, flatten
        )
        
        return train_images, train_labels, val_images, val_labels, test_images, test_labels


class DataBatchIterator:
    """数据批处理迭代器"""
    
    def __init__(self, images, labels, batch_size=32, shuffle=True):
        """
        初始化批处理迭代器
        
        Args:
            images: 图像数据
            labels: 标签数据
            batch_size: 批大小
            shuffle: 是否打乱数据
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = images.shape[0]
        self.indices = np.arange(self.num_samples)
        self.current_index = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """返回迭代器对象"""
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """获取下一个批次"""
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        # 获取当前批次的索引
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_index:end_index]
        
        # 获取批次数据
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        self.current_index = end_index
        
        return batch_images, batch_labels
