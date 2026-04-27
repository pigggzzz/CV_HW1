"""
模型训练循环
"""
import numpy as np
import os
from .model import MLP
from .optim import SGD, LearningRateDecay
from .data_loader import DataBatchIterator


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, optimizer, data_loader, checkpoint_dir='./checkpoints'):
        """
        初始化训练器
        
        Args:
            model: MLP 模型
            optimizer: 优化器
            data_loader: 数据加载器
            checkpoint_dir: 模型保存目录
        """
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.best_val_accuracy = 0.0
        self.best_model_params = None
    
    def train_epoch(self, train_images, train_labels, batch_size=32, l2_lambda=0.0):
        """
        训练一个epoch
        
        Args:
            train_images: 训练集图像
            train_labels: 训练集标签
            batch_size: 批大小
            l2_lambda: L2 正则化强度
            
        Returns:
            平均损失
        """
        batch_iterator = DataBatchIterator(train_images, train_labels, batch_size, shuffle=True)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_images, batch_labels in batch_iterator:
            # 前向传播
            logits = self.model.forward(batch_images)
            
            # 计算损失
            loss, _ = self.model.compute_loss(logits, batch_labels, l2_lambda)
            
            # 反向传播
            dout = self.model.loss_fn.backward()
            self.model.backward(dout)
            
            # 获取梯度（包括L2正则化）
            gradients = self.model.get_gradients_with_l2(l2_lambda)
            
            # 参数更新
            params = self.model.get_params()
            self.optimizer.update(params, gradients)
            self.model.set_params(params)
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_images, val_labels, batch_size=32, l2_lambda=0.0):
        """
        验证集评估
        
        Args:
            val_images: 验证集图像
            val_labels: 验证集标签
            batch_size: 批大小
            l2_lambda: L2 正则化强度
            
        Returns:
            (准确率, 损失)
        """
        batch_iterator = DataBatchIterator(val_images, val_labels, batch_size, shuffle=False)
        
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        for batch_images, batch_labels in batch_iterator:
            # 前向传播
            logits = self.model.forward(batch_images)
            
            # 计算损失
            loss, _ = self.model.compute_loss(logits, batch_labels, l2_lambda)
            total_loss += loss
            
            # 计算准确率
            predictions = np.argmax(logits, axis=1)
            labels = np.argmax(batch_labels, axis=1)
            correct += np.sum(predictions == labels)
            total += batch_images.shape[0]
            
            num_batches += 1
        
        accuracy = correct / total
        avg_loss = total_loss / num_batches
        
        return accuracy, avg_loss
    
    def train(self, train_images, train_labels, val_images, val_labels,
              epochs=100, batch_size=32, l2_lambda=0.0, learning_rate_decay=None,
              patience=10, verbose=True):
        """
        完整的训练循环
        
        Args:
            train_images: 训练集图像
            train_labels: 训练集标签
            val_images: 验证集图像
            val_labels: 验证集标签
            epochs: 训练轮数
            batch_size: 批大小
            l2_lambda: L2 正则化强度
            learning_rate_decay: 学习率衰减调度器
            patience: 早停耐心值
            verbose: 是否打印日志
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 学习率衰减
            if learning_rate_decay is not None:
                learning_rate_decay.step(epoch)
            
            # 训练
            train_loss = self.train_epoch(train_images, train_labels, batch_size, l2_lambda)
            self.train_losses.append(train_loss)
            
            # 验证
            val_acc, val_loss = self.evaluate(val_images, val_labels, batch_size, l2_lambda)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                lr = self.optimizer.learning_rate
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"LR: {lr:.6f}")
            
            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_accuracy = best_val_acc
                self.best_model_params = self.model.get_params()
                patience_counter = 0
                
                # 保存模型权重
                self.save_checkpoint(epoch, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最优模型
        if self.best_model_params is not None:
            self.model.set_params(self.best_model_params)
    
    def save_checkpoint(self, epoch, val_accuracy):
        """保存模型检查点"""
        filepath = os.path.join(self.checkpoint_dir, f'best_model.npz')
        
        params = self.model.get_params()
        
        # 分别保存每层的权重和偏置
        save_dict = {'epoch': epoch, 'val_accuracy': val_accuracy}
        for i, (w, b) in enumerate(params):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b
        
        np.savez(filepath, **save_dict)
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        data = np.load(filepath, allow_pickle=True)
        
        # 从保存的文件中重新构建参数
        params = []
        i = 0
        while f'weight_{i}' in data.files:
            w = data[f'weight_{i}']
            b = data[f'bias_{i}']
            params.append((w, b))
            i += 1
        
        self.model.set_params(params)
        
        return {
            'epoch': int(data['epoch']),
            'val_accuracy': float(data['val_accuracy'])
        }
