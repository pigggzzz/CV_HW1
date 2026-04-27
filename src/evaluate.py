"""
模型评估与测试功能
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from .data_loader import DataBatchIterator


class Evaluator:
    """模型评估工具"""
    
    def __init__(self, model):
        """
        初始化评估工具
        
        Args:
            model: MLP 模型
        """
        self.model = model
    
    def evaluate(self, test_images, test_labels, batch_size=32):
        """
        在测试集上进行评估
        
        Args:
            test_images: 测试集图像
            test_labels: 测试集标签（独热编码）
            batch_size: 批大小
            
        Returns:
            评估结果字典
        """
        batch_iterator = DataBatchIterator(test_images, test_labels, batch_size, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        for batch_images, batch_labels in batch_iterator:
            # 前向传播
            logits = self.model.forward(batch_images)
            
            # 获取预测
            predictions = np.argmax(logits, axis=1)
            labels = np.argmax(batch_labels, axis=1)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # 计算各类别的指标
        class_report = classification_report(all_labels, all_predictions, 
                                            target_names=[f"Class_{i}" for i in range(10)])
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'labels': all_labels,
            'class_report': class_report
        }
    
    def print_results(self, results):
        """
        打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*50}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"{'='*50}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        print("\nClassification Report:")
        print(results['class_report'])
    
    def print_confusion_matrix_readable(self, results, class_names=None):
        """
        以可读的格式打印混淆矩阵
        
        Args:
            results: 评估结果字典
            class_names: 类别名称列表
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(10)]
        
        conf_matrix = results['confusion_matrix']
        
        print("\n" + "="*80)
        print("Confusion Matrix (rows=true labels, columns=predictions)")
        print("="*80)
        
        # 打印表头
        print(f"{'':>15} | ", end="")
        for i, name in enumerate(class_names):
            print(f"{name[:10]:>10} ", end="")
        print()
        print("-"*80)
        
        # 打印矩阵
        for i, name in enumerate(class_names):
            print(f"{name[:15]:>15} | ", end="")
            for j in range(len(class_names)):
                print(f"{conf_matrix[i, j]:>10} ", end="")
            print()
    
    def get_misclassified_samples(self, test_images, test_labels, num_samples=5):
        """
        获取分类错误的样本
        
        Args:
            test_images: 测试集图像
            test_labels: 测试集标签
            num_samples: 返回的样本数
            
        Returns:
            [(image, true_label, predicted_label, logits), ...]
        """
        batch_iterator = DataBatchIterator(test_images, test_labels, 
                                          batch_size=test_images.shape[0], shuffle=False)
        
        batch_images, batch_labels = next(iter(batch_iterator))
        logits = self.model.forward(batch_images)
        
        predictions = np.argmax(logits, axis=1)
        labels = np.argmax(batch_labels, axis=1)
        
        # 找到错误分类的样本
        misclassified_mask = predictions != labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # 随机选择样本
        selected_indices = np.random.choice(misclassified_indices, 
                                          min(num_samples, len(misclassified_indices)), 
                                          replace=False)
        
        misclassified_samples = []
        for idx in selected_indices:
            image = batch_images[idx]
            true_label = labels[idx]
            predicted_label = predictions[idx]
            logit = logits[idx]
            
            misclassified_samples.append({
                'image': image,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'logits': logit
            })
        
        return misclassified_samples
