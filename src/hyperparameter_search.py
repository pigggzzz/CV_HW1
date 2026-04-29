"""
超参数搜索与调调
"""
import os
import csv
import json
import itertools
import numpy as np
from .model import MLP
from .optim import SGD, LearningRateDecay
from .train import Trainer


def _ensure_int(name, v):
    if name in ("hidden_dim", "batch_size") and v is not None:
        return int(v)
    return v


def save_hparam_results_json_csv(results, base_path_without_ext, best_params, best_score):
    """
    将超参搜索结果保存为 JSON（完整列表）和 CSV（表格便于 Excel / 画图）。

    base_path_without_ext: 无后缀路径，如 ./results/hparam_sweep_2025
    """
    _dir = os.path.dirname(os.path.abspath(base_path_without_ext))
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    out_json = base_path_without_ext + ".json"
    out_csv = base_path_without_ext + ".csv"
    meta = {
        "best_params": best_params,
        "best_val_accuracy": float(best_score) if best_score is not None else None,
    }
    def _row_jsonable(r):
        out = {}
        for k, v in r.items():
            if isinstance(v, np.generic):
                out[k] = v.item()
            else:
                out[k] = v
        return out

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "runs": [_row_jsonable(x) for x in results]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    if not results:
        return out_json, out_csv
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(_row_jsonable(row))
    return out_json, out_csv


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
               input_dim, num_classes, epochs=50, verbose=True,
               patience=None, output_dir=None, save_prefix="hparam_sweep"):
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
            patience: 早停；默认 None 表示在搜索阶段禁用早停（跑满 epochs）
            output_dir: 若设置，将每轮指标写入 JSON/CSV
            save_prefix: 保存文件前缀
        """
        if patience is None:
            patience = 10**9
        # 生成所有参数组合
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        
        for idx, pvals in enumerate(param_combinations):
            params = dict(zip(param_names, pvals))
            for k, v in list(params.items()):
                params[k] = _ensure_int(k, v)
            
            if verbose:
                print(f"\n[{idx+1}/{total_combinations}] Testing parameters: {params}")
            
            try:
                metrics = self._train_model(
                    train_images, train_labels,
                    val_images, val_labels,
                    input_dim, num_classes,
                    params, epochs, verbose=False, patience=patience
                )
                val_acc = metrics["best_val_accuracy"]
                result = {
                    "run_id": idx,
                    **{k: params[k] for k in param_names},
                    "epochs_trained": metrics["epochs_trained"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "final_val_accuracy": metrics["final_val_accuracy"],
                    "final_val_loss": metrics["final_val_loss"],
                    "final_train_loss": metrics["final_train_loss"],
                }
                self.results.append(result)
                
                if verbose:
                    print(
                        f"  -> best_val_acc={val_acc:.4f} | final_val_acc={result['final_val_accuracy']:.4f} | "
                        f"epochs={result['epochs_trained']}"
                    )
                
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_params = params.copy()
                    if verbose:
                        print("  ** New best! **")
                if output_dir:
                    save_hparam_results_json_csv(
                        self.results,
                        os.path.join(output_dir, save_prefix),
                        self.best_params,
                        self.best_score,
                    )
            
            except Exception as e:
                print(f"  Error: {e}")
        if output_dir and self.results:
            save_hparam_results_json_csv(
                self.results,
                os.path.join(output_dir, save_prefix),
                self.best_params,
                self.best_score,
            )
    
    def _train_model(self, train_images, train_labels, val_images, val_labels,
                    input_dim, num_classes, params, epochs, verbose=True, patience=20):
        """
        训练单个模型
        
        Returns:
            指标 dict: best/final 验证准确率、末轮损失、实际训练轮数
        """
        learning_rate = params.get('learning_rate', 0.01)
        hidden_dim = int(params.get('hidden_dim', 128))
        l2_lambda = params.get('l2_lambda', 0.0)
        batch_size = int(params.get('batch_size', 32))
        activation = params.get('activation', 'relu')
        
        model = MLP(input_dim, hidden_dim, num_classes, activation=activation)
        optimizer = SGD(learning_rate=learning_rate)
        lr_decay = LearningRateDecay(optimizer, epochs, decay_type='step', decay_rate=0.95)
        trainer = Trainer(model, optimizer, None, checkpoint_dir="./checkpoints")
        
        trainer.train(
            train_images, train_labels,
            val_images, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            l2_lambda=l2_lambda,
            learning_rate_decay=lr_decay,
            patience=patience,
            verbose=verbose
        )
        n = len(trainer.train_losses)
        return {
            "best_val_accuracy": float(trainer.best_val_accuracy),
            "final_val_accuracy": float(trainer.val_accuracies[-1]) if trainer.val_accuracies else 0.0,
            "final_val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
            "final_train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else 0.0,
            "epochs_trained": n,
        }
    
    def get_results_sorted(self):
        """按 best_val_accuracy 降序。"""
        return sorted(self.results, key=lambda x: x.get("best_val_accuracy", 0.0), reverse=True)


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
               input_dim, num_classes, epochs=50, verbose=True,
               patience=None, output_dir=None, save_prefix="hparam_sweep"):
        """
        执行随机搜索；参数含义同 GridSearchCV.search。
        """
        if patience is None:
            patience = 10**9
        for idx in range(self.n_iter):
            params = self._sample_params()
            for k, v in list(params.items()):
                params[k] = _ensure_int(k, v)
            
            if verbose:
                print(f"\n[{idx+1}/{self.n_iter}] Testing parameters: {params}")
            
            try:
                metrics = self._train_model(
                    train_images, train_labels,
                    val_images, val_labels,
                    input_dim, num_classes,
                    params, epochs, verbose=False, patience=patience
                )
                val_acc = metrics["best_val_accuracy"]
                result = {
                    "run_id": idx,
                    **params,
                    "epochs_trained": metrics["epochs_trained"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "final_val_accuracy": metrics["final_val_accuracy"],
                    "final_val_loss": metrics["final_val_loss"],
                    "final_train_loss": metrics["final_train_loss"],
                }
                self.results.append(result)
                
                if verbose:
                    print(
                        f"  -> best_val_acc={val_acc:.4f} | final_val_acc={result['final_val_accuracy']:.4f} | "
                        f"epochs={result['epochs_trained']}"
                    )
                
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_params = params.copy()
                    if verbose:
                        print("  ** New best! **")
                if output_dir:
                    save_hparam_results_json_csv(
                        self.results,
                        os.path.join(output_dir, save_prefix),
                        self.best_params,
                        self.best_score,
                    )
            
            except Exception as e:
                print(f"  Error: {e}")
        if output_dir and self.results:
            save_hparam_results_json_csv(
                self.results,
                os.path.join(output_dir, save_prefix),
                self.best_params,
                self.best_score,
            )
    
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
                    input_dim, num_classes, params, epochs, verbose=True, patience=20):
        learning_rate = params.get('learning_rate', 0.01)
        hidden_dim = int(params.get('hidden_dim', 128))
        l2_lambda = params.get('l2_lambda', 0.0)
        batch_size = int(params.get('batch_size', 32))
        activation = params.get('activation', 'relu')
        
        model = MLP(input_dim, hidden_dim, num_classes, activation=activation)
        optimizer = SGD(learning_rate=learning_rate)
        lr_decay = LearningRateDecay(optimizer, epochs, decay_type='step', decay_rate=0.95)
        trainer = Trainer(model, optimizer, None, checkpoint_dir="./checkpoints")
        
        trainer.train(
            train_images, train_labels,
            val_images, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            l2_lambda=l2_lambda,
            learning_rate_decay=lr_decay,
            patience=patience,
            verbose=verbose
        )
        n = len(trainer.train_losses)
        return {
            "best_val_accuracy": float(trainer.best_val_accuracy),
            "final_val_accuracy": float(trainer.val_accuracies[-1]) if trainer.val_accuracies else 0.0,
            "final_val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
            "final_train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else 0.0,
            "epochs_trained": n,
        }
    
    def get_results_sorted(self):
        return sorted(self.results, key=lambda x: x.get("best_val_accuracy", 0.0), reverse=True)
