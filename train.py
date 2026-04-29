"""
Fashion-MNIST 三层 MLP 实验：唯一推荐入口（数据 → 超参搜索 → 训练 → 评估 → 可视化与报告）。

用法:
  python train.py                    # 完整流程（网格短训 + 长训 + 评估 + 图表）
  python train.py --eval-only        # 仅加载 ./checkpoints/best_model.npz 在测试集上评估
  python train.py --skip-hparam-search  # 跳过超参搜索，用默认/内置超参直接长训
"""
import argparse
import json
import os
import sys

import numpy as np

# 包根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import FashionMNISTLoader
from src.model import MLP
from src.optim import SGD, LearningRateDecay
from src.train import Trainer
from src.evaluate import Evaluator
from src.hyperparameter_search import GridSearchCV, RandomSearchCV
from visualization import (
    plot_hyperparameter_search_results,
    plot_training_curves,
    visualize_first_layer_weights,
    visualize_confusion_matrix,
    visualize_misclassified_samples,
    analyze_misclassified_samples,
)


# ---------------------- 数据 ----------------------


def download_and_prepare_data():
    print("=" * 60)
    print("准备数据集...")
    print("=" * 60)
    loader = FashionMNISTLoader("./data")
    try:
        loader.download_data()
    except Exception as e:
        print(f"下载失败: {e}（如已有 data 可忽略）")
    loader.load_data()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        loader.get_train_test_split(val_ratio=0.1, normalize=True, flatten=True)
    print(f"训练集: {train_images.shape}")
    print(f"验证集: {val_images.shape}")
    print(f"测试集: {test_images.shape}")
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# ---------------------- 训练 / 评估 ----------------------


def train_model(
    train_images,
    train_labels,
    val_images,
    val_labels,
    hidden_dim=256,
    learning_rate=0.01,
    l2_lambda=0.0001,
    batch_size=32,
    epochs=100,
    activation="relu",
):
    print("\n" + "=" * 60)
    print("完整训练")
    print("=" * 60)
    for k, v in [
        ("隐藏层维度", hidden_dim),
        ("学习率", learning_rate),
        ("L2 正则", l2_lambda),
        ("批大小", batch_size),
        ("激活函数", activation),
        ("轮数", epochs),
    ]:
        print(f"  - {k}: {v}")
    input_dim = 784
    num_classes = 10
    model = MLP(input_dim, int(hidden_dim), num_classes, activation=activation)
    optimizer = SGD(learning_rate=float(learning_rate))
    lr_decay = LearningRateDecay(optimizer, epochs, decay_type="step", decay_rate=0.95)
    trainer = Trainer(model, optimizer, None, checkpoint_dir="./checkpoints")
    trainer.train(
        train_images,
        train_labels,
        val_images,
        val_labels,
        epochs=epochs,
        batch_size=int(batch_size),
        l2_lambda=float(l2_lambda),
        learning_rate_decay=lr_decay,
        patience=20,
        verbose=True,
    )
    print(f"\n训练完成，最优验证准确率: {trainer.best_val_accuracy:.4f}")
    return model, trainer


def evaluate_on_test(model, test_images, test_labels, batch_size=32):
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    evaluator = Evaluator(model)
    results = evaluator.evaluate(test_images, test_labels, batch_size=batch_size)
    evaluator.print_results(results)
    evaluator.print_confusion_matrix_readable(results, FashionMNISTLoader.CLASS_NAMES)
    return evaluator, results


def save_hparams_json(path, hparams):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: (int(v) if k in ("hidden_dim", "batch_size") else v) for k, v in hparams.items()}, f, ensure_ascii=False, indent=2)
    print(f"超参已写入 {path}")


def load_hparams_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_model_from_hparams(h):
    return MLP(784, int(h["hidden_dim"]), 10, activation=str(h["activation"]))


def load_trained_for_eval(
    checkpoint_path="./checkpoints/best_model.npz",
    hparams_path="./checkpoints/best_hparams.json",
):
    if not os.path.isfile(hparams_path):
        raise FileNotFoundError(f"未找到 {hparams_path}，请先完成一次完整训练或指定路径。")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到 {checkpoint_path}，请先完成训练并保存权重。")
    h = load_hparams_json(hparams_path)
    model = build_model_from_hparams(h)
    trainer = Trainer(model, SGD(0.01), None, checkpoint_dir="./checkpoints")
    meta = trainer.load_checkpoint(checkpoint_path)
    return model, h, meta


# ---------------------- 超参搜索 ----------------------


def run_hyperparameter_search(
    train_images,
    train_labels,
    val_images,
    val_labels,
    search_type="grid",
    search_epochs=20,
    n_iter=12,
    output_dir="./results",
    save_prefix="hparam_sweep_short",
):
    print("\n" + "=" * 60)
    print(f"超参数搜索 ({search_type})，每组 {search_epochs} epoch（短训阶段关闭早停）")
    print("=" * 60)
    input_dim = 784
    num_classes = 10
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    hparam_desc = ""

    if search_type == "grid":
        # 学习率、隐藏维、L2、批大小、激活均参与网格
        param_grid = {
            "learning_rate": [0.01, 0.02, 0.05],
            "hidden_dim": [128, 256],
            "l2_lambda": [0.0, 0.0001],
            "batch_size": [32],
            "activation": ["relu", "sigmoid"],
        }
        hparam_desc = f"网格: {param_grid}"
        search = GridSearchCV(param_grid)
        search.search(
            train_images,
            train_labels,
            val_images,
            val_labels,
            input_dim,
            num_classes,
            epochs=search_epochs,
            verbose=True,
            patience=None,
            output_dir=output_dir,
            save_prefix=save_prefix,
        )
    else:
        param_distributions = {
            "learning_rate": (0.001, 0.1),
            "hidden_dim": [128, 256, 512],
            "l2_lambda": (1e-6, 0.01),
            "batch_size": [16, 32, 64],
            "activation": ["relu", "sigmoid", "tanh"],
        }
        hparam_desc = f"随机搜索分布: {param_distributions}, n_iter={n_iter}"
        search = RandomSearchCV(param_distributions, n_iter=n_iter)
        search.search(
            train_images,
            train_labels,
            val_images,
            val_labels,
            input_dim,
            num_classes,
            epochs=search_epochs,
            verbose=True,
            patience=None,
            output_dir=output_dir,
            save_prefix=save_prefix,
        )

    print("\n" + "=" * 60)
    print("短训搜索结束")
    print("=" * 60)
    print(f"最优参数: {search.best_params}")
    print(f"短训中的最优验证准确率: {search.best_score:.4f}")
    print("\nTop 10:")
    for i, r in enumerate(search.get_results_sorted()[:10]):
        print(f"  {i+1}. {r}")
    if search.results:
        plot_hyperparameter_search_results(
            search.results, save_path="./figures", metric="best_val_accuracy", top_n=16
        )
    return search, hparam_desc


def _default_hparams():
    return {
        "hidden_dim": 256,
        "learning_rate": 0.01,
        "l2_lambda": 0.0001,
        "batch_size": 32,
        "activation": "relu",
    }


#可视化与报告


def format_confusion_matrix(conf_matrix, class_names):
    header = "\t" + "\t".join([name[:6] for name in class_names])
    lines = [header]
    for i, name in enumerate(class_names):
        row = name[:6] + "\t" + "\t".join(str(int(conf_matrix[i, j])) for j in range(len(class_names)))
        lines.append(row)
    return "\n".join(lines)


def _step_decay_desc(total_epochs, decay_rate):
    n = max(total_epochs // 10, 1)
    return f"Step: 每 {n} 个 epoch 将学习率乘以 {decay_rate}（以 total_epochs={total_epochs} 计）"





def run_figures_and_report(
    model,
    trainer,
    results,
    test_images,
    test_labels,
    evaluator,
    hparams,
    hparam_section,
    class_names,
    max_epochs_planned=100,
):
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    print("\n[可视化] 训练曲线、权重、混淆矩阵、错例...")
    if len(getattr(trainer, "train_losses", [])) > 0:
        plot_training_curves(
            trainer.train_losses, trainer.val_losses, trainer.val_accuracies, save_path="./figures"
        )
    else:
        print("  (跳过训练曲线: 无训练历史；若需曲线请将 training_history.npz 置于 ./checkpoints/)")
    visualize_first_layer_weights(model, save_path="./figures")
    visualize_confusion_matrix(
        results["confusion_matrix"], class_names, save_path="./figures"
    )
    mis = evaluator.get_misclassified_samples(test_images, test_labels, num_samples=5)
    visualize_misclassified_samples(
        mis, class_names, save_path="./figures", num_samples=5
    )
    analyze_misclassified_samples(mis, class_names)
    last_epoch = max(0, len(trainer.train_losses) - 1)
    trainer.save_checkpoint(last_epoch, trainer.best_val_accuracy)
    np.savez(
        "./checkpoints/training_history.npz",
        train_losses=np.array(trainer.train_losses),
        val_losses=np.array(trainer.val_losses),
        val_accuracies=np.array(trainer.val_accuracies),
    )
    print("已保存: ./figures/training_curves.png, first_layer_weights.png, confusion_matrix.png, misclassified_samples.png")
    print("已保存: ./checkpoints/best_model.npz, training_history.npz")


def run_eval_only_from_checkpoint():
    model, hparams, ck_meta = load_trained_for_eval()
    print(f"已加载 hparams: {hparams}；checkpoint: {ck_meta}")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        download_and_prepare_data()
    evaluator, results = evaluate_on_test(
        model, test_images, test_labels, batch_size=int(hparams.get("batch_size", 32))
    )
    hist_path = "./checkpoints/training_history.npz"
    if os.path.isfile(hist_path):
        d = np.load(hist_path, allow_pickle=True)
        tl, vl, va = d["train_losses"], d["val_losses"], d["val_accuracies"]
        fake_trainer = type("T", (), {})()
        fake_trainer.train_losses = list(tl)
        fake_trainer.val_losses = list(vl)
        fake_trainer.val_accuracies = list(va)
        fake_trainer.best_val_accuracy = float(np.max(va)) if len(va) else float(ck_meta.get("val_accuracy", 0.0))
    else:
        fake_trainer = type("T", (), {})()
        fake_trainer.train_losses = []
        fake_trainer.val_losses = []
        fake_trainer.val_accuracies = []
        fake_trainer.best_val_accuracy = float(ck_meta.get("val_accuracy", 0.0))
    hparam_section = "本次为 `--eval-only`，未重新做超参搜索；长训时保存的 `results/hparam_sweep_short.*` 可供对照。"
    run_figures_and_report(
        model,
        fake_trainer,
        results,
        test_images,
        test_labels,
        evaluator,
        hparams,
        hparam_section,
        FashionMNISTLoader.CLASS_NAMES,
        max_epochs_planned=100,
    )


# ---------------------- 主流程 ----------------------


def main():
    p = argparse.ArgumentParser(description="Fashion-MNIST MLP：训练 / 超参搜索 / 测试")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="仅加载 checkpoints/best_model.npz 与 best_hparams.json，在测试集上评估并出图/报告",
    )
    p.add_argument(
        "--skip-hparam-search",
        action="store_true",
        help="跳过超参短训，直接使用默认超参进行长训",
    )
    p.add_argument(
        "--search-type",
        choices=("grid", "random"),
        default="grid",
        help="超参短训使用网格或随机搜索",
    )
    p.add_argument(
        "--search-epochs", type=int, default=20, help="超参短训每组合 epoch 数"
    )
    p.add_argument(
        "--full-epochs", type=int, default=100, help="长训 epoch 数"
    )
    args = p.parse_args()

    if args.eval_only:
        run_eval_only_from_checkpoint()
        return

    print("\n" + "* " * 30)
    print("Fashion-MNIST 三层 MLP | 推荐入口: train.py")
    print("* " * 30)

    full_epochs = int(args.full_epochs)
    search_epochs = int(args.search_epochs)
    hparam_section = ""

    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        download_and_prepare_data()

    if not args.skip_hparam_search:
        hsearch, hparam_desc = run_hyperparameter_search(
            train_images,
            train_labels,
            val_images,
            val_labels,
            search_type=args.search_type,
            search_epochs=search_epochs,
            output_dir="./results",
            save_prefix="hparam_sweep_short",
        )
        hparam_section = f"""{hparam_desc}
短训每组合 {search_epochs} epoch；详细指标见 `results/hparam_sweep_short.json` 与 `.csv`，汇总图 `figures/hyperparameter_search_summary.png`。
**短训后选取的最优超参**（将用于 {full_epochs} epoch 长训）: {hsearch.best_params}（短训验证准确率 {hsearch.best_score:.4f}）。"""
        best = hsearch.best_params or _default_hparams()
    else:
        best = _default_hparams()
        hparam_section = f"已使用 `--skip-hparam-search`，未做短训。默认/内置超参: {best}。"
        print(f"\n已跳过超参搜索，长训将使用: {best}")

    model, trainer = train_model(
        train_images,
        train_labels,
        val_images,
        val_labels,
        hidden_dim=best["hidden_dim"],
        learning_rate=best["learning_rate"],
        l2_lambda=best["l2_lambda"],
        batch_size=best["batch_size"],
        epochs=full_epochs,
        activation=best["activation"],
    )
    # 以最终长训所用地超参再写一次（与 checkpoint 一致）
    save_hparams_json("./checkpoints/best_hparams.json", best)

    evaluator, results = evaluate_on_test(
        model, test_images, test_labels, batch_size=int(best["batch_size"])
    )
    run_figures_and_report(
        model,
        trainer,
        results,
        test_images,
        test_labels,
        evaluator,
        best,
        hparam_section,
        FashionMNISTLoader.CLASS_NAMES,
        max_epochs_planned=full_epochs,
    )

    print("\n" + "=" * 60)
    print("全部完成。图表见 ./figures/，权重与历史见 ./checkpoints/，超参搜索表见 ./results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
