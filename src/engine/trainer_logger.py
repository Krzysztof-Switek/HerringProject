from __future__ import annotations

zimport csv
from pathlib import Path

import torch


def init_metrics_logger(trainer, log_dir, full_name):
    metrics_file_path = log_dir / f"{full_name}_training_metrics.csv"
    trainer.metrics_file = open(metrics_file_path, mode="w", newline="", encoding="utf-8")
    trainer.metrics_writer = csv.writer(trainer.metrics_file)

    class_indices = list(range(len(trainer.population_mapper.active_populations)))
    biologic_labels = [trainer.population_mapper.to_pop(idx) for idx in class_indices]
    class_headers = [
        f"Train Class {bio} (idx {idx})"
        for bio, idx in zip(biologic_labels, class_indices)
    ]
    val_class_headers = [
        f"Val Class {bio} (idx {idx})"
        for bio, idx in zip(biologic_labels, class_indices)
    ]

    multitask_headers = [
        "Train Classification Loss",
        "Val Classification Loss",
        "Train Regression Loss",
        "Val Regression Loss",
    ]

    trainer.metrics_writer.writerow([
        "Epoch",
        "Train Samples",
        "Val Samples",
        *class_headers,
        *val_class_headers,
        *multitask_headers,
        "Train Loss",
        "Train Accuracy",
        "Train Precision",
        "Train Recall",
        "Train F1",
        "Train AUC",
        "Val Loss",
        "Val Accuracy",
        "Val Precision",
        "Val Recall",
        "Val F1",
        "Val AUC",
        "Train Time (s)",
    ])


def log_epoch_metrics(trainer, epoch, loss_name, train_metrics, val_metrics, epoch_time):
    class_indices = list(range(len(trainer.population_mapper.active_populations)))
    biologic_labels = [trainer.population_mapper.to_pop(idx) for idx in class_indices]

    train_targets_idx = train_metrics.get("targets_idx", [])
    val_targets_idx = val_metrics.get("targets_idx", [])

    train_class_counts = get_class_distribution(train_targets_idx, class_indices)
    val_class_counts = get_class_distribution(val_targets_idx, class_indices)

    train_samples = len(train_targets_idx) if train_targets_idx else len(train_metrics.get("targets", []))
    val_samples = len(val_targets_idx) if val_targets_idx else len(val_metrics.get("targets", []))

    class_dist_str = ", ".join(
        [
            f"{idx}({bio}): {cnt}"
            for idx, bio, cnt in zip(class_indices, biologic_labels, train_class_counts)
        ]
    )

    print(f"\nEpoch {epoch + 1}/{trainer.cfg.training.epochs} ({loss_name}):")
    print(
        f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%, "
        f"Precision: {train_metrics['precision']:.2f}, Recall: {train_metrics['recall']:.2f}, "
        f"F1: {train_metrics['f1']:.2f}, AUC: {train_metrics['auc']:.2f}"
    )
    print(
        f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%, "
        f"Precision: {val_metrics['precision']:.2f}, Recall: {val_metrics['recall']:.2f}, "
        f"F1: {val_metrics['f1']:.2f}, AUC: {val_metrics['auc']:.2f}"
    )
    print(f"Train class dist: {class_dist_str}, Time: {epoch_time:.1f}s")

    train_cls_loss = train_metrics.get("classification_loss", None)
    val_cls_loss = val_metrics.get("classification_loss", None)
    train_reg_loss = train_metrics.get("regression_loss", None)
    val_reg_loss = val_metrics.get("regression_loss", None)

    trainer.metrics_writer.writerow([
        f"{loss_name}-e{epoch + 1}",
        train_samples,
        val_samples,
        *train_class_counts,
        *val_class_counts,
        train_cls_loss,
        val_cls_loss,
        train_reg_loss,
        val_reg_loss,
        train_metrics["loss"],
        train_metrics["acc"],
        train_metrics["precision"],
        train_metrics["recall"],
        train_metrics["f1"],
        train_metrics["auc"],
        val_metrics["loss"],
        val_metrics["acc"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
        round(epoch_time, 2),
    ])


def save_best_model(trainer, val_acc, val_cm, model_name, loss_name, checkpoint_dir):
    if val_acc > trainer.best_acc:
        trainer.best_acc = val_acc
        trainer.best_cm = val_cm

        model_path = checkpoint_dir / f"{model_name}_{loss_name}_ACC_{val_acc:.2f}.pth"
        torch.save(trainer.model.state_dict(), model_path)

        trainer.last_model_path = model_path
        trainer.early_stop_counter = 0

        print(f"📂 Zapisano najlepszy model do: {model_path}")
        return trainer, True

    return trainer, False


def should_stop_early(trainer):
    return trainer.early_stop_counter >= trainer.cfg.training.early_stopping_patience


def get_class_distribution(targets, class_labels):
    targets = [int(t.item()) if hasattr(t, "item") else int(t) for t in targets]
    class_dist = {int(lbl): 0 for lbl in class_labels}

    for target in targets:
        class_dist[target] = class_dist.get(target, 0) + 1

    return tuple(class_dist.get(int(lbl), 0) for lbl in class_labels)


def log_augmentation_summary(augment_applied_dict, full_name, log_dir: Path = None):
    if log_dir is None:
        log_dir = Path("results/logs")

    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"augmentation_summary_{full_name}.csv"
    output_path = log_dir / filename

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Populacja", "Wiek", "Liczba_augmentacji"])

        for (pop, wiek), count in sorted(augment_applied_dict.items()):
            writer.writerow([pop, wiek, count])

    print(f"📝 Zapisano log augmentacji: {output_path}")