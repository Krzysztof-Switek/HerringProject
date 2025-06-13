import csv
import torch

def init_metrics_logger(trainer, log_dir):
    metrics_file_path = log_dir / "training_metrics.csv"
    trainer.metrics_file = open(metrics_file_path, mode="w", newline="")
    trainer.metrics_writer = csv.writer(trainer.metrics_file)
    trainer.metrics_writer.writerow([
        'Epoch', 'Train Samples', 'Val Samples', 'Train Class 0', 'Train Class 1',
        'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
        'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
        'Train Time (s)'
    ])

def log_epoch_metrics(trainer, epoch, loss_name, train_metrics, val_metrics, epoch_time):
    train_c0, train_c1 = get_class_distribution(train_metrics[-1])
    val_samples = len(val_metrics[-1])
    train_samples = len(train_metrics[-1])

    print(f"\nEpoch {epoch + 1}/{trainer.cfg.training.epochs} ({loss_name}):")
    print(f"Train - Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.2f}%, Precision: {train_metrics[2]:.2f}, Recall: {train_metrics[3]:.2f}, F1: {train_metrics[4]:.2f}, AUC: {train_metrics[5]:.2f}")
    print(f"Val   - Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.2f}%, Precision: {val_metrics[2]:.2f}, Recall: {val_metrics[3]:.2f}, F1: {val_metrics[4]:.2f}, AUC: {val_metrics[5]:.2f}")
    print(f"Train class dist: 0: {train_c0}, 1: {train_c1}, Time: {epoch_time:.1f}s")

    trainer.metrics_writer.writerow([
        f"{loss_name}-e{epoch + 1}", train_samples, val_samples, train_c0, train_c1,
        *train_metrics[:-1], *val_metrics[:-2], round(epoch_time, 2)
    ])

def save_best_model(trainer, val_acc, val_cm, model_name, loss_name, checkpoint_dir):
    if val_acc > trainer.best_acc:
        trainer.best_acc = val_acc
        trainer.best_cm = val_cm
        model_path = checkpoint_dir / f"{model_name}_{loss_name}_ACC_{val_acc:.2f}.pth"
        torch.save(trainer.model.state_dict(), model_path)
        trainer.last_model_path = model_path
        print(f"📂 Zapisano najlepszy model do: {model_path}")
        trainer.early_stop_counter = 0
        return trainer, True
    return trainer, False

def should_stop_early(trainer):
    return trainer.early_stop_counter >= trainer.cfg.training.early_stopping_patience

def get_class_distribution(targets):
    import numpy as np
    values, counts = np.unique(targets, return_counts=True)
    class_dist = {int(v): int(c) for v, c in zip(values, counts)}
    return class_dist.get(0, 0), class_dist.get(1, 0)
