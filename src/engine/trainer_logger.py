import csv
import torch
from datetime import datetime
from pathlib import Path

def init_metrics_logger(trainer, log_dir, full_name):
    metrics_file_path = log_dir / f"{full_name}_training_metrics.csv"
    trainer.metrics_file = open(metrics_file_path, mode="w", newline="")
    trainer.metrics_writer = csv.writer(trainer.metrics_file)
    # dynamicznie nazwy kolumn dla każdej populacji
    class_labels = list(trainer.cfg.data.active_populations)
    class_headers = [f"Train Class {pop}" for pop in class_labels]
    trainer.metrics_writer.writerow([
        'Epoch', 'Train Samples', 'Val Samples', *class_headers,
        'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
        'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
        'Train Time (s)'
    ])

# dodaj przekazanie listy populacji z configa
def log_epoch_metrics(trainer, epoch, loss_name, train_metrics, val_metrics, epoch_time):
    class_labels = list(trainer.cfg.data.active_populations)
    train_class_counts = get_class_distribution(train_metrics[-1], class_labels)
    val_samples = len(val_metrics[-1])
    train_samples = len(train_metrics[-1])

    print(f"\nEpoch {epoch + 1}/{trainer.cfg.training.epochs} ({loss_name}):")
    print(f"Train - Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.2f}%, Precision: {train_metrics[2]:.2f}, Recall: {train_metrics[3]:.2f}, F1: {train_metrics[4]:.2f}, AUC: {train_metrics[5]:.2f}")
    print(f"Val   - Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.2f}%, Precision: {val_metrics[2]:.2f}, Recall: {val_metrics[3]:.2f}, F1: {val_metrics[4]:.2f}, AUC: {val_metrics[5]:.2f}")
    print(f"Train class dist: {dict(zip(class_labels, train_class_counts))}, Time: {epoch_time:.1f}s")

    trainer.metrics_writer.writerow([
        f"{loss_name}-e{epoch + 1}", train_samples, val_samples, *train_class_counts,
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

# 🟢 ZMIANA: przyjmuje class_labels jako argument
def get_class_distribution(targets, class_labels):
    import numpy as np
    values, counts = np.unique(targets, return_counts=True)
    class_dist = {int(v): int(c) for v, c in zip(values, counts)}
    return tuple(class_dist.get(int(lbl), 0) for lbl in class_labels)

def log_augmentation_summary(augment_applied_dict, full_name, log_dir: Path = None):
    """Zapisuje informacje o augmentacji do pliku CSV w katalogu log_dir (domyślnie results/logs)."""
    if log_dir is None:
        log_dir = Path("results/logs")

    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"augmentation_summary_{full_name}.csv"
    output_path = log_dir / filename

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Populacja", "Wiek", "Liczba_augmentacji"])
        for (pop, wiek), count in sorted(augment_applied_dict.items()):
            writer.writerow([pop, wiek, count])

    print(f"📝 Zapisano log augmentacji: {output_path}")
