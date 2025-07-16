import csv
import torch
import numpy as np
from pathlib import Path

def init_metrics_logger(trainer, log_dir, full_name):
    metrics_file_path = log_dir / f"{full_name}_training_metrics.csv"
    trainer.metrics_file = open(metrics_file_path, mode="w", newline="")
    trainer.metrics_writer = csv.writer(trainer.metrics_file)

    class_labels = list(range(len(trainer.population_mapper.active_populations)))   # 🟢 ZMIANA
    biologic_labels = [trainer.population_mapper.to_pop(idx) for idx in class_labels]  # 🟢 ZMIANA
    class_headers = [f"Train Class {bio} (idx {idx})" for bio, idx in zip(biologic_labels, class_labels)]  # 🟢 ZMIANA

    multitask_headers = [
        'Train Classification Loss', 'Val Classification Loss',
        'Train Regression Loss', 'Val Regression Loss'
    ]

    trainer.metrics_writer.writerow([
        'Epoch', 'Train Samples', 'Val Samples', *class_headers,
        # --- multitask columns
        *multitask_headers,
        # --- klasyczne metryki
        'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC',
        'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUC',
        # --- nowe metryki walidacyjne ---
        'Val MAE Age', 'Val F1 Pop2 Age3-6', 'Val Composite Score',
        'Train Time (s)'
    ])

def log_epoch_metrics(trainer, epoch, loss_name, train_metrics, val_metrics, epoch_time):
    class_labels = list(range(len(trainer.population_mapper.active_populations)))   # 🟢 ZMIANA
    biologic_labels = [trainer.population_mapper.to_pop(idx) for idx in class_labels]  # 🟢 ZMIANA

    train_class_counts = get_class_distribution(train_metrics["targets"], class_labels)
    val_samples = len(val_metrics["targets"])
    train_samples = len(train_metrics["targets"])

    class_dist_str = ", ".join([f"{lbl}({bio}): {cnt}" for lbl, bio, cnt in zip(class_labels, biologic_labels, train_class_counts)])  # 🟢 ZMIANA
    print(f"\nEpoch {epoch + 1}/{trainer.cfg.training.epochs} ({loss_name}):")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%, "
          f"Precision: {train_metrics['precision']:.2f}, Recall: {train_metrics['recall']:.2f}, "
          f"F1: {train_metrics['f1']:.2f}, AUC: {train_metrics['auc']:.2f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%, "
          f"Precision: {val_metrics['precision']:.2f}, Recall: {val_metrics['recall']:.2f}, "
          f"F1: {val_metrics['f1']:.2f}, AUC: {val_metrics['auc']:.2f}")
    print(f"Train class dist: {class_dist_str}, Time: {epoch_time:.1f}s")

    train_cls_loss = train_metrics.get("classification_loss", np.nan)
    val_cls_loss = val_metrics.get("classification_loss", np.nan)
    train_reg_loss = train_metrics.get("regression_loss", np.nan)
    val_reg_loss = val_metrics.get("regression_loss", np.nan)


    trainer.metrics_writer.writerow([
        f"{loss_name}-e{epoch + 1}", train_samples, val_samples, *train_class_counts,
        # --- multitask columns
        train_cls_loss, val_cls_loss, train_reg_loss, val_reg_loss,
        # --- klasyczne metryki
        train_metrics["loss"], train_metrics["acc"], train_metrics["precision"], train_metrics["recall"],
        train_metrics["f1"], train_metrics["auc"], val_metrics["loss"], val_metrics["acc"],
        val_metrics["precision"], val_metrics["recall"], val_metrics["f1"], val_metrics["auc"],
        # --- nowe metryki walidacyjne ---
        val_metrics.get("mae_age", np.nan),
        val_metrics.get("f1_pop2_age3_6", np.nan),
        val_metrics.get("composite_score", np.nan),
        round(epoch_time, 2)
    ])

    # Logowanie do TensorBoard
    if trainer.writer:
        # Metryki treningowe
        trainer.writer.add_scalar(f'Loss/train_{loss_name}', train_metrics["loss"], epoch)
        trainer.writer.add_scalar(f'Accuracy/train_{loss_name}', train_metrics["acc"], epoch)
        trainer.writer.add_scalar(f'F1_score/train_{loss_name}', train_metrics["f1"], epoch) # Global F1 for train

        # Metryki walidacyjne
        trainer.writer.add_scalar(f'Loss/val_{loss_name}', val_metrics["loss"], epoch)
        trainer.writer.add_scalar(f'Accuracy/val_{loss_name}', val_metrics["acc"], epoch)
        trainer.writer.add_scalar(f'F1_score/val_GLOBAL_{loss_name}', val_metrics["f1"], epoch) # Global F1 for val

        # Nowe metryki walidacyjne dla TensorBoard
        if "mae_age" in val_metrics and not np.isnan(val_metrics["mae_age"]):
            trainer.writer.add_scalar(f'MAE_Age/val_{loss_name}', val_metrics["mae_age"], epoch)

        if "f1_pop2_age3_6" in val_metrics and not np.isnan(val_metrics["f1_pop2_age3_6"]):
            trainer.writer.add_scalar(f'F1_score/val_Pop2Age3-6_{loss_name}', val_metrics["f1_pop2_age3_6"], epoch)

        if "composite_score" in val_metrics and not np.isnan(val_metrics["composite_score"]):
            trainer.writer.add_scalar(f'Score/val_Composite_{loss_name}', val_metrics["composite_score"], epoch)

        # Dodatkowo logowanie komponentów strat, jeśli są dostępne (zgodnie z istniejącą logiką)
        if trainer.cfg.multitask_model.log_loss_components:
            if not np.isnan(train_cls_loss):
                 trainer.writer.add_scalar(f'Loss_Train/classification_{loss_name}', train_cls_loss, epoch)
            if not np.isnan(train_reg_loss):
                 trainer.writer.add_scalar(f'Loss_Train/regression_{loss_name}', train_reg_loss, epoch)
            if not np.isnan(val_cls_loss):
                 trainer.writer.add_scalar(f'Loss_Val/classification_{loss_name}', val_cls_loss, epoch)
            if not np.isnan(val_reg_loss):
                 trainer.writer.add_scalar(f'Loss_Val/regression_{loss_name}', val_reg_loss, epoch)

def save_best_model(trainer, current_composite_score, val_cm, model_name, loss_name, checkpoint_dir):
    """
    Zapisuje najlepszy model na podstawie composite_score.
    val_cm jest nadal przekazywane, aby można było zapisać macierz pomyłek najlepszego modelu.
    """
    # Obsługa przypadku, gdy current_composite_score jest nan
    if np.isnan(current_composite_score):
        print(f"⚠️ Composite score is NaN. Nie można porównać i zapisać modelu.")
        return trainer, False # Nie było poprawy, nie resetuj licznika early stopping

    # Porównujemy z trainer.best_score (który został zainicjalizowany na -float('inf'))
    if current_composite_score > trainer.best_score:
        trainer.best_score = current_composite_score
        trainer.best_cm = val_cm # Zapisujemy CM dla modelu z najlepszym score

        # Zmieniamy nazwę pliku, aby odzwierciedlała composite_score
        # Używamy formatowania f-string z 3 miejscami po przecinku dla score
        model_filename = f"{model_name}_{loss_name}_SCORE_{current_composite_score:.3f}.pth"
        model_path = checkpoint_dir / model_filename

        torch.save(trainer.model.state_dict(), model_path)
        trainer.last_model_path = model_path # Ścieżka do ostatnio zapisanego *najlepszego* modelu
        print(f"📂 Zapisano najlepszy model (Score: {current_composite_score:.3f}) do: {model_path}")
        trainer.early_stop_counter = 0 # Reset licznika, bo nastąpiła poprawa
        return trainer, True

    # Jeśli nie było poprawy
    return trainer, False

def should_stop_early(trainer):
    # Warunek early stopping pozostaje taki sam, ale bazuje na liczniku,
    # który jest resetowany tylko gdy `best_score` się poprawia.
    return trainer.early_stop_counter >= trainer.cfg.training.early_stopping_patience

def get_class_distribution(targets, class_labels):
    targets = [int(t.item()) if hasattr(t, 'item') else int(t) for t in targets]
    values, counts = np.unique(targets, return_counts=True)
    class_dist = {int(v): int(c) for v, c in zip(values, counts)}
    return tuple(class_dist.get(int(lbl), 0) for lbl in class_labels)

def log_augmentation_summary(augment_applied_dict, full_name, log_dir: Path = None):
    if log_dir is None:
        log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"augmentation_summary_{full_name}.csv"
    output_path = log_dir / filename

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Populacja", "Wiek", "Liczba_augmentacji"])
        for (pop, wiek), count in sorted(augment_applied_dict.items()):
            writer.writerow([pop, wiek, count])

    print(f"📝 Zapisano log augmentacji: {output_path}")
