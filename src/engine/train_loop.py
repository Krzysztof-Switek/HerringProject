from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _extract_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _safe_multiclass_auc(all_targets_idx, all_probs_matrix):
    """
    Zwraca AUC. Rozroznia przypadek binarny i wieloklasowy:
    - Binary (2 klasy): roc_auc_score(y_true, y_score[:, 1]) — wymaga 1D probs
    - Multiclass (>2): roc_auc_score(y_true, y_score, multi_class="ovr") — wymaga 2D
    Zwraca 0.0 jesli tylko jedna klasa w danych lub blad obliczen.
    """
    if not all_targets_idx or all_probs_matrix is None:
        return 0.0

    y_true = np.asarray(all_targets_idx)
    y_score = np.asarray(all_probs_matrix)

    if y_score.ndim != 2:
        return 0.0

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.0

    try:
        n_classes = y_score.shape[1]
        if n_classes == 2:
            # Binary: roc_auc_score wymaga 1D (prob klasy pozytywnej)
            return float(roc_auc_score(y_true, y_score[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except Exception as e:
        print(f"WARN: Blad obliczania AUC: {e}")
        return 0.0


def _finalize_epoch_metrics(stats, all_targets_idx, all_preds_idx, all_probs_matrix, population_mapper, include_cm=False):
    mapper = population_mapper

    loss = stats["loss"] / max(stats["num_batches"], 1)
    acc = 100.0 * stats["correct"] / max(stats["total"], 1)
    cls_loss = stats.get("cls_loss", 0.0) / max(stats["num_batches"], 1)
    reg_loss = stats.get("reg_loss", 0.0) / max(stats["num_batches"], 1)

    all_targets_pop = [mapper.to_pop(idx) for idx in all_targets_idx]
    all_preds_pop = [mapper.to_pop(idx) for idx in all_preds_idx]

    print(
        f"    Unikalne klasy w targetach: {set(all_targets_pop)} | "
        f"w predykcjach: {set(all_preds_pop)}"
    )

    labels_pop = mapper.all_pops()

    precision = precision_score(
        all_targets_pop,
        all_preds_pop,
        labels=labels_pop,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        all_targets_pop,
        all_preds_pop,
        labels=labels_pop,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        all_targets_pop,
        all_preds_pop,
        labels=labels_pop,
        average="macro",
        zero_division=0,
    )
    auc = _safe_multiclass_auc(all_targets_idx, all_probs_matrix)

    result = {
        "loss": float(loss),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "targets": all_targets_pop,
        "preds": all_preds_pop,
        "targets_idx": list(all_targets_idx),
        "preds_idx": list(all_preds_idx),
    }

    if cls_loss > 0.0 or reg_loss > 0.0:
        result["classification_loss"] = float(cls_loss)
        result["regression_loss"] = float(reg_loss)

    if include_cm:
        result["cm"] = confusion_matrix(
            all_targets_pop,
            all_preds_pop,
            labels=labels_pop,
        )

    return result


def train_epoch(model, device, dataloader, loss_fn, optimizer, population_mapper):
    model.train()

    stats = {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
        "num_batches": 0,
        "cls_loss": 0.0,
        "reg_loss": 0.0,
    }

    all_targets_idx = []
    all_preds_idx = []
    all_probs_matrix = []

    print(f"\n⏩ [train_epoch] Start trenowania. Liczba batchy: {len(dataloader)}")

    for batch_idx, (inputs, targets, meta) in enumerate(dataloader):
        if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
            print(
                f"    [train_epoch] Batch {batch_idx + 1}/{len(dataloader)} "
                f"({round(100 * (batch_idx + 1) / max(len(dataloader), 1))}%)"
            )

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets, meta)
        loss.backward()
        optimizer.step()

        logits = _extract_logits(outputs)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        targets_np = targets.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        stats["loss"] += float(loss.item())
        stats["correct"] += int(np.sum(preds_np == targets_np))
        stats["total"] += int(targets.size(0))
        stats["num_batches"] += 1

        if hasattr(loss_fn, "_last_cls_loss"):
            stats["cls_loss"] += float(loss_fn._last_cls_loss.item())
            stats["reg_loss"] += float(loss_fn._last_reg_loss.item())

        all_targets_idx.extend(targets_np.tolist())
        all_preds_idx.extend(preds_np.tolist())
        all_probs_matrix.extend(probs_np.tolist())

    print("✅ [train_epoch] Epoka zakończona. Obliczam metryki...")

    result = _finalize_epoch_metrics(
        stats=stats,
        all_targets_idx=all_targets_idx,
        all_preds_idx=all_preds_idx,
        all_probs_matrix=all_probs_matrix,
        population_mapper=population_mapper,
        include_cm=False,
    )

    print(
        f"    [train_epoch] Loss: {result['loss']:.4f} | "
        f"Acc: {result['acc']:.2f}% | "
        f"F1: {result['f1']:.3f} | "
        f"AUC: {result['auc']:.3f}"
    )

    return result


def validate(model, device, dataloader, loss_fn, population_mapper):
    model.eval()

    stats = {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
        "num_batches": 0,
        "cls_loss": 0.0,
        "reg_loss": 0.0,
    }

    all_targets_idx = []
    all_preds_idx = []
    all_probs_matrix = []

    print(f"\n⏩ [validate] Start walidacji. Liczba batchy: {len(dataloader)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
                print(
                    f"    [validate] Batch {batch_idx + 1}/{len(dataloader)} "
                    f"({round(100 * (batch_idx + 1) / max(len(dataloader), 1))}%)"
                )

            if len(batch) != 3:
                raise ValueError(
                    "Batch walidacyjny musi zawierać 3 elementy: "
                    "(inputs, targets, meta). "
                    f"Aktualnie: {len(batch)} elementów."
                )

            inputs, targets, meta = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets, meta)

            logits = _extract_logits(outputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            targets_np = targets.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

            stats["loss"] += float(loss.item())
            stats["correct"] += int(np.sum(preds_np == targets_np))
            stats["total"] += int(targets.size(0))
            stats["num_batches"] += 1

            if hasattr(loss_fn, "_last_cls_loss"):
                stats["cls_loss"] += float(loss_fn._last_cls_loss.item())
                stats["reg_loss"] += float(loss_fn._last_reg_loss.item())

            all_targets_idx.extend(targets_np.tolist())
            all_preds_idx.extend(preds_np.tolist())
            all_probs_matrix.extend(probs_np.tolist())

    print("✅ [validate] Walidacja zakończona. Obliczam metryki...")

    result = _finalize_epoch_metrics(
        stats=stats,
        all_targets_idx=all_targets_idx,
        all_preds_idx=all_preds_idx,
        all_probs_matrix=all_probs_matrix,
        population_mapper=population_mapper,
        include_cm=True,
    )

    print(
        f"    [validate] Loss: {result['loss']:.4f} | "
        f"Acc: {result['acc']:.2f}% | "
        f"F1: {result['f1']:.3f} | "
        f"AUC: {result['auc']:.3f}"
    )

    return result