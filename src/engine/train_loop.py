import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    # pobierz klasy z dataloadera, najlepiej z configu
    class_labels = None
    if hasattr(dataloader.dataset, 'base_dataset') and hasattr(dataloader.dataset.base_dataset, 'classes'):
        class_labels = [int(x) for x in dataloader.dataset.base_dataset.classes]
    elif hasattr(dataloader.dataset, 'classes'):
        class_labels = [int(x) for x in dataloader.dataset.classes]

    for inputs, targets, meta in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, targets, meta)
        loss.backward()
        optimizer.step()

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets_np = targets.cpu().numpy()

        stats['loss'] += loss.item()
        stats['correct'] += int(np.sum(preds == targets_np))
        stats['total'] += targets.size(0)
        # 🟢 ZMIANA: filtruj tylko dozwolone klasy
        if class_labels is not None:
            for target, pred, prob in zip(targets_np, preds, probs):
                if int(target) in class_labels:
                    all_targets.append(int(target))
                    all_preds.append(int(pred))
                    all_probs.append(float(prob))
        else:
            all_targets.extend(targets_np)
            all_preds.extend(preds)
            all_probs.extend(probs)

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "targets": all_targets
    }

def validate(model, device, dataloader, loss_fn):
    model.eval()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    # pobierz klasy z dataloadera, najlepiej z configu
    class_labels = None
    if hasattr(dataloader.dataset, 'base_dataset') and hasattr(dataloader.dataset.base_dataset, 'classes'):
        class_labels = [int(x) for x in dataloader.dataset.base_dataset.classes]
    elif hasattr(dataloader.dataset, 'classes'):
        class_labels = [int(x) for x in dataloader.dataset.classes]

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) != 3:
                raise ValueError(
                    f"Batch walidacyjny musi zawierać 3 elementy: (inputs, targets, meta). "
                    f"Aktualnie: {len(batch)} elementów! Popraw DataLoader dla walidacji."
                )
            inputs, targets, meta = batch

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets, meta)
            stats['loss'] += loss.item()

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            stats['correct'] += int(np.sum(preds == targets_np))
            stats['total'] += targets.size(0)
            # 🟢 ZMIANA: filtruj tylko dozwolone klasy
            if class_labels is not None:
                for target, pred, prob in zip(targets_np, preds, probs):
                    if int(target) in class_labels:
                        all_targets.append(int(target))
                        all_preds.append(int(pred))
                        all_probs.append(float(prob))
            else:
                all_targets.extend(targets_np)
                all_preds.extend(preds)
                all_probs.extend(probs)

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "targets": all_targets
    }
