import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_epoch(model, device, dataloader, loss_fn, optimizer, population_mapper):
    model.train()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []
    mapper = population_mapper

    print(f"\n⏩ [train_epoch] Start trenowania. Liczba batchy: {len(dataloader)}")

    for batch_idx, (inputs, targets, meta) in enumerate(dataloader):
        if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
            print(f"    [train_epoch] Batch {batch_idx+1}/{len(dataloader)} ({round(100*(batch_idx+1)/len(dataloader))}%)")

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, targets, meta)
        loss.backward()
        optimizer.step()

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)  # [batch, num_classes]
        preds = logits.argmax(dim=1)
        targets_np = targets.cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        stats['loss'] += loss.item()
        stats['correct'] += int(np.sum(preds_np == targets_np))
        stats['total'] += targets.size(0)

        all_targets.extend(targets_np)
        all_preds.extend(preds_np)
        all_probs.extend(probs_np[range(len(preds_np)), preds_np])

    print(f"✅ [train_epoch] Epoka zakończona. Obliczam metryki...")

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    all_targets_pop = [mapper.to_pop(idx) for idx in all_targets]
    all_preds_pop = [mapper.to_pop(idx) for idx in all_preds]

    print(f"    [train_epoch] Unikalne klasy w targetach: {set(all_targets_pop)} | w predykcjach: {set(all_preds_pop)}")

    precision = precision_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    recall = recall_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    f1 = f1_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')

    print(f"    [train_epoch] Loss: {loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.3f} | AUC: {auc:.3f}")

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "targets": all_targets_pop
    }


def validate(model, device, dataloader, loss_fn, population_mapper):
    model.eval()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []
    mapper = population_mapper

    print(f"\n⏩ [validate] Start walidacji. Liczba batchy: {len(dataloader)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
                print(f"    [validate] Batch {batch_idx+1}/{len(dataloader)} ({round(100*(batch_idx+1)/len(dataloader))}%)")

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
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            stats['correct'] += int(np.sum(preds_np == targets_np))
            stats['total'] += targets.size(0)

            all_targets.extend(targets_np)
            all_preds.extend(preds_np)
            all_probs.extend(probs_np[range(len(preds_np)), preds_np])

    print(f"✅ [validate] Walidacja zakończona. Obliczam metryki...")

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    all_targets_pop = [mapper.to_pop(idx) for idx in all_targets]
    all_preds_pop = [mapper.to_pop(idx) for idx in all_preds]

    print(f"    [validate] Unikalne klasy w targetach: {set(all_targets_pop)} | w predykcjach: {set(all_preds_pop)}")

    precision = precision_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    recall = recall_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    f1 = f1_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    cm = confusion_matrix(all_targets_pop, all_preds_pop, labels=mapper.all_pops())

    print(f"    [validate] Loss: {loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.3f} | AUC: {auc:.3f}")

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "targets": all_targets_pop
    }
