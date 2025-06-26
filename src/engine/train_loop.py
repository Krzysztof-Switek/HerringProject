import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def train_epoch(model, device, dataloader, loss_fn, optimizer,
                population_mapper):  # ZMIENIONO: dodano population_mapper
    model.train()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    # *** MAPOWANIE: pobieramy idx->pop oraz pop->idx ***
    mapper = population_mapper  # DODANO: jawny mapper

    for inputs, targets, meta in dataloader:
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

        # *** ZAMIANA: zapamiętujemy indeksy, NIE numery populacji ***
        all_targets.extend(targets_np)
        all_preds.extend(preds_np)
        all_probs.extend(probs_np[range(len(preds_np)), preds_np])  # prawdopodobieństwo trafionej klasy

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    # *** MAPOWANIE: wyświetl metryki dla populacji (numery biologiczne) ***
    all_targets_pop = [mapper.to_pop(idx) for idx in all_targets]  # DODANO
    all_preds_pop = [mapper.to_pop(idx) for idx in all_preds]  # DODANO

    precision = precision_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    recall = recall_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    f1 = f1_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')  # ZMIENIONO (indexy klas)

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "targets": all_targets_pop  # ZAMIANA: zwracaj jako numery populacji
    }


def validate(model, device, dataloader, loss_fn, population_mapper):  # ZMIENIONO: dodano population_mapper
    model.eval()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    mapper = population_mapper  # DODANO

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

    loss = stats['loss'] / len(dataloader)
    acc = 100. * stats['correct'] / stats['total']
    # MAPUJEMY do numerów populacji:
    all_targets_pop = [mapper.to_pop(idx) for idx in all_targets]  # DODANO
    all_preds_pop = [mapper.to_pop(idx) for idx in all_preds]  # DODANO

    precision = precision_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    recall = recall_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    f1 = f1_score(all_targets_pop, all_preds_pop, labels=mapper.all_pops(), zero_division=0)  # ZMIENIONO
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')  # ZMIENIONO
    cm = confusion_matrix(all_targets_pop, all_preds_pop, labels=mapper.all_pops())  # ZMIENIONO

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

