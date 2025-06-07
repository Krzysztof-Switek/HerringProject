import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from engine.loss_utils import get_sample_weights

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    for inputs, targets, meta in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        weights = get_sample_weights(meta['populacja'], meta['wiek']).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()

        stats['loss'] += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        targets_np = targets.cpu().numpy()

        stats['correct'] += (preds == targets_np).sum()
        stats['total'] += targets.size(0)
        all_targets.extend(targets_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

    loss = stats['loss'] / len(train_loader)
    acc = 100. * stats['correct'] / stats['total']
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    return loss, acc, precision, recall, f1, auc, all_targets

def validate(model, device, val_loader, criterion):
    model.eval()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss.mean()
            stats['loss'] += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            stats['correct'] += (preds == targets_np).sum()
            stats['total'] += targets.size(0)
            all_targets.extend(targets_np)
            all_preds.extend(preds)
            all_probs.extend(probs)

    loss = stats['loss'] / len(val_loader)
    acc = 100. * stats['correct'] / stats['total']
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    cm = confusion_matrix(all_targets, all_preds)
    return loss, acc, precision, recall, f1, auc, cm, all_targets
