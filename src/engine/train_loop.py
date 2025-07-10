import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error

# NOWA FUNKCJA
def create_subgroup_mask(meta_data_list, target_population, age_range_min, age_range_max):
    """
    Tworzy maskę booleanową dla próbek należących do określonej populacji i zakresu wieku.
    Args:
        meta_data_list (list[dict]): Lista słowników, gdzie każdy słownik zawiera 'populacja' i 'wiek'.
        target_population (int): Docelowa wartość populacji.
        age_range_min (int): Minimalny wiek (włącznie).
        age_range_max (int): Maksymalny wiek (włącznie).
    Returns:
        np.ndarray: Maska booleanowa.
    """
    mask = []
    for meta in meta_data_list:
        is_pop = (meta['populacja'] == target_population)
        is_age_in_range = (age_range_min <= meta['wiek'] <= age_range_max)
        mask.append(is_pop and is_age_in_range)
    return np.array(mask)

# NOWA FUNKCJA
def calculate_f1_subgroup(targets, predictions, mask, population_mapper):
    """
    Oblicza F1-score dla podgrupy określonej przez maskę.
    Args:
        targets (list|np.ndarray): Rzeczywiste etykiety (indeksy klas).
        predictions (list|np.ndarray): Przewidziane etykiety (indeksy klas).
        mask (np.ndarray): Maska booleanowa wskazująca elementy podgrupy.
        population_mapper (PopulationMapper): Obiekt mappera do konwersji etykiet.
    Returns:
        float: F1-score dla podgrupy lub np.nan jeśli podgrupa jest pusta.
    """
    targets_np = np.array(targets)
    predictions_np = np.array(predictions)

    subgroup_targets = targets_np[mask]
    subgroup_predictions = predictions_np[mask]

    if len(subgroup_targets) == 0:
        # print("    [calculate_f1_subgroup] Pusta podgrupa, F1 nie może być obliczone.")
        return np.nan

    # Konwersja na etykiety populacji dla spójności z globalnym F1
    subgroup_targets_pop = [population_mapper.to_pop(idx) for idx in subgroup_targets]
    subgroup_predictions_pop = [population_mapper.to_pop(idx) for idx in subgroup_predictions]

    # Używamy tych samych etykiet co dla globalnego F1, aby uniknąć problemów
    # gdyby w podgrupie nie wystąpiły wszystkie możliwe klasy.
    # Sklearn może rzucić warning jeśli labels zawierają klasy nieobecne w danych,
    # ale f1_score z average='macro' (domyślnie jest binary, potrzebujemy macro/micro/weighted)
    # lub specific label, powinien sobie poradzić.
    # Dla F1 globalnego używamy labels=mapper.all_pops() i average='macro' (domyślnie jest 'binary')
    # Tutaj dla podgrupy, jeśli chcemy F1 dla konkretnej klasy w tej podgrupie, musimy to inaczej podejść.
    # Zadanie mówi "Obliczenie F1-score dla podgrupy (pop2, wiek 3-6)" - to sugeruje, że interesuje nas
    # ogólna jakość klasyfikacji *wewnątrz* tej podgrupy, a nie F1 dla klasy "pop2" w tej podgrupie.
    # Dlatego użyjemy average='macro' lub 'weighted'.
    # labels=population_mapper.all_pops() zapewnia, że wszystkie klasy są uwzględnione w obliczeniach,
    # nawet jeśli nie wszystkie są obecne w podgrupie.
    f1 = f1_score(subgroup_targets_pop, subgroup_predictions_pop, labels=population_mapper.all_pops(), average='macro', zero_division=0)
    # print(f"    [calculate_f1_subgroup] F1 dla podgrupy ({len(subgroup_targets)} próbek): {f1:.3f}")
    return f1

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


def validate(model, device, dataloader, loss_fn, population_mapper, cfg): # Dodano cfg
    model.eval()
    stats = {'loss': 0.0, 'correct': 0, 'total': 0}
    all_targets, all_preds, all_probs = [], [], []
    all_meta_data = []
    all_age_preds = [] # <-- NOWA LINIA: do przechowywania predykcji wieku
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

            if isinstance(outputs, tuple):
                logits = outputs[0]
                age_predictions_batch = outputs[1]
                all_age_preds.extend(age_predictions_batch.detach().cpu().numpy())
            else:
                logits = outputs
                # W przypadku braku multitask, można dodać placeholder np. listę NaN lub pustą listę,
                # ale MAE nie będzie obliczane, więc można to obsłużyć później.
                # Na razie zakładamy, że jeśli chcemy MAE, to model jest multitask.

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

            # Meta jest słownikiem, gdzie klucze 'wiek' i 'populacja'
            # mapują na tensory zawierające dane dla całego batcha.
            # Musimy przekształcić to z powrotem na listę słowników,
            # po jednym dla każdej próbki w batchu.
            batch_meta_ages = meta['wiek'].cpu().tolist()
            batch_meta_populations = meta['populacja'].cpu().tolist()

            for i in range(len(batch_meta_ages)):
                all_meta_data.append({
                    'wiek': batch_meta_ages[i],
                    'populacja': batch_meta_populations[i]
                })

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

    # Obliczanie MAE dla wieku
    mae_age = np.nan
    if all_age_preds and all_meta_data:
        true_ages = [m['wiek'] for m in all_meta_data]
        # Upewnijmy się, że długości list są takie same
        if len(true_ages) == len(all_age_preds):
            # Filtrujemy próbki, dla których wiek jest znany (nie -9)
            valid_age_indices = [i for i, age in enumerate(true_ages) if age != -9]
            if valid_age_indices:
                filtered_true_ages = np.array(true_ages)[valid_age_indices]
                filtered_pred_ages = np.array(all_age_preds)[valid_age_indices]
                mae_age = mean_absolute_error(filtered_true_ages, filtered_pred_ages)
                print(f"    [validate] MAE Age: {mae_age:.3f} (dla {len(filtered_true_ages)} próbek z znanym wiekiem)")
            else:
                print("    [validate] MAE Age: Brak próbek z znanym wiekiem do obliczenia MAE.")
        else:
            print(f"    [validate] MAE Age: Niezgodność liczby metadanych ({len(true_ages)}) i predykcji wieku ({len(all_age_preds)}). Nie można obliczyć MAE.")
    else:
        print("    [validate] MAE Age: Brak predykcji wieku lub metadanych do obliczenia MAE (prawdopodobnie model nie jest multitask).")

    # Obliczanie F1-score dla podgrupy: populacja 2, wiek 3-6
    TARGET_POPULATION_SUBGROUP = 2
    AGE_MIN_SUBGROUP = 3
    AGE_MAX_SUBGROUP = 6

    subgroup_mask = create_subgroup_mask(
        meta_data_list=all_meta_data,
        target_population=TARGET_POPULATION_SUBGROUP,
        age_range_min=AGE_MIN_SUBGROUP,
        age_range_max=AGE_MAX_SUBGROUP
    )

    f1_pop2_age3_6 = calculate_f1_subgroup(
        targets=all_targets, # Używamy oryginalnych all_targets (indeksy klas)
        predictions=all_preds, # Używamy oryginalnych all_preds (indeksy klas)
        mask=subgroup_mask,
        population_mapper=mapper
    )

    if not np.isnan(f1_pop2_age3_6):
        print(f"    [validate] F1 Pop2 Age[3-6]: {f1_pop2_age3_6:.3f} (dla {np.sum(subgroup_mask)} próbek w podgrupie)")
    else:
        print(f"    [validate] F1 Pop2 Age[3-6]: Podgrupa pusta lub błąd w obliczeniach (liczba próbek: {np.sum(subgroup_mask)}).")

    # Obliczanie composite_score
    composite_score = np.nan
    try:
        weights = cfg.multitask_model.metrics_weights
        alpha = weights.alpha
        beta = weights.beta
        gamma = weights.gamma

        # Normalizacja MAE
        # Zakładamy, że MAE jest zawsze nieujemne.
        # Chcemy, aby (1 - MAE_normalized) było bliskie 1 dla małego MAE i bliskie 0 dla dużego MAE.
        # MAE_normalized powinno być w przybliżeniu w zakresie [0, 1].
        # Użyjemy MAX_EXPECTED_MAE jako wartości, przy której MAE_normalized = 1.
        MAX_EXPECTED_MAE_FOR_NORMALIZATION = 10.0 # Można to przenieść do configu później

        mae_for_score = mae_age
        if np.isnan(mae_for_score):
            # Jeśli MAE nie jest dostępne (np. nie-multitask), ten komponent powinien mieć neutralny wkład lub być pominięty.
            # W przypadku ważenia, jeśli beta > 0, to nan * beta = nan.
            # Możemy zdecydować, że jeśli MAE jest nan, to ten człon wyniku jest 0 lub średni (0.5).
            # Na razie, jeśli jest nan, cały composite_score będzie nan.
            print("    [validate] MAE is NaN, composite_score component for MAE will be NaN.")
            normalized_mae_term = np.nan
        elif MAX_EXPECTED_MAE_FOR_NORMALIZATION <= 0:
            print(f"    [validate] MAX_EXPECTED_MAE_FOR_NORMALIZATION ({MAX_EXPECTED_MAE_FOR_NORMALIZATION}) musi być dodatnie. Nie można znormalizować MAE.")
            normalized_mae_term = np.nan
        else:
            normalized_mae = min(mae_for_score / MAX_EXPECTED_MAE_FOR_NORMALIZATION, 1.0) # Ograniczenie do max 1.0
            normalized_mae_term = 1.0 - normalized_mae

        # Komponenty composite_score (global f1, (1-norm_mae), f1_subgroup)
        # Jeśli któryś z komponentów jest NaN, wynik też będzie NaN, co jest akceptowalne.
        # Wartości NaN dla F1 mogą się zdarzyć, jeśli nie ma próbek w danej klasie/podgrupie.

        # Używamy globalnego 'f1' (który jest F1 macro dla wszystkich klas)
        f1_global_component = f1
        mae_component = normalized_mae_term
        f1_subgroup_component = f1_pop2_age3_6

        # Sprawdzanie, czy wagi sumują się do 1 (opcjonalne, ale dobra praktyka)
        # if not np.isclose(alpha + beta + gamma, 1.0):
        #     print(f"    [validate] UWAGA: Wagi (alpha, beta, gamma) nie sumują się do 1 (suma: {alpha+beta+gamma:.2f})")

        # Jeśli którakolwiek z metryk składowych jest NaN, a jej waga > 0, wynik będzie NaN.
        # To jest poprawne zachowanie - nie możemy policzyć sensownego score.
        if np.isnan(f1_global_component) and alpha > 0:
            print("    [validate] Global F1 is NaN, composite_score będzie NaN.")
        if np.isnan(mae_component) and beta > 0:
            print("    [validate] MAE component is NaN, composite_score będzie NaN.")
        if np.isnan(f1_subgroup_component) and gamma > 0:
            print("    [validate] Subgroup F1 is NaN, composite_score będzie NaN.")

        composite_score = alpha * f1_global_component + \
                          beta * mae_component + \
                          gamma * f1_subgroup_component

        if not np.isnan(composite_score):
            print(f"    [validate] Composite Score: {composite_score:.3f} (alpha:{alpha}*F1g:{f1_global_component:.3f} + beta:{beta}*(1-MAE_norm):{mae_component:.3f} + gamma:{gamma}*F1s:{f1_subgroup_component:.3f})")
        else:
            print(f"    [validate] Composite Score: NaN (F1g:{f1_global_component}, MAE_comp:{mae_component}, F1s:{f1_subgroup_component})")

    except AttributeError as e:
        print(f"    [validate] Błąd przy odczycie wag dla composite_score z konfiguracji: {e}. Composite_score pozostanie NaN.")
    except Exception as e:
        print(f"    [validate] Nieoczekiwany błąd podczas obliczania composite_score: {e}. Composite_score pozostanie NaN.")


    print(f"    [validate] Loss: {loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.3f} | AUC: {auc:.3f}")

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "targets": all_targets_pop,
        "meta_data": all_meta_data,
        "mae_age": mae_age,
        "f1_pop2_age3_6": f1_pop2_age3_6,
        "composite_score": composite_score # <-- NOWA LINIA: dodanie composite_score
    }
