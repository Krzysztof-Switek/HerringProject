from __future__ import annotations


def get_class_metadata(trainer) -> dict:
    """
    Zwraca metadane klas obliczone wyłącznie z danych treningowych (train/).

    Zwracane klucze:
    - class_counts : lista [count_idx_0, count_idx_1, ...] — do LDAMLoss / SeesawLoss
    - class_freq   : dict {idx -> count} — do ClassBalancedFocalLoss
    - age_counts   : dict {wiek -> count} — do WeightedMSELoss (regresja wieku)

    HerringDataset.class_counts jest teraz obliczane wyłącznie z katalogu train/,
    więc wszystkie trzy struktury są wolne od data leakage z val/test.
    """
    population_mapper = trainer.population_mapper
    data_loader = trainer.data_loader

    num_classes = len(population_mapper.active_populations)
    counts_by_idx = {i: 0 for i in range(num_classes)}
    age_counts: dict[int, int] = {}

    for (pop, wiek), count in data_loader.class_counts.items():
        if pop in population_mapper.pop_to_idx:
            idx = population_mapper.to_idx(pop)
            counts_by_idx[idx] += count
            age_counts[wiek] = age_counts.get(wiek, 0) + count

    class_counts = [counts_by_idx[i] for i in range(num_classes)]
    class_freq = {i: counts_by_idx[i] for i in range(num_classes)}

    return {
        "class_counts": class_counts,
        "class_freq": class_freq,
        "age_counts": age_counts,
    }