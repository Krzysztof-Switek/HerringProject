from __future__ import annotations


def get_class_metadata(trainer) -> dict:
    """
    Zwraca class_counts (lista po indeksie klasy) i class_freq (dict indeks→liczebność)
    potrzebne do LDAMLoss, SeesawLoss i ClassBalancedFocalLoss.
    Agreguje dane z HerringDataset.class_counts (Counter po (pop, wiek)).
    """
    population_mapper = trainer.population_mapper
    data_loader = trainer.data_loader

    num_classes = len(population_mapper.active_populations)
    counts_by_idx = {i: 0 for i in range(num_classes)}

    for (pop, _wiek), count in data_loader.class_counts.items():
        if pop in population_mapper.pop_to_idx:
            idx = population_mapper.to_idx(pop)
            counts_by_idx[idx] += count

    class_counts = [counts_by_idx[i] for i in range(num_classes)]
    class_freq = {i: counts_by_idx[i] for i in range(num_classes)}

    return {
        "class_counts": class_counts,
        "class_freq": class_freq,
    }