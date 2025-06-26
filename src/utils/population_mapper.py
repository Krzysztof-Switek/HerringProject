class PopulationMapper:
    def __init__(self, active_populations):
        # Sprawdź unikalność i jawnie konwertuj do int
        self.active_populations = [int(p) for p in active_populations]
        self.pop_to_idx = {pop: idx for idx, pop in enumerate(self.active_populations)}
        self.idx_to_pop = {idx: pop for idx, pop in enumerate(self.active_populations)}

    def to_idx(self, pop):
        """Konwertuj numer populacji (Excel) na indeks klasy (model)."""
        if int(pop) not in self.pop_to_idx:
            raise ValueError(f"Nieznana populacja: {pop} (dostępne: {self.active_populations})")
        return self.pop_to_idx[int(pop)]

    def to_pop(self, idx):
        """Konwertuj indeks klasy na numer populacji (Excel)."""
        if int(idx) not in self.idx_to_pop:
            raise ValueError(f"Nieznany indeks klasy: {idx} (dostępne: {list(self.idx_to_pop.keys())})")
        return self.idx_to_pop[int(idx)]

    def map_targets(self, targets):
        """Konwertuj listę indeksów na numery populacji."""
        return [self.to_pop(idx) for idx in targets]

    def map_preds(self, preds):
        """Konwertuj listę indeksów na numery populacji."""
        return [self.to_pop(idx) for idx in preds]
