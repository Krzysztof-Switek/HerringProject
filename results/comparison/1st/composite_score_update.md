# Composite Score — zmiana kryterium selekcji modelu

## Wersja poprzednia: accuracy-only

Najlepszy checkpoint modelu był wybierany według jednego kryterium:

    composite_score (stary) = val_accuracy

    zapis checkpointu gdy: val_acc > best_acc

Model zapisywany przy najwyższej accuracy klasyfikacji populacji,
niezależnie od jakości predykcji wieku.

**Konsekwencja:** W trybie multitask model realizuje dwa zadania.
Optymalizacja wyłącznie pod kątem klasyfikacji może prowadzić do wyboru
epoki, w której model "wyspecjalizował się" w populacji kosztem regresji.
Early stopping zatrzymuje trening w punkcie maksymalnej accuracy — który
może nie pokrywać się z punktem minimalnego MAE.

Dodatkowo: MAE nie było obliczane podczas walidacji. Wartość
`regression_loss` logowana do CSV to wartość funkcji straty (MSE lub
WeightedMSE) — nie interpretowalna biologicznie miara błędu w latach.

---

## Wersja aktualna: combined score

Kryterium selekcji łączy oba zadania w jednej metryce:

    composite_score (nowy) = accuracy_weight · (1 − val_acc/100)
                           + mae_weight · (val_mae / mae_reference)

    zapis checkpointu gdy: composite_score < best_composite_score

Parametry (konfigurowane w `config.yaml / training.early_stopping_metric`):

| Parametr | Wartość | Rola |
|----------|---------|------|
| `accuracy_weight` | 0.5 | waga klasyfikacji |
| `mae_weight` | 0.5 | waga regresji |
| `mae_reference` | 3.0 lat | normalizacja MAE |

### Interpretacja składników

**Składnik klasyfikacji:** `(1 − val_acc/100)`

Przekształca accuracy [%] na metrykę minimalizowaną [0, 1]:
- accuracy = 100% → składnik = 0.00 (idealny)
- accuracy = 80%  → składnik = 0.20
- accuracy = 70%  → składnik = 0.30

**Składnik regresji:** `val_mae / mae_reference`

Normalizuje MAE [lata] do skali porównywalnej z klasyfikacją:
- MAE = 0.0 lat → składnik = 0.00 (idealny)
- MAE = 1.5 lat → składnik = 0.50
- MAE = 3.0 lat → składnik = 1.00 (baseline: model bez uczenia)
- MAE > 3.0 lat → składnik > 1.00 (gorszy od baseline)

`mae_reference = 3.0` odpowiada MAE modelu przewidującego stale medianę
zbioru treningowego (~wiek 4–5 lat), gdzie dla starszych ryb błąd wynosi
3–8 lat. Jest to naturalny punkt odniesienia "bez uczenia" dla tego datasetu.

### Przykład porównania dwóch epok

| Epoka | val_acc | val_mae | Stary: zapisz? | Nowy composite | Nowy: zapisz? |
|-------|---------|---------|----------------|----------------|---------------|
| 10 | **82.0%** | 1.8 lat | **Tak** (wyższa acc) | 0.5·0.18 + 0.5·0.60 = **0.390** | Tak (niższy) |
| 14 | 81.5% | **1.2 lat** | Nie (niższa acc) | 0.5·0.185 + 0.5·0.40 = **0.293** | **Tak** (niższy) |

W powyższym przykładzie stare kryterium wybiera epokę 10 (wyższa accuracy),
nowe kryterium wybiera epokę 14 (niższy błąd wieku, nieznacznie niższa accuracy).
Epoka 14 jest biologicznie lepsza — przewiduje wiek o 0.6 roku dokładniej
przy zaniedbywalnej stracie 0.5 pp w klasyfikacji populacji.

---

## Uzasadnienie naukowe

### Problem selekcji modelu w multitask learning

Wybór optymalnego punktu zatrzymania treningu w multitask learning jest
nietrywialny: zadania mogą osiągać maksimum w różnych epokach, a poprawa
jednego zadania może następować kosztem drugiego (negative transfer).

Standardowym podejściem jest zdefiniowanie skalowanej kombinowanej metryki
walidacyjnej, która uwzględnia wszystkie zadania z wagami odzwierciedlającymi
ich względną ważność biologiczną lub aplikacyjną.

### Normalizacja jako warunek konieczny

Bezpośrednie sumowanie accuracy i MAE jest niepoprawne bez normalizacji —
obie metryki mają różne jednostki i zakresy:
- accuracy ∈ [0%, 100%]
- MAE ∈ [0 lat, ~10 lat]

Bez normalizacji MAE dominowałoby nad accuracy w surowej sumie.
Zastosowana normalizacja `(1 − acc/100)` i `mae / mae_ref` sprowadza
obie metryki do bezwymiarowej skali ∈ [0, ~1].

### Wybór wag 0.5 / 0.5

Równe wagi odzwierciedlają traktowanie obu zadań jako równorzędnych.
W kontekście badań biologicznych śledzia bałtyckiego:
- klasyfikacja populacji — ważna dla identyfikacji stad
- predykcja wieku — ważna dla oceny struktury wiekowej i dynamiki populacji

Parametry są konfigurowalne w `config.yaml`, co pozwala dostosować
priorytety bez zmiany kodu w przypadku zmiany założeń badawczych.

---

## Zmiana techniczna: obliczanie MAE w validate()

Poprzednia implementacja funkcji `validate()` w `train_loop.py` nie zbierała
predykcji wieku (`age_pred`). Wartość `regression_loss` w CSV odpowiadała
wartości funkcji straty (WeightedMSE), a nie MAE w latach — metryki różnią
się jednostkami i nie są bezpośrednio porównywalne.

Obecna implementacja:
1. Zbiera `age_pred` per próbka podczas walidacji
2. Oblicza `age_mae = mean(|age_pred − age_true|)` po całym zbiorze val
3. Loguje `age_mae` do CSV i używa go w combined score

---

## Literatura

1. Vandenhende, S. et al. (2021). *Multi-Task Learning for Dense Prediction
   Tasks: A Survey.* IEEE TPAMI.
   https://doi.org/10.1109/TPAMI.2021.3054719

2. Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using
   Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018.
   https://arxiv.org/abs/1705.07115

3. Sener, O., & Koltun, V. (2018). *Multi-Task Learning as Multi-Objective
   Optimization.* NeurIPS 2018.
   https://arxiv.org/abs/1810.04650

4. Zhang, Y., & Yang, Q. (2022). *A Survey on Multi-Task Learning.*
   IEEE TKDE, 34(12), 5586–5609.
   https://doi.org/10.1109/TKDE.2021.3070203
