# Uzasadnienie zmiany kryterium zatrzymania treningu — accuracy → combined score

## Problem: early stopping wyłącznie na accuracy klasyfikacji

W pierwotnej implementacji najlepszy checkpoint modelu był wybierany
według jednego kryterium:

    if val_acc > trainer.best_acc:
        save_checkpoint()

Model realizuje dwa zadania jednocześnie (multitask learning):
- klasyfikację populacji (metryka: accuracy, F1, AUC)
- regresję wieku (metryka: MAE [lata])

Ignorowanie zadania regresji przy wyborze checkpointu oznacza,
że zapisany model niekoniecznie jest optymalny dla predykcji wieku —
może to być epoka, w której accuracy jest najwyższe, ale model
wykazuje silny bias dla starszych ryb.

### Dodatkowy problem: MAE nie było obliczane podczas walidacji

Wartość `regression_loss` logowana do CSV to wartość funkcji straty
(MSE lub WeightedMSE) — nie jest to interpretowalna biologicznie miara
błędu w latach. Funkcja `validate()` zbierała predykcje klas populacji,
ale nie zbierała predykcji wieku (`age_pred`), dlatego MAE nie mogło
być obliczone ani użyte do selekcji modelu.

## Rozwiązanie: combined score

Wprowadzono kombinowaną metrykę jakości modelu (im niższa, tym lepszy model):

    combined_score = accuracy_weight · (1 − val_acc/100)
                   + mae_weight · (val_mae / mae_reference)

Gdzie:
- `val_acc`       — accuracy klasyfikacji populacji na zbiorze walidacyjnym [%]
- `val_mae`       — średni błąd bezwzględny predykcji wieku [lata], obliczany
                    jako mean(|age_pred − age_true|) po wszystkich próbkach walidacyjnych
- `mae_reference` — punkt odniesienia normalizacji MAE [lata]
- `accuracy_weight`, `mae_weight` — wagi zadań (suma = 1.0)

Checkpoint jest zapisywany gdy `combined_score` spada poniżej
dotychczasowego minimum (`trainer.best_combined_score`).

### Uzasadnienie mae_reference = 3.0

`mae_reference` pełni rolę normalizatora — skaluje MAE do zakresu
porównywalnego z `(1 − accuracy)`. Wartość 3.0 odpowiada MAE modelu
baseline przewidującego stały wiek równy medianie zbioru treningowego
(wiek ~4–5 lat), gdzie dla starszych ryb (7+) błąd wynosi 3–8 lat.

Przy `mae_reference = 3.0`:
- MAE = 1.0 lat → składnik mae = 0.33
- MAE = 3.0 lat → składnik mae = 1.00 (pełna "kara")
- MAE = 1.5 lat → składnik mae = 0.50

Dla accuracy:
- acc = 80% → składnik acc = 0.20
- acc = 70% → składnik acc = 0.30

### Zachowanie trybu "accuracy" (fallback)

Parametr `mode: "accuracy"` w konfiguracji przywraca pierwotne zachowanie
(selekcja wyłącznie po `val_acc`). Umożliwia to porównanie obu strategii
bez zmiany kodu.

## Parametry konfiguracyjne (config.yaml)

```yaml
training:
  early_stopping_metric:
    mode: "combined"       # "accuracy" | "combined"
    accuracy_weight: 0.5   # waga dla (1 − val_acc/100)
    mae_weight: 0.5        # waga dla (val_mae / mae_reference)
    mae_reference: 3.0     # punkt odniesienia MAE [lata]
```

Wagi `0.5 / 0.5` odzwierciedlają równorzędne traktowanie obu zadań.
W przypadku zmiany priorytetów biologicznych parametry można dostosować
bez modyfikacji kodu.

## Zmiany w kodzie

| Plik | Funkcja | Zmiana |
|------|---------|--------|
| `src/engine/train_loop.py` | `validate()` | Zbieranie `age_pred`, obliczanie MAE, dodanie `age_mae` do słownika wynikowego |
| `src/engine/trainer_logger.py` | `save_best_model()` | Nowy parametr `val_mae`, obliczanie `combined_score`, śledzenie `trainer.best_combined_score` |
| `src/engine/trainer_setup.py` | pętla treningowa | Przekazywanie `val_mae` do `save_best_model()`, inicjalizacja `trainer.best_combined_score = inf` |
| `src/config/config.yaml` | sekcja `training` | Nowa podsekcja `early_stopping_metric` |

## Literatura

1. Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M.,
   Dai, D., & Van Gool, L. (2021).
   *Multi-Task Learning for Dense Prediction Tasks: A Survey.*
   IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
   https://doi.org/10.1109/TPAMI.2021.3054719

2. Kendall, A., Gal, Y., & Cipolla, R. (2018).
   *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
   Geometry and Semantics.*
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018).
   https://arxiv.org/abs/1705.07115

3. Ruder, S. (2017).
   *An Overview of Multi-Task Learning in Deep Neural Networks.*
   https://arxiv.org/abs/1706.05098

4. Zhang, Y., & Yang, Q. (2022).
   *A Survey on Multi-Task Learning.*
   IEEE Transactions on Knowledge and Data Engineering, 34(12), 5586–5609.
   https://doi.org/10.1109/TKDE.2021.3070203
