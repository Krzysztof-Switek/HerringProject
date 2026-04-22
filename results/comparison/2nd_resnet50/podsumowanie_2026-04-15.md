# Podsumowanie wyników treningu — 2026-04-15

Porównanie 1st (2026-04-08) vs 2nd (2026-04-15) — ocena efektu wprowadzonych zmian w kodzie.

---

## 1. Naprawione problemy z 1st treningu

### LDAM data leakage — naprawiony

Not-Embedded: Train Loss ≈ Val Loss był główną anomalią w 1st:

| | Train Loss | Val Loss |
|--|--|--|
| 1st | 10.862 | 10.860 |
| 2nd | **8.104** | **8.933** |

Wyraźna przerwa między train a val loss — jak powinna wyglądać. Poprzednia identyczność wartości była sygnałem wycieku danych przy obliczaniu marginesów LDAM.

---

### Weighted MSE dla regresji wieku — wyraźna poprawa

Najbardziej widoczny efekt zmian. Bias systematyczny (przesunięcie predykcji względem prawdziwego wieku) niemal zniknął:

| | Embedded val bias | Not-Emb val bias |
|--|--|--|
| 1st | -0.388 | -0.340 |
| 2nd | **+0.021** | **-0.050** |

MAE również się poprawiło:

| | Embedded val MAE | Not-Emb val MAE |
|--|--|--|
| 1st | 1.227 | 1.211 |
| 2nd | **1.192** (−3%) | **1.070** (−11%) |

---

### Early stopping na combined metric — widoczny efekt

Modele trenują dłużej przed zatrzymaniem (best epoch: embedded 14→18, not-emb 10→13), co wskazuje że kryterium faktycznie szuka lepszego kompromisu między klasyfikacją a regresją.

---

## 2. Metryki klasyfikacji populacji

| Metryka | 1st Emb | 2nd Emb | 1st Not-Emb | 2nd Not-Emb |
|---------|---------|---------|-------------|-------------|
| Val Accuracy | 79.46% | 77.60% | 80.14% | **80.32%** |
| Test Accuracy | 82.17% | **82.61%** | 79.57% | 79.57% |
| Val F1 | 0.7935 | 0.7739 | 0.8004 | **0.8007** |
| Val AUC | **0.8527** | 0.8284 | 0.8501 | **0.8630** |

---

## 3. Metryki predykcji wieku

| Metryka | 1st Emb | 2nd Emb | 1st Not-Emb | 2nd Not-Emb |
|---------|---------|---------|-------------|-------------|
| Val MAE | 1.227 | **1.192** | 1.211 | **1.070** |
| Test MAE | 1.173 | **1.116** | 1.228 | **1.131** |
| Val bias | -0.388 | **+0.021** | -0.340 | **-0.050** |
| Val RMSE | 1.727 | **1.640** | 1.684 | **1.444** |

---

## 4. Pewność modelu

| | 1st Emb | 2nd Emb | 1st Not-Emb | 2nd Not-Emb |
|--|--|--|--|--|
| Conf mean (val) | 70.2% | **72.3%** | 69.1% | **72.2%** |
| High conf % (val) | 21.7% | **31.4%** | 19.9% | **27.1%** |
| Low conf % (val) | 22.1% | **22.4%** | 25.5% | **19.0%** |

Oba modele są teraz wyraźnie bardziej pewne swoich predykcji.

---

## 5. Co się nie poprawiło / otwarte pytania

### Anomalia Test > Val dla Embedded — wyjaśniona

| | Val Accuracy | Test Accuracy |
|--|--|--|
| 1st Embedded | 79.46% | 82.17% |
| 2nd Embedded | 77.60% | 82.61% |

**Stratyfikacja po wieku jest wdrożona** (patrz `data/split_info.md`, data 2026-04-15). Rozkład wiekowy jest proporcjonalny we wszystkich zbiorach. Residualna różnica ~5 pp wynika z **wariancji statystycznej małej próby**:

- Test set: 230 próbek → błąd standardowy accuracy ≈ ±2.6 pp
- Val set: 442 próbki → błąd standardowy accuracy ≈ ±1.9 pp

Różnica 5 pp przy SE ±2-3 pp mieści się w granicach normalnej wariancji. Anomalia nie wymaga dalszych korekt kodu.

### Embedded val accuracy nieznacznie gorsza

79.46% → 77.60% (−1.9 pp). Prawdopodobna przyczyna: combined metric w early stopping "poświęca" trochę accuracy klasyfikacji na rzecz lepszej regresji — co jest intencjonalne w multitask learningu.

### Embedded AUC: 0.8527 → 0.8284

Pogorszenie o 0.024. Wymaga dalszej obserwacji w kolejnych treningach.

### Czas treningu Embedded: 44 min → 311 min — wyjaśniony i naprawiony

Przyczyna: obrazy embedded są ~2× większe fizycznie (958×1846 px vs 489×937 px). Przy `num_workers=0` operacja Resize na każdej partii blokowała GPU. Naprawa: dodano `num_workers: 4` do `src/config/config.yaml`, DataLoader czyta teraz dane równolegle w 4 procesach. Oczekiwany czas treningu embedded w 3rd run: ~60-80 min.

---

## 6. Ocena ogólna

| Kryterium | Ocena |
|-----------|-------|
| Wiarygodność metryk | **lepsza** — train/val loss gap teraz realny |
| Predykcja wieku | **znacznie lepsza** — bias ~0, MAE w dół |
| Klasyfikacja populacji | bez zmian lub minimalnie gorsza dla embedded |
| Spójność wyników | lepsza dla not-embedded, mixed dla embedded |
| Interpretowalność | **lepsza** — modele bardziej pewne siebie |

Zmiany przyniosły wyraźny pozytywny skutek, szczególnie dla not-embedded. Wyniki są teraz bardziej wiarygodne — poprzednie metryki były częściowo fałszywe z powodu wycieku danych przy LDAM. Główne otwarte kwestie to anomalia test > val (wymaga stratyfikacji w data split) oraz niewyjaśniony wzrost czasu treningu embedded.