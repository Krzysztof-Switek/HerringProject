# Adaptacyjne ważenie funkcji strat w uczeniu wielozadaniowym: uzasadnienie metodologiczne

**Projekt:** Klasyfikacja populacji i predykcja wieku śledzi na podstawie zdjęć otolitów  
**Data:** 2026-04-16  

---

## 1. Wprowadzenie

Uczenie wielozadaniowe (*multi-task learning*, MTL) polega na jednoczesnym trenowaniu modelu na kilku powiązanych zadaniach przy użyciu wspólnej reprezentacji [Caruana, 1997]. W niniejszym projekcie model realizuje dwa zadania:

1. **Klasyfikację populacji** — przypisanie próbki do jednej z populacji biologicznych (zadanie klasyfikacyjne).
2. **Predykcję wieku** — estymacja wieku ryby jako wartości ciągłej (zadanie regresyjne).

Łączna funkcja straty ma postać:

$$\mathcal{L}_{total} = w_1 \cdot \mathcal{L}_{cls} + w_2 \cdot \mathcal{L}_{reg}$$

gdzie $w_1, w_2$ to wagi kontrolujące względny wpływ każdego zadania. Dobór tych wag jest nietrywialny: zbyt wysoka waga zadania regresyjnego może zdominować sygnał klasyfikacyjny i odwrotnie. Skale gradientów obu strat mogą różnić się o kilka rzędów wielkości, co prowadzi do niestabilności treningu.

---

## 2. Klasyczne podejście: zewnętrzne przeszukiwanie przestrzeni wag

Tradycyjne podejście polega na potraktowaniu $w_1, w_2$ jako hiperparametrów i przeszukaniu przestrzeni ich wartości za pomocą zewnętrznego optymizatora, np. biblioteki Optuna (Bayesian optimization, TPE sampler) [Akiba et al., 2019].

**Schemat działania:**

```
for trial t = 1..N:
    (w1, w2) ← suggest(trial)        # Optuna proponuje wartości
    train model from scratch          # pełny trening: E epok
    score ← evaluate(val_set)
    optuna.report(score)
```

**Wady przy ograniczonych zasobach:**

- Każda próba wymaga pełnego treningu od początku — całkowity koszt to $N \times E$ epok.
- Przy $N = 50$ próbach i $E = 150$ epokach daje to 7 500 epok treningu.
- Przeszukiwana jest przestrzeń **stałych** wag globalnych, co zakłada że optymalne proporcje zadań nie zmieniają się w trakcie treningu — założenie biologicznie nieuzasadnione.
- Wymaga wielokrotnego dostępu do GPU przez wiele dni lub tygodni.

---

## 3. Uncertainty Weighting — adaptacyjne wagi przez homoskedastyczną niepewność zadań

### 3.1 Podstawa teoretyczna

Kendall, Gal i Cipolla [2018] zaproponowali metodę wywodzącą się z bayesowskiej interpretacji funkcji straty. Rozważmy model probabilistyczny, gdzie dla każdego zadania definiujemy likelihood:

$$p(\mathbf{y} \mid \mathbf{f}^\mathbf{W}(\mathbf{x})) = \mathcal{N}\!\left(\mathbf{f}^\mathbf{W}(\mathbf{x}),\, \sigma^2\right)$$

gdzie $\sigma$ jest parametrem szumu obserwacyjnego (*observation noise*), a nie parametrem modelu. Maksymalizacja log-likelihood przy dwóch zadaniach prowadzi do:

$$\mathcal{L}_{total} = \frac{1}{2\sigma_1^2}\,\mathcal{L}_{cls} + \frac{1}{2\sigma_2^2}\,\mathcal{L}_{reg} + \log\sigma_1 + \log\sigma_2$$

Parametry $\sigma_1, \sigma_2$ (lub ich logarytmy $s_i = \log\sigma_i^2$) są **uczonymi parametrami sieci**, aktualizowanymi tym samym algorytmem backpropagation co wagi modelu. W implementacji używa się $s_i = \log\sigma_i^2$ dla stabilności numerycznej:

$$\mathcal{L}_{total} = e^{-s_1}\,\mathcal{L}_{cls} + e^{-s_2}\,\mathcal{L}_{reg} + s_1 + s_2$$

### 3.2 Interpretacja

- $e^{-s_i}$ to **precyzja** (odwrotność wariancji) zadania $i$ — gdy zadanie jest trudne/niepewne, $s_i$ rośnie, a jego wpływ na łączną stratę maleje automatycznie.
- Człon regularyzacyjny $s_i$ zapobiega degeneracji (model nie może po prostu ustawić $s_i \to \infty$ i zignorować zadanie).
- Wagi **ewoluują dynamicznie** w trakcie treningu, dostosowując się do aktualnego stanu modelu.

### 3.3 Implementacja w projekcie

Plik: `src/engine/loss_utills.py`, klasa `MultiTaskLossWrapper`

```python
# Uczalne parametry log-wariancji
self.log_vars = nn.Parameter(torch.zeros(2))

# Forward pass
precision_cls = torch.exp(-self.log_vars[0])
precision_reg = torch.exp(-self.log_vars[1])
loss = (precision_cls * cls_loss + self.log_vars[0] +
        precision_reg * reg_loss + self.log_vars[1])
```

Parametry `log_vars` są przekazywane do optymizatora AdamW razem z wagami modelu (`src/engine/trainer_setup.py:205-209`).

---

## 4. GradNorm — normalizacja gradientów między zadaniami

### 4.1 Podstawa teoretyczna

Chen et al. [2018] zaproponowali metodę opartą na bezpośredniej obserwacji norm gradientów. Uczalne wagi $w_i(t)$ są aktualizowane tak, aby normy gradientów wszystkich zadań były zbliżone i rosły w podobnym tempie. Dla każdego zadania definiuje się:

$$\tilde{\mathcal{L}}_i(t) = \frac{\mathcal{L}_i(t)}{\mathcal{L}_i(0)}$$

jako znormalizowaną stratę względem wartości początkowej. Następnie oblicza się:

$$\bar{g}(t) = \mathbb{E}_i\!\left[\|\nabla_W w_i \mathcal{L}_i\|_2\right]$$

jako średnią normę gradientu po zadaniach i minimalizuje się:

$$\mathcal{L}_{grad} = \sum_i \left| \|\nabla_W w_i \mathcal{L}_i\|_2 - \bar{g}(t) \cdot [\tilde{\mathcal{L}}_i(t)]^\alpha \right|_1$$

gdzie hiperparametr $\alpha$ kontroluje jak silnie zadania uczące się wolniej powinny zostać "przyspieszone" (w projekcie: `alpha=1.5`).

### 4.2 Porównanie z Uncertainty Weighting

| Kryterium | Uncertainty Weighting | GradNorm |
|-----------|----------------------|----------|
| Podstawa | Bayesowska likelihood | Balans gradientów |
| Dodatkowe hiperparametry | brak | $\alpha$ |
| Interpretacja $w_i$ | niepewność zadania | tempo uczenia zadania |
| Złożoność obliczeniowa | minimalna (+2 parametry) | umiarkowana (wymaga norm gradientów) |
| Stabilność | wysoka | zależy od $\alpha$ |

---

## 5. Dlaczego te metody są lepsze przy ograniczonych zasobach

### 5.1 Redukcja kosztu obliczeniowego

Kluczowa różnica polega na tym, że optymalizacja wag odbywa się **wewnątrz** pętli treningowej, nie poza nią:

| Podejście | Całkowity koszt |
|-----------|----------------|
| Optuna (N=50 prób, E=150 epok) | ~7 500 epok GPU |
| Uncertainty/GradNorm | ~150 epok GPU (+2 parametry) |

Przyspieszenie jest rzędu $N\times$ — przy 50 próbach Optuny jest to **50-krotna** redukcja czasu obliczeń.

### 5.2 Lepsza jakość optymalizacji

Podejście Optuny szuka **stałej** pary $(w_1, w_2)$ — jednego punktu w przestrzeni 2D. Tymczasem:

- Na początku treningu model klasyfikuje losowo — sygnał z regresji może być dominującym sygnałem uczącym.
- W późniejszych epokach, gdy klasyfikacja się stabilizuje, regresja może potrzebować większego udziału.
- Optymalne proporcje w epoce 10 nie są tymi samymi, co w epoce 100.

Metody adaptacyjne śledzą tę dynamikę automatycznie, co odpowiada bardziej realistycznemu modelowi procesu uczenia.

### 5.3 Mniejsze ryzyko overfittingu do zbioru walidacyjnego

W podejściu z Optuna wyniki każdej próby są oceniane na zbiorze walidacyjnym, a Optuna dostosowuje następną próbę na podstawie tych wyników. Przy małej liczbie prób istnieje ryzyko przypadkowej optymalizacji pod konkretny podział danych. W podejściu adaptacyjnym wagi uczą się z sygnału treningowego, bez dodatkowego "spojrzenia" na walidację.

### 5.4 Stosowalność przy małych zbiorach danych

W badaniach biologicznych zbiory danych są często małe i niezrównoważone. Mała liczba próbek oznacza, że:
- Wariancja estymaty na zbiorze walidacyjnym jest wysoka.
- Optuna może źle ocenić jakość próby z powodu tej wariancji.
- Metody adaptacyjne uczą się wag z całego sygnału treningowego, co jest bardziej odporne na szum małych zbiorów.

---

## 6. Konfiguracja w projekcie

Wybór metody odbywa się w `src/config/config.yaml`:

```yaml
loss_weighting:
  method: "uncertainty"   # "none" | "static" | "uncertainty" | "gradnorm"
  static:
    classification: 1.0
    age: 0.3
```

Implementacja: `src/engine/loss_utills.py`, klasa `MultiTaskLossWrapper` (linie 290–363).  
Integracja z optymizatorem: `src/engine/trainer_setup.py` (linie 205–209).

---

## 7. Literatura

[Caruana, 1997] Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28(1), 41–75. https://doi.org/10.1023/A:1007379606734

[Kendall et al., 2018] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 7482–7491. https://doi.org/10.1109/CVPR.2018.00781

[Chen et al., 2018] Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, 794–803.

[Akiba et al., 2019] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

[Vandenhende et al., 2021] Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. (2021). Multi-task learning for dense prediction tasks: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 44(7), 3614–3633. https://doi.org/10.1109/TPAMI.2021.3054719
