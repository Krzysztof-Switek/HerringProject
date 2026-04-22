# Uzasadnienie zmiany funkcji straty regresji wieku — MSELoss → WeightedMSELoss

## Problem: niezbalansowany rozkład wiekowy

Dane treningowe wykazują silną nierównowagę klas wiekowych charakterystyczną
dla połowów śledzi bałtyckich — dominują osobniki w wieku 4–6 lat:

| Wiek | Liczba próbek | Udział |
|------|--------------|--------|
| 4    | 525          | 23.5%  |
| 5    | 399          | 17.9%  |
| 3    | 397          | 17.8%  |
| 6    | 190          | 8.5%   |
| 7    | 150          | 6.7%   |
| 8    | 96           | 4.3%   |
| 9    | 34           | 1.5%   |
| 10   | 18           | 0.8%   |
| 11   | 14           | 0.6%   |
| 12   | 14           | 0.6%   |
| 14   | 5            | 0.2%   |
| 15   | 5            | 0.2%   |

## Skutek dla standardowego MSELoss

Standardowa funkcja straty MSE minimalizuje globalny błąd kwadratowy:

    L = (1/N) · Σ (ŷᵢ − yᵢ)²

Przy niezbalansowanym rozkładzie gradient z klas dominujących (wiek 4–6)
przytłacza gradient z klas rzadkich (wiek 7+) proporcjonalnie do ich liczebności.
Model konwerguje do przewidywania wartości bliskich medianie rozkładu
treningowego, ignorując rzadkie przypadki starszych ryb.

Potwierdza to analiza błędów po treningu — bias predykcji rośnie monotonicznie
z wiekiem i osiąga wartości niedopuszczalne biologicznie:

| Wiek prawdziwy | N próbek | MAE (Embedded) | Bias   | Najczęstsza predykcja |
|----------------|----------|----------------|--------|-----------------------|
| 4              | 525      | 0.61           | +0.12  | 4                     |
| 5              | 399      | 0.85           | −0.43  | 5                     |
| 7              | 150      | 1.63           | −1.56  | 6                     |
| 8              | 96       | 2.37           | −2.36  | 6                     |
| 10             | 18       | 4.16           | −4.16  | 6                     |
| 12             | 14       | 5.15           | −5.15  | 6                     |
| 14             | 5        | 6.37           | −6.37  | 7                     |
| 15             | 5        | 8.07           | −8.07  | 5                     |

Model systematycznie "ucieka" w kierunku wieku 4–6, który dominuje w danych
treningowych. Dla ryb w wieku 15 lat model przewiduje wiek 5 lat — błąd
o wartości biologicznie nieakceptowalnej.

## Rozwiązanie: Weighted MSELoss

Zastosowano ważony MSE z wagami odwrotnie proporcjonalnymi do pierwiastka
kwadratowego z liczebności klasy wiekowej:

    w(age) = 1 / sqrt(N_age)

Wagi są następnie normalizowane tak, by ich średnia wynosiła 1.0,
co zachowuje porównywalną skalę straty względem klasyfikacji:

    w_norm(age) = w(age) / mean(w)

Ważona funkcja straty:

    L_weighted = (1/N) · Σ w_norm(yᵢ) · (ŷᵢ − yᵢ)²

### Uzasadnienie formy wag 1/sqrt(N)

Forma `1/sqrt(N)` jest kompromisem na skali agresywności ważenia:

| Forma wagi      | Skutek                                              |
|-----------------|-----------------------------------------------------|
| `1/N`           | Zbyt agresywne — całkowita dominacja rzadkich klas, niestabilny trening |
| `1/sqrt(N)`     | Umiarkowane — zachowuje wpływ klas dominujących, wzmacnia rzadkie |
| `1/sqrt(sqrt(N))` | Łagodne — minimalna korekta                       |
| brak ważenia    | Klasy rzadkie praktycznie ignorowane                |

Forma `1/sqrt(N)` jest powszechnie stosowana w literaturze dla danych
o umiarkowanej do silnej nierównowadze (stosunek klas do ~100:1).

### Przykładowe wagi dla tego datasetu

| Wiek | N     | w(age) = 1/sqrt(N) | w_norm (przybliżone) |
|------|-------|--------------------|----------------------|
| 4    | 525   | 0.044              | 0.21                 |
| 5    | 399   | 0.050              | 0.24                 |
| 7    | 150   | 0.082              | 0.39                 |
| 10   | 18    | 0.236              | 1.12                 |
| 12   | 14    | 0.267              | 1.27                 |
| 14   | 5     | 0.447              | 2.12                 |
| 15   | 5     | 0.447              | 2.12                 |

Ryby w wieku 14–15 lat są ważone ~10× silniej niż ryby w wieku 4 lat,
zamiast identycznie jak w standardowym MSE.

## Ważny aspekt implementacyjny: obliczanie wag wyłącznie z danych treningowych

Wagi `w(age)` są obliczane wyłącznie na podstawie zbioru treningowego.
Użycie statystyk z całego datasetu (train + val + test) stanowiłoby
wyciek informacji (data leakage) ze zbiorów walidacyjnego i testowego,
co mogłoby zawyżać wyniki walidacji i dawać fałszywe poczucie jakości modelu.

## Implementacja

```python
class WeightedMSELoss(nn.Module):
    """MSELoss z wagami odwrotnie proporcjonalnymi do sqrt(liczebności klasy wiekowej)."""

    def __init__(self, age_counts: dict):
        """
        age_counts: {wiek_int -> liczba_próbek} — obliczone wyłącznie z danych train.
        """
        super().__init__()
        max_age = max(age_counts.keys()) + 1
        weights = torch.ones(max_age)
        for age, count in age_counts.items():
            weights[int(age)] = 1.0 / (count ** 0.5)
        weights = weights / weights.mean()
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_int = target.long().clamp(0, len(self.weights) - 1)
        w = self.weights[target_int]
        return (w * (pred - target) ** 2).mean()
```

## Literatura

1. Yang, Y., Zha, K., Chen, Y., Wang, H., & Katabi, D. (2021).
   *Delving into Deep Imbalanced Regression.*
   Proceedings of the 38th International Conference on Machine Learning (ICML 2021).
   https://arxiv.org/abs/2102.09554

2. Ren, J., Zhang, M., Yu, C., & Liu, Z. (2022).
   *Balanced MSE for Imbalanced Visual Regression.*
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).
   https://arxiv.org/abs/2203.16427

3. Steininger, M., Kobs, K., Davidson, P., Krause, A., & Hotho, A. (2021).
   *Density-based weighting for imbalanced regression with continuous target variable.*
   Machine Learning, 110(8), 2187–2211.
   https://doi.org/10.1007/s10994-021-06023-5

4. Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).
   *SMOTE for Regression.*
   Progress in Artificial Intelligence, EPIA 2013, LNAI 8154, 378–389.
   https://doi.org/10.1007/978-3-642-40669-0_33
