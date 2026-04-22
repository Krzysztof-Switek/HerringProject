# Opis podziału danych — split_info

Data wykonania: 2026-04-15
Skrypt: `tools/data_split_pipeline.py`
Proporcje: train=0.7, val=0.2, test=0.1
Seed: 42

## Metoda podziału

Podział jest wykonywany na poziomie ryby (`fish_key`), nie na poziomie obrazu.
Gwarantuje to:
- brak tej samej ryby w więcej niż jednym zbiorze (train/val/test)
- identyczny podział dla zdjęć Embedded i Not-Embedded tej samej ryby

### Stratyfikacja po wieku (zmiana względem poprzedniej wersji)

Poprzednia wersja wykonywała `random.shuffle` wyłącznie w obrębie populacji,
bez uwzględnienia wieku. Powodowało to losowy rozkład wiekowy w zbiorach,
co skutkowało anomalią Test Accuracy > Val Accuracy (test set przypadkowo
zawierał więcej łatwych, młodych ryb).

Obecna wersja stratyfikuje ryby według kombinacji:
**(populacja × bin wiekowy)** przed losowym tasowaniem.

Biny wiekowe:
| Bin | Zakres | Uzasadnienie |
|-----|--------|--------------|
| young | 1–3 lat | Rzadkie, pozytywny bias modelu |
| middle | 4–6 lat | Dominujące w datasecie (~60% próbek) |
| old | 7+ lat | Dramatycznie niedoreprezentowane, najgorzej przewidywane |

Każdy bin jest tasowany i dzielony niezależnie, co zapewnia proporcjonalną
reprezentację ryb w każdym wieku we wszystkich trzech zbiorach.

## Rozkład ryb (fish_key) po podziale

| Populacja | Bin wiekowy | Train | Val | Test | Razem |
|-----------|-------------|-------|-----|------|-------|
| 1 | young (1–3) | 88 | 25 | 13 | 126 |
| 1 | middle (4–6) | 245 | 70 | 36 | 351 |
| 1 | old (7–+) | 118 | 34 | 18 | 170 |
| 2 | young (1–3) | 205 | 58 | 31 | 294 |
| 2 | middle (4–6) | 164 | 47 | 24 | 235 |
| 2 | old (7–+) | 7 | 2 | 1 | 10 |

## Liczba obrazów w zbiorach

| Split | Embedded | Not-Embedded |
|-------|----------|-------------|
| train | 1562 | 1562 |
| val | 442 | 442 |
| test | 230 | 230 |

Każdy obraz Embedded ma dokładnie jeden odpowiednik Not-Embedded
(ten sam otolith, różne metody preparacji).

## Weryfikacja

Po wykonaniu podziału uruchom etap 4 (`--steps 4`) aby sprawdzić:
- równą liczbę plików Embedded i Not-Embedded w każdym split/pop
- te same ryby w Embedded i Not-Embedded
- brak leakage (ta sama ryba nie w więcej niż 1 zbiorze)
