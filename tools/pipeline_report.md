# Raport przygotowania danych — Otolity Sledzia

Data: 2026-04-15
Skrypt: `tools/data_split_pipeline.py`


## Zrodla danych

| Zrodlo | Sciezka |
|--------|---------|
| Dysk sieciowy Processed | `Z:\Photo\Otolithes\HER\Processed` |
| Dysk sieciowy Raw | `Z:\Photo\Otolithes\HER\Raw` |
| Metadane (populacje) | `tools/analysisWithOtolithPhoto.xlsx` |

---

## Etap 1 — Skanowanie i budowanie par

| Metryka | Wartosc |
|---------|---------|
| Wszystkie pliki JPG w Processed | 18 727 |
| Rozpoznane (9 segmentow, view Right/Left) | 17 592 |
| Pominiete (nieznany view) | 1 135 |
| Embedded + Sharpest | 8 780 |
| NotEmbedded + WithoutPostproc | 8 812 |
| Pary potwierdzone w Raw | 2 650 |
| **Finalne pary wybrane** | **2791** |
| Odrzucone (brak wspolnego view) | 1 798 |
| Odrzucone (brak Emb lub NotEmb) | 646 |
| Unikalne ryby (fish_key) | 1505 |

Para = jedno zdjecie Embedded + jedno NotEmbedded tego samego otolitu, ten sam view (preferowany Right).

---

## Etap 2 — Przypisanie populacji i kopiowanie

| Metryka | Wartosc |
|---------|---------|
| Par z przypisana populacja (1 lub 2) | **2234** |
| Par bez populacji (pominiete) | 557 |
| Embedded pop. 1 | 1232 |
| Embedded pop. 2 | 1002 |
| Not-embedded pop. 1 | 1232 |
| Not-embedded pop. 2 | 1002 |

Pominiete 557 par to ryby z lokalizacji KolobrzeskoDarlowskie bez wypelnionej kolumny Populacja w Excelu.

---

## Etap 3 — Podzial train / val / test

Proporcje: **70% train / 20% val / 10% test**, seed=42
Podzial na poziomie ryby (fish_key) — brak leakage, ten sam podzial dla embedded i not_embedded.

### Liczba plikow per split i populacja

| Split | Emb pop.1 | Emb pop.2 | Emb lacznie | NotEmb pop.1 | NotEmb pop.2 | NotEmb lacznie | Ryb pop.1 | Ryb pop.2 | Ryb lacznie |
|-------|-----------|-----------|-------------|--------------|--------------|----------------|-----------|-----------|-------------|
| train | 863 | 699 | 1562 | 863 | 699 | 1562 | 451 | 376 | 827 |
| val   | 243 | 199 | 442 | 243 | 199 | 442 | 129 | 107 | 236 |
| test  | 126 | 104 | 230 | 126 | 104 | 230 | 67 | 56 | 123 |
| **SUMA** | **1232** | **1002** | **2234** | **1232** | **1002** | **2234** | **647** | **539** | **1186** |

Lacznie w final_pairs/: **4468 plikow** (2234 embedded + 2234 not_embedded w splitach)
plus kopie zrodlowe: 4468 plikow w katalogach 1/ i 2/

### Struktura katalogow

```
final_pairs/
  embedded/
    1/  (1232 plikow)    not_embedded/1/  (1232 plikow)
    2/  (1002 plikow)                 2/  (1002 plikow)
    train/1/ (863)  not_embedded/train/1/ (863)
    train/2/ (699)               train/2/ (699)
    val/1/   (243)                 val/1/   (243)
    val/2/   (199)                 val/2/   (199)
    test/1/  (126)                test/1/  (126)
    test/2/  (104)                test/2/  (104)
```

---

## Etap 4 — Weryfikacja

Wszystkie trzy sprawdzenia zakonczone pozytywnie:

| Sprawdzenie | Wynik |
|-------------|-------|
| Rowna liczba plikow embedded i not_embedded w kazdym split/pop | OK |
| Te same ryby w embedded i not_embedded w kazdym split/pop | OK |
| Brak leakage (ta sama ryba nie w wiecej niz 1 secie) | OK |

---

## Etap 5 — Excel (kolumna SET)

Plik: `tools/analysisWithOtolithPhoto.xlsx` (14498 rekordow lacznie)

| SET | Rekordow |
|-----|---------|
| TRAIN | 1562 |
| VAL | 442 |
| TEST | 230 |
| Bez SET (poza zbiorem treningowym) | 12264 |

Rekordy bez SET to zdjecia embedded bez pary NotEmbedded lub bez przypisanej populacji.

---

## Lokalizacje polowow w zbiorze

| Lokalizacja | Par |
|-------------|-----|
| 2023_BITS4q_HER_KolobrzeskoDarlowskie | 209 |
| 2024_BITS1q_HER_ZatokaGdanska | 188 |
| 2024_BITS1q_HER_Wladyslawowskie | 174 |
| 2022_BITS4q_HER_ZatokaGdanska | 163 |
| 2023_BITS4q_HER_RynnaSlupska | 161 |
| 2023_BITS4q_HER_UsteckoLebskie | 156 |
| 2023_BIAS_HER_RynnaSlupska | 132 |
| 2024_BITS1q_HER_UsteckoLebskie | 128 |
| 2024_BITS1q_HER_BornholmskieS | 125 |
| 2023_BITS4q_HER_GotlandzkieS | 109 |
| 2022_BIAS_HER_KolobrzeskoDarlowskie | 106 |
| 2024_BITS1q_HER_GlebiaGdanska | 100 |
| 2022_BITS4q_HER_GlebiaGdanska | 95 |
| 2023_BIAS_HER_Wladyslawowskie | 93 |
| 2023_BIAS_HER_ZatokaGdanska | 85 |
| 2022_BIAS_HER_ZatokaGdanska | 83 |
| 2023_BIAS_HER_KolobrzeskoDarlowskie | 83 |
| 2023_BIAS_HER_GlebiaGdanska | 79 |
| 2023_BITS4q_HER_GlebiaGdanska | 79 |
| 2024_BITS4q_HER_BornholmskieS | 77 |
| 2022_BITS4q_HER_GotlandzkieS | 75 |
| 2023_BITS4q_HER_LawicaSrodkowa | 66 |
| 2022_BIAS_HER_GlebiaGdanska | 60 |
| 2022_BIAS_HER_Wladyslawowskie | 56 |
| 2022_BIAS_HER_UsteckoLebskie | 47 |
| 2022_BITS4q_HER_LawicaSrodkowa | 46 |
| 2022_BITS4q_HER_RynnaSlupska | 16 |


---

## Konfiguracja treningu

```yaml
# Model Embedded — config.yaml:
data:
  data_dir: C:\Users\kswitek\Documents\HerringProject\data\embedded

# Model Not-Embedded — config.yaml:
data:
  data_dir: C:\Users\kswitek\Documents\HerringProject\data\not_embedded
```

Oba katalogi maja identyczna strukture train/val/test/{1,2}/ z tymi samymi rybami w tych samych setach.

---

## Ponowne uruchomienie pipeline'u

```bash
# Pelny pipeline (wymaga sieci Z:\):
.venv\Scripts\python tools\data_split_pipeline.py

# Tylko split + weryfikacja (bez sieci, pliki juz skopiowane):
.venv\Scripts\python tools\data_split_pipeline.py --steps 3 4 5

# Podglad bez kopiowania:
.venv\Scripts\python tools\data_split_pipeline.py --steps 3 --dry-run
```
