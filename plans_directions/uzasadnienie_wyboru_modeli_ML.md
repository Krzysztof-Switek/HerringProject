# Uzasadnienie wyboru architektur modeli uczenia maszynowego
## dla systemu klasyfikacji populacji i szacowania wieku śledzi na podstawie obrazów otolitów

**Data:** 22 kwietnia 2026  
**Dotyczy:** Rozszerzenia pipeline'u uczenia głębokiego o dwa nowe warianty eksperymentalne  
**Adresat:** Kierownik naukowy instytutu badawczego

---

## 1. Kontekst i cel

Niniejszy dokument uzasadnia wybór dwóch nowych architektur modeli uczenia głębokiego proponowanych do wdrożenia w systemie klasyfikacji populacji i szacowania wieku śledzi (*Clupea harengus*) na podstawie obrazów otolitów. Modele te stanowią rozszerzenie wobec dotychczas stosowanych architektur i zostały dobrane z uwzględnieniem specyfiki analizowanych danych — obrazów biologicznych o charakterze mikroskopowym, wymagających detekcji subtelnych struktur (pierścieni rocznych, wzorów krystalicznych, tekstur).

---

## 2. Dotychczas stosowane modele — stan wyjściowy

W aktualnym pipeline stosowane są następujące architektury (wszystkie pretrenowane na zbiorze ImageNet):

| Model | Parametry | Rok | Typ | Maks. rozdzielczość |
|-------|-----------|-----|-----|---------------------|
| ResNet-50 | 25M | 2015 | CNN | 224×224 px |
| SwinV2-B | 87M | 2022 | Transformer | 256×256 px |
| ConvNeXt-Large | 197M | 2022 | CNN (Transformer-inspired) | 384×384 px |
| EfficientNet V2-L | 120M | 2022 | CNN (NAS) | 480×480 px |
| RegNet-Y-32GF | 145M | 2020 | CNN | 384×384 px |

**Wspólna cecha wszystkich dotychczasowych modeli**: pretrening wyłącznie nadzorowany (supervised) na zbiorze ImageNet (fotografie obiektów codziennych). Żaden z modeli nie był trenowany na danych medycznych, mikroskopowych ani biologicznych, co ogranicza jakość reprezentacji dla obrazów otolitów.

---

## 3. Proponowane nowe warianty

### 3.1. Wariant `hybrid_768` — MaxViT Base, 768×768 px

**Pełna nazwa modelu:** `maxvit_base_tf_512.in21k_ft_in1k` (biblioteka timm)  
**Architektura:** MaxViT — Multi-Axis Vision Transformer (Google Research, 2022)  
**Parametry:** ~119M  
**Rozdzielczość wejściowa:** 768×768 px

#### Opis architektury

MaxViT (Tu i in., 2022) łączy dwa mechanizmy w jednej, skalowalnej jednostce obliczeniowej:

1. **Blok MBConv (CNN)** — wydobywa lokalne cechy przestrzenne przy użyciu konwolucji depthwise separable, analogicznie do EfficientNet. Odpowiada za rozpoznawanie lokalnych wzorów i krawędzi.
2. **Block Attention (lokalny Transformer)** — analizuje relacje przestrzenne wewnątrz nieoverlappingowych okien lokalnych (podobnie jak Swin Transformer).
3. **Grid Attention (globalny Transformer)** — analizuje relacje rozrzedzone globalnie, stosując siatkę co *k*-tego piksela, uchwytując korelacje długodystansowe przy koszcie liniowym.

Kluczowa przewaga: architektura osiąga **globalny kontekst przy złożoności liniowej** (nie kwadratowej jak standardowy ViT), co pozwala efektywnie przetwarzać obrazy o dużej rozdzielczości (768 px i powyżej).

#### Uzasadnienie wyboru dla obrazów otolitów

Otolity wykazują struktury o naturze wieloskalowej: grube pierścienie roczne widoczne w skali globalnej oraz subtelne wzory krystaliczne i tekstury na poziomie lokalnym. MaxViT jest architektonicznie zaprojektowany do jednoczesnego wychwytywania cech lokalnych (przez CNN i lokalną uwagę okienkową) oraz globalnych korelacji (przez uwagę siatkową). Jest to bezpośredni odpowiednik wymagań percepcyjnych przy czytaniu otolitów przez eksperta.

#### Wyniki benchmarkowe (ImageNet)

Zgodnie z publikacją oryginalną (Tu i in., ECCV 2022):

- MaxViT-B przy 224 px: **84.9% top-1 accuracy** na ImageNet-1k
- MaxViT-B przy 384 px: **86.34% top-1** (przewyższa EfficientNetV2-L o 0.64%)
- MaxViT-L przy 512 px: **86.7% top-1** (SOTA pod normalnym trybem treningu)
- Z pretreningiem ImageNet-21k: **88.38% top-1** (przy 43% mniej parametrów niż CoAtNet-4)
- W detekcji i segmentacji: przewyższa Swin-B przy **40% niższym koszcie obliczeniowym**

#### Zastosowania medyczne

MaxViT-UNet (Rehman i in., 2023) zastosował architekturę MaxViT w segmentacji obrazów medycznych (skóra, polipy jelita), uzyskując poprawę w rozróżnianiu struktur tła i obiektu dzięki wieloosiowej uwadze w każdym bloku dekodera. Multi-Scale Fusion MaxViT (MDPI Electronics, 2025) osiągnął poprawione wyniki w klasyfikacji obrazów medycznych poprzez połączenie lokalnej i globalnej uwagi z mechanizmem fuzji cech.

---

### 3.2. Wariant `advanced_dinov2` — DINOv2 ViT-L/14, 518×518 px

**Pełna nazwa modelu:** `vit_large_patch14_dinov2.lvd142m` (biblioteka timm)  
**Architektura:** Vision Transformer ViT-L/14 z pretreningiem DINOv2 (Meta AI, 2023)  
**Parametry:** ~307M  
**Rozdzielczość wejściowa:** 518×518 px (natywna — 14 px patch × 37 = 518)

#### Opis architektury i metody pretreningu

DINOv2 (Oquab i in., 2023) to model fundacyjny wizji komputerowej trenowany metodą **samouczenia się bez nadzoru (self-supervised learning)** na zbiorze 142 milionów kuratowanych obrazów (LVD-142M). Metoda łączy:

- **Destylację nauczyciel–uczeń** przez mechanizm cross-view
- **Masked Image Modeling** (dla przestrzennych cech lokalnych)
- **Self-Distillation with No Labels (DINO)** (dla globalnych reprezentacji)

Kluczowa różnica od wszystkich dotychczas stosowanych modeli: **DINOv2 nie uczy się od etykiet klas, lecz od struktury samych obrazów**. W efekcie uczy się cech strukturalnych, teksturalnych i przestrzennych, a nie semantyki kategorii obiektów z fotografii. To właśnie sprawia, że model ten wykazuje znacznie lepszy transfer na domeny odległe od ImageNet — w tym obrazy mikroskopowe i biologiczne.

#### Uzasadnienie wyboru dla obrazów otolitów

Otolity to obrazy biologiczne o charakterze mikroskopowym. Wzory, które decydują o klasyfikacji populacji i szacowaniu wieku, to przede wszystkim:
- koncentryczne pierścienie o różnej gęstości i grubości,
- tekstury krystalograficzne,
- subtelne różnice morfologiczne między populacjami.

Te cechy są analogiczne do cech obecnych w preparatach histologicznych i mikroskopii komórkowej — dziedzinie, w której DINOv2 i jego pochodne wykazały przewagę nad modelami trenowanymi na ImageNet.

#### Wyniki benchmarkowe i zastosowania biologiczne/medyczne

1. **Cell-DINO** (Caron i in., 2024 — *PLOS Computational Biology*): adaptacja DINOv2 do fenotypowania komórek w mikroskopii fluorescencyjnej. Wyniki:
   - Przewyższa nadzorowanego ViT o **20% średnio** na zbiorze Human Protein Atlas
   - Przy użyciu zaledwie 1% etykiet: przewyższa nadzorowane metody o **70%**
   - Przewyższa MAE o 34% i SimCLR o 20% na klasyfikacji lokalizacji białek

2. **Virchow** (Vorontsov i in., 2023, *arXiv 2309.07778*): model fundacyjny histopatologii zbudowany na DINOv2, trenowany na **1,5 miliona skrawków histopatologicznych**. Osiągnął AUC 0.949 na 17 typach nowotworów i AUC 0.937 na 7 rzadkich typach.

3. **UNI** (Chen i in., 2024, *Nature Medicine*): oparty na DINOv2 ViT-L, trenowany na **100 000+ preparatach WSI** (Whole Slide Images). Obecna wersja UNI 2 używa ponad 200 milionów obrazów patologicznych.

4. **DINOv2 w radiologii** (Keicher i in., 2023, *arXiv 2312.02366*): DINOv2-L/14 dorównuje lub przewyższa CNN i nadzorowane ViT-L/16 na benchmarkach klatki piersiowej (NIH Chest X-ray, CheXpert).

5. **Explainable DINOv2 w diagnostyce** (*Scientific Reports*, 2025): DINOv2 jako backbone w interpretowalnej diagnostyce medycznej z przeszukiwaniem semantycznym.

#### Uzasadnienie rozdzielczości 518 px

Wielkość patcha w DINOv2 wynosi 14 px. Rozdzielczość 518 px = 14 × 37 jest natywną rozdzielczością, przy której model był trenowany, co zapewnia optymalną jakość reprezentacji. Użycie wyższych rozdzielczości (np. 1024 px) jest możliwe, ale przy liczbie tokenów ~5300 prowadzi do wykładniczego wzrostu zużycia VRAM (batch_size=1–2), bez proporcjonalnej poprawy jakości — embeddingi pozycyjne są interpolowane, a wzrost rozdzielczości powyżej natywnej nie gwarantuje lepszych wyników przy fine-tuningu.

---

## 4. Porównanie z dotychczasowymi modelami

| Cecha | ResNet-50 | SwinV2-B | ConvNeXt-L | MaxViT-B | DINOv2 ViT-L |
|-------|-----------|----------|------------|----------|--------------|
| Rok publikacji | 2015 | 2022 | 2022 | 2022 | 2023 |
| Typ | CNN | Transformer | CNN* | Hybrid CNN+T | Transformer |
| Parametry | 25M | 87M | 197M | 119M | 307M |
| Rozdzielczość | 224 | 256 | 384 | 768 | 518 |
| Pretrening | ImageNet-1k | ImageNet-1k/21k | ImageNet-1k/21k | ImageNet-21k | Self-supervised (142M) |
| Lokalne cechy | Tak (silne) | Ograniczone | Tak (silne) | Tak (silne) | Średnie |
| Globalne korelacje | Nie | Tak (okienkowy) | Nie | Tak (dilated grid) | Tak (pełna uwaga) |
| Transfer na biologię | Słaby | Średni | Średni | Dobry | Bardzo dobry |
| Domena pretreningu | Zdjęcia | Zdjęcia | Zdjęcia | Zdjęcia | Strukturalne/teksturalne |

*ConvNeXt inspirowany architektonicznie Transformerami, ale pozostaje CNN.

### Kluczowe różnice jakościowe

**ResNet-50 vs. nowe modele**: ResNet operuje wyłącznie lokalnie — recepcyjne pole jest ograniczone przez głębokość sieci. Nie jest w stanie bezpośrednio modelować korelacji między odległymi regionami otolitu. Jest najstarszym i najsłabszym modelem w pipeline.

**SwinV2-B vs. MaxViT**: Swin używa wyłącznie lokalnych okien z przesunięciem (shifted window) — globalny kontekst buduje hierarchicznie. MaxViT uzupełnia lokalną uwagę okienkową o rozmytą uwagę siatkową na każdym poziomie, co daje bezpośredni globalny kontekst bez dodatkowych warstw. MaxViT-B przewyższa Swin-B przy 40% niższych kosztach obliczeniowych (Tu i in., 2022).

**ConvNeXt-Large vs. MaxViT**: ConvNeXt to modernizacja ResNet bez mechanizmu uwagi — brak explicite modelowanych relacji długodystansowych. MaxViT łączy zalety konwolucji z mechanizmem uwagi, co jest architektonicznie nowocześniejsze i odpowiedniejsze dla analizy wzorów wymagających rozumienia globalnego kontekstu.

**Wszystkie dotychczasowe modele vs. DINOv2**: Fundamentalna różnica leży w metodzie pretreningu. Modele nadzorowane uczą się rozróżniać 1000 kategorii obiektów codziennych (ImageNet) — otolity nie należą do tej domeny. DINOv2 uczy się ogólnych reprezentacji wizualnych ze struktury obrazów, co przekłada się na lepszy transfer do domen biologicznych i mikroskopowych.

---

## 5. Podstawy literaturowe

1. **Tu, Z. i in.** (2022). *MaxViT: Multi-Axis Vision Transformer*. ECCV 2022. DOI: 10.1007/978-3-031-20053-3_27. [[arxiv]](https://arxiv.org/abs/2204.01697)

2. **Oquab, M. i in.** (2023). *DINOv2: Learning Robust Visual Features without Supervision*. Transactions on Machine Learning Research. [[arxiv]](https://arxiv.org/abs/2304.07193)

3. **Rehman, M.Z.U. i in.** (2023). *MaxViT-UNet: Multi-Axis Attention for Medical Image Segmentation*. [[arxiv]](https://arxiv.org/abs/2305.08396)

4. **Siddiqui, M.A. i in.** (2025). *Multi-Scale Fusion MaxViT for Medical Image Classification*. MDPI Electronics, 14(5), 912. [[link]](https://www.mdpi.com/2079-9292/14/5/912)

5. **Caron, M. i in.** (2024). *Cell-DINO: Self-supervised image-based embeddings for cell fluorescent microscopy*. PLOS Computational Biology. [[link]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828)

6. **Vorontsov, E. i in.** (2023). *Virchow: A Million-Slide Digital Pathology Foundation Model*. [[arxiv]](https://arxiv.org/abs/2309.07778)

7. **Chen, R.J. i in.** (2024). *A General-Purpose Self-Supervised Model for Computational Pathology (UNI)*. Nature Medicine. [[ResearchGate]](https://www.researchgate.net/publication/373487557_A_General-Purpose_Self-Supervised_Model_for_Computational_Pathology)

8. **Keicher, M. i in.** (2023). *Evaluating General Purpose Vision Foundation Models for Medical Image Analysis: DINOv2 on Radiology Benchmarks*. [[arxiv]](https://arxiv.org/abs/2312.02366)

9. **Salberg, A.B. i in.** (2021). *Automating fish age estimation combining otolith images and deep learning: The role of multitask learning*. Fisheries Research, 242, 105968. [[link]](https://www.sciencedirect.com/science/article/abs/pii/S0165783621001612)

10. **Vignon, M. i in.** (2022). *Age prediction by deep learning applied to Greenland halibut otolith images*. PLOS ONE. [[link]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277244)

11. **Liu, Z. i in.** (2022). *A ConvNet for the 2020s (ConvNeXt)*. CVPR 2022. [[arxiv]](https://arxiv.org/pdf/2201.03545)

12. **Muñoz, D. i in.** (2023). *Comparing the Robustness of ResNet, Swin-Transformer under Distribution Shifts in Fundus Images*. MDPI Bioengineering, 10(12). [[link]](https://www.mdpi.com/2306-5354/10/12/1383)

---

## 6. Wnioski i rekomendacje

Obydwa proponowane modele stanowią uzasadnione architektonicznie i literaturowo rozszerzenie wobec dotychczas stosowanych:

- **MaxViT-Base (hybrid_768)** wypełnia lukę w architekturze hybrydowej CNN+Transformer, operując przy wyższej rozdzielczości (768 px) i modelując jednocześnie cechy lokalne i globalne. Przewyższa SwinV2-B przy niższym koszcie obliczeniowym. Jest gotowy do zastosowania w klasyfikacji obrazów biologicznych, co potwierdza użycie w segmentacji medycznej (MaxViT-UNet).

- **DINOv2 ViT-L/14 (advanced_dinov2)** wnosi jakościowo nowy typ reprezentacji — oparty na uczeniu się struktury wizualnej, nie kategorii ImageNet. Jego pochodne (Virchow, UNI, Cell-DINO) potwierdzają skuteczność w analizie obrazów biologicznych o charakterze mikroskopowym, co jest domeną bezpośrednio zbliżoną do obrazów otolitów. DINOv2-L/14 dorównuje lub przewyższa nadzorowane modele CNN i ViT w benchmarkach medycznych.

Wdrożenie obu wariantów jako eksperymentów porównawczych, obok dotychczasowych modeli, pozwoli empirycznie zweryfikować hipotezę o korzyści z pretreningu samouczącego i architektury hybrydowej dla specyfiki danych otolitów.

---

*Dokument przygotowany na podstawie przeglądu literatury naukowej (kwiecień 2026).*  
*Implementacja techniczna: pipeline PyTorch + biblioteka timm, pełna kompatybilność wsteczna z istniejącym systemem.*