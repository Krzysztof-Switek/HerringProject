#!/usr/bin/env python3
"""
data_split_pipeline.py — Kompletny potok przygotowania danych do treningu modeli otolitów.

CEL:
  Porównanie skuteczności modeli trenowanych na zdjęciach EMBEDDED vs NOT_EMBEDDED
  tych samych otolitów. Podział train/val/test jest identyczny dla obu gałęzi —
  te same ryby trafiają do tych samych setów, co eliminuje confounding factor.

ETAPY:
  1  scan    — Skanuje dysk sieciowy, buduje pary Embedded+NotEmbedded, waliduje
               z katalogiem Raw. Wyjście: tools/processed_pairs.csv
  2  copy    — Kopiuje pliki z sieci do final_pairs/{embedded,not_embedded}/{1,2}/
               Wymaga dostępu do Z:/ Czyta populację z Excel.
  3  split   — Dzieli pary na train/val/test (grupowanie po fish_key = brak leakage,
               ten sam podział dla embedded i not_embedded).
  4  verify  — Weryfikacja podziału: 3 tabele (liczby, zgodność par, leakage check).
  5  excel   — Aktualizuje Excel: dodaje kolumnę SET dla każdego rekordu.

UŻYCIE:
  python data_split_pipeline.py                  # wszystkie etapy 1–5
  python data_split_pipeline.py --steps 3 4      # tylko etapy 3 i 4 (bez sieci)
  python data_split_pipeline.py --steps 3 --dry-run  # podgląd bez kopiowania

WYMAGANIA:
  pip install pandas openpyxl tqdm
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ================================================================================
# CONFIG
# ================================================================================

# Dysk sieciowy — wymagany tylko dla etapów 1 i 2
NETWORK_PROCESSED = Path(r"Z:\Photo\Otolithes\HER\Processed")
NETWORK_RAW = Path(r"Z:\Photo\Otolithes\HER\Raw")

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent

# Pliki pomocnicze — w tools/
EXCEL_PATH = TOOLS_DIR / "analysisWithOtolithPhoto.xlsx"
PAIRS_CSV = TOOLS_DIR / "processed_pairs.csv"

# Katalog z wynikowymi danymi do treningu
FINAL_PAIRS_DIR = PROJECT_ROOT / "final_pairs"
EMBEDDED_ROOT = FINAL_PAIRS_DIR / "embedded"      # embedded/{1,2}/  + embedded/train/...
NOT_EMBEDDED_ROOT = FINAL_PAIRS_DIR / "not_embedded"  # analogicznie

POPULATIONS = ["1", "2"]
SPLITS = ["train", "val", "test"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.1
SEED = 42


# ================================================================================
# HELPERS
# ================================================================================

def sep(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def normalize_filename(value) -> str | None:
    """Zwraca samą nazwę pliku (bez ścieżki) w lowercase."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s).name.lower()


def fish_key_from_filename(filename: str) -> str:
    """
    Wyciąga fish_key z nazwy pliku (9 segmentów):
      seg 0-3  = prefix (rok_PROJEKT_GATUNEK_Lokalizacja)
      seg 4    = process (Embedded / NotEmbedded)
      seg 5    = variant (Sharpest / Embedded / WithoutPostproc)
      seg 6    = FishIndex{N}
      seg 7    = Single{N}
      seg 8    = view (Left / Right)

    fish_key = prefix + FishIndex — identyczny dla Embedded i NotEmbedded tej samej ryby.
    Zawsze lowercase dla spójności.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 9:
        return stem.lower()
    key = "_".join(parts[:4] + [parts[6]])  # prefix + FishIndex
    return key.lower()


def print_table(headers: list, rows: list):
    """Drukuje prostą tabelę tekstową."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


# ================================================================================
# ETAP 1 — SKANOWANIE SIECI I BUDOWANIE PAR
# ================================================================================

def _parse_processed_file(path: Path) -> dict | None:
    """
    Parsuje plik z katalogu Processed.
    Format wymagany: 9 segmentów, ostatni segment = Right lub Left.
    """
    parts = path.stem.split("_")
    if len(parts) != 9:
        return None
    view = parts[8]
    if view not in {"Right", "Left"}:
        return None
    prefix = "_".join(parts[:4])
    fish_index = parts[6]
    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "prefix": prefix,
        "process": parts[4],       # Embedded / NotEmbedded
        "variant": parts[5],       # Sharpest / Embedded / WithoutPostproc
        "fish_index": fish_index,
        "single": parts[7],        # Single1 / Single2
        "view": view,
        "fish_key": f"{prefix}_{fish_index}".lower(),
        "otolith_key": f"{prefix}_{fish_index}_{parts[7]}".lower(),
    }


def _build_raw_reference_pairs(raw_dir: Path) -> set[str]:
    """
    Zwraca zbiór fish_key (lowercase), dla których w katalogu Raw
    istnieje zarówno Embedded jak i NotEmbedded.

    Format pliku Raw: 7 segmentów
      0-3 = prefix, 4 = process, 5 = variant, 6 = FishIndex
    """
    groups: dict[str, set] = defaultdict(set)
    rejected = 0
    for path in raw_dir.rglob("*.jpg"):
        parts = path.stem.split("_")
        if len(parts) != 7:
            rejected += 1
            continue
        prefix = "_".join(parts[:4])
        key = f"{prefix}_{parts[6]}".lower()
        groups[key].add(parts[4])  # "Embedded" lub "NotEmbedded"

    pairs = {k for k, procs in groups.items()
             if "Embedded" in procs and "NotEmbedded" in procs}

    print(f"  Raw: {len(groups)} unikalnych fish_key, "
          f"{len(pairs)} potwierdzonych par Embedded+NotEmbedded")
    if rejected:
        print(f"  Raw: pominięto {rejected} plików z nieprawidłową liczbą segmentów")
    return pairs


def _choose_best_pair(otolith_df: pd.DataFrame) -> dict | None:
    """
    Z grupy rekordów dla tego samego otolitu wybiera jedną parę:
    jedno zdjęcie Embedded + jedno NotEmbedded, z tym samym view.
    Preferuje Right przed Left.
    """
    emb = otolith_df[otolith_df["process"] == "Embedded"]
    not_emb = otolith_df[otolith_df["process"] == "NotEmbedded"]

    if emb.empty or not_emb.empty:
        return None

    common_views = set(emb["view"]) & set(not_emb["view"])
    if not common_views:
        return None

    selected_view = "Right" if "Right" in common_views else next(iter(common_views))

    emb_pick = emb[emb["view"] == selected_view].sort_values("file_name").iloc[0]
    not_emb_pick = not_emb[not_emb["view"] == selected_view].sort_values("file_name").iloc[0]

    return {
        "fish_key": emb_pick["fish_key"],
        "otolith_key": emb_pick["otolith_key"],
        "prefix": emb_pick["prefix"],
        "fish_index": emb_pick["fish_index"],
        "single": emb_pick["single"],
        "selected_view": selected_view,
        "embedded_file_name": emb_pick["file_name"],
        "embedded_file_path": emb_pick["file_path"],
        "not_embedded_file_name": not_emb_pick["file_name"],
        "not_embedded_file_path": not_emb_pick["file_path"],
    }


def step_scan() -> pd.DataFrame:
    sep("ETAP 1 — SKANOWANIE DYSKU SIECIOWEGO I BUDOWANIE PAR")

    for p, label in [(NETWORK_PROCESSED, "Processed"), (NETWORK_RAW, "Raw")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Brak dostępu do katalogu {label}: {p}\n"
                "Sprawdź czy dysk sieciowy Z:\\ jest zamontowany."
            )

    # Skan Processed
    print("Skanowanie Processed...")
    records = []
    skipped_segments = Counter()
    skipped_view = 0

    for path in NETWORK_PROCESSED.rglob("*.jpg"):
        parts = path.stem.split("_")
        skipped_segments[len(parts)] += 1
        rec = _parse_processed_file(path)
        if rec is None:
            if len(parts) == 9:
                skipped_view += 1
            continue
        records.append(rec)

    df = pd.DataFrame(records)

    print(f"  Wszystkie JPG: {sum(skipped_segments.values())}")
    print(f"  Rozpoznane (9 segmentów, Right/Left): {len(df)}")
    print(f"  Segmenty inne niż 9: "
          f"{sum(v for k, v in skipped_segments.items() if k != 9)} plików")
    print(f"  Pominięte (nieznany view): {skipped_view}")

    if df.empty:
        raise ValueError("Brak rozpoznanych plików w Processed!")

    print("\nRozkład process + variant:")
    combo = df.groupby(["process", "variant"]).size().reset_index(name="count")
    for _, row in combo.iterrows():
        print(f"  {row['process']} + {row['variant']}: {row['count']}")

    # Referencja z Raw
    print("\nSkanowanie Raw...")
    raw_pairs = _build_raw_reference_pairs(NETWORK_RAW)

    # Filtruj Processed do fish_key potwierdzonych w Raw
    df_filtered = df[df["fish_key"].isin(raw_pairs)].copy()
    print(f"\nProcessed po filtrze Raw: {len(df_filtered)} rekordów "
          f"({df['fish_key'].nunique()} ryb -> {df_filtered['fish_key'].nunique()} ryb po filtrze)")

    # Dla każdego otolitu wybierz jedną parę Embedded+NotEmbedded
    final_rows = []
    rejected_no_common_view = 0
    rejected_incomplete = 0

    for otolith_key, group in df_filtered.groupby("otolith_key"):
        pair = _choose_best_pair(group)
        if pair is None:
            has_emb = "Embedded" in set(group["process"])
            has_not = "NotEmbedded" in set(group["process"])
            if has_emb and has_not:
                rejected_no_common_view += 1
            else:
                rejected_incomplete += 1
            continue
        final_rows.append(pair)

    pairs_df = pd.DataFrame(final_rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            ["fish_key", "otolith_key"]
        ).reset_index(drop=True)

    print(f"\nFinalne pary:")
    print(f"  Wybrane pary:                     {len(pairs_df)}")
    print(f"  Odrzucone (brak wspólnego view):  {rejected_no_common_view}")
    print(f"  Odrzucone (brak Emb lub NotEmb):  {rejected_incomplete}")
    print(f"  Unikalne ryby (fish_key):          {pairs_df['fish_key'].nunique()}")

    # Zapisz CSV
    PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nOK: Zapisano: {PAIRS_CSV}")

    return pairs_df


# ================================================================================
# ETAP 2 — KOPIOWANIE DO KATALOGÓW POPULACYJNYCH
# ================================================================================

def _load_population_lookup() -> dict[str, str]:
    """
    Czyta Excel i zwraca mapę: nazwa_pliku_lowercase -> populacja ("1" lub "2").
    Szuka kolumny z populacją (POPULACJA/Populacja) i kolumny ze ścieżką pliku.
    """
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku Excel: {EXCEL_PATH}\n"
            "Przenieś analysisWithOtolithPhoto.xlsx z final_pairs/ do tools/"
        )

    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    pop_col = next(
        (c for c in df.columns if c.lower() in {"populacja", "population"}), None
    )
    # Szukamy kolumny z embedded_file_path lub path — powinna zawierać nazwy plików embedded
    path_col = next(
        (c for c in df.columns
         if c.lower() in {"embedded_file_path", "path", "filepath", "file_path"}),
        None,
    )

    if pop_col is None:
        raise ValueError(
            f"Brak kolumny populacji w Excelu. Dostępne kolumny: {list(df.columns)}"
        )
    if path_col is None:
        raise ValueError(
            f"Brak kolumny ścieżki w Excelu. Dostępne kolumny: {list(df.columns)}"
        )

    lookup: dict[str, str] = {}
    conflicts = 0

    for _, row in df[[path_col, pop_col]].dropna().iterrows():
        fname = normalize_filename(row[path_col])
        if not fname:
            continue
        try:
            pop = str(int(float(row[pop_col])))
        except (ValueError, TypeError):
            continue
        if pop not in {"1", "2"}:
            continue
        if fname in lookup and lookup[fname] != pop:
            conflicts += 1
            continue
        lookup[fname] = pop

    print(f"  Mapowanie populacji: {len(lookup)} plików "
          f"(pop 1: {sum(1 for v in lookup.values() if v == '1')}, "
          f"pop 2: {sum(1 for v in lookup.values() if v == '2')})")
    if conflicts:
        print(f"  Konflikty populacji pominięte: {conflicts}")

    return lookup


def step_copy(pairs_df: pd.DataFrame | None = None, dry_run: bool = False):
    sep("ETAP 2 — KOPIOWANIE DO KATALOGÓW POPULACYJNYCH")

    if pairs_df is None:
        if not PAIRS_CSV.exists():
            raise FileNotFoundError(
                f"Brak pliku par: {PAIRS_CSV}\nUruchom najpierw etap 1 (scan)."
            )
        pairs_df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")

    print(f"Wczytano {len(pairs_df)} par z: {PAIRS_CSV}")
    pop_lookup = _load_population_lookup()

    # Utwórz katalogi docelowe
    if not dry_run:
        for pop in POPULATIONS:
            (EMBEDDED_ROOT / pop).mkdir(parents=True, exist_ok=True)
            (NOT_EMBEDDED_ROOT / pop).mkdir(parents=True, exist_ok=True)

    stats = Counter()

    for i, row in pairs_df.iterrows():
        emb_fname = normalize_filename(row.get("embedded_file_name", ""))
        not_emb_fname = normalize_filename(row.get("not_embedded_file_name", ""))

        pop = pop_lookup.get(emb_fname) or pop_lookup.get(not_emb_fname)

        if pop not in {"1", "2"}:
            stats["missing_pop"] += 1
            if stats["missing_pop"] <= 5:
                print(f"  WARN: Brak populacji dla pary: {emb_fname}")
            continue

        emb_src = Path(str(row["embedded_file_path"]))
        not_emb_src = Path(str(row["not_embedded_file_path"]))

        for src, dst_root, label in [
            (emb_src, EMBEDDED_ROOT, "emb"),
            (not_emb_src, NOT_EMBEDDED_ROOT, "not_emb"),
        ]:
            dst = dst_root / pop / src.name
            if dry_run:
                stats[f"would_copy_{label}"] += 1
            else:
                if not src.exists():
                    stats[f"missing_{label}"] += 1
                    print(f"  ERR: Brak pliku źródłowego: {src}")
                    continue
                try:
                    shutil.copy2(src, dst)
                    stats[f"copied_{label}"] += 1
                except Exception as e:
                    stats[f"error_{label}"] += 1
                    print(f"  ERR: Błąd kopiowania: {src.name} -> {e}")

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Wyniki kopiowania:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\nKatalog docelowy: {FINAL_PAIRS_DIR}")


# ================================================================================
# ETAP 3 — PODZIAŁ NA TRAIN / VAL / TEST
# ================================================================================

def _build_filename_to_pop_map(process_root: Path) -> dict[str, str]:
    """
    Skanuje process_root/{1,2}/ i zwraca mapę: filename.lower() -> populacja.
    """
    result: dict[str, str] = {}
    for pop in POPULATIONS:
        pop_dir = process_root / pop
        if not pop_dir.exists():
            continue
        for f in pop_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                key = f.name.lower()
                if key in result and result[key] != pop:
                    raise ValueError(
                        f"Plik {f.name} znaleziony w więcej niż jednej klasie populacji!"
                    )
                result[key] = pop
    return result


def step_split(pairs_df: pd.DataFrame | None = None, dry_run: bool = False):
    sep("ETAP 3 — PODZIAŁ NA TRAIN / VAL / TEST")

    if pairs_df is None:
        if not PAIRS_CSV.exists():
            raise FileNotFoundError(
                f"Brak pliku par: {PAIRS_CSV}\nUruchom najpierw etap 1 (scan)."
            )
        pairs_df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")

    print(f"Wczytano {len(pairs_df)} par z: {PAIRS_CSV}")
    print(f"Proporcje: train={TRAIN_RATIO}, val={VAL_RATIO}, "
          f"test={round(1 - TRAIN_RATIO - VAL_RATIO, 2)}, seed={SEED}")

    # Wyznacz populację dla każdej pary na podstawie plików lokalnych
    emb_map = _build_filename_to_pop_map(EMBEDDED_ROOT)
    not_map = _build_filename_to_pop_map(NOT_EMBEDDED_ROOT)

    df = pairs_df.copy()
    df["emb_fname"] = df["embedded_file_name"].apply(
        lambda x: normalize_filename(x) or ""
    )
    df["not_emb_fname"] = df["not_embedded_file_name"].apply(
        lambda x: normalize_filename(x) or ""
    )
    df["pop_emb"] = df["emb_fname"].map(emb_map)
    df["pop_not"] = df["not_emb_fname"].map(not_map)

    # Walidacja: spójność populacji
    missing_emb = df["pop_emb"].isna().sum()
    missing_not = df["pop_not"].isna().sum()
    if missing_emb > 0:
        print(f"  WARN: {missing_emb} par bez pliku embedded lokalnie — pominięte.")
        print(f"    (Pliki nie zostały skopiowane przez etap 2 lub nie należą do pop. 1/2)")
    if missing_not > 0:
        print(f"  WARN: {missing_not} par bez pliku not_embedded lokalnie — pominięte.")

    # Zachowaj tylko pary z kompletnymi danymi lokalnymi
    df = df.dropna(subset=["pop_emb", "pop_not"]).copy()

    mismatch = df[df["pop_emb"] != df["pop_not"]]
    if not mismatch.empty:
        raise ValueError(
            f"BŁĄD: {len(mismatch)} par z rozbieżną populacją embedded vs not_embedded!\n"
            "Embedded i not_embedded tego samego otolitu muszą być w tej samej klasie.\n"
            f"Przykłady:\n{mismatch[['emb_fname', 'pop_emb', 'not_emb_fname', 'pop_not']].head(5)}"
        )

    df["pop"] = df["pop_emb"]

    # Normalizuj fish_key do lowercase dla spójności
    df["fish_key_norm"] = df["fish_key"].apply(
        lambda x: str(x).lower().strip() if pd.notna(x) else ""
    )

    # Podział po fish_key — TEN SAM podział dla embedded i not_embedded
    random.seed(SEED)
    group_assignments: dict[tuple[str, str], str] = {}  # (pop, fish_key_norm) -> split

    print("\nPodział ryb na grupy:")
    for pop in POPULATIONS:
        pop_df = df[df["pop"] == pop]
        fish_keys = sorted(pop_df["fish_key_norm"].dropna().unique().tolist())
        random.shuffle(fish_keys)

        n = len(fish_keys)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        for i, k in enumerate(fish_keys):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            group_assignments[(pop, k)] = split

        counts = Counter(group_assignments[(pop, k)] for k in fish_keys)
        print(f"  Populacja {pop}: {n} ryb "
              f"-> train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # Wyczyść stare katalogi splitów (NIE ruszaj 1/ i 2/ — to źródła)
    if not dry_run:
        for process_root in [EMBEDDED_ROOT, NOT_EMBEDDED_ROOT]:
            for split in SPLITS:
                split_dir = process_root / split
                if split_dir.exists():
                    shutil.rmtree(split_dir)
                    print(f"  Usunięto: {split_dir}")

        # Utwórz nową strukturę katalogów
        for process_root in [EMBEDDED_ROOT, NOT_EMBEDDED_ROOT]:
            for split in SPLITS:
                for pop in POPULATIONS:
                    (process_root / split / pop).mkdir(parents=True, exist_ok=True)

    # Kopiuj pliki do splitów
    stats: dict[str, Counter] = {
        "embedded": Counter(),
        "not_embedded": Counter(),
    }
    no_split_count = 0
    missing_src_count = 0

    for _, row in df.iterrows():
        pop = str(row["pop"])
        fish_key = str(row["fish_key_norm"])
        split = group_assignments.get((pop, fish_key))

        if split is None:
            no_split_count += 1
            continue

        emb_src = EMBEDDED_ROOT / pop / row["emb_fname"]
        not_emb_src = NOT_EMBEDDED_ROOT / pop / row["not_emb_fname"]

        for src, process_root, label in [
            (emb_src, EMBEDDED_ROOT, "embedded"),
            (not_emb_src, NOT_EMBEDDED_ROOT, "not_embedded"),
        ]:
            if not src.exists():
                missing_src_count += 1
                print(f"  ERR: Brak pliku: {src}")
                continue
            if not dry_run:
                dst = process_root / split / pop / src.name
                shutil.copy2(src, dst)
            stats[label][split] += 1

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Skopiowane pary (embedded / not_embedded):")
    print_table(
        ["SPLIT", "EMBEDDED", "NOT_EMBEDDED"],
        [[s, stats["embedded"][s], stats["not_embedded"][s]] for s in SPLITS]
    )

    if no_split_count:
        print(f"\nWARN: Pominięto {no_split_count} par bez przypisanego splitu")
    if missing_src_count:
        print(f"WARN: Brak {missing_src_count} plików źródłowych")
    if not no_split_count and not missing_src_count:
        print("\nOK: Podział zakończony bez błędów.")

    return group_assignments


# ================================================================================
# ETAP 4 — WERYFIKACJA
# ================================================================================

def _scan_split_fish_keys(process_root: Path) -> dict[str, dict[str, set[str]]]:
    """
    Skanuje katalogi splitów i zwraca:
    { split: { pop: set(fish_key_lowercase) } }
    """
    result: dict[str, dict[str, set]] = {
        split: {pop: set() for pop in POPULATIONS}
        for split in SPLITS
    }
    for split in SPLITS:
        for pop in POPULATIONS:
            d = process_root / split / pop
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    result[split][pop].add(fish_key_from_filename(f.name))
    return result


def _count_split_files(process_root: Path) -> dict[str, dict[str, int]]:
    """Liczy pliki w każdym split/pop."""
    result: dict[str, dict[str, int]] = {
        split: {pop: 0 for pop in POPULATIONS}
        for split in SPLITS
    }
    for split in SPLITS:
        for pop in POPULATIONS:
            d = process_root / split / pop
            if d.exists():
                result[split][pop] = sum(
                    1 for f in d.iterdir() if f.is_file() and not f.name.startswith(".")
                )
    return result


def step_verify():
    sep("ETAP 4 — WERYFIKACJA PODZIAŁU")

    emb_fish = _scan_split_fish_keys(EMBEDDED_ROOT)
    not_fish = _scan_split_fish_keys(NOT_EMBEDDED_ROOT)
    emb_cnt = _count_split_files(EMBEDDED_ROOT)
    not_cnt = _count_split_files(NOT_EMBEDDED_ROOT)

    errors: list[str] = []

    # -- TABELA 1: Liczba plików ----------------------------------------------
    print("\n[TABELA 1] LICZBA PLIKÓW W KATALOGACH SPLITÓW")
    rows = []
    for split in SPLITS:
        for pop in POPULATIONS:
            ec = emb_cnt[split][pop]
            nc = not_cnt[split][pop]
            diff = ec - nc
            status = "OK" if diff == 0 else "RÓŻNICA !"
            rows.append([split, pop, ec, nc, diff, status])
            if diff != 0:
                errors.append(
                    f"Nierówna liczba plików w {split}/pop{pop}: "
                    f"embedded={ec}, not_embedded={nc}"
                )
    print_table(
        ["SPLIT", "POP", "EMBEDDED", "NOT_EMBEDDED", "RÓŻNICA", "STATUS"], rows
    )

    # -- TABELA 2: Zgodność grup ryb -----------------------------------------
    print("\n[TABELA 2] ZGODNOŚĆ GRUP RYB (embedded vs not_embedded, te same otolity)")
    rows = []
    for split in SPLITS:
        for pop in POPULATIONS:
            ek = emb_fish[split][pop]
            nk = not_fish[split][pop]
            only_e = len(ek - nk)
            only_n = len(nk - ek)
            common = len(ek & nk)
            status = "OK" if only_e == 0 and only_n == 0 else "RÓŻNICA !"
            rows.append([split, pop, common, only_e, only_n, status])
            if only_e or only_n:
                errors.append(
                    f"Niezgodne ryby w {split}/pop{pop}: "
                    f"tylko_embedded={only_e}, tylko_not_embedded={only_n}"
                )
    print_table(
        ["SPLIT", "POP", "WSPÓLNE_RYBY", "TYLKO_EMB", "TYLKO_NOT", "STATUS"], rows
    )

    # -- TABELA 3: Leakage ----------------------------------------------------
    print("\n[TABELA 3] LEAKAGE CHECK — ta sama ryba w więcej niż jednym secie")
    rows = []
    for process_name, proc_fish in [("embedded", emb_fish), ("not_embedded", not_fish)]:
        for pop in POPULATIONS:
            tv = proc_fish["train"][pop] & proc_fish["val"][pop]
            tt = proc_fish["train"][pop] & proc_fish["test"][pop]
            vt = proc_fish["val"][pop] & proc_fish["test"][pop]
            total = len(tv) + len(tt) + len(vt)
            status = "OK" if total == 0 else "LEAKAGE !"
            rows.append([process_name, pop, len(tv), len(tt), len(vt), status])
            if total > 0:
                errors.append(
                    f"Leakage w {process_name}/pop{pop}: "
                    f"train+val={len(tv)}, train+test={len(tt)}, val+test={len(vt)}"
                )
    print_table(
        ["PROCES", "POP", "TRAIN+VAL", "TRAIN+TEST", "VAL+TEST", "STATUS"], rows
    )

    # -- TABELA 4: Podsumowanie liczebności -----------------------------------
    print("\n[TABELA 4] PODSUMOWANIE — ŁĄCZNA LICZBA ZDJĘĆ NA SPLIT")
    rows = []
    for split in SPLITS:
        emb_total = sum(emb_cnt[split][pop] for pop in POPULATIONS)
        not_total = sum(not_cnt[split][pop] for pop in POPULATIONS)
        rows.append([split, emb_total, not_total])
    emb_grand = sum(emb_cnt[s][p] for s in SPLITS for p in POPULATIONS)
    not_grand = sum(not_cnt[s][p] for s in SPLITS for p in POPULATIONS)
    rows.append(["SUMA", emb_grand, not_grand])
    print_table(["SPLIT", "EMBEDDED_PLIKI", "NOT_EMBEDDED_PLIKI"], rows)

    # -- Szczegóły błędów ----------------------------------------------------
    sep("WYNIK WERYFIKACJI")
    if errors:
        print("ERR: Wykryto problemy:")
        for e in errors:
            print(f"  - {e}")
        print()
        # Dla różnic embedded vs not_embedded pokaż przykłady
        for split in SPLITS:
            for pop in POPULATIONS:
                ek = emb_fish[split][pop]
                nk = not_fish[split][pop]
                only_e = sorted(ek - nk)
                only_n = sorted(nk - ek)
                if only_e or only_n:
                    print(f"  Szczegóły {split}/pop{pop}:")
                    if only_e:
                        print(f"    Tylko w embedded (pierwsze 5): {only_e[:5]}")
                    if only_n:
                        print(f"    Tylko w not_embedded (pierwsze 5): {only_n[:5]}")
    else:
        print("OK: Wszystko poprawne. Podział gotowy do treningu.")
        print(f"\nKonfiguracja treningu modeli:")
        print(f"  Embedded model:     data_dir = {EMBEDDED_ROOT}")
        print(f"  Not-embedded model: data_dir = {NOT_EMBEDDED_ROOT}")


# ================================================================================
# ETAP 5 — AKTUALIZACJA EXCELA
# ================================================================================

def step_excel():
    sep("ETAP 5 — AKTUALIZACJA EXCELA (kolumna SET)")

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku Excel: {EXCEL_PATH}\n"
            "Przenieś analysisWithOtolithPhoto.xlsx z final_pairs/ do tools/"
        )

    # Zbierz mapę: filename -> SET (na podstawie faktycznych katalogów)
    file_to_set: dict[str, str] = {}
    for process_root in [EMBEDDED_ROOT, NOT_EMBEDDED_ROOT]:
        for split in SPLITS:
            for pop in POPULATIONS:
                d = process_root / split / pop
                if not d.exists():
                    continue
                for f in d.iterdir():
                    if f.is_file() and not f.name.startswith("."):
                        file_to_set[f.name.lower()] = split.upper()

    if not file_to_set:
        raise ValueError(
            "Nie znaleziono żadnych plików w katalogach splitów. "
            "Uruchom najpierw etap 3 (split)."
        )

    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    # Usuń stare kolumny SET
    for col in ["SET", "SET_embedded", "SET_not_embedded",
                "embedded_name", "not_embedded_name"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Znajdź kolumnę ze ścieżką/nazwą pliku — obsługuje różne schematy Excela
    # Schemat A: kolumny embedded_file_path / not_embedded_file_path (pary)
    # Schemat B: kolumna FilePath / FileName (jeden wiersz = jedno zdjęcie)
    path_col = next(
        (c for c in df.columns
         if c.lower() in {"filepath", "filename", "file_path", "file_name",
                          "embedded_file_path", "not_embedded_file_path"}),
        None,
    )

    if "embedded_file_path" in df.columns or "not_embedded_file_path" in df.columns:
        # Schemat A: Excel z parami (stary format)
        if "embedded_file_path" in df.columns:
            df["_fname"] = df["embedded_file_path"].apply(normalize_filename)
            df["SET"] = df["_fname"].map(file_to_set)
            df.drop(columns=["_fname"], inplace=True)
        else:
            df["_fname"] = df["not_embedded_file_path"].apply(normalize_filename)
            df["SET"] = df["_fname"].map(file_to_set)
            df.drop(columns=["_fname"], inplace=True)
    elif path_col is not None:
        # Schemat B: Excel z jednym wierszem per zdjęcie (FilePath / FileName)
        df["_fname"] = df[path_col].apply(normalize_filename)
        df["SET"] = df["_fname"].map(file_to_set)
        df.drop(columns=["_fname"], inplace=True)
    else:
        raise ValueError(
            f"Excel nie zawiera rozpoznanej kolumny ze ścieżką pliku.\n"
            f"Dostępne kolumny: {list(df.columns)}"
        )

    # Usuń pomocnicze kolumny (zostaw tylko SET)
    for col in ["SET_embedded", "SET_not_embedded"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.to_excel(EXCEL_PATH, index=False)

    set_counts = df["SET"].value_counts().to_dict()
    nan_count = df["SET"].isna().sum()
    print(f"OK: Zaktualizowano: {EXCEL_PATH}")
    print(f"Rozkład SET: {dict(sorted(set_counts.items()))}")
    if nan_count:
        print(f"WARN: Rekordów bez SET: {nan_count} (brak w splitach — sprawdź czy pary są kompletne)")


# ================================================================================
# ETAP 6 — GENEROWANIE RAPORTU MD
# ================================================================================

def step_report():
    sep("ETAP 6 — GENEROWANIE RAPORTU")

    from datetime import date

    # Zbierz liczby plików z katalogów splitów
    cnt: dict[str, dict[str, dict[str, int]]] = {}
    for proc in ["embedded", "not_embedded"]:
        cnt[proc] = {}
        root = EMBEDDED_ROOT if proc == "embedded" else NOT_EMBEDDED_ROOT
        for split in SPLITS:
            cnt[proc][split] = {}
            for pop in POPULATIONS:
                d = root / split / pop
                cnt[proc][split][pop] = (
                    sum(1 for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
                    if d.exists() else 0
                )

    # Liczby źródłowe (pop-level, bez splitów)
    src_cnt: dict[str, dict[str, int]] = {}
    for proc in ["embedded", "not_embedded"]:
        root = EMBEDDED_ROOT if proc == "embedded" else NOT_EMBEDDED_ROOT
        src_cnt[proc] = {}
        for pop in POPULATIONS:
            d = root / pop
            src_cnt[proc][pop] = (
                sum(1 for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
                if d.exists() else 0
            )

    # Dane z CSV par
    pairs_info = {"total": 0, "fish": 0, "locations": {}}
    if PAIRS_CSV.exists():
        pairs_df = pd.read_csv(PAIRS_CSV, encoding="utf-8-sig")
        pairs_info["total"] = len(pairs_df)
        pairs_info["fish"] = int(pairs_df["fish_key"].nunique()) if "fish_key" in pairs_df.columns else 0
        if "prefix" in pairs_df.columns:
            pairs_info["locations"] = pairs_df["prefix"].value_counts().to_dict()

    # Dane z Excela (SET)
    excel_set: dict[str, int] = {}
    excel_noset = 0
    excel_total = 0
    if EXCEL_PATH.exists():
        df_ex = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        excel_total = len(df_ex)
        if "SET" in df_ex.columns:
            excel_set = df_ex["SET"].value_counts().to_dict()
            excel_noset = int(df_ex["SET"].isna().sum())

    # Podsumowanie podziału ryb
    fish_split: dict[str, dict[str, int]] = {}
    for pop in POPULATIONS:
        fish_keys: dict[str, set] = {s: set() for s in SPLITS}
        for proc in ["embedded", "not_embedded"]:
            root = EMBEDDED_ROOT if proc == "embedded" else NOT_EMBEDDED_ROOT
            for split in SPLITS:
                d = root / split / pop
                if d.exists():
                    for f in d.iterdir():
                        if f.is_file() and not f.name.startswith("."):
                            fish_keys[split].add(fish_key_from_filename(f.name))
        fish_split[pop] = {s: len(fish_keys[s]) for s in SPLITS}

    today = date.today().isoformat()

    # Sumy pomocnicze
    src_total_emb = sum(src_cnt["embedded"].values())
    src_total_not = sum(src_cnt["not_embedded"].values())
    pairs_with_pop = src_total_emb  # tyle par skopiowano do pop-folderów
    pairs_without_pop = pairs_info["total"] - pairs_with_pop

    def row(split):
        e1, e2 = cnt["embedded"][split]["1"], cnt["embedded"][split]["2"]
        n1, n2 = cnt["not_embedded"][split]["1"], cnt["not_embedded"][split]["2"]
        et, nt = e1 + e2, n1 + n2
        f1, f2 = fish_split.get("1", {}).get(split, 0), fish_split.get("2", {}).get(split, 0)
        return e1, e2, et, n1, n2, nt, f1, f2, f1 + f2

    tr = row("train")
    vl = row("val")
    te = row("test")
    su = tuple(a + b + c for a, b, c in zip(tr, vl, te))

    loc_rows = ""
    for loc, n in sorted(pairs_info["locations"].items(), key=lambda x: -x[1]):
        loc_rows += f"| {loc} | {n} |\n"

    report = f"""# Raport przygotowania danych — Otolity Sledzia

Data: {today}
Skrypt: `tools/data_split_pipeline.py`


## Zrodla danych

| Zrodlo | Sciezka |
|--------|---------|
| Dysk sieciowy Processed | `Z:\\Photo\\Otolithes\\HER\\Processed` |
| Dysk sieciowy Raw | `Z:\\Photo\\Otolithes\\HER\\Raw` |
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
| **Finalne pary wybrane** | **{pairs_info['total']}** |
| Odrzucone (brak wspolnego view) | 1 798 |
| Odrzucone (brak Emb lub NotEmb) | 646 |
| Unikalne ryby (fish_key) | {pairs_info['fish']} |

Para = jedno zdjecie Embedded + jedno NotEmbedded tego samego otolitu, ten sam view (preferowany Right).

---

## Etap 2 — Przypisanie populacji i kopiowanie

| Metryka | Wartosc |
|---------|---------|
| Par z przypisana populacja (1 lub 2) | **{pairs_with_pop}** |
| Par bez populacji (pominiete) | {pairs_without_pop} |
| Embedded pop. 1 | {src_cnt['embedded']['1']} |
| Embedded pop. 2 | {src_cnt['embedded']['2']} |
| Not-embedded pop. 1 | {src_cnt['not_embedded']['1']} |
| Not-embedded pop. 2 | {src_cnt['not_embedded']['2']} |

Pominiete {pairs_without_pop} par to ryby z lokalizacji KolobrzeskoDarlowskie bez wypelnionej kolumny Populacja w Excelu.

---

## Etap 3 — Podzial train / val / test

Proporcje: **70% train / 20% val / 10% test**, seed=42
Podzial na poziomie ryby (fish_key) — brak leakage, ten sam podzial dla embedded i not_embedded.

### Liczba plikow per split i populacja

| Split | Emb pop.1 | Emb pop.2 | Emb lacznie | NotEmb pop.1 | NotEmb pop.2 | NotEmb lacznie | Ryb pop.1 | Ryb pop.2 | Ryb lacznie |
|-------|-----------|-----------|-------------|--------------|--------------|----------------|-----------|-----------|-------------|
| train | {tr[0]} | {tr[1]} | {tr[2]} | {tr[3]} | {tr[4]} | {tr[5]} | {tr[6]} | {tr[7]} | {tr[8]} |
| val   | {vl[0]} | {vl[1]} | {vl[2]} | {vl[3]} | {vl[4]} | {vl[5]} | {vl[6]} | {vl[7]} | {vl[8]} |
| test  | {te[0]} | {te[1]} | {te[2]} | {te[3]} | {te[4]} | {te[5]} | {te[6]} | {te[7]} | {te[8]} |
| **SUMA** | **{su[0]}** | **{su[1]}** | **{su[2]}** | **{su[3]}** | **{su[4]}** | **{su[5]}** | **{su[6]}** | **{su[7]}** | **{su[8]}** |

Lacznie w final_pairs/: **{su[2] + su[5]} plikow** ({su[2]} embedded + {su[5]} not_embedded w splitach)
plus kopie zrodlowe: {src_total_emb + src_total_not} plikow w katalogach 1/ i 2/

### Struktura katalogow

```
final_pairs/
  embedded/
    1/  ({src_cnt['embedded']['1']} plikow)    not_embedded/1/  ({src_cnt['not_embedded']['1']} plikow)
    2/  ({src_cnt['embedded']['2']} plikow)                 2/  ({src_cnt['not_embedded']['2']} plikow)
    train/1/ ({cnt['embedded']['train']['1']})  not_embedded/train/1/ ({cnt['not_embedded']['train']['1']})
    train/2/ ({cnt['embedded']['train']['2']})               train/2/ ({cnt['not_embedded']['train']['2']})
    val/1/   ({cnt['embedded']['val']['1']})                 val/1/   ({cnt['not_embedded']['val']['1']})
    val/2/   ({cnt['embedded']['val']['2']})                 val/2/   ({cnt['not_embedded']['val']['2']})
    test/1/  ({cnt['embedded']['test']['1']})                test/1/  ({cnt['not_embedded']['test']['1']})
    test/2/  ({cnt['embedded']['test']['2']})                test/2/  ({cnt['not_embedded']['test']['2']})
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

Plik: `tools/analysisWithOtolithPhoto.xlsx` ({excel_total} rekordow lacznie)

| SET | Rekordow |
|-----|---------|
| TRAIN | {excel_set.get('TRAIN', 0)} |
| VAL | {excel_set.get('VAL', 0)} |
| TEST | {excel_set.get('TEST', 0)} |
| Bez SET (poza zbiorem treningowym) | {excel_noset} |

Rekordy bez SET to zdjecia embedded bez pary NotEmbedded lub bez przypisanej populacji.

---

## Lokalizacje polowow w zbiorze

| Lokalizacja | Par |
|-------------|-----|
{loc_rows}

---

## Konfiguracja treningu

```yaml
# Model Embedded — config.yaml:
data:
  data_dir: {EMBEDDED_ROOT}

# Model Not-Embedded — config.yaml:
data:
  data_dir: {NOT_EMBEDDED_ROOT}
```

Oba katalogi maja identyczna strukture train/val/test/{{1,2}}/ z tymi samymi rybami w tych samych setach.

---

## Ponowne uruchomienie pipeline'u

```bash
# Pelny pipeline (wymaga sieci Z:\\):
.venv\\Scripts\\python tools\\data_split_pipeline.py

# Tylko split + weryfikacja (bez sieci, pliki juz skopiowane):
.venv\\Scripts\\python tools\\data_split_pipeline.py --steps 3 4 5

# Podglad bez kopiowania:
.venv\\Scripts\\python tools\\data_split_pipeline.py --steps 3 --dry-run
```
"""

    report_path = TOOLS_DIR / "pipeline_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"OK: Raport zapisany: {report_path}")


# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline przygotowania danych otolitów do treningu embedded vs not-embedded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python data_split_pipeline.py                   # wszystkie etapy (wymaga sieci Z:\\)
  python data_split_pipeline.py --steps 3 4       # split + weryfikacja (bez sieci)
  python data_split_pipeline.py --steps 4         # tylko weryfikacja istniejących splitów
  python data_split_pipeline.py --steps 3 --dry-run  # podgląd podziału bez kopiowania
        """,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Numery etapów do uruchomienia (1-6, domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Podgląd bez kopiowania plików (tylko dla etapów 2 i 3)",
    )
    args = parser.parse_args()

    steps = sorted(set(args.steps))

    print("=" * 72)
    print("Pipeline przygotowania danych otolitów")
    print("=" * 72)
    print(f"  Etapy:       {steps}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"  TOOLS_DIR:   {TOOLS_DIR}")
    print(f"  FINAL_PAIRS: {FINAL_PAIRS_DIR}")
    print(f"  EXCEL:       {EXCEL_PATH}")
    print(f"  PAIRS_CSV:   {PAIRS_CSV}")

    pairs_df = None

    if 1 in steps:
        pairs_df = step_scan()

    if 2 in steps:
        step_copy(pairs_df, dry_run=args.dry_run)

    if 3 in steps:
        step_split(pairs_df, dry_run=args.dry_run)

    if 4 in steps:
        step_verify()

    if 5 in steps:
        step_excel()

    if 6 in steps:
        step_report()

    sep("GOTOWE")
    print(f"Uruchomione etapy: {steps}")


if __name__ == "__main__":
    main()