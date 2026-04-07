from pathlib import Path
from collections import Counter, defaultdict
import shutil
import pandas as pd

# =========================
# CONFIG
# =========================

PROCESSED_DIR = Path(r"Z:\Photo\Otolithes\HER\Processed")
RAW_DIR = Path(r"Z:\Photo\Otolithes\HER\Raw")

OUTPUT_CSV = Path(__file__).resolve().parent / "processed_pairs.csv"

EXCEL_PATH = Path(r"C:\Users\kswitek\Documents\HerringProject\tools\analysisWithOtolithPhoto.xlsx")
OUTPUT_DIR = Path(__file__).resolve().parent / "final_pairs"
EMBEDDED_DIR = OUTPUT_DIR / "embedded"
NOT_EMBEDDED_DIR = OUTPUT_DIR / "not_embedded"


# =========================
# HELPERS
# =========================

def print_separator(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalize_name(value):
    if pd.isna(value):
        return None
    return Path(str(value).strip()).name.lower()


def build_population_lookup():
    if not EXCEL_PATH.exists():
        print(f"❌ Nie znaleziono pliku Excel: {EXCEL_PATH}")
        return {}

    try:
        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    except Exception as e:
        print(f"❌ Błąd wczytywania Excela: {e}")
        return {}

    population_col = None
    for candidate in ["POPULACJA", "Populacja", "populacja"]:
        if candidate in df.columns:
            population_col = candidate
            break

    if population_col is None:
        print("❌ Nie znaleziono kolumny 'POPULACJA' / 'Populacja' w Excelu.")
        return {}

    path_col = None
    for candidate in ["path", "Path", "FILEPATH", "FilePath", "file_path"]:
        if candidate in df.columns:
            path_col = candidate
            break

    if path_col is None:
        print("❌ Nie znaleziono kolumny ze ścieżką pliku w Excelu.")
        return {}

    df = df[[path_col, population_col]].dropna().copy()

    lookup = {}
    duplicates = 0

    for _, row in df.iterrows():
        file_name = normalize_name(row[path_col])
        if not file_name:
            continue

        try:
            population = str(int(row[population_col]))
        except Exception:
            continue

        if population not in {"1", "2"}:
            continue

        if file_name in lookup and lookup[file_name] != population:
            duplicates += 1
            continue

        lookup[file_name] = population

    print_separator("MAPOWANIE POPULACJI Z EXCELA")
    print(f"Zmapowane pliki: {len(lookup)}")
    print(f"Konflikty pominięte: {duplicates}")

    return lookup


def build_processed_records():
    records = []
    all_files = list(PROCESSED_DIR.rglob("*"))
    jpg_files = [p for p in all_files if p.is_file() and p.suffix.lower() == ".jpg"]

    segment_counts = Counter()
    segment_9_counter = Counter()
    right_left_count = 0
    segment_5_counter = Counter()
    segment_6_counter = Counter()
    segment_5_6_combo = Counter()

    for path in jpg_files:
        parts = path.stem.split("_")
        segment_counts[len(parts)] += 1

        if len(parts) != 9:
            continue

        prefix = "_".join(parts[0:4])
        process = parts[4]
        variant = parts[5]
        fish_index = parts[6]
        single = parts[7]
        view = parts[8]

        segment_9_counter[view] += 1

        if view in {"Right", "Left"}:
            right_left_count += 1
            segment_5_counter[process] += 1
            segment_6_counter[variant] += 1
            segment_5_6_combo[(process, variant)] += 1

        records.append(
            {
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "prefix": prefix,
                "process": process,
                "variant": variant,
                "fish_index": fish_index,
                "single": single,
                "view": view,
                "fish_key": f"{prefix}_{fish_index}",
                "otolith_key": f"{prefix}_{fish_index}_{single}",
            }
        )

    df = pd.DataFrame(records)

    return {
        "df": df,
        "all_files": len(all_files),
        "jpg_files": len(jpg_files),
        "segment_counts": segment_counts,
        "segment_9_counter": segment_9_counter,
        "right_left_count": right_left_count,
        "segment_5_counter": segment_5_counter,
        "segment_6_counter": segment_6_counter,
        "segment_5_6_combo": segment_5_6_combo,
    }


def build_raw_reference():
    raw_files = list(RAW_DIR.rglob("*"))
    raw_jpg = [p for p in raw_files if p.is_file() and p.suffix.lower() == ".jpg"]

    segment_counts = Counter()
    segment_5_counter = Counter()
    segment_6_counter = Counter()
    segment_5_6_combo = Counter()

    raw_groups = defaultdict(set)

    for path in raw_jpg:
        parts = path.stem.split("_")
        segment_counts[len(parts)] += 1

        if len(parts) != 7:
            continue

        prefix = "_".join(parts[0:4])
        process = parts[4]
        variant = parts[5]
        fish_index = parts[6]

        segment_5_counter[process] += 1
        segment_6_counter[variant] += 1
        segment_5_6_combo[(process, variant)] += 1

        fish_key = f"{prefix}_{fish_index}"
        raw_groups[fish_key].add(process)

    raw_reference_pairs = {
        fish_key
        for fish_key, classes in raw_groups.items()
        if "Embedded" in classes and "NotEmbedded" in classes
    }

    return {
        "all_files": len(raw_files),
        "jpg_files": len(raw_jpg),
        "segment_counts": segment_counts,
        "segment_5_counter": segment_5_counter,
        "segment_6_counter": segment_6_counter,
        "segment_5_6_combo": segment_5_6_combo,
        "raw_reference_pairs": raw_reference_pairs,
        "raw_groups": raw_groups,
    }


def choose_best_pair(group: pd.DataFrame):
    embedded = group[group["process"] == "Embedded"].copy()
    not_embedded = group[group["process"] == "NotEmbedded"].copy()

    if embedded.empty or not_embedded.empty:
        return None

    embedded_views = set(embedded["view"])
    not_embedded_views = set(not_embedded["view"])
    common_views = embedded_views & not_embedded_views

    preferred_view = None
    for candidate in ["Right", "Left"]:
        if candidate in common_views:
            preferred_view = candidate
            break

    if preferred_view is None:
        return None

    embedded_pick = (
        embedded[embedded["view"] == preferred_view]
        .sort_values(["file_name"])
        .iloc[0]
    )
    not_embedded_pick = (
        not_embedded[not_embedded["view"] == preferred_view]
        .sort_values(["file_name"])
        .iloc[0]
    )

    return {
        "fish_key": embedded_pick["fish_key"],
        "otolith_key": embedded_pick["otolith_key"],
        "prefix": embedded_pick["prefix"],
        "fish_index": embedded_pick["fish_index"],
        "single": embedded_pick["single"],
        "selected_view": preferred_view,
        "embedded_file_name": embedded_pick["file_name"],
        "embedded_file_path": embedded_pick["file_path"],
        "not_embedded_file_name": not_embedded_pick["file_name"],
        "not_embedded_file_path": not_embedded_pick["file_path"],
    }


def copy_pairs_to_population_folders(final_pairs_df: pd.DataFrame):
    if final_pairs_df.empty:
        print_separator("KOPIOWANIE PLIKÓW")
        print("Brak par do kopiowania.")
        return

    population_lookup = build_population_lookup()
    if not population_lookup:
        print_separator("KOPIOWANIE PLIKÓW")
        print("Brak mapowania populacji. Kopiowanie pominięte.")
        return

    for population in ["1", "2"]:
        (EMBEDDED_DIR / population).mkdir(parents=True, exist_ok=True)
        (NOT_EMBEDDED_DIR / population).mkdir(parents=True, exist_ok=True)

    copied_embedded = 0
    copied_not_embedded = 0
    missing_population = 0
    copy_errors = 0

    for i, row in final_pairs_df.iterrows():
        embedded_src = Path(row["embedded_file_path"])
        not_embedded_src = Path(row["not_embedded_file_path"])

        population = population_lookup.get(normalize_name(row["embedded_file_name"]))
        if population is None:
            population = population_lookup.get(normalize_name(row["not_embedded_file_name"]))

        if population not in {"1", "2"}:
            print(f"❌ [{i}] Brak populacji dla pary: {row['embedded_file_name']} / {row['not_embedded_file_name']}")
            missing_population += 1
            continue

        embedded_dst = EMBEDDED_DIR / population / embedded_src.name
        not_embedded_dst = NOT_EMBEDDED_DIR / population / not_embedded_src.name

        try:
            shutil.copy2(embedded_src, embedded_dst)
            copied_embedded += 1
        except Exception as e:
            print(f"❌ [{i}] Błąd kopiowania Embedded: {embedded_src} -> {e}")
            copy_errors += 1

        try:
            shutil.copy2(not_embedded_src, not_embedded_dst)
            copied_not_embedded += 1
        except Exception as e:
            print(f"❌ [{i}] Błąd kopiowania NotEmbedded: {not_embedded_src} -> {e}")
            copy_errors += 1

    print_separator("KOPIOWANIE PLIKÓW — PODSUMOWANIE")
    print(f"Skopiowane Embedded:        {copied_embedded}")
    print(f"Skopiowane NotEmbedded:     {copied_not_embedded}")
    print(f"Brak populacji:             {missing_population}")
    print(f"Błędy kopiowania:           {copy_errors}")
    print(f"Katalog docelowy:           {OUTPUT_DIR.resolve()}")


# =========================
# MAIN ANALYSIS
# =========================

def analyze_and_export():
    processed = build_processed_records()
    raw = build_raw_reference()

    processed_df = processed["df"]

    # -------------------------
    # KROK 1 — PROCESSED NAZWY
    # -------------------------
    print_separator("KROK 1 — PROCESSED: ROZKŁAD LICZBY SEGMENTÓW")
    for k, v in sorted(processed["segment_counts"].items()):
        print(f"{k} segmentów: {v} plików")

    print_separator("KROK 2 — PROCESSED: UNIKALNE WARTOŚCI SEGMENTU 9")
    for val, cnt in processed["segment_9_counter"].most_common():
        print(f"{val}: {cnt}")

    print_separator("KROK 3 — PROCESSED: TYLKO RIGHT + LEFT")
    print(f"Zdjęcia po filtrze Right/Left: {processed['right_left_count']}")

    print_separator("KROK 3A — PROCESSED: SEGMENT 5")
    for val, cnt in processed["segment_5_counter"].most_common():
        print(f"{val}: {cnt}")

    print_separator("KROK 3B — PROCESSED: SEGMENT 6")
    for val, cnt in processed["segment_6_counter"].most_common():
        print(f"{val}: {cnt}")

    print_separator("KROK 4 — PROCESSED: KOMBINACJE SEGMENTU 5 + 6")
    for (s5, s6), cnt in processed["segment_5_6_combo"].most_common():
        print(f"{s5} + {s6}: {cnt}")

    # -------------------------
    # KROK 5 — RAW REFERENCJA
    # -------------------------
    print_separator("KROK 5 — RAW: ROZKŁAD LICZBY SEGMENTÓW")
    for k, v in sorted(raw["segment_counts"].items()):
        print(f"{k} segmentów: {v} plików")

    print_separator("KROK 5A — RAW: SEGMENT 5")
    for val, cnt in raw["segment_5_counter"].most_common():
        print(f"{val}: {cnt}")

    print_separator("KROK 5B — RAW: SEGMENT 6")
    for val, cnt in raw["segment_6_counter"].most_common():
        print(f"{val}: {cnt}")

    print_separator("KROK 5C — RAW: KOMBINACJE SEGMENTU 5 + 6")
    for (s5, s6), cnt in raw["segment_5_6_combo"].most_common():
        print(f"{s5} + {s6}: {cnt}")

    print_separator("KROK 5D — RAW: REFERENCYJNE PARY")
    print(f"RAW pary referencyjne Embedded + NotEmbedded: {len(raw['raw_reference_pairs'])}")

    # -------------------------
    # KROK 6 — PROCESSED VS RAW
    # -------------------------
    processed_right_left = processed_df[processed_df["view"].isin(["Right", "Left"])].copy()

    processed_fish_groups = defaultdict(set)
    for _, row in processed_right_left.iterrows():
        processed_fish_groups[row["fish_key"]].add(row["process"])

    processed_fish_pairs = {
        fish_key
        for fish_key, classes in processed_fish_groups.items()
        if "Embedded" in classes and "NotEmbedded" in classes
    }

    raw_reference_pairs = raw["raw_reference_pairs"]

    correct_fish_pairs = processed_fish_pairs & raw_reference_pairs
    extra_fish_pairs = processed_fish_pairs - raw_reference_pairs
    missing_fish_pairs = raw_reference_pairs - processed_fish_pairs

    print_separator("KROK 6 — WALIDACJA PROCESSED VS RAW")
    print(f"RAW pary referencyjne:        {len(raw_reference_pairs)}")
    print(f"Processed pary fish-level:    {len(processed_fish_pairs)}")
    print(f"Poprawne (RAW ∩ Processed):   {len(correct_fish_pairs)}")
    print(f"Nadmiarowe:                   {len(extra_fish_pairs)}")
    print(f"Brakujące:                    {len(missing_fish_pairs)}")

    # -------------------------
    # KROK 7 — WYBÓR FINALNYCH PAR
    # -------------------------
    candidate_df = processed_right_left[
        processed_right_left["fish_key"].isin(raw_reference_pairs)
    ].copy()

    final_rows = []
    rejected_no_same_view = 0
    candidate_otolith_keys = 0

    for otolith_key, group in candidate_df.groupby("otolith_key"):
        candidate_otolith_keys += 1
        selected = choose_best_pair(group)
        if selected is None:
            rejected_no_same_view += 1
            continue
        final_rows.append(selected)

    final_pairs_df = pd.DataFrame(final_rows)

    if not final_pairs_df.empty:
        final_pairs_df = final_pairs_df.sort_values(
            ["prefix", "fish_index", "single", "selected_view"]
        ).reset_index(drop=True)

    final_pairs_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print_separator("KROK 7 — FINALNE PARY PROCESSED")
    print(f"Kandydaci otolith-level po filtrze RAW:   {candidate_otolith_keys}")
    print(f"Odrzucone bez wspólnego view:             {rejected_no_same_view}")
    print(f"Finalne pary zapisane do CSV:             {len(final_pairs_df)}")

    print_separator("KROK 8 — DLACZEGO TE PARY")
    print("1. RAW najpierw wyznacza, które fish_key naprawdę mają Embedded + NotEmbedded.")
    print("2. Processed bierzemy tylko dla fish_key potwierdzonych w RAW.")
    print("3. W Processed schodzimy poziom niżej: ten sam otolit = ten sam Single.")
    print("4. Finalna para to dokładnie 2 zdjęcia: jedno Embedded i jedno NotEmbedded.")
    print("5. Zachowujemy tylko pary ze wspólnym view, preferując Right przed Left.")
    print("6. Dzięki temu odrzucamy pary niepewne i zostawiamy tylko spójne 2-zdjęciowe dopasowania.")

    # -------------------------
    # KROK 9 — KOPIOWANIE DO embedded/not_embedded + populacja
    # -------------------------
    copy_pairs_to_population_folders(final_pairs_df)

    print_separator("ZAPISANO")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    analyze_and_export()