from pathlib import Path
import re
import pandas as pd

# =========================
# CONFIG
# =========================

DATA_DIR = Path(r"Z:\Photo\Otolithes\HER\Processed")

# =========================
# PARSING
# =========================

def parse_filename(file_name: str):
    name = Path(file_name).name

    pattern = re.compile(
        r"^(?P<prefix>.+?)_"
        r"(?P<process>Embedded|NotEmbedded)_"
        r"(?P<variant>[^_]+)_"
        r"FishIndex(?P<fish_index>\d+)_"
        r"Single(?P<single>\d+)_"
        r"(?P<view>[^_]+)"
        r"\.jpg$",
        flags=re.IGNORECASE,
    )

    match = pattern.match(name)
    if not match:
        return None

    prefix = match.group("prefix")
    process = match.group("process")
    variant = match.group("variant")
    fish_index = match.group("fish_index")
    single = match.group("single")
    view = match.group("view").lower()

    cls = "embedded" if process.lower() == "embedded" else "not_embedded"

    fish_id = f"{prefix}_FishIndex{fish_index}"
    pair_key = f"{prefix}_FishIndex{fish_index}"

    return {
        "file_name": name,
        "class": cls,
        "prefix": prefix,
        "variant": variant,
        "fish_index": fish_index,
        "single": f"single{single}",
        "view": view,
        "fish_id": fish_id,
        "pair_key": pair_key,
    }


# =========================
# LOAD FILES
# =========================

def load_files():
    all_files = list(DATA_DIR.rglob("*"))
    jpg_files = [p for p in all_files if p.is_file() and p.suffix.lower() == ".jpg"]

    records = []

    for path in jpg_files:
        parsed = parse_filename(path.name)
        if parsed:
            parsed["file_path"] = str(path.resolve())
            records.append(parsed)

    df = pd.DataFrame(records)
    return df, len(all_files), len(jpg_files)


# =========================
# HELPERS
# =========================

def print_separator(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_nunique(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].nunique())


# =========================
# BUILD
# =========================

def build():
    df, total_all_files, total_jpg_files = load_files()

    if df.empty:
        print("Brak rozpoznanych danych.")
        return

    recognized_before_filter = len(df)

    # tylko Right / Left
    df = df[df["view"].isin(["right", "left"])].copy()

    embedded_full = df[df["class"] == "embedded"].copy()
    not_embedded_full = df[df["class"] == "not_embedded"].copy()

    embedded_keys = set(embedded_full["pair_key"])
    not_embedded_keys = set(not_embedded_full["pair_key"])
    common_keys = embedded_keys & not_embedded_keys

    embedded_unpaired = embedded_full[~embedded_full["pair_key"].isin(common_keys)].copy()
    not_embedded_unpaired = not_embedded_full[
        ~not_embedded_full["pair_key"].isin(common_keys)
    ].copy()

    common_df = df[df["pair_key"].isin(common_keys)].copy()

    # =========================
    # RAPORT PODSTAWOWY
    # =========================

    print_separator("TABELA 1 — WEJŚCIE")
    print(f"Wszystkie pliki:                  {total_all_files}")
    print(f"Wszystkie JPG:                    {total_jpg_files}")
    print(f"Rozpoznane rekordy:               {recognized_before_filter}")
    print(f"Odrzucone przez parser:           {total_jpg_files - recognized_before_filter}")
    print(f"Odrzucone przez filtr view:       {recognized_before_filter - len(df)}")
    print(f"Po filtrze Right/Left:            {len(df)}")

    print_separator("TABELA 2 — ZBIORY")
    print(f"Embedded:     {len(embedded_full)}")
    print(f"NotEmbedded:  {len(not_embedded_full)}")

    print_separator("TABELA 3 — WSPÓLNY ZBIÓR")
    print(f"PARY (common):        {len(common_keys)}")
    print(f"Ryb w common:         {safe_nunique(common_df, 'fish_id')}")
    print(f"Embedded bez pary:    {len(embedded_unpaired)}")
    print(f"NotEmbedded bez pary: {len(not_embedded_unpaired)}")

    # =========================
    # ANALIZA BRAKU PAR
    # =========================

    print_separator("ANALIZA BRAKU PAR — NOT EMBEDDED")

    embedded_fish_ids = set(embedded_full["fish_id"])
    missing_fish = 0

    for _, row in not_embedded_unpaired.iterrows():
        fish_id = row["fish_id"]
        if fish_id not in embedded_fish_ids:
            missing_fish += 1

    print(f"Brak EMBEDDED dla fish_id:        {missing_fish}")

    print_separator("ANALIZA BRAKU PAR — EMBEDDED")

    not_embedded_fish_ids = set(not_embedded_full["fish_id"])
    missing_fish_e = 0

    for _, row in embedded_unpaired.iterrows():
        fish_id = row["fish_id"]
        if fish_id not in not_embedded_fish_ids:
            missing_fish_e += 1

    print(f"Brak NOT EMBEDDED dla fish_id:    {missing_fish_e}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    build()