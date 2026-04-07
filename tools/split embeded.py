from pathlib import Path
import pandas as pd
import shutil

# =========================
# CONFIG
# =========================

CSV_PATH = Path(r"C:\Users\kswitek\Documents\HerringProject\tools\processed_pairs.csv")

OUTPUT_DIR = Path(r"C:\Users\kswitek\Documents\HerringProject\final_pairs")

EMBEDDED_DIR = OUTPUT_DIR / "embedded"
NOT_EMBEDDED_DIR = OUTPUT_DIR / "not_embedded"


# =========================
# COPY LOGIC
# =========================

def copy_files():
    if not CSV_PATH.exists():
        print("Brak pliku CSV:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)
    NOT_EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)

    copied_embedded = 0
    copied_not_embedded = 0

    for _, row in df.iterrows():

        emb_src = Path(row["embedded_file_path"])
        not_emb_src = Path(row["not_embedded_file_path"])

        emb_dst = EMBEDDED_DIR / emb_src.name
        not_emb_dst = NOT_EMBEDDED_DIR / not_emb_src.name

        if emb_src.exists():
            shutil.copy2(emb_src, emb_dst)
            copied_embedded += 1
        else:
            print(f"Brak pliku EMBEDDED: {emb_src}")

        if not_emb_src.exists():
            shutil.copy2(not_emb_src, not_emb_dst)
            copied_not_embedded += 1
        else:
            print(f"Brak pliku NOT EMBEDDED: {not_emb_src}")

    print("\n" + "=" * 60)
    print("PODSUMOWANIE KOPIOWANIA")
    print("=" * 60)
    print(f"Embedded skopiowane:     {copied_embedded}")
    print(f"NotEmbedded skopiowane:  {copied_not_embedded}")
    print(f"Katalog docelowy:        {OUTPUT_DIR.resolve()}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    copy_files()