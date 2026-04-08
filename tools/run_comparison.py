#!/usr/bin/env python3
"""
run_comparison.py — Trening porownawczy: Embedded vs Not-Embedded.

Uruchamia pelny trening dla obu typow zdjec (embedded i not_embedded)
korzystajac z tego samego podzialu ryb (train/val/test). Generuje raport
porownujacy metryki obu modeli.

WARUNKI WSTEPNE:
  1. Dane podzielone — katalogi final_pairs/embedded/ i final_pairs/not_embedded/
     z podkatalogami train/val/test/{1,2}/ (uruchom data_split_pipeline.py)
  2. Metadata wygenerowana — src/data_loader/metadata_embedded.xlsx
     i src/data_loader/metadata_not_embedded.xlsx (uruchom etap 7 w pipeline)

UZYCIE:
  .venv\\Scripts\\python tools\\run_comparison.py           # pelny trening obu modeli
  .venv\\Scripts\\python tools\\run_comparison.py --embedded-only
  .venv\\Scripts\\python tools\\run_comparison.py --not-embedded-only
  .venv\\Scripts\\python tools\\run_comparison.py --report-only  # tylko raport (po treningu)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sciezki do danych treningowych (data/ = konwencja projektu)
EMBEDDED_ROOT = PROJECT_ROOT / "data" / "embedded"
NOT_EMBEDDED_ROOT = PROJECT_ROOT / "data" / "not_embedded"

METADATA_EMB = SRC_ROOT / "data_loader" / "metadata_embedded.xlsx"
METADATA_NOT = SRC_ROOT / "data_loader" / "metadata_not_embedded.xlsx"

CONFIG_PATH = SRC_ROOT / "config" / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"


# ================================================================================
# WALIDACJA PRZED TRENINGIEM
# ================================================================================

def validate_prerequisites(image_type: str):
    """Sprawdza czy dane i metadata istnieja przed uruchomieniem treningu."""
    errors = []

    root = EMBEDDED_ROOT if image_type == "embedded" else NOT_EMBEDDED_ROOT
    meta = METADATA_EMB if image_type == "embedded" else METADATA_NOT

    for split in ["train", "val"]:
        for pop in ["1", "2"]:
            d = root / split / pop
            if not d.exists():
                errors.append(f"Brak katalogu: {d}")
            elif not any(d.iterdir()):
                errors.append(f"Pusty katalog: {d}")

    if not meta.exists():
        errors.append(
            f"Brak pliku metadata: {meta}\n"
            f"  Uruchom: .venv\\Scripts\\python tools\\data_split_pipeline.py --steps 7"
        )
    else:
        df = pd.read_excel(meta, engine="openpyxl")
        for col in ["FileName", "Populacja", "Wiek"]:
            if col not in df.columns:
                errors.append(f"Brak kolumny '{col}' w {meta.name}")

    if errors:
        print(f"\nBLAD — brakuje wymaganych danych dla '{image_type}':")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"OK: Walidacja '{image_type}' — dane gotowe.")
    return True


# ================================================================================
# URUCHOMIENIE TRENINGU
# ================================================================================

def run_training(image_type: str) -> Path | None:
    """
    Uruchamia pelny trening dla danego typu zdjec.
    Zwraca sciezke do katalogu wynikowego (results/logs/...) lub None przy bledzie.
    """
    print(f"\n{'=' * 72}")
    print(f"TRENING: {image_type.upper()}")
    print(f"{'=' * 72}")

    if not validate_prerequisites(image_type):
        return None

    root = EMBEDDED_ROOT if image_type == "embedded" else NOT_EMBEDDED_ROOT
    meta = METADATA_EMB if image_type == "embedded" else METADATA_NOT

    # Zaladuj bazowy config i nadpisz sciezki dla tego image_type
    cfg = OmegaConf.load(CONFIG_PATH)

    # Sciezki relatywne do project_root (dla PathManager)
    cfg.data.root_dir = str(root.relative_to(PROJECT_ROOT)).replace("\\", "/")
    # Sciezka relatywna do src/ (dla PathManager.metadata_file)
    cfg.data.metadata_file = str(meta.relative_to(SRC_ROOT)).replace("\\", "/")

    # Upewnij sie ze stop_after_one_epoch jest false
    if cfg.training.get("stop_after_one_epoch", False):
        print("WARN: stop_after_one_epoch=true w config — nadpisuje na false dla treningu!")
        cfg.training.stop_after_one_epoch = False

    # Zapisz tymczasowy config (potrzebny do PathManager.config_path())
    tmp_config_path = TOOLS_DIR / f"_tmp_config_{image_type}.yaml"
    OmegaConf.save(cfg, tmp_config_path)

    timestamp_before = datetime.now()

    try:
        from src.engine.trainer import Trainer

        trainer = Trainer(
            config_path=str(tmp_config_path),
            project_root=PROJECT_ROOT,
        )
        trainer.train()

        print(f"\nOK: Trening '{image_type}' zakończony.")

    except Exception as e:
        print(f"\nBLAD podczas treningu '{image_type}': {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp_config_path.exists():
            tmp_config_path.unlink()

    # Znajdz katalog wynikowy (najnowszy po timestamp_before)
    log_dir = _find_latest_log_dir(timestamp_before)
    if log_dir:
        print(f"Wyniki zapisane w: {log_dir}")
    else:
        print("WARN: Nie znaleziono katalogu wynikowego — sprawdz results/logs/")

    return log_dir


def _find_latest_log_dir(after: datetime) -> Path | None:
    """Zwraca najnowszy katalog w results/logs/ utworzony po podanym czasie."""
    logs_root = RESULTS_DIR / "logs"
    if not logs_root.exists():
        return None

    candidates = [
        d for d in logs_root.iterdir()
        if d.is_dir() and datetime.fromtimestamp(d.stat().st_ctime) >= after
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda d: d.stat().st_ctime)


# ================================================================================
# RAPORT POROWNAWCZY
# ================================================================================

def _load_best_metrics(log_dir: Path) -> dict | None:
    """
    Wczytuje najlepsze metryki z pliku training_metrics.csv w katalogu wynikowym.
    Najlepsza epoka = maksimum Val Accuracy.
    """
    csv_files = list(log_dir.glob("*_training_metrics.csv"))
    if not csv_files:
        print(f"  WARN: Brak pliku metrics CSV w {log_dir}")
        return None

    # Moze byc wiele plikow (jeden na loss_type) — laczone
    all_rows = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)

    # Znajdz kolumny z metrykami
    val_acc_col = next((c for c in df.columns if "Val Accuracy" in c), None)
    val_f1_col = next((c for c in df.columns if "Val F1" in c), None)
    val_prec_col = next((c for c in df.columns if "Val Precision" in c), None)
    val_rec_col = next((c for c in df.columns if "Val Recall" in c), None)
    val_auc_col = next((c for c in df.columns if "Val AUC" in c), None)
    val_loss_col = next((c for c in df.columns if "Val Loss" in c and "Classification" not in c and "Regression" not in c), None)

    if val_acc_col is None:
        print(f"  WARN: Brak kolumny 'Val Accuracy' w {csv_path.name}")
        return None

    best_idx = df[val_acc_col].idxmax()
    best_row = df.loc[best_idx]

    return {
        "log_dir": str(log_dir),
        "epochs_trained": len(df),
        "best_epoch": best_row.get("Epoch", best_idx),
        "val_accuracy": round(float(best_row[val_acc_col]), 4) if val_acc_col else None,
        "val_f1": round(float(best_row[val_f1_col]), 4) if val_f1_col else None,
        "val_precision": round(float(best_row[val_prec_col]), 4) if val_prec_col else None,
        "val_recall": round(float(best_row[val_rec_col]), 4) if val_rec_col else None,
        "val_auc": round(float(best_row[val_auc_col]), 4) if val_auc_col else None,
        "val_loss": round(float(best_row[val_loss_col]), 6) if val_loss_col else None,
    }


def generate_comparison_report(emb_log_dir: Path | None, not_log_dir: Path | None):
    """Generuje raport porownawczy embedded vs not_embedded."""
    print(f"\n{'=' * 72}")
    print("RAPORT POROWNAWCZY: EMBEDDED vs NOT-EMBEDDED")
    print(f"{'=' * 72}")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for label, log_dir in [("embedded", emb_log_dir), ("not_embedded", not_log_dir)]:
        if log_dir is None:
            print(f"  WARN: Brak katalogu wynikow dla '{label}' — pomijam.")
            results[label] = None
            continue

        if not log_dir.exists():
            # Sprobuj znalezc recznie przez nazwe (jesli user podaje sciezke)
            print(f"  WARN: Katalog nie istnieje: {log_dir}")
            results[label] = None
            continue

        metrics = _load_best_metrics(log_dir)
        if metrics:
            results[label] = metrics
            print(f"\n  [{label.upper()}]")
            for k, v in metrics.items():
                if k not in ("log_dir", "best_epoch"):
                    print(f"    {k:20s}: {v}")

    # Zapisz CSV
    rows = []
    for label, metrics in results.items():
        if metrics is None:
            rows.append({"image_type": label, **{k: None for k in
                ["epochs_trained", "val_accuracy", "val_f1", "val_precision",
                 "val_recall", "val_auc", "val_loss"]}})
        else:
            rows.append({"image_type": label, **{k: v for k, v in metrics.items()
                                                  if k != "log_dir"}})

    df_report = pd.DataFrame(rows)
    csv_out = COMPARISON_DIR / f"comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    df_report.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\nZapisano: {csv_out}")

    # Zapisz MD
    _write_comparison_md(results, df_report)


def _write_comparison_md(results: dict, df: pd.DataFrame):
    emb = results.get("embedded")
    not_e = results.get("not_embedded")

    def fmt(val, other_val, higher_is_better=True):
        """Dodaje strzalke gdy wartosc jest lepsza."""
        if val is None or other_val is None:
            return str(val)
        if higher_is_better:
            marker = " <--" if val > other_val else ""
        else:
            marker = " <--" if val < other_val else ""
        return f"{val}{marker}"

    lines = [
        "# Raport porownawczy: Embedded vs Not-Embedded",
        "",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Cel",
        "Porownanie skutecznosci modeli klasyfikacji populacji sledzia",
        "trenowanych na zdjeciach zalewanych (embedded) vs niezalewanych (not-embedded).",
        "Oba modele uzywaly tych samych ryb w train/val/test.",
        "",
        "## Wyniki (najlepsza epoka wg Val Accuracy)",
        "",
        "| Metryka | Embedded | Not-Embedded |",
        "|---------|----------|--------------|",
    ]

    metrics_config = [
        ("Val Accuracy (%)", "val_accuracy", True),
        ("Val F1", "val_f1", True),
        ("Val Precision", "val_precision", True),
        ("Val Recall", "val_recall", True),
        ("Val AUC", "val_auc", True),
        ("Val Loss", "val_loss", False),
        ("Epoki treningu", "epochs_trained", None),
    ]

    for label, key, higher_better in metrics_config:
        e_val = emb.get(key) if emb else None
        n_val = not_e.get(key) if not_e else None

        if higher_better is not None:
            e_str = fmt(e_val, n_val, higher_better)
            n_str = fmt(n_val, e_val, higher_better)
        else:
            e_str = str(e_val) if e_val is not None else "-"
            n_str = str(n_val) if n_val is not None else "-"

        lines.append(f"| {label} | {e_str} | {n_str} |")

    lines += [
        "",
        "Strzalka `<--` wskazuje lepszy wynik.",
        "",
        "## Katalogi wynikow",
        "",
    ]
    if emb:
        lines.append(f"- Embedded:     `{emb.get('log_dir', '-')}`")
    if not_e:
        lines.append(f"- Not-embedded: `{not_e.get('log_dir', '-')}`")

    md_out = COMPARISON_DIR / f"comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Zapisano: {md_out}")


# ================================================================================
# SZUKANIE ISTNIEJACYCH WYNIKOW (dla --report-only)
# ================================================================================

def find_existing_log_dirs() -> tuple[Path | None, Path | None]:
    """
    Probuje znalezc najnowsze katalogi wynikow dla embedded i not_embedded
    na podstawie zawartosci (sciezki data_dir w metrics CSV lub nazwy katalogu).
    """
    logs_root = RESULTS_DIR / "logs"
    if not logs_root.exists():
        return None, None

    emb_dir = None
    not_dir = None

    # Sortuj od najnowszego
    for d in sorted(logs_root.iterdir(), key=lambda x: x.stat().st_ctime, reverse=True):
        if not d.is_dir():
            continue
        # Szukaj metrics CSV i sprawdz jakie dane byly uzywane
        csv_files = list(d.glob("*_training_metrics.csv"))
        if not csv_files:
            continue

        name = d.name.lower()
        # Heurystyka na podstawie nazwy katalogu (zawiera timestamp)
        # Lepsze: sprawdz config saved w katalogu lub przyjmij ze user poda sciezki
        if emb_dir is None:
            emb_dir = d
        elif not_dir is None:
            not_dir = d
            break

    return emb_dir, not_dir


# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trening porownawczy: Embedded vs Not-Embedded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przyklady:
  python tools/run_comparison.py                        # pelny trening obu modeli
  python tools/run_comparison.py --embedded-only        # tylko embedded
  python tools/run_comparison.py --not-embedded-only    # tylko not-embedded
  python tools/run_comparison.py --report-only          # tylko raport (po treningu)
  python tools/run_comparison.py --report-only \\
    --emb-log results/logs/resnet50_focal_2025-01-01 \\
    --not-log results/logs/resnet50_focal_2025-01-02
        """,
    )
    parser.add_argument("--embedded-only", action="store_true")
    parser.add_argument("--not-embedded-only", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Generuj tylko raport na podstawie istniejacych wynikow")
    parser.add_argument("--emb-log", type=str, default=None,
                        help="Sciezka do katalogu wynikow embedded (dla --report-only)")
    parser.add_argument("--not-log", type=str, default=None,
                        help="Sciezka do katalogu wynikow not_embedded (dla --report-only)")
    args = parser.parse_args()

    if args.report_only:
        emb_log = Path(args.emb_log) if args.emb_log else None
        not_log = Path(args.not_log) if args.not_log else None

        if emb_log is None or not_log is None:
            print("WARN: Nie podano sciezek --emb-log i --not-log.")
            print("  Probuje znalezc automatycznie najnowsze wyniki...")
            emb_log, not_log = find_existing_log_dirs()
            if emb_log:
                print(f"  Embedded log:     {emb_log}")
            if not_log:
                print(f"  Not-embedded log: {not_log}")

        generate_comparison_report(emb_log, not_log)
        return

    emb_log_dir = None
    not_log_dir = None

    if not args.not_embedded_only:
        emb_log_dir = run_training("embedded")

    if not args.embedded_only:
        not_log_dir = run_training("not_embedded")

    generate_comparison_report(emb_log_dir, not_log_dir)

    print("\n" + "=" * 72)
    print("ZAKONCZONE")
    print("=" * 72)
    if emb_log_dir:
        print(f"  Embedded wyniki:     {emb_log_dir}")
    if not_log_dir:
        print(f"  Not-embedded wyniki: {not_log_dir}")
    print(f"  Raport porownawczy:  {COMPARISON_DIR}")


if __name__ == "__main__":
    main()