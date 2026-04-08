#!/usr/bin/env python3
"""
run_comparison.py - Trening porownawczy: Embedded vs Not-Embedded.

Uruchamia pelny trening dla obu typow zdjec (embedded i not_embedded)
korzystajac z tego samego podzialu ryb (train/val/test). Generuje raport
porownujacy metryki obu modeli.

WARUNKI WSTEPNE:
  1. Dane podzielone - katalogi data/embedded/ i data/not_embedded/
     z podkatalogami train/val/test/{1,2}/ (uruchom data_split_pipeline.py)
  2. Metadata wygenerowana - src/data_loader/metadata_embedded.xlsx
     i src/data_loader/metadata_not_embedded.xlsx (uruchom etap 7 w pipeline)

UZYCIE:
  .venv\\Scripts\\python tools\\run_comparison.py           # pelny trening obu modeli
  .venv\\Scripts\\python tools\\run_comparison.py --embedded-only
  .venv\\Scripts\\python tools\\run_comparison.py --not-embedded-only
  .venv\\Scripts\\python tools\\run_comparison.py --report-only  # tylko raport (po treningu)
  .venv\\Scripts\\python tools\\run_comparison.py --report-only \\
    --emb-log results/logs/resnet50_ldam_multi_2026-01-01 \\
    --not-log results/logs/resnet50_ldam_multi_2026-01-02
"""

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

def validate_prerequisites(image_type: str) -> bool:
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
        print(f"\nBLAD - brakuje wymaganych danych dla '{image_type}':")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"OK: Walidacja '{image_type}' - dane gotowe.")
    return True


# ================================================================================
# URUCHOMIENIE TRENINGU
# ================================================================================

def run_training(image_type: str) -> Path | None:
    print(f"\n{'=' * 72}")
    print(f"TRENING: {image_type.upper()}")
    print(f"{'=' * 72}")

    if not validate_prerequisites(image_type):
        return None

    root = EMBEDDED_ROOT if image_type == "embedded" else NOT_EMBEDDED_ROOT
    meta = METADATA_EMB if image_type == "embedded" else METADATA_NOT

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.data.root_dir = str(root.relative_to(PROJECT_ROOT)).replace("\\", "/")
    cfg.data.metadata_file = str(meta.relative_to(SRC_ROOT)).replace("\\", "/")

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

        print(f"\nOK: Trening '{image_type}' zakonczony.")

    except Exception as e:
        print(f"\nBLAD podczas treningu '{image_type}': {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp_config_path.exists():
            tmp_config_path.unlink()

    log_dir = _find_latest_log_dir(timestamp_before)
    if log_dir:
        # Oznacz log_dir aby pozniej wiedziec, ze to jest embedded/not_embedded
        _save_run_info(log_dir, image_type)
        print(f"Wyniki zapisane w: {log_dir}")
    else:
        print("WARN: Nie znaleziono katalogu wynikowego - sprawdz results/logs/")

    return log_dir


def _save_run_info(log_dir: Path, image_type: str):
    """Zapisuje metadane runu do log_dir (ulatwia pozniejsze szukanie)."""
    info = {
        "image_type": image_type,
        "log_dir": str(log_dir),
        "timestamp": datetime.now().isoformat(),
    }
    (log_dir / "_run_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )


def _find_latest_log_dir(after: datetime) -> Path | None:
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
# WCZYTYWANIE DANYCH Z LOG_DIR
# ================================================================================

def _load_metrics_df(log_dir: Path) -> pd.DataFrame | None:
    """Wczytuje metrics CSV z log_dir. Dodaje kolumne 'epoch_num' (1, 2, 3...)."""
    csv_files = list(log_dir.glob("*_training_metrics.csv"))
    if not csv_files:
        return None

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df["epoch_num"] = range(1, len(df) + 1)
    return df


def _load_predictions_df(log_dir: Path) -> pd.DataFrame | None:
    """Wczytuje predictions Excel z log_dir."""
    xlsx_files = list(log_dir.glob("*_predictions.xlsx"))
    if not xlsx_files:
        return None
    return pd.read_excel(xlsx_files[0])


def _extract_loss_name(pred_df: pd.DataFrame) -> str | None:
    """Wyciaga nazwe funkcji straty z nazw kolumn predictions.
    Szuka kolumny {loss}_pred - wyklucza {loss}_age_pred.
    """
    pred_cols = [c for c in pred_df.columns
                 if c.endswith("_pred") and not c.endswith("_age_pred")]
    return pred_cols[0][:-5] if pred_cols else None


def _compute_predictions_metrics(pred_df: pd.DataFrame) -> dict:
    """
    Liczy metryki z predictions Excel per split (train/val/test):
      - accuracy, per-populacja accuracy
      - confusion matrix (lista list)
      - confidence stats (srednia, mediana, %>=80%, %<60%)
      - age MAE i RMSE (jezeli multitask)
    """
    loss_name = _extract_loss_name(pred_df)
    if loss_name is None:
        return {}

    pred_col = f"{loss_name}_pred"
    prob_col = f"{loss_name}_prob"
    age_col = f"{loss_name}_age_pred"

    results = {}

    for split in ["train", "val", "test", "all"]:
        if split == "all":
            subset = pred_df.copy()
        elif "set" in pred_df.columns:
            subset = pred_df[pred_df["set"] == split].copy()
        elif split == "val":
            subset = pred_df.copy()
        else:
            continue

        if len(subset) == 0:
            continue

        if pred_col not in subset.columns or "Populacja" not in subset.columns:
            continue

        valid = subset.dropna(subset=[pred_col, "Populacja"])
        if len(valid) == 0:
            continue

        all_pops = sorted(pred_df["Populacja"].dropna().unique())

        acc = (valid[pred_col] == valid["Populacja"]).mean() * 100

        per_class = {}
        for pop in all_pops:
            pop_sub = valid[valid["Populacja"] == pop]
            if len(pop_sub) > 0:
                per_class[int(pop)] = round(
                    (pop_sub[pred_col] == pop_sub["Populacja"]).mean() * 100, 2
                )

        # Macierz pomylek
        cm = []
        for true_pop in all_pops:
            row = []
            true_sub = valid[valid["Populacja"] == true_pop]
            for pred_pop in all_pops:
                row.append(int((true_sub[pred_col] == pred_pop).sum()))
            cm.append(row)

        # Statystyki pewnosci predykcji
        conf_stats = {}
        if prob_col in valid.columns:
            probs = valid[prob_col].dropna()
            if len(probs) > 0:
                conf_stats = {
                    "mean": round(float(probs.mean()), 1),
                    "median": round(float(probs.median()), 1),
                    "pct_high": round(float((probs >= 80).mean() * 100), 1),
                    "pct_low": round(float((probs < 60).mean() * 100), 1),
                }

        # Age regression (multitask)
        age_stats = {}
        if age_col in subset.columns and "Wiek" in subset.columns:
            age_valid = subset.dropna(subset=[age_col, "Wiek"])
            if len(age_valid) > 0:
                errors = age_valid[age_col] - age_valid["Wiek"]
                rounded_pred = age_valid[age_col].round().astype(int)
                age_acc = round(float((rounded_pred == age_valid["Wiek"]).mean() * 100), 2)

                # Per-age-group stats
                per_age = {}
                for true_age in sorted(age_valid["Wiek"].unique()):
                    grp = age_valid[age_valid["Wiek"] == true_age]
                    grp_err = grp[age_col] - grp["Wiek"]
                    per_age[int(true_age)] = {
                        "n": int(len(grp)),
                        "mae": round(float(grp_err.abs().mean()), 2),
                        "bias": round(float(grp_err.mean()), 2),
                        "most_common_pred": int(grp[age_col].round().mode().iloc[0]),
                    }

                # Macierz pomylek wieku (true_age x rounded_pred_age)
                all_ages = sorted(age_valid["Wiek"].unique())
                pred_ages_unique = sorted(
                    set(int(x) for x in age_valid[age_col].round())
                )
                all_age_labels = sorted(set(all_ages) | set(pred_ages_unique))
                age_cm = []
                for true_age in all_ages:
                    grp = age_valid[age_valid["Wiek"] == true_age]
                    row_vals = []
                    for pa in all_age_labels:
                        row_vals.append(int((grp[age_col].round().astype(int) == pa).sum()))
                    age_cm.append(row_vals)

                age_stats = {
                    "mae": round(float(errors.abs().mean()), 3),
                    "rmse": round(float((errors ** 2).mean() ** 0.5), 3),
                    "bias": round(float(errors.mean()), 3),
                    "accuracy_rounded": age_acc,
                    "per_age": per_age,
                    "age_cm": age_cm,
                    "age_cm_true_labels": [int(a) for a in all_ages],
                    "age_cm_pred_labels": [int(a) for a in all_age_labels],
                }

        results[split] = {
            "n": int(len(valid)),
            "accuracy": round(float(acc), 2),
            "per_class": per_class,
            "cm": cm,
            "cm_labels": [int(p) for p in all_pops],
            "confidence": conf_stats,
            "age": age_stats,
        }

    return results


def _load_best_val_metrics(metrics_df: pd.DataFrame) -> dict:
    """Najlepsze metryki walidacyjne z CSV (z epoki o max Val Accuracy)."""
    val_acc_col = next((c for c in metrics_df.columns if "Val Accuracy" in c), None)
    if val_acc_col is None:
        return {}

    idx = metrics_df[val_acc_col].idxmax()
    row = metrics_df.loc[idx]

    def _get(col_substr):
        col = next((c for c in metrics_df.columns if col_substr in c), None)
        return round(float(row[col]), 4) if col else None

    # Sumaryczny czas treningu
    time_col = next((c for c in metrics_df.columns if "Train Time" in c), None)
    total_time_min = None
    if time_col:
        total_sec = metrics_df[time_col].sum()
        total_time_min = round(total_sec / 60, 1)

    return {
        "best_epoch": int(row.get("epoch_num", idx + 1)),
        "total_epochs": len(metrics_df),
        "total_time_min": total_time_min,
        "val_accuracy": _get("Val Accuracy"),
        "val_f1": _get("Val F1"),
        "val_precision": _get("Val Precision"),
        "val_recall": _get("Val Recall"),
        "val_auc": _get("Val AUC"),
        "val_loss": _get("Val Loss"),
        "train_accuracy": _get("Train Accuracy"),
        "train_loss": _get("Train Loss"),
        "train_f1": _get("Train F1"),
    }


# ================================================================================
# WYKRESY POROWNAWCZE
# ================================================================================

def _fig_to_base64(fig) -> str:
    """Zamienia figure matplotlib na base64 PNG do osadzenia w HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _generate_comparison_plots(
    emb_df: pd.DataFrame | None,
    not_df: pd.DataFrame | None,
) -> dict[str, str]:
    """
    Generuje wykresy porownawcze (overlaid) dla obu typow zdjec.
    Zwraca slownik {tytuł -> base64_png}.
    """
    plots = {}

    metrics_config = [
        ("Accuracy (%)", "Val Accuracy", "Train Accuracy"),
        ("Loss", "Val Loss", "Train Loss"),
        ("F1", "Val F1", "Train F1"),
        ("AUC", "Val AUC", "Train AUC"),
    ]

    for metric_label, val_col_substr, train_col_substr in metrics_config:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Porownanie: {metric_label}", fontsize=13, fontweight="bold")

        for ax, col_substr, curve_label in [
            (axes[0], val_col_substr, "Walidacja"),
            (axes[1], train_col_substr, "Trening"),
        ]:
            for df, label, color in [
                (emb_df, "Embedded", "#1f77b4"),
                (not_df, "Not-Embedded", "#ff7f0e"),
            ]:
                if df is None or df.empty:
                    continue
                col = next((c for c in df.columns if col_substr in c), None)
                if col is None:
                    continue
                epochs = df["epoch_num"].tolist()
                values = df[col].tolist()
                ax.plot(epochs, values, label=label, color=color, marker="o",
                        markersize=3, linewidth=1.5)

            ax.set_title(curve_label)
            ax.set_xlabel("Epoka")
            ax.set_ylabel(metric_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plots[metric_label] = _fig_to_base64(fig)

    return plots


def _generate_multitask_loss_plots(
    emb_df: pd.DataFrame | None,
    not_df: pd.DataFrame | None,
) -> dict[str, str]:
    """Dodatkowe wykresy dla trybu multitask (Classification + Regression loss)."""
    plots = {}

    for col_substr, title in [
        ("Classification Loss", "Classification Loss"),
        ("Regression Loss", "Regression Loss"),
    ]:
        # Sprawdz czy kolumna istnieje w ktorymkolwiek df
        has_col = any(
            df is not None and any(col_substr in c for c in df.columns)
            for df in [emb_df, not_df]
        )
        if not has_col:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Multitask: {title}", fontsize=13, fontweight="bold")

        for ax, split_prefix, curve_label in [
            (axes[0], "Val", "Walidacja"),
            (axes[1], "Train", "Trening"),
        ]:
            for df, label, color in [
                (emb_df, "Embedded", "#1f77b4"),
                (not_df, "Not-Embedded", "#ff7f0e"),
            ]:
                if df is None or df.empty:
                    continue
                col = next(
                    (c for c in df.columns
                     if col_substr in c and split_prefix in c),
                    None,
                )
                if col is None:
                    continue
                epochs = df["epoch_num"].tolist()
                values = df[col].tolist()
                ax.plot(epochs, values, label=label, color=color, marker="o",
                        markersize=3, linewidth=1.5)

            ax.set_title(curve_label)
            ax.set_xlabel("Epoka")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plots[title] = _fig_to_base64(fig)

    return plots


def _generate_age_plots(
    emb_pred_df: pd.DataFrame | None,
    not_pred_df: pd.DataFrame | None,
) -> dict[str, str]:
    """
    Generuje wykresy analizy predykcji wieku:
    - Scatter plot (predykcja vs prawda) osobno dla embedded i not_embedded
    - Histogram bledow predykcji (bias distribution)
    """
    plots = {}

    # --- Scatter plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predykcja wieku: wartosci przewidziane vs rzeczywiste",
                 fontsize=13, fontweight="bold")

    for ax, pred_df, model_label in [
        (axes[0], emb_pred_df, "Embedded"),
        (axes[1], not_pred_df, "Not-Embedded"),
    ]:
        if pred_df is None or pred_df.empty:
            ax.set_title(f"{model_label} — brak danych")
            continue

        loss_name = _extract_loss_name(pred_df)
        age_col = f"{loss_name}_age_pred" if loss_name else None
        if age_col is None or age_col not in pred_df.columns or "Wiek" not in pred_df.columns:
            ax.set_title(f"{model_label} — brak kolumny age_pred")
            continue

        colors = {"train": "#aec7e8", "val": "#1f77b4", "test": "#d62728", "all": "#999"}
        for split, color in [("train", "#aec7e8"), ("val", "#1f77b4"), ("test", "#d62728")]:
            if "set" in pred_df.columns:
                sub = pred_df[pred_df["set"] == split].dropna(subset=[age_col, "Wiek"])
            else:
                sub = pred_df.dropna(subset=[age_col, "Wiek"])
            if len(sub) == 0:
                continue
            ax.scatter(sub["Wiek"], sub[age_col], alpha=0.35, s=12,
                       color=color, label=split)

        # Linia idealnej predykcji
        all_ages = pred_df["Wiek"].dropna()
        if len(all_ages) > 0:
            age_min, age_max = int(all_ages.min()), int(all_ages.max())
            ax.plot([age_min, age_max], [age_min, age_max],
                    "k--", linewidth=1.2, label="idealna")

        ax.set_title(f"{model_label}")
        ax.set_xlabel("Wiek rzeczywisty (lata)")
        ax.set_ylabel("Wiek przewidziany (lata)")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots["Scatter: wiek przewidziany vs rzeczywisty"] = _fig_to_base64(fig)

    # --- Histogram bledow ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Rozklad bledow predykcji wieku (pred - prawda)",
                  fontsize=13, fontweight="bold")

    for ax, pred_df, model_label in [
        (axes2[0], emb_pred_df, "Embedded"),
        (axes2[1], not_pred_df, "Not-Embedded"),
    ]:
        if pred_df is None or pred_df.empty:
            continue
        loss_name = _extract_loss_name(pred_df)
        age_col = f"{loss_name}_age_pred" if loss_name else None
        if age_col is None or age_col not in pred_df.columns:
            continue

        age_valid = pred_df.dropna(subset=[age_col, "Wiek"])
        if len(age_valid) == 0:
            continue

        errors = age_valid[age_col] - age_valid["Wiek"]
        ax.hist(errors, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="blad=0")
        ax.axvline(float(errors.mean()), color="orange", linestyle="--",
                   linewidth=1.5, label=f"srednia={errors.mean():.2f}")
        ax.set_title(model_label)
        ax.set_xlabel("Blad predykcji (lata)")
        ax.set_ylabel("Liczba probek")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots["Histogram bledow predykcji wieku"] = _fig_to_base64(fig2)

    # --- MAE per wiek ---
    fig3, ax3 = plt.subplots(figsize=(11, 4))
    fig3.suptitle("MAE predykcji wieku per grupa wiekowa",
                  fontsize=13, fontweight="bold")

    for pred_df, model_label, color in [
        (emb_pred_df, "Embedded", "#1f77b4"),
        (not_pred_df, "Not-Embedded", "#ff7f0e"),
    ]:
        if pred_df is None or pred_df.empty:
            continue
        loss_name = _extract_loss_name(pred_df)
        age_col = f"{loss_name}_age_pred" if loss_name else None
        if age_col is None or age_col not in pred_df.columns:
            continue
        age_valid = pred_df.dropna(subset=[age_col, "Wiek"])
        if len(age_valid) == 0:
            continue
        ages = sorted(age_valid["Wiek"].unique())
        maes = [float(abs(age_valid[age_valid["Wiek"] == a][age_col] - a).mean())
                for a in ages]
        ax3.plot(ages, maes, marker="o", label=model_label, color=color, linewidth=1.8)

    ax3.set_xlabel("Wiek rzeczywisty (lata)")
    ax3.set_ylabel("MAE (lata)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plots["MAE predykcji wieku per grupa wiekowa"] = _fig_to_base64(fig3)

    return plots


# ================================================================================
# RAPORT HTML
# ================================================================================

def _better(a, b, higher_is_better=True):
    """Zwraca True jesli a jest lepsze od b."""
    if a is None or b is None:
        return False
    return (a > b) if higher_is_better else (a < b)


def _fmt(val, ref=None, higher_is_better=True, suffix=""):
    if val is None:
        return "<td>-</td>"
    text = f"{val}{suffix}"
    style = ""
    if ref is not None and _better(val, ref, higher_is_better):
        style = ' style="font-weight:bold; color:#1a7a1a;"'
    return f"<td{style}>{text}</td>"


def _age_cm_html(age_stats: dict, label: str) -> str:
    """Macierz pomylek dla predykcji wieku (true_age x rounded_pred_age)."""
    cm = age_stats.get("age_cm")
    true_labels = age_stats.get("age_cm_true_labels", [])
    pred_labels = age_stats.get("age_cm_pred_labels", [])
    if not cm or not true_labels:
        return ""

    header = "<tr><th>Prawda \\ Pred</th>" + "".join(
        f"<th>{p}</th>" for p in pred_labels
    ) + "</tr>"
    body = ""
    for i, true_age in enumerate(true_labels):
        total = sum(cm[i])
        cells = ""
        for j, val in enumerate(cm[i]):
            is_diag = (pred_labels[j] == true_age)
            pct = f"<br><small>{val/total*100:.0f}%</small>" if total > 0 else ""
            style = ' style="background:#d4edda; font-weight:bold;"' if is_diag else (
                ' style="background:#f8d7da;"' if val > 0 else ""
            )
            cells += f"<td{style}>{val}{pct}</td>"
        body += f"<tr><th>{true_age}</th>{cells}</tr>"

    return f"""
    <p style="margin-bottom:4px;"><b>Macierz pomylek wieku — {label}</b>
    <small>(wiersze = wiek prawdziwy, kolumny = wiek przewidziany po zaokragleniu)</small></p>
    <div style="overflow-x:auto;">
    <table style="width:auto; font-size:0.85em;">
    {header}{body}
    </table></div>"""


def _per_age_table_html(emb_age: dict, not_age: dict) -> str:
    """Tabela MAE/bias per grupa wiekowa — Embedded vs Not-Embedded."""
    per_age_emb = emb_age.get("per_age", {})
    per_age_not = not_age.get("per_age", {})
    all_ages = sorted(set(list(per_age_emb.keys()) + list(per_age_not.keys())))
    if not all_ages:
        return ""

    rows = ""
    for age in all_ages:
        e = per_age_emb.get(age, {})
        n = per_age_not.get(age, {})
        e_mae = e.get("mae")
        n_mae = n.get("mae")
        e_n = e.get("n", "-")
        n_n = n.get("n", "-")
        e_bias = e.get("bias")
        n_bias = n.get("bias")
        e_mode = e.get("most_common_pred", "-")
        n_mode = n.get("most_common_pred", "-")

        # Pogrubienie lepszego MAE
        def fmt_mae(val, ref):
            if val is None:
                return "<td>-</td>"
            style = ' style="font-weight:bold; color:#1a7a1a;"' if (
                ref is not None and val < ref
            ) else ""
            return f"<td{style}>{val}</td>"

        rows += (
            f"<tr><td>{age}</td>"
            f"<td>{e_n}</td>{fmt_mae(e_mae, n_mae)}<td>{e_bias}</td><td>{e_mode}</td>"
            f"<td>{n_n}</td>{fmt_mae(n_mae, e_mae)}<td>{n_bias}</td><td>{n_mode}</td>"
            f"</tr>"
        )

    return f"""
    <table style="font-size:0.9em;">
    <tr>
      <th rowspan="2">Wiek</th>
      <th colspan="4" style="background:#dbeafe;">Embedded</th>
      <th colspan="4" style="background:#fef3c7;">Not-Embedded</th>
    </tr>
    <tr>
      <th style="background:#dbeafe;">N</th>
      <th style="background:#dbeafe;">MAE</th>
      <th style="background:#dbeafe;">Bias</th>
      <th style="background:#dbeafe;">Najcz. pred.</th>
      <th style="background:#fef3c7;">N</th>
      <th style="background:#fef3c7;">MAE</th>
      <th style="background:#fef3c7;">Bias</th>
      <th style="background:#fef3c7;">Najcz. pred.</th>
    </tr>
    {rows}
    </table>"""


def _build_age_section_html(
    emb_pred_metrics: dict,
    not_pred_metrics: dict,
    row_fn,
    age_plots: dict[str, str] | None = None,
) -> str:
    """Buduje cala sekcje HTML poswiecona predykcji wieku."""
    section_title = '<h2 style="margin-top:2em; border-bottom:2px solid #ccc;">Regresja wieku (multitask)</h2>'

    # Tabela podsumowujaca metryki wieku per split
    splits_order = [("all", "Caly zbior"), ("test", "Test"), ("val", "Walidacja")]
    summary_rows = ""
    for split_key, split_label in splits_order:
        emb_s = emb_pred_metrics.get(split_key, {})
        not_s = not_pred_metrics.get(split_key, {})
        ea = emb_s.get("age", {})
        na = not_s.get("age", {})
        if not ea and not na:
            continue
        summary_rows += f'<tr><td><b>{split_label}</b></td>'
        summary_rows += f'<td>{ea.get("mae", "-")}</td>'
        summary_rows += f'<td>{ea.get("rmse", "-")}</td>'
        summary_rows += f'<td>{ea.get("bias", "-")}</td>'
        summary_rows += f'<td>{ea.get("accuracy_rounded", "-")}%</td>'
        summary_rows += f'<td>{na.get("mae", "-")}</td>'
        summary_rows += f'<td>{na.get("rmse", "-")}</td>'
        summary_rows += f'<td>{na.get("bias", "-")}</td>'
        summary_rows += f'<td>{na.get("accuracy_rounded", "-")}%</td>'
        summary_rows += '</tr>'

    if not summary_rows:
        return ""

    summary_table = f"""
    <table style="font-size:0.9em;">
    <tr>
      <th rowspan="2">Zbior</th>
      <th colspan="4" style="background:#dbeafe;">Embedded</th>
      <th colspan="4" style="background:#fef3c7;">Not-Embedded</th>
    </tr>
    <tr>
      <th style="background:#dbeafe;">MAE</th>
      <th style="background:#dbeafe;">RMSE</th>
      <th style="background:#dbeafe;">Bias</th>
      <th style="background:#dbeafe;">Acc (zaokr.)</th>
      <th style="background:#fef3c7;">MAE</th>
      <th style="background:#fef3c7;">RMSE</th>
      <th style="background:#fef3c7;">Bias</th>
      <th style="background:#fef3c7;">Acc (zaokr.)</th>
    </tr>
    {summary_rows}
    </table>
    <p style="font-size:0.85em; color:#555;">
    MAE = sredni blad bezwzgledny [lata] &nbsp;|&nbsp;
    RMSE = pierwiastek ze sredniokwadratowego bledu &nbsp;|&nbsp;
    Bias = systematyczne przeszacowanie (+) lub niedoszacowanie (-) &nbsp;|&nbsp;
    Acc (zaokr.) = % trafnych predykcji po zaokragleniu do najblizszego roku
    </p>"""

    # Tabela per-wiek (caly zbior)
    emb_all = emb_pred_metrics.get("all", {})
    not_all = not_pred_metrics.get("all", {})
    per_age_html = _per_age_table_html(emb_all.get("age", {}), not_all.get("age", {}))

    # Macierze pomylek wieku (test + all)
    cm_html_parts = []
    for split_key, split_label in [("test", "Test"), ("all", "Caly zbior")]:
        emb_s = emb_pred_metrics.get(split_key, {})
        not_s = not_pred_metrics.get(split_key, {})
        if emb_s.get("age", {}).get("age_cm") or not_s.get("age", {}).get("age_cm"):
            cm_html_parts.append(f"<h3>Macierze pomylek wieku — {split_label}</h3>")
            cm_html_parts.append('<div style="display:flex; gap:2em; flex-wrap:wrap;">')
            cm_html_parts.append(f'<div>{_age_cm_html(emb_s.get("age", {}), "Embedded")}</div>')
            cm_html_parts.append(f'<div>{_age_cm_html(not_s.get("age", {}), "Not-Embedded")}</div>')
            cm_html_parts.append('</div>')
            cm_html_parts.append('<hr style="margin:1.5em 0; border:1px solid #eee;">')

    # Wykresy analizy wieku
    age_plot_html = ""
    if age_plots:
        for plot_title, b64 in age_plots.items():
            age_plot_html += (
                f'<h3>{plot_title}</h3>'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%; border:1px solid #ddd; border-radius:4px;"/>'
            )

    return f"""
{section_title}
<h3>Podsumowanie metryk regresji</h3>
{summary_table}
<hr style="margin:2em 0; border:1px solid #ccc;">
<h3>Analiza per grupa wiekowa (caly zbior)</h3>
{per_age_html if per_age_html else "<p><i>Brak danych.</i></p>"}
<hr style="margin:2em 0; border:1px solid #ccc;">
{"".join(cm_html_parts)}
{age_plot_html}
"""


def _write_comparison_html(
    emb_best: dict,
    not_best: dict,
    emb_pred_metrics: dict,
    not_pred_metrics: dict,
    plots: dict[str, str],
    age_plots: dict[str, str],
    emb_log_dir: Path | None,
    not_log_dir: Path | None,
    timestamp: str,
):
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    html_path = COMPARISON_DIR / f"comparison_{timestamp}.html"

    def row(label, e_val, n_val, higher=True, suffix=""):
        return (
            f"<tr><td>{label}</td>"
            f"{_fmt(e_val, n_val, higher, suffix)}"
            f"{_fmt(n_val, e_val, higher, suffix)}"
            f"</tr>"
        )

    def section(title):
        return f'<h2 style="margin-top:2em; border-bottom:2px solid #ccc;">{title}</h2>'

    # --- Tabela: metryki z CSV (najlepsza epoka walidacyjna) ---
    emb_e = emb_best.get
    not_e = not_best.get

    csv_table = f"""
    <table>
    <tr><th>Metryka (CSV — najlepsza epoka val)</th><th>Embedded</th><th>Not-Embedded</th></tr>
    {row("Najlepsza epoka", emb_e("best_epoch"), not_e("best_epoch"), higher=False)}
    {row("Liczba epok", emb_e("total_epochs"), not_e("total_epochs"), higher=False)}
    {row("Czas treningu (min)", emb_e("total_time_min"), not_e("total_time_min"), higher=False)}
    {row("Val Accuracy (%)", emb_e("val_accuracy"), not_e("val_accuracy"), higher=True, suffix="%")}
    {row("Val F1", emb_e("val_f1"), not_e("val_f1"), higher=True)}
    {row("Val Precision", emb_e("val_precision"), not_e("val_precision"), higher=True)}
    {row("Val Recall", emb_e("val_recall"), not_e("val_recall"), higher=True)}
    {row("Val AUC", emb_e("val_auc"), not_e("val_auc"), higher=True)}
    {row("Val Loss", emb_e("val_loss"), not_e("val_loss"), higher=False)}
    {row("Train Accuracy (%)", emb_e("train_accuracy"), not_e("train_accuracy"), higher=True, suffix="%")}
    {row("Train Loss", emb_e("train_loss"), not_e("train_loss"), higher=False)}
    {row("Train F1", emb_e("train_f1"), not_e("train_f1"), higher=True)}
    </table>
    """

    def _cm_html(s: dict, label: str) -> str:
        """Generuje macierz pomylek jako HTML table."""
        cm = s.get("cm")
        labels = s.get("cm_labels", [])
        if not cm or not labels:
            return ""
        header = "<tr><th>True \\ Pred</th>" + "".join(f"<th>Pop {p}</th>" for p in labels) + "</tr>"
        body = ""
        for i, true_pop in enumerate(labels):
            total = sum(cm[i])
            cells = ""
            for j, val in enumerate(cm[i]):
                pct = f" ({val/total*100:.0f}%)" if total > 0 else ""
                style = ' style="background:#d4edda;"' if i == j else ' style="background:#f8d7da;"'
                cells += f"<td{style}>{val}{pct}</td>"
            body += f"<tr><th>Pop {true_pop}</th>{cells}</tr>"
        return f"""
        <p><b>Macierz pomylek — {label}</b></p>
        <table style="width:auto; margin-bottom:0.5em;">
        {header}{body}
        </table>"""

    # --- Tabele per split z predictions Excel ---
    def pred_split_section(s_label, s_key):
        emb_s = emb_pred_metrics.get(s_key, {})
        not_s = not_pred_metrics.get(s_key, {})

        def prow(lbl, key, higher=True, suffix=""):
            e_v = emb_s.get(key)
            n_v = not_s.get(key)
            return row(lbl, e_v, n_v, higher=higher, suffix=suffix)

        all_pops = sorted(set(
            list(emb_s.get("per_class", {}).keys()) +
            list(not_s.get("per_class", {}).keys())
        ))
        per_class_rows = ""
        for pop in all_pops:
            e_pc = emb_s.get("per_class", {}).get(pop)
            n_pc = not_s.get("per_class", {}).get(pop)
            per_class_rows += row(f"Acc Populacja {pop} (%)", e_pc, n_pc, higher=True, suffix="%")

        age_stats_rows = ""
        if emb_s.get("age") or not_s.get("age"):
            e_age = emb_s.get("age", {})
            n_age = not_s.get("age", {})
            age_stats_rows += row("Age MAE (lata)", e_age.get("mae"), n_age.get("mae"), higher=False)
            age_stats_rows += row("Age RMSE (lata)", e_age.get("rmse"), n_age.get("rmse"), higher=False)
            age_stats_rows += row("Age bias (lata)", e_age.get("bias"), n_age.get("bias"), higher=False)

        conf_rows = ""
        if emb_s.get("confidence") or not_s.get("confidence"):
            e_c = emb_s.get("confidence", {})
            n_c = not_s.get("confidence", {})
            conf_rows += row("Pewnosc srednia (%)", e_c.get("mean"), n_c.get("mean"), higher=True)
            conf_rows += row("Wysoka pewnosc >=80% (%)", e_c.get("pct_high"), n_c.get("pct_high"), higher=True)
            conf_rows += row("Niska pewnosc <60% (%)", e_c.get("pct_low"), n_c.get("pct_low"), higher=False)

        n_emb = emb_s.get("n", "-")
        n_not = not_s.get("n", "-")

        return f"""
        <h3>{s_label} (N: Embedded={n_emb}, Not-Embedded={n_not})</h3>
        <table>
        <tr><th>Metryka</th><th>Embedded</th><th>Not-Embedded</th></tr>
        {prow("Accuracy (%)", "accuracy", higher=True, suffix="%")}
        {per_class_rows}
        {age_stats_rows}
        {conf_rows}
        </table>
        <div style="display:flex; gap:2em; flex-wrap:wrap;">
          <div>{_cm_html(emb_s, "Embedded")}</div>
          <div>{_cm_html(not_s, "Not-Embedded")}</div>
        </div>
        """

    pred_sections = ""
    for split_key, split_label in [
        ("all",   "Caly zbior (train + val + test)"),
        ("test",  "Test"),
        ("val",   "Walidacja"),
        ("train", "Trening"),
    ]:
        if split_key in emb_pred_metrics or split_key in not_pred_metrics:
            pred_sections += '<hr style="margin:2em 0; border:1px solid #ccc;">'
            pred_sections += pred_split_section(split_label, split_key)

    # --- Wykresy ---
    plot_html = ""
    for title, b64 in plots.items():
        plot_html += f"""
        <h3>{title}</h3>
        <img src="data:image/png;base64,{b64}" style="max-width:100%; border:1px solid #ddd; border-radius:4px;"/>
        """

    # --- Sciezki ---
    paths_html = "<ul>"
    if emb_log_dir:
        paths_html += f"<li><b>Embedded:</b> <code>{emb_log_dir}</code></li>"
    if not_log_dir:
        paths_html += f"<li><b>Not-Embedded:</b> <code>{not_log_dir}</code></li>"
    paths_html += f"<li><b>Raport:</b> <code>{html_path}</code></li>"
    paths_html += "</ul>"

    css = """
    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background: #f0f0f0; font-weight: bold; }
    tr:nth-child(even) { background: #f9f9f9; }
    code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; }
    """

    # --- Sekcja analizy wieku ---
    age_section_html = _build_age_section_html(
        emb_pred_metrics, not_pred_metrics, row, age_plots
    )

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Raport porownawczy: Embedded vs Not-Embedded</title>
<style>{css}</style>
</head>
<body>
<h1>Raport porownawczy: Embedded vs Not-Embedded</h1>
<p><b>Data:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
<b>Legenda:</b> <span style="font-weight:bold; color:#1a7a1a;">pogrubiony</span> = lepszy wynik</p>

{section("Metryki treningowe (CSV)")}
{csv_table}

{section("Klasyfikacja populacji — predykcje na zbiorze danych")}
{pred_sections if pred_sections else "<p><i>Brak danych predictions Excel.</i></p>"}

{age_section_html}

{section("Wykresy porownawcze krzywych treningowych")}
{plot_html if plot_html else "<p><i>Brak danych do wykresu.</i></p>"}

{section("Katalogi wynikow")}
{paths_html}
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    print(f"Zapisano HTML: {html_path}")
    return html_path


# ================================================================================
# GLOWNA FUNKCJA RAPORTU
# ================================================================================

def generate_comparison_report(
    emb_log_dir: Path | None,
    not_log_dir: Path | None,
):
    print(f"\n{'=' * 72}")
    print("RAPORT POROWNAWCZY: EMBEDDED vs NOT-EMBEDDED")
    print(f"{'=' * 72}")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # --- Wczytaj dane ---
    emb_metrics_df = _load_metrics_df(emb_log_dir) if emb_log_dir else None
    not_metrics_df = _load_metrics_df(not_log_dir) if not_log_dir else None

    emb_pred_df = _load_predictions_df(emb_log_dir) if emb_log_dir else None
    not_pred_df = _load_predictions_df(not_log_dir) if not_log_dir else None

    emb_best = _load_best_val_metrics(emb_metrics_df) if emb_metrics_df is not None else {}
    not_best = _load_best_val_metrics(not_metrics_df) if not_metrics_df is not None else {}

    emb_pred_metrics = _compute_predictions_metrics(emb_pred_df) if emb_pred_df is not None else {}
    not_pred_metrics = _compute_predictions_metrics(not_pred_df) if not_pred_df is not None else {}

    # --- Wyswietl podsumowanie w konsoli ---
    for label, bst, pred_m in [
        ("EMBEDDED", emb_best, emb_pred_metrics),
        ("NOT-EMBEDDED", not_best, not_pred_metrics),
    ]:
        print(f"\n[{label}]")
        if bst:
            print(f"  Najlepsza epoka val:  {bst.get('best_epoch')} / {bst.get('total_epochs')}")
            print(f"  Val Accuracy:         {bst.get('val_accuracy')}%")
            print(f"  Val F1:               {bst.get('val_f1')}")
            print(f"  Val AUC:              {bst.get('val_auc')}")
        for split in ["test", "val"]:
            s = pred_m.get(split)
            if s:
                print(f"  {split.capitalize()} Acc (predictions): {s.get('accuracy')}%  (N={s.get('n')})")
                for pop, acc in s.get("per_class", {}).items():
                    print(f"    Populacja {pop}: {acc}%")
                age = s.get("age", {})
                if age.get("mae") is not None:
                    print(f"  {split.capitalize()} Age MAE={age.get('mae')} RMSE={age.get('rmse')} bias={age.get('bias')}")
                conf = s.get("confidence", {})
                if conf:
                    print(f"  {split.capitalize()} Confidence: mean={conf.get('mean')}% high={conf.get('pct_high')}% low={conf.get('pct_low')}%")

    # --- Wygeneruj wykresy ---
    plots = _generate_comparison_plots(emb_metrics_df, not_metrics_df)
    plots.update(_generate_multitask_loss_plots(emb_metrics_df, not_metrics_df))
    age_plots = _generate_age_plots(emb_pred_df, not_pred_df)

    # --- HTML ---
    _write_comparison_html(
        emb_best, not_best,
        emb_pred_metrics, not_pred_metrics,
        plots,
        age_plots,
        emb_log_dir, not_log_dir,
        timestamp,
    )

    # --- CSV (backward compat) ---
    rows = []
    for label, bst, pred_m in [
        ("embedded", emb_best, emb_pred_metrics),
        ("not_embedded", not_best, not_pred_metrics),
    ]:
        base = {"image_type": label, **bst}
        for split in ["train", "val", "test"]:
            s = pred_m.get(split, {})
            base[f"{split}_accuracy_pred"] = s.get("accuracy")
            base[f"{split}_n"] = s.get("n")
            age = s.get("age", {})
            base[f"{split}_age_mae"] = age.get("mae")
            base[f"{split}_age_rmse"] = age.get("rmse")
            base[f"{split}_age_bias"] = age.get("bias")
            conf = s.get("confidence", {})
            base[f"{split}_conf_mean"] = conf.get("mean")
            base[f"{split}_conf_pct_high"] = conf.get("pct_high")
            base[f"{split}_conf_pct_low"] = conf.get("pct_low")
            for pop, acc in s.get("per_class", {}).items():
                base[f"{split}_pop{pop}_accuracy"] = acc
        rows.append(base)

    df_csv = pd.DataFrame(rows)
    csv_path = COMPARISON_DIR / f"comparison_{timestamp}.csv"
    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Zapisano CSV:  {csv_path}")


# ================================================================================
# SZUKANIE ISTNIEJACYCH WYNIKOW (--report-only)
# ================================================================================

def find_existing_log_dirs() -> tuple[Path | None, Path | None]:
    """
    Szuka log_dir dla embedded i not_embedded uzywajac _run_info.json.
    Fallback: dwa najnowsze katalogi.
    """
    logs_root = RESULTS_DIR / "logs"
    if not logs_root.exists():
        return None, None

    emb_dir = None
    not_dir = None

    # Priorytet: katalogi z _run_info.json
    for d in sorted(logs_root.iterdir(), key=lambda x: x.stat().st_ctime, reverse=True):
        if not d.is_dir():
            continue
        info_file = d / "_run_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
                img_type = info.get("image_type")
                if img_type == "embedded" and emb_dir is None:
                    emb_dir = d
                elif img_type == "not_embedded" and not_dir is None:
                    not_dir = d
            except Exception:
                pass

        if emb_dir and not_dir:
            break

    # Fallback: najnowsze dwa katalogi z metrics CSV
    if emb_dir is None or not_dir is None:
        print("WARN: Brak _run_info.json - uzywam heurystyki (2 najnowsze katalogi).")
        candidates = sorted(
            [d for d in logs_root.iterdir()
             if d.is_dir() and list(d.glob("*_training_metrics.csv"))],
            key=lambda x: x.stat().st_ctime,
            reverse=True,
        )
        if len(candidates) >= 1 and emb_dir is None:
            emb_dir = candidates[0]
        if len(candidates) >= 2 and not_dir is None:
            not_dir = candidates[1]

    return emb_dir, not_dir


# ================================================================================
# MAIN
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trening porownawczy: Embedded vs Not-Embedded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embedded-only", action="store_true")
    parser.add_argument("--not-embedded-only", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Generuj tylko raport (bez treningu)")
    parser.add_argument("--emb-log", type=str, default=None,
                        help="Sciezka do log_dir embedded (dla --report-only)")
    parser.add_argument("--not-log", type=str, default=None,
                        help="Sciezka do log_dir not_embedded (dla --report-only)")
    args = parser.parse_args()

    if args.report_only:
        emb_log = Path(args.emb_log) if args.emb_log else None
        not_log = Path(args.not_log) if args.not_log else None

        if emb_log is None or not_log is None:
            print("WARN: Brak --emb-log / --not-log, szukam automatycznie...")
            emb_log, not_log = find_existing_log_dirs()
            if emb_log:
                print(f"  Embedded:     {emb_log}")
            if not_log:
                print(f"  Not-embedded: {not_log}")

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