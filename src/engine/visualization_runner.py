from __future__ import annotations

import base64
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ..data_loader.transforms import get_eval_transform
from ..models.checkpoint_utils import load_model_checkpoint
from ..models.model import build_model
from ..models.model_heatmaps import GradCAM, GradCAMPP, GuidedBackprop
from ..utils.config_helpers import get_active_model_name

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class _ClassificationWrapper(torch.nn.Module):
    """Opakowuje model multitaskowy — zwraca tylko logity klasyfikacji.
    GradCAM wymaga pojedynczego tensora wyjściowego do backward pass."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


class VisualizationRunner:
    def __init__(self, cfg, project_root: Path, model_path: Optional[Path] = None):
        self.cfg = cfg
        self.project_root = Path(project_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_populations = list(cfg.data.active_populations)

        # Ładowanie modelu
        checkpoint_path = model_path or (self.project_root / cfg.prediction.model_path)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Nie znaleziono checkpointu: {checkpoint_path}\n"
                "Zaktualizuj prediction.model_path w src/config/config.yaml."
            )

        self.model = build_model(cfg).to(self.device)
        self.model.eval()
        load_model_checkpoint(self.model, checkpoint_path, self.device)
        print(f"[Visualizer] Model: {get_active_model_name(cfg)}, checkpoint: {checkpoint_path}")

        # Wrapper do GradCAM (obsługuje multitask)
        self._clf_wrapper = _ClassificationWrapper(self.model).to(self.device)
        self._clf_wrapper.eval()

        self.transform = get_eval_transform(cfg)
        self.vis_cfg = cfg.visualization

        # Inicjalizacja metod wizualizacji
        requested = [m.lower() for m in self.vis_cfg.methods]
        self.generators = {}
        target_layer = self.vis_cfg.target_layer

        if "gradcam" in requested:
            try:
                self.generators["gradcam"] = GradCAM(self._clf_wrapper, target_layer)
            except ValueError as e:
                print(f"[Visualizer] UWAGA: GradCAM niedostępny — {e}")

        if "gradcam++" in requested:
            try:
                self.generators["gradcam++"] = GradCAMPP(self._clf_wrapper, target_layer)
            except ValueError as e:
                print(f"[Visualizer] UWAGA: GradCAM++ niedostępny — {e}")

        if "guided_backprop" in requested:
            try:
                gb = GuidedBackprop(self._clf_wrapper)
                relu_count = sum(1 for m in self._clf_wrapper.modules()
                                 if isinstance(m, torch.nn.ReLU))
                if relu_count == 0:
                    print("[Visualizer] INFO: Guided Backprop — model nie używa ReLU "
                          "(np. Swin używa GELU). Saliency mapy będą oparte na zwykłych "
                          "gradientach wejściowych.")
                self.generators["guided_backprop"] = gb
            except Exception as e:
                print(f"[Visualizer] UWAGA: GuidedBackprop niedostępny — {e}")

        if not self.generators:
            raise RuntimeError("Żadna metoda wizualizacji nie jest dostępna. "
                               "Sprawdź target_layer w konfiguracji.")

        print(f"[Visualizer] Aktywne metody: {list(self.generators.keys())}")

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def run(self, image_dir: Path, output_dir: Path, max_images: Optional[int] = None,
            generate_html: bool = True) -> list[dict]:
        """Przetwarza obrazy z image_dir, zapisuje heatmapy i zwraca listę wyników."""
        image_paths = self._collect_images(image_dir)
        if max_images:
            image_paths = image_paths[:max_images]

        if not image_paths:
            print(f"[Visualizer] Brak obrazów w: {image_dir}")
            return []

        print(f"[Visualizer] Przetwarzanie {len(image_paths)} obrazów → {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for idx, (img_path, true_pop) in enumerate(image_paths, start=1):
            print(f"  [{idx}/{len(image_paths)}] {img_path.name}", end=" ")
            try:
                result = self._process_image(img_path, true_pop, idx, output_dir)
                results.append(result)
                correct = "✓" if result["pred_population"] == result["true_population"] else "✗"
                age_str = f"wiek={result['pred_age']:.1f}" if result["pred_age"] is not None else ""
                print(f"→ pop={result['pred_population']} {correct} conf={result['confidence']:.0f}% {age_str}")
            except Exception as e:
                print(f"→ BŁĄD: {e}")

        self._save_csv(results, output_dir / "predictions.csv")

        if generate_html:
            self._generate_html(results, output_dir)
            print(f"[Visualizer] Raport HTML: {output_dir / 'report.html'}")

        return results

    # ------------------------------------------------------------------
    # Wewnętrzne metody
    # ------------------------------------------------------------------

    def _collect_images(self, image_dir: Path) -> list[tuple[Path, Optional[int]]]:
        """Zbiera pliki obrazów z katalogu. Jeśli podfoldery = numery populacji,
        wyciąga true_pop z nazwy folderu."""
        image_dir = Path(image_dir)
        items = []

        # Sprawdź czy mamy strukturę pop/
        subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
        pop_dirs = []
        for d in subdirs:
            try:
                pop_dirs.append((d, int(d.name)))
            except ValueError:
                pass

        if pop_dirs:
            for pop_dir, pop_num in sorted(pop_dirs):
                for f in sorted(pop_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS:
                        items.append((f, pop_num))
        else:
            for f in sorted(image_dir.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS:
                    items.append((f, None))

        return items

    def _process_image(self, img_path: Path, true_pop: Optional[int],
                       idx: int, output_dir: Path) -> dict:
        img_pil = Image.open(img_path).convert("RGB")
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Predykcja
        pred = self._predict(tensor)
        class_idx = self.active_populations.index(pred["population"])

        # Obraz oryginalny (w rozmiarze po resize/crop)
        img_np = self._tensor_to_bgr(tensor[0])
        h, w = img_np.shape[:2]

        # Heatmapy
        heatmap_paths = {}
        best_overlay = None
        best_method = list(self.generators.keys())[0]

        for method_name, generator in self.generators.items():
            try:
                heatmap = generator.generate(tensor, class_idx)
                heatmap_resized = cv2.resize(heatmap, (w, h))
                colormap_id = getattr(cv2, f"COLORMAP_{self.vis_cfg.colormap.upper()}",
                                      cv2.COLORMAP_JET)
                heatmap_color = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8), colormap_id
                )
                overlay = cv2.addWeighted(img_np, 1 - self.vis_cfg.alpha,
                                          heatmap_color, self.vis_cfg.alpha, 0)

                safe_name = method_name.replace("+", "p").replace(" ", "_")
                stem = f"{idx:04d}_pop{true_pop or 'X'}_{img_path.stem}"

                hm_path = output_dir / f"{stem}_{safe_name}.jpg"
                cv2.imwrite(str(hm_path), overlay)
                heatmap_paths[method_name] = hm_path

                if best_overlay is None:
                    best_overlay = overlay
                    best_method = method_name

            except Exception as e:
                print(f"\n    [UWAGA] {method_name}: {e}", end="")

        # Zapis oryginału
        stem = f"{idx:04d}_pop{true_pop or 'X'}_{img_path.stem}"
        orig_path = output_dir / f"{stem}_original.jpg"
        cv2.imwrite(str(orig_path), img_np)

        return {
            "index": idx,
            "filename": img_path.name,
            "true_population": true_pop,
            "pred_population": pred["population"],
            "confidence": pred["confidence"],
            "pred_age": pred["age"],
            "correct": true_pop == pred["population"] if true_pop is not None else None,
            "original_path": orig_path,
            "heatmap_paths": heatmap_paths,
            "best_method": best_method,
        }

    def _predict(self, tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            output = self.model(tensor)

        if isinstance(output, tuple):
            logits, age_tensor = output
            age = float(age_tensor[0].cpu())
        else:
            logits = output
            age = None

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx]) * 100
        population = self.active_populations[pred_idx]

        return {"population": population, "confidence": confidence, "age": age}

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalizuje tensor ImageNet i konwertuje do BGR uint8."""
        mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD).reshape(3, 1, 1)
        img = tensor.cpu().numpy() * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _save_csv(self, results: list[dict], csv_path: Path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "filename", "true_population", "pred_population",
                             "correct", "confidence_pct", "pred_age"])
            for r in results:
                writer.writerow([
                    r["index"], r["filename"], r["true_population"],
                    r["pred_population"],
                    "TAK" if r["correct"] else ("NIE" if r["correct"] is False else ""),
                    f"{r['confidence']:.1f}",
                    f"{r['pred_age']:.2f}" if r["pred_age"] is not None else "",
                ])

    def _generate_html(self, results: list[dict], output_dir: Path):
        def img_to_b64(path: Path) -> str:
            if not path.exists():
                return ""
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        methods = list(self.generators.keys())
        method_headers = "".join(f"<th>{m}</th>" for m in methods)

        rows_html = ""
        for r in results:
            orig_b64 = img_to_b64(r["original_path"])
            hm_cells = ""
            for m in methods:
                hm_path = r["heatmap_paths"].get(m)
                if hm_path and hm_path.exists():
                    b64 = img_to_b64(hm_path)
                    hm_cells += f'<td><img src="data:image/jpeg;base64,{b64}"></td>'
                else:
                    hm_cells += "<td>—</td>"

            correct_icon = "✓" if r["correct"] else ("✗" if r["correct"] is False else "")
            correct_color = "#2ecc71" if r["correct"] else ("#e74c3c" if r["correct"] is False else "#aaa")
            age_str = f" | Wiek: <b>{r['pred_age']:.1f}</b>" if r["pred_age"] is not None else ""
            caption = (
                f"<span style='color:{correct_color};font-weight:bold'>{correct_icon}</span> "
                f"Pop. prawdziwa: {r['true_population'] or '?'} | "
                f"Predykcja: <b>{r['pred_population']}</b> | "
                f"Pewność: <b>{r['confidence']:.0f}%</b>{age_str}"
            )

            rows_html += f"""
            <tr>
              <td>
                <img src="data:image/jpeg;base64,{orig_b64}">
                <div class="caption">{r['filename']}</div>
              </td>
              {hm_cells}
              <td class="caption-cell">{caption}</td>
            </tr>"""

        model_name = get_active_model_name(self.cfg)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        accuracy = ""
        scored = [r for r in results if r["correct"] is not None]
        if scored:
            acc = sum(1 for r in scored if r["correct"]) / len(scored) * 100
            accuracy = f"<p>Dokładność: <b>{acc:.1f}%</b> ({sum(1 for r in scored if r['correct'])}/{len(scored)})</p>"

        html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>GradCAM — {model_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
  h1 {{ color: #e94560; }}
  p {{ color: #aaa; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #16213e; color: #e94560; padding: 8px 12px; text-align: center; }}
  td {{ padding: 6px; border-bottom: 1px solid #333; text-align: center; vertical-align: middle; }}
  td img {{ max-width: 200px; max-height: 200px; border-radius: 4px; }}
  .caption {{ font-size: 11px; color: #aaa; margin-top: 4px; }}
  .caption-cell {{ font-size: 13px; text-align: left; min-width: 200px; }}
  tr:hover td {{ background: #16213e; }}
</style>
</head>
<body>
<h1>GradCAM — {model_name}</h1>
<p>Wygenerowano: {timestamp} | Obrazów: {len(results)}</p>
{accuracy}
<table>
  <thead>
    <tr>
      <th>Oryginał</th>
      {method_headers}
      <th>Predykcja</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</body>
</html>"""

        with open(output_dir / "report.html", "w", encoding="utf-8") as f:
            f.write(html)
