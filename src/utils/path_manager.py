from pathlib import Path
from datetime import datetime

class PathManager:
    def __init__(self, project_root: Path, cfg):
        self.project_root = project_root.resolve()
        self.cfg = cfg
        self.experiment_date = datetime.now().strftime("%Y-%m-%d")

    def config_path(self) -> Path:
        return self.project_root / "src" / "config" / "config.yaml"

    def metadata_file(self) -> Path:
        return self._resolve(self.cfg.data.metadata_file, subdir="src")

    def data_root(self) -> Path:
        return self._resolve(self.cfg.data.root_dir)

    def checkpoint_dir(self) -> Path:
        return self._resolve(self.cfg.training.checkpoint_dir)

    def model_path(self) -> Path:
        return self._resolve(self.cfg.prediction.model_path)

    def results_dir(self) -> Path:
        return self.project_root / "results"

    def logs_dir(self) -> Path:
        return self.results_dir() / "logs"

    def excel_predictions_output(self) -> Path:
        filename = f"all_predictions_{self.experiment_date}.xlsx"
        return self.project_root / "src" / "data_loader" / filename

    def _resolve(self, path_str: str, subdir: str = None) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        base = self.project_root / subdir if subdir else self.project_root
        return (base / path).resolve()
