import torch
from pathlib import Path
from omegaconf import OmegaConf
from data_loader.dataset import HerringDataset
from utils.path_manager import PathManager
from engine.trainer_setup import run_training_loop

class Trainer:
    def __init__(self, config_path: str = None, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        print(f"\nProject root: {self.project_root}")
        self.cfg = self._load_config(config_path)
        self.path_manager = PathManager(self.project_root, self.cfg)
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        self._validate_data_structure()
        self.model = None
        self.data_loader = HerringDataset(self.cfg)
        self.last_model_path = None

    def _load_config(self, config_path: str = None):
        if config_path is None:
            temp_path_manager = PathManager(self.project_root, cfg=None)
            config_path = temp_path_manager.config_path()
        return OmegaConf.load(config_path)

    def _init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_data_structure(self):
        print("\nValidating data structure...")
        for split in ["train", "val"]:
            split_path = self.path_manager.data_root() / split
            if not split_path.exists():
                raise FileNotFoundError(f"Missing directory: {split_path}")
            if not any(split_path.iterdir()):
                raise RuntimeError(f"Katalog {split_path} istnieje, ale jest pusty")
        print("Data structure validated.")

    def train(self):
        run_training_loop(self)