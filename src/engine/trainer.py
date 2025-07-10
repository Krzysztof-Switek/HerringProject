import torch
from pathlib import Path
from omegaconf import OmegaConf
from data_loader.dataset import HerringDataset
from utils.path_manager import PathManager
from engine.trainer_setup import run_training_loop
from utils.population_mapper import PopulationMapper

class Trainer:
    def __init__(self, project_root: Path = None, config_path_override: str = None, debug_mode: bool = False):
        self.project_root = project_root or Path(__file__).parent.parent
        print(f"\nProject root: {self.project_root}")
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("🔥 Uruchomiono w trybie DEBUG 🔥")

        self.cfg = self._load_config(config_path_override) # Przekazujemy config_path_override

        # Modyfikacja konfiguracji, jeśli debug_mode jest True
        if self.debug_mode:
            print("   Modyfikowanie konfiguracji dla trybu DEBUG:")
            # Ustawienie flagi stop_after_one_epoch, jeśli istnieje w konfiguracji
            if 'training' in self.cfg and hasattr(self.cfg.training, 'stop_after_one_epoch'):
                print(f"   - training.stop_after_one_epoch: {self.cfg.training.stop_after_one_epoch} -> True")
                self.cfg.training.stop_after_one_epoch = True
            else: # Jeśli nie ma, dodajmy ją
                if 'training' not in self.cfg:
                    OmegaConf.update(self.cfg, "training", {}, merge=True)
                print(f"   - training.stop_after_one_epoch: (brak) -> True (dodano)")
                OmegaConf.update(self.cfg.training, "stop_after_one_epoch", True, merge=True)

            # Zmniejszenie liczby epok, jeśli stop_after_one_epoch nie jest używane lub dla pewności
            if 'training' in self.cfg and hasattr(self.cfg.training, 'epochs'):
                 print(f"   - training.epochs: {self.cfg.training.epochs} -> 1 (lub min(aktualna, 1))")
                 self.cfg.training.epochs = 1 # Można też min(self.cfg.training.epochs, 1 lub 2)

            # Można dodać inne modyfikacje, np. zmniejszenie batch_size
            # if 'data' in self.cfg and hasattr(self.cfg.data, 'batch_size'):
            #     print(f"   - data.batch_size: {self.cfg.data.batch_size} -> min({self.cfg.data.batch_size}, 4)")
            #     self.cfg.data.batch_size = min(self.cfg.data.batch_size, 4) # Np. max 4
            print("   Konfiguracja zmodyfikowana.")

        self.population_mapper = PopulationMapper(self.cfg.data.active_populations)
        self.path_manager = PathManager(self.project_root, self.cfg)
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        self._validate_data_structure()
        self.model = None
        self.data_loader = HerringDataset(self.cfg, population_mapper=self.population_mapper)
        self.last_model_path = None

    def _load_config(self, config_path_override: str = None):
        # Użyj config_path_override jeśli jest dostępny, w przeciwnym razie domyślna ścieżka
        if config_path_override is not None:
            final_config_path = self.project_root / config_path_override # Zakładamy, że jest to ścieżka względna do roota lub absolutna
            if not final_config_path.is_file():
                 # Spróbuj jako ścieżkę absolutną, jeśli nie jest to ścieżka względna do roota
                final_config_path = Path(config_path_override)
                if not final_config_path.is_file():
                    raise FileNotFoundError(f"Plik konfiguracyjny '{config_path_override}' nie został znaleziony (sprawdzono jako względny i absolutny).")
            print(f"Ładowanie konfiguracji z (override): {final_config_path}")
        else:
            # Użyj domyślnej ścieżki z PathManager
            temp_path_manager = PathManager(self.project_root, cfg=None) # cfg=None, bo jeszcze go nie mamy
            final_config_path = temp_path_manager.config_path()
            print(f"Ładowanie konfiguracji z (domyślna): {final_config_path}")

        if not final_config_path.exists():
            raise FileNotFoundError(f"Plik konfiguracyjny nie istnieje: {final_config_path}")

        return OmegaConf.load(final_config_path)

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