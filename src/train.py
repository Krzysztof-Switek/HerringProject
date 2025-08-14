import torch
from pathlib import Path
from omegaconf import OmegaConf
from utils.path_manager import PathManager
from engine.trainer_setup import run_training_loop
from utils.population_mapper import PopulationMapper

from omegaconf import DictConfig

class Trainer:
    def __init__(self, project_root: Path = None, config_path_override: str = None, config_override: DictConfig = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        print(f"\nProject root: {self.project_root}")
        self.log_dir = None # Atrybut do przechowywania ścieżki logów dla Optuny

        # Priorytet dla obiektu config_override
        if config_override:
            self.cfg = config_override
            print("INFO: Konfiguracja załadowana z przekazanego obiektu OmegaConf.")
        else:
            self.cfg = self._load_config(config_path_override)

        # NOWA LOGIKA: Ustaw debug_mode na podstawie wczytanej konfiguracji
        self.debug_mode = self.cfg.get('training', {}).get('stop_after_one_epoch', False)

        if self.debug_mode:
            print("🔥 Uruchomiono w trybie DEBUG (na podstawie 'stop_after_one_epoch: true' w konfiguracji) 🔥")
            # Upewnij się, że liczba epok jest 1, jeśli debug_mode jest aktywny
            if self.cfg.training.get('epochs', 1) > 1:
                 print(f"   - Wymuszanie liczby epok na 1 w trybie debug.")
                 self.cfg.training.epochs = 1

        self.population_mapper = PopulationMapper(self.cfg.data.active_populations)
        self.path_manager = PathManager(self.project_root, self.cfg)
        self.device = self._init_device()
        print(f"Using device: {self.device}")
        self._validate_data_structure()
        self.model = None
        self.data_loader = None
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
        # ZMODYFIKOWANA METODA: Przechwytuje i zwraca wynik
        best_score, final_log_dir = run_training_loop(self)
        self.log_dir = final_log_dir  # Zapisz ścieżkę do logów
        return best_score