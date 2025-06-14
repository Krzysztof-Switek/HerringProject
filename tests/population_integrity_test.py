from pathlib import Path
from omegaconf import OmegaConf
from src.utils.path_manager import PathManager
import pandas as pd


def load_metadata_df():
    config_path = Path(__file__).parent.parent / "src" / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    project_root = Path(__file__).parent.parent
    path_manager = PathManager(project_root=project_root, cfg=cfg)
    df = pd.read_excel(path_manager.metadata_file())
    df.columns = df.columns.str.lower()
    return df



def test_population_separation_if_required():
    df = load_metadata_df()
    df_train = df[df["set"].str.lower() == "train"]
    allowed_pops = {1, 2}
    actual_pops = set(df_train["populacja"].unique())
    assert actual_pops.issubset(allowed_pops), f"Niedozwolone populacje: {actual_pops - allowed_pops}"


def test_wiek_value_range():
    df = load_metadata_df()
    df_train = df[df["set"].str.lower() == "train"]
    invalid_wiek = df_train[~df_train["wiek"].between(1, 20)]
    assert invalid_wiek.empty, f"Znaleziono nieprawidłowe wartości wieku:\n{invalid_wiek[['wiek', 'populacja', 'set']]}"

