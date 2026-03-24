from __future__ import annotations

import pandas as pd

from src.engine.trainer_logger import log_augmentation_summary


def test_log_augmentation_summary_creates_csv(tmp_path):
    augment_applied = {
        (1, 0): 12,
        (2, 1): 5,
        (2, 3): 3,
    }
    model_name = "testmodel"

    log_augmentation_summary(augment_applied, model_name, log_dir=tmp_path)

    files = list(tmp_path.glob("augmentation_summary_*.csv"))
    assert len(files) == 1, "Nie znaleziono zapisanego pliku CSV z logiem augmentacji."

    df = pd.read_csv(files[0])

    assert set(df.columns) == {"Populacja", "Wiek", "Liczba_augmentacji"}, "Niepoprawne kolumny CSV."
    assert df.shape[0] == 3, "Niepoprawna liczba wierszy w logu augmentacji."

    expected_rows = {
        (1, 0, 12),
        (2, 1, 5),
        (2, 3, 3),
    }

    actual_rows = set(df.itertuples(index=False, name=None))
    assert expected_rows == actual_rows, "Zawartość CSV nie odpowiada przekazanym danym."