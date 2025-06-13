import pandas as pd
from collections import Counter

def get_class_metadata(trainer):
    df = pd.read_excel(trainer.path_manager.metadata_file())
    df_train = df[df["SET"].str.lower() == "train"]

    # Liczność klas (dla LDAM, Seesaw)
    age_counts = df_train["Wiek"].value_counts().sort_index().to_dict()
    class_counts = [age_counts.get(age, 0) for age in sorted(age_counts)]

    # Częstości klas (dla ClassBalancedFocal)
    total = sum(age_counts.values())
    class_freq = {int(k): v for k, v in age_counts.items() if v > 0}

    return {
        "class_counts": class_counts,
        "class_freq": class_freq
    }
