import pandas as pd

def get_class_metadata(trainer):
    df = pd.read_excel(trainer.path_manager.metadata_file())
    df_train = df[df["SET"].str.lower() == "train"]

    # Filtrujemy tylko wiek od 1 do 20
    df_train = df_train[df_train["Wiek"].between(1, 20, inclusive='both')]

    # Liczność klas (np. dla LDAM, Seesaw)
    age_counts_series = df_train["Wiek"].value_counts().sort_index()
    age_counts = age_counts_series.to_dict()
    class_counts = [age_counts.get(age, 0) for age in range(1, 21)]

    # Częstości klas (np. dla ClassBalancedFocal)
    class_freq = {age: count for age, count in age_counts.items() if count > 0}

    return {
        "class_counts": class_counts,
        "class_freq": class_freq
    }
