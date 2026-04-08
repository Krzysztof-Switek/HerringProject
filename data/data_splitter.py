import os
import random
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path


class DataSplitter:
    # =========================================================
    # CONFIG
    # =========================================================
    # wybór źródła list txt:
    # "embedded" albo "not_embedded"
    SOURCE_PROCESS = "embedded"

    def __init__(
        self,
        base_dir="C:/Users/kswitek/Documents/HerringProject",
        source_subdir="final_pairs",
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42,
    ):
        self.base_dir = Path(base_dir)
        self.final_pairs_dir = self.base_dir / source_subdir

        self.embedded_dir = self.final_pairs_dir / "embedded"
        self.not_embedded_dir = self.final_pairs_dir / "not_embedded"

        self.source_process = self.SOURCE_PROCESS.lower().strip()
        if self.source_process not in {"embedded", "not_embedded"}:
            raise ValueError("SOURCE_PROCESS musi być ustawione na 'embedded' albo 'not_embedded'.")

        self.source_dir = self.embedded_dir if self.source_process == "embedded" else self.not_embedded_dir

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

        total = train_ratio + val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError("Suma proporcji musi wynosić 1.0")

        random.seed(seed)

        self.split_files_selected = {"train": [], "val": [], "test": []}
        self.file_to_set = {}
        self.group_assignments = {}
        self.pairs_df = pd.DataFrame()

        self.embedded_file_class_map = {}
        self.not_embedded_file_class_map = {}

    # =========================================================
    # MAIN
    # =========================================================
    def split_data(self):
        self._validate_source_structure()
        self._load_pairs_source_of_truth()

        class_dirs = ["1", "2"]

        self._clear_old_split_dirs()
        self._create_split_dirs(class_dirs)

        for class_dir in class_dirs:
            self._build_group_assignments_from_pairs(class_dir)

        self._copy_pairs_by_assignments()
        self._save_file_lists()
        self._update_excel_with_sets()

    # =========================================================
    # VALIDATION
    # =========================================================
    def _validate_source_structure(self):
        for root in [self.embedded_dir, self.not_embedded_dir]:
            if not root.exists():
                raise FileNotFoundError(f"Brak katalogu: {root}")

        for root in [self.embedded_dir, self.not_embedded_dir]:
            for class_dir in ["1", "2"]:
                class_path = root / class_dir
                if not class_path.exists():
                    raise FileNotFoundError(f"Brak katalogu klasy: {class_path}")

    # =========================================================
    # SOURCE OF TRUTH = GOTOWE PARY
    # =========================================================
    def _load_pairs_source_of_truth(self):
        csv_candidates = [
            self.final_pairs_dir / "processed_pairs.csv",
            Path(__file__).resolve().parent / "processed_pairs.csv",
        ]
        excel_path = self.final_pairs_dir / "analysisWithOtolithPhoto.xlsx"

        csv_path = next((p for p in csv_candidates if p.exists()), None)
        df = None

        if csv_path is not None:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            print(f"📥 Wczytano pary z CSV: {csv_path}")
        elif excel_path.exists():
            df = pd.read_excel(excel_path, engine="openpyxl")
            print(f"📥 Wczytano pary z Excela: {excel_path}")
        else:
            raise FileNotFoundError(
                "Nie znaleziono ani processed_pairs.csv, ani analysisWithOtolithPhoto.xlsx."
            )

        required_cols = {"embedded_file_path", "not_embedded_file_path"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                "Źródło danych zostało wczytane, ale nie zawiera kolumn par zdjęć.\n"
                f"Brakuje: {sorted(missing)}\n"
                "Użyj pliku processed_pairs.csv wygenerowanego przez wcześniejszy skrypt."
            )

        # mapa: nazwa pliku -> klasa 1/2 na podstawie realnych katalogów final_pairs
        self.embedded_file_class_map = self._build_filename_class_map(self.embedded_dir)
        self.not_embedded_file_class_map = self._build_filename_class_map(self.not_embedded_dir)

        df["embedded_file_name"] = df["embedded_file_path"].apply(lambda p: Path(str(p)).name)
        df["not_embedded_file_name"] = df["not_embedded_file_path"].apply(lambda p: Path(str(p)).name)

        if "fish_key" not in df.columns:
            df["fish_key"] = df["embedded_file_name"].apply(self._extract_fish_key_from_filename)

        df["class_dir_embedded"] = df["embedded_file_name"].apply(
            lambda name: self._resolve_class_from_filename(name, "embedded")
        )
        df["class_dir_not_embedded"] = df["not_embedded_file_name"].apply(
            lambda name: self._resolve_class_from_filename(name, "not_embedded")
        )

        missing_class_emb = df[df["class_dir_embedded"].isna()]
        missing_class_not = df[df["class_dir_not_embedded"].isna()]

        if not missing_class_emb.empty:
            raise ValueError(
                f"Nie znaleziono klasy dla {len(missing_class_emb)} plików embedded w final_pairs/embedded/1-2."
            )

        if not missing_class_not.empty:
            raise ValueError(
                f"Nie znaleziono klasy dla {len(missing_class_not)} plików not_embedded w final_pairs/not_embedded/1-2."
            )

        class_mismatch = df[df["class_dir_embedded"] != df["class_dir_not_embedded"]]
        if not class_mismatch.empty:
            raise ValueError(
                f"Wykryto {len(class_mismatch)} par, gdzie embedded i not_embedded są w różnych klasach."
            )

        df["class_dir"] = df["class_dir_embedded"]

        self.pairs_df = df.copy()

        print("📊 PODSUMOWANIE ŹRÓDŁA PAR")
        print(f"   liczba par:        {len(self.pairs_df)}")
        print(f"   unikalne fish_key: {self.pairs_df['fish_key'].nunique()}")

    def _build_filename_class_map(self, process_root: Path):
        file_map = {}

        for class_dir in ["1", "2"]:
            class_path = process_root / class_dir
            if not class_path.exists():
                continue

            for f in class_path.iterdir():
                if not f.is_file() or f.name.startswith("."):
                    continue

                key = f.name.lower()
                if key in file_map and file_map[key] != class_dir:
                    raise ValueError(
                        f"Ten sam plik występuje w więcej niż jednej klasie: {f.name}"
                    )
                file_map[key] = class_dir

        return file_map

    def _resolve_class_from_filename(self, filename: str, process_type: str) -> str:
        key = Path(str(filename)).name.lower()

        if process_type == "embedded":
            return self.embedded_file_class_map.get(key)
        if process_type == "not_embedded":
            return self.not_embedded_file_class_map.get(key)

        raise ValueError(f"Nieznany process_type: {process_type}")

    def _extract_fish_key_from_filename(self, filename: str) -> str:
        """
        Oczekiwany format 9 segmentów:
        1-4 = prefix
        5   = process (Embedded / NotEmbedded)
        6   = variant
        7   = fish_index
        8   = single
        9   = view

        fish-level key:
        1-4 + 6 + 7
        czyli bez:
        - process
        - single
        - view
        """
        stem = Path(filename).stem
        parts = stem.split("_")

        if len(parts) < 9:
            return stem.lower()

        key_parts = parts[0:4] + [parts[5], parts[6]]
        return "_".join(key_parts).lower()

    # =========================================================
    # DIRS
    # =========================================================
    def _clear_old_split_dirs(self):
        for process_root in [self.embedded_dir, self.not_embedded_dir]:
            for split in ["train", "val", "test"]:
                split_dir = process_root / split
                if split_dir.exists():
                    shutil.rmtree(split_dir)

            for txt_name in ["train_files.txt", "val_files.txt", "test_files.txt"]:
                txt_path = process_root / txt_name
                if txt_path.exists():
                    txt_path.unlink()

    def _create_split_dirs(self, class_dirs):
        for process_root in [self.embedded_dir, self.not_embedded_dir]:
            for split in ["train", "val", "test"]:
                split_dir = process_root / split
                split_dir.mkdir(parents=True, exist_ok=True)

                for class_dir in class_dirs:
                    (split_dir / class_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # GROUPING / NO LEAKAGE
    # =========================================================
    def _build_group_assignments_from_pairs(self, class_dir: str):
        class_df = self.pairs_df[self.pairs_df["class_dir"] == class_dir].copy()

        if class_df.empty:
            print(f"⚠️ Klasa {class_dir}: brak par w tabeli źródłowej")
            return

        group_keys = sorted(class_df["fish_key"].dropna().unique().tolist())
        random.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = int(n_groups * self.train_ratio)
        n_val = int(n_groups * self.val_ratio)

        train_groups = group_keys[:n_train]
        val_groups = group_keys[n_train:n_train + n_val]
        test_groups = group_keys[n_train + n_val:]

        for g in train_groups:
            self.group_assignments[(class_dir, g)] = "train"
        for g in val_groups:
            self.group_assignments[(class_dir, g)] = "val"
        for g in test_groups:
            self.group_assignments[(class_dir, g)] = "test"

        print(f"📦 Klasa {class_dir}: grup ryb = {n_groups}")
        print(f"   train: {len(train_groups)}, val: {len(val_groups)}, test: {len(test_groups)}")

    # =========================================================
    # COPY
    # =========================================================
    def _copy_pairs_by_assignments(self):
        if self.pairs_df.empty:
            print("⚠️ Brak par do kopiowania.")
            return

        split_buckets = {
            "embedded": {"train": [], "val": [], "test": []},
            "not_embedded": {"train": [], "val": [], "test": []},
        }

        skipped = 0

        for _, row in self.pairs_df.iterrows():
            class_dir = str(row["class_dir"])
            fish_key = str(row["fish_key"]).lower().strip()

            split_name = self.group_assignments.get((class_dir, fish_key))
            if split_name is None:
                skipped += 1
                continue

            embedded_name = str(row["embedded_file_name"])
            not_embedded_name = str(row["not_embedded_file_name"])

            embedded_src = self.embedded_dir / class_dir / embedded_name
            not_embedded_src = self.not_embedded_dir / class_dir / not_embedded_name

            if not embedded_src.exists():
                raise FileNotFoundError(f"Brak pliku embedded do kopiowania: {embedded_src}")

            if not not_embedded_src.exists():
                raise FileNotFoundError(f"Brak pliku not_embedded do kopiowania: {not_embedded_src}")

            split_buckets["embedded"][split_name].append((class_dir, embedded_src))
            split_buckets["not_embedded"][split_name].append((class_dir, not_embedded_src))

            self.file_to_set[embedded_src.name.lower()] = split_name.upper()
            self.file_to_set[not_embedded_src.name.lower()] = split_name.upper()

            if self.source_process == "embedded":
                self.split_files_selected[split_name].append(os.path.join(class_dir, embedded_src.name))
            else:
                self.split_files_selected[split_name].append(os.path.join(class_dir, not_embedded_src.name))

        if skipped:
            print(f"⚠️ Pominięto {skipped} rekordów bez przypisanego splitu")

        for process_name, process_root in [
            ("embedded", self.embedded_dir),
            ("not_embedded", self.not_embedded_dir),
        ]:
            for split_name in ["train", "val", "test"]:
                items = split_buckets[process_name][split_name]

                for class_dir, src in tqdm(items, desc=f"Kopiowanie {process_name}/{split_name}"):
                    dst = process_root / split_name / class_dir / src.name
                    shutil.copy2(src, dst)

    # =========================================================
    # TXT
    # =========================================================
    def _save_file_lists(self):
        txt_root = self.source_dir

        for split, files in self.split_files_selected.items():
            list_path = txt_root / f"{split}_files.txt"
            with open(list_path, "w", encoding="utf-8") as f:
                for file in sorted(files):
                    f.write(file + "\n")
            print(f"📝 Zapisano listę plików do {list_path}")

    # =========================================================
    # EXCEL
    # =========================================================
    def _extract_filename(self, path_value: str) -> str:
        return Path(str(path_value)).name.lower()

    def _update_excel_with_sets(self):
        excel_path = self.final_pairs_dir / "analysisWithOtolithPhoto.xlsx"

        if not excel_path.exists():
            print(f"⚠️ Nie znaleziono pliku Excel: {excel_path} — aktualizacja SET pominięta.")
            return

        df_excel = pd.read_excel(excel_path, engine="openpyxl")

        if "SET" in df_excel.columns:
            df_excel = df_excel.drop(columns=["SET"])

        if "SET_embedded" in df_excel.columns:
            df_excel = df_excel.drop(columns=["SET_embedded"])

        if "SET_not_embedded" in df_excel.columns:
            df_excel = df_excel.drop(columns=["SET_not_embedded"])

        if "embedded_name" in df_excel.columns:
            df_excel = df_excel.drop(columns=["embedded_name"])

        if "not_embedded_name" in df_excel.columns:
            df_excel = df_excel.drop(columns=["not_embedded_name"])

        # mapowanie po obu kolumnach, jeśli istnieją
        if "embedded_file_path" in df_excel.columns:
            df_excel["embedded_name"] = df_excel["embedded_file_path"].apply(self._extract_filename)
            df_excel["SET_embedded"] = df_excel["embedded_name"].map(self.file_to_set)

        if "not_embedded_file_path" in df_excel.columns:
            df_excel["not_embedded_name"] = df_excel["not_embedded_file_path"].apply(self._extract_filename)
            df_excel["SET_not_embedded"] = df_excel["not_embedded_name"].map(self.file_to_set)

        if "SET_embedded" in df_excel.columns and "SET_not_embedded" in df_excel.columns:
            df_excel["SET"] = df_excel["SET_embedded"].combine_first(df_excel["SET_not_embedded"])

            mismatch = df_excel[
                df_excel["SET_embedded"].notna()
                & df_excel["SET_not_embedded"].notna()
                & (df_excel["SET_embedded"] != df_excel["SET_not_embedded"])
            ]
            if not mismatch.empty:
                raise ValueError(
                    f"W Excelu wykryto {len(mismatch)} rekordów, gdzie embedded i not_embedded dostały różne SET."
                )

        elif "SET_embedded" in df_excel.columns:
            df_excel["SET"] = df_excel["SET_embedded"]
        elif "SET_not_embedded" in df_excel.columns:
            df_excel["SET"] = df_excel["SET_not_embedded"]
        else:
            raise ValueError("Excel nie zawiera kolumn embedded_file_path ani not_embedded_file_path.")

        df_excel.to_excel(excel_path, index=False)
        print(f"✅ Zaktualizowany plik Excel zapisany: {excel_path}")

    # =========================================================
    # CHECKER
    # =========================================================
    def _scan_split_groups(self, process_root: Path):
        result = {}
        for split in ["train", "val", "test"]:
            result[split] = {}
            for class_dir in ["1", "2"]:
                dir_path = process_root / split / class_dir
                if not dir_path.exists():
                    result[split][class_dir] = set()
                    continue

                files = [
                    f.name for f in dir_path.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]

                fish_groups = {self._extract_fish_key_from_filename(file_name) for file_name in files}
                result[split][class_dir] = fish_groups

        return result

    def _print_table(self, headers, rows):
        widths = [len(str(h)) for h in headers]

        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        def fmt(row):
            return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

        separator = "-+-".join("-" * w for w in widths)

        print(fmt(headers))
        print(separator)
        for row in rows:
            print(fmt(row))

    def run_checker(self):
        print("\n" + "=" * 72)
        print("CHECKER — WERYFIKACJA SPLITÓW")
        print("=" * 72)

        embedded_map = self._scan_split_groups(self.embedded_dir)
        not_embedded_map = self._scan_split_groups(self.not_embedded_dir)

        count_rows = []
        total_emb = 0
        total_not = 0

        for split in ["train", "val", "test"]:
            for class_dir in ["1", "2"]:
                emb_count = len(embedded_map[split][class_dir])
                not_emb_count = len(not_embedded_map[split][class_dir])
                diff = emb_count - not_emb_count
                status = "OK" if diff == 0 else "RÓŻNICA"
                count_rows.append([split, class_dir, emb_count, not_emb_count, diff, status])

                total_emb += emb_count
                total_not += not_emb_count

        print("\n[TABELA 1] LICZBA GRUP RYB W SPLITACH")
        self._print_table(
            ["SET", "KLASA", "EMBEDDED", "NOT_EMBEDDED", "DIFF", "STATUS"],
            count_rows
        )

        match_rows = []
        mismatch_details = []

        for split in ["train", "val", "test"]:
            for class_dir in ["1", "2"]:
                emb_groups = embedded_map[split][class_dir]
                not_emb_groups = not_embedded_map[split][class_dir]

                only_emb = sorted(emb_groups - not_emb_groups)
                only_not = sorted(not_emb_groups - emb_groups)

                status = "OK" if not only_emb and not only_not else "RÓŻNICA"
                match_rows.append([
                    split,
                    class_dir,
                    len(only_emb),
                    len(only_not),
                    status
                ])

                if only_emb or only_not:
                    mismatch_details.append({
                        "split": split,
                        "class_dir": class_dir,
                        "only_embedded": only_emb,
                        "only_not_embedded": only_not,
                    })

        print("\n[TABELA 2] ZGODNOŚĆ GRUP RYB: EMBEDDED VS NOT_EMBEDDED")
        self._print_table(
            ["SET", "KLASA", "TYLKO_EMBEDDED", "TYLKO_NOT_EMBEDDED", "STATUS"],
            match_rows
        )

        leakage_rows = []
        leakage_details = []

        for process_name, process_map in [
            ("embedded", embedded_map),
            ("not_embedded", not_embedded_map)
        ]:
            for class_dir in ["1", "2"]:
                train_set = process_map["train"][class_dir]
                val_set = process_map["val"][class_dir]
                test_set = process_map["test"][class_dir]

                train_val = sorted(train_set & val_set)
                train_test = sorted(train_set & test_set)
                val_test = sorted(val_set & test_set)

                total_leaks = len(train_val) + len(train_test) + len(val_test)
                status = "OK" if total_leaks == 0 else "LEAKAGE"

                leakage_rows.append([
                    process_name,
                    class_dir,
                    len(train_val),
                    len(train_test),
                    len(val_test),
                    status
                ])

                if total_leaks > 0:
                    leakage_details.append({
                        "process": process_name,
                        "class_dir": class_dir,
                        "train_val": train_val,
                        "train_test": train_test,
                        "val_test": val_test,
                    })

        print("\n[TABELA 3] LEAKAGE CHECK — CZY TA SAMA RYBA WPADA DO WIĘCEJ NIŻ 1 SETU")
        self._print_table(
            ["PROCESS", "KLASA", "TRAIN∩VAL", "TRAIN∩TEST", "VAL∩TEST", "STATUS"],
            leakage_rows
        )

        any_errors = False

        if total_emb == 0 or total_not == 0:
            any_errors = True
            print("\n❌ SANITY CHECK: splity są puste. To NIE jest poprawny wynik.")

        if mismatch_details:
            any_errors = True
            print("\n" + "=" * 72)
            print("SZCZEGÓŁY RÓŻNIC: EMBEDDED VS NOT_EMBEDDED")
            print("=" * 72)
            for item in mismatch_details:
                print(f"\nSET={item['split']} | KLASA={item['class_dir']}")
                if item["only_embedded"]:
                    print(f"  Tylko w embedded ({len(item['only_embedded'])}):")
                    for x in item["only_embedded"][:20]:
                        print(f"    - {x}")
                    if len(item["only_embedded"]) > 20:
                        print(f"    ... +{len(item['only_embedded']) - 20} więcej")

                if item["only_not_embedded"]:
                    print(f"  Tylko w not_embedded ({len(item['only_not_embedded'])}):")
                    for x in item["only_not_embedded"][:20]:
                        print(f"    - {x}")
                    if len(item["only_not_embedded"]) > 20:
                        print(f"    ... +{len(item['only_not_embedded']) - 20} więcej")

        if leakage_details:
            any_errors = True
            print("\n" + "=" * 72)
            print("SZCZEGÓŁY LEAKAGE")
            print("=" * 72)
            for item in leakage_details:
                print(f"\nPROCESS={item['process']} | KLASA={item['class_dir']}")

                if item["train_val"]:
                    print(f"  TRAIN ∩ VAL ({len(item['train_val'])}):")
                    for x in item["train_val"][:20]:
                        print(f"    - {x}")
                    if len(item["train_val"]) > 20:
                        print(f"    ... +{len(item['train_val']) - 20} więcej")

                if item["train_test"]:
                    print(f"  TRAIN ∩ TEST ({len(item['train_test'])}):")
                    for x in item["train_test"][:20]:
                        print(f"    - {x}")
                    if len(item["train_test"]) > 20:
                        print(f"    ... +{len(item['train_test']) - 20} więcej")

                if item["val_test"]:
                    print(f"  VAL ∩ TEST ({len(item['val_test'])}):")
                    for x in item["val_test"][:20]:
                        print(f"    - {x}")
                    if len(item["val_test"]) > 20:
                        print(f"    ... +{len(item['val_test']) - 20} więcej")

        print("\n" + "=" * 72)
        if any_errors:
            print("WYNIK CHECKERA: WYKRYTO PROBLEMY")
        else:
            print("WYNIK CHECKERA: WSZYSTKO OK")
        print("=" * 72)


if __name__ == "__main__":
    splitter = DataSplitter(
        base_dir="C:/Users/kswitek/Documents/HerringProject",
        source_subdir="final_pairs",
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42,
    )

    print("🔄 Rozpoczynanie podziału danych z katalogu final_pairs/")
    print(f"📌 Źródło list txt: {splitter.source_process}")
    splitter.split_data()

    print("\n✅ Podział zakończony pomyślnie!")
    splitter.run_checker()