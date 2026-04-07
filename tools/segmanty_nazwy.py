from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(r"Z:\Photo\Otolithes\HER\Processed")
RAW_DIR = Path(r"Z:\Photo\Otolithes\HER\Raw")


def analyze_structure():
    all_files = list(DATA_DIR.rglob("*"))
    jpg_files = [p for p in all_files if p.is_file() and p.suffix.lower() == ".jpg"]

    segment_counts = Counter()
    last_segment_counter = Counter()

    right_left_rows = []
    segment_5_counter = Counter()
    segment_6_counter = Counter()
    segment_5_6_combo = Counter()

    for path in jpg_files:
        name = path.stem
        parts = name.split("_")
        count = len(parts)

        segment_counts[count] += 1

        if count == 9:
            last_segment = parts[8]
            last_segment_counter[last_segment] += 1

            if last_segment in {"Right", "Left"}:
                right_left_rows.append(parts)
                segment_5_counter[parts[4]] += 1
                segment_6_counter[parts[5]] += 1
                segment_5_6_combo[(parts[4], parts[5])] += 1

    print("\n" + "=" * 60)
    print("KROK 1 — ROZKŁAD LICZBY SEGMENTÓW (_)")
    print("=" * 60)

    for k, v in sorted(segment_counts.items()):
        print(f"{k} segmentów: {v} plików")

    print("\n" + "=" * 60)
    print("KROK 2 — UNIKALNE WARTOŚCI SEGMENTU 9 (VIEW)")
    print("=" * 60)

    for val, cnt in last_segment_counter.most_common():
        print(f"{val} -> {cnt}")

    print("\n" + "=" * 60)
    print(f"LICZBA UNIKALNYCH VIEW: {len(last_segment_counter)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("KROK 3 — TYLKO RIGHT + LEFT")
    print("=" * 60)
    print(f"Zdjęcia po filtrze Right/Left: {len(right_left_rows)}")

    print("\n" + "=" * 60)
    print("KROK 3A — UNIKALNE WARTOŚCI SEGMENTU 5")
    print("=" * 60)

    for val, cnt in segment_5_counter.most_common():
        print(f"{val} -> {cnt}")

    print("\n" + "=" * 60)
    print(f"LICZBA UNIKALNYCH SEGMENTU 5: {len(segment_5_counter)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("KROK 3B — UNIKALNE WARTOŚCI SEGMENTU 6")
    print("=" * 60)

    for val, cnt in segment_6_counter.most_common():
        print(f"{val} -> {cnt}")

    print("\n" + "=" * 60)
    print(f"LICZBA UNIKALNYCH SEGMENTU 6: {len(segment_6_counter)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("KROK 4 — KOMBINACJE SEGMENTU 5 + 6")
    print("=" * 60)

    for (s5, s6), cnt in segment_5_6_combo.most_common():
        print(f"{s5} + {s6} -> {cnt}")

    print("\n" + "=" * 60)
    print(f"LICZBA UNIKALNYCH KOMBINACJI: {len(segment_5_6_combo)}")
    print("=" * 60)

    # =========================
    # KROK 5 — RAW ANALIZA
    # =========================

    raw_files = list(RAW_DIR.rglob("*"))
    raw_jpg = [p for p in raw_files if p.is_file() and p.suffix.lower() == ".jpg"]

    raw_segment_5 = Counter()
    raw_segment_6 = Counter()
    raw_segment_5_6_combo = Counter()
    raw_groups = defaultdict(set)

    # nowy słownik: key = prefix + FishIndex, value = zestaw kombinacji (seg5, seg6)
    raw_key_to_combos = defaultdict(set)

    valid_raw = []

    for path in raw_jpg:
        name = path.stem
        parts = name.split("_")

        if len(parts) != 7:
            continue

        valid_raw.append(parts)

        seg5 = parts[4]
        seg6 = parts[5]

        raw_segment_5[seg5] += 1
        raw_segment_6[seg6] += 1
        raw_segment_5_6_combo[(seg5, seg6)] += 1

        prefix = "_".join(parts[0:4])
        fish_index = parts[6]
        key = f"{prefix}_{fish_index}"

        raw_groups[key].add(seg5)
        raw_key_to_combos[key].add((seg5, seg6))

    print("\n" + "=" * 60)
    print("KROK 5 — RAW SEGMENTY 5 I 6")
    print("=" * 60)

    print("\nSegment 5:")
    for val, cnt in raw_segment_5.most_common():
        print(f"{val} -> {cnt}")

    print("\nSegment 6:")
    for val, cnt in raw_segment_6.most_common():
        print(f"{val} -> {cnt}")

    print("\n" + "=" * 60)
    print("KROK 5A — KOMBINACJE SEGMENTU 5 + 6 (RAW)")
    print("=" * 60)

    for (s5, s6), cnt in raw_segment_5_6_combo.most_common():
        print(f"{s5} + {s6} -> {cnt}")

    print("\n" + "=" * 60)
    print(f"LICZBA KOMBINACJI RAW: {len(raw_segment_5_6_combo)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("KROK 5B — PARY W RAW")
    print("=" * 60)

    pair_count = 0

    for key, classes in raw_groups.items():
        if "Embedded" in classes and "NotEmbedded" in classes:
            pair_count += 1

    print(f"Unikalne pary (prefix + FishIndex): {pair_count}")
    print(f"Łączne grupy: {len(raw_groups)}")

    # =========================
    # KROK 5C — CZY TO TE SAME RYBY W 3 KOMBINACJACH
    # =========================

    target_a = ("Embedded", "Embedded")
    target_b = ("Embedded", "Sharpest")
    target_c = ("NotEmbedded", "WithoutPostproc")

    keys_a = {key for key, combos in raw_key_to_combos.items() if target_a in combos}
    keys_b = {key for key, combos in raw_key_to_combos.items() if target_b in combos}
    keys_c = {key for key, combos in raw_key_to_combos.items() if target_c in combos}

    common_ab = keys_a & keys_b
    common_ac = keys_a & keys_c
    common_bc = keys_b & keys_c
    common_abc = keys_a & keys_b & keys_c

    print("\n" + "=" * 60)
    print("KROK 5C — NAKŁADANIE 3 KOMBINACJI NA TE SAME RYBY")
    print("=" * 60)
    print(f"Embedded + Embedded:                {len(keys_a)}")
    print(f"Embedded + Sharpest:                {len(keys_b)}")
    print(f"NotEmbedded + WithoutPostproc:      {len(keys_c)}")
    print("-" * 60)
    print(f"Wspólne A ∩ B:                      {len(common_ab)}")
    print(f"Wspólne A ∩ C:                      {len(common_ac)}")
    print(f"Wspólne B ∩ C:                      {len(common_bc)}")
    print(f"Wspólne A ∩ B ∩ C:                  {len(common_abc)}")

    print("\n" + "=" * 60)
    print("KROK 5D — PRZYKŁADY KLUCZY A ∩ B ∩ C (max 20)")
    print("=" * 60)

    for idx, key in enumerate(sorted(common_abc)):
        print(key)
        if idx >= 19:
            break


if __name__ == "__main__":
    analyze_structure()