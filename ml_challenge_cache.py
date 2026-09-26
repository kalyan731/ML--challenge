import os
import re
import time
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\kulth\OneDrive\ML CHALLENGE"

SOURCE1 = os.path.join(
    BASE,
    "resource",
    "student_resource",
    "dataset",
    "train",
    "train_source1.tsv",
)

SOURCE2 = os.path.join(
    BASE,
    "resource",
    "student_resource",
    "dataset",
    "train",
    "train_source2.tsv",
)

SOURCE3 = os.path.join(
    BASE,
    "resource",
    "student_resource",
    "dataset",
    "train",
    "train_source3.tsv",
)

CACHE_DIR = os.path.join(
    BASE,
    "normalized_cache",
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = x.replace("&", " and ")

    x = re.sub(
        r"[^\w\s]",
        " ",
        x,
        flags=re.UNICODE,
    )

    x = re.sub(
        r"\s+",
        " ",
        x,
    ).strip()

    return x


# ============================================================
# NORMALIZE ONE SOURCE
# ============================================================

def process_source(input_path, output_path, source_name):

    if os.path.exists(output_path):

        print(
            f"\n{source_name} cache already exists:"
        )
        print(output_path)
        print("Skipping.")

        return

    print("\n" + "=" * 70)
    print(f"Processing {source_name}")
    print("=" * 70)

    start = time.time()

    print("Loading...")

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(
        f"Loaded: {len(df):,} rows"
    )

    print("Normalizing name...")

    name = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    print("Normalizing address...")

    address = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    print("Normalizing country...")

    country = (
        df["country"]
        .fillna("")
        .map(normalize_text)
    )

    cache = pd.DataFrame({
        "entity_id": df["entity_id"].fillna(""),
        "name_norm": name,
        "addr_norm": address,
        "country_norm": country,
    })

    print("Saving Parquet...")

    cache.to_parquet(
        output_path,
        index=False,
        compression="snappy",
    )

    elapsed = time.time() - start

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Runtime: {elapsed:.2f} sec"
    )


# ============================================================
# START
# ============================================================

overall_start = time.time()

print("=" * 70)
print("NORMALIZED DATA CACHE BUILDER")
print("=" * 70)

process_source(
    SOURCE1,
    os.path.join(
        CACHE_DIR,
        "source1_normalized.parquet",
    ),
    "SOURCE1",
)

process_source(
    SOURCE2,
    os.path.join(
        CACHE_DIR,
        "source2_normalized.parquet",
    ),
    "SOURCE2",
)

process_source(
    SOURCE3,
    os.path.join(
        CACHE_DIR,
        "source3_normalized.parquet",
    ),
    "SOURCE3",
)

print("\n" + "=" * 70)
print("CACHE BUILD COMPLETE")
print("=" * 70)

print(
    f"Total runtime: "
    f"{time.time() - overall_start:.2f} sec"
)

print("\nCache directory:")
print(CACHE_DIR)