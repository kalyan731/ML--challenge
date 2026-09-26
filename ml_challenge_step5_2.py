import time
from pathlib import Path

import pandas as pd


START = time.time()

BASE = Path("resource/student_resource/dataset/test")
CACHE_DIR = Path("normalized_cache_test")

CACHE_DIR.mkdir(exist_ok=True)


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = x.replace("&", " and ")

    import re

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


FILES = {
    "source1": BASE / "test_source1.tsv",
    "source2": BASE / "test_source2.tsv",
    "source3": BASE / "test_source3.tsv",
}


print("=" * 75)
print("STEP 5.2 - BUILD TEST NORMALIZED CACHE")
print("=" * 75)


for name, input_file in FILES.items():

    output_file = CACHE_DIR / f"{name}_normalized.parquet"

    print("\n" + "-" * 75)
    print(f"Processing {name}")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    if output_file.exists():

        print("Cache already exists.")
        print("Skipping.")

        existing = pd.read_parquet(output_file)

        print(
            f"Cached shape: "
            f"{existing.shape}"
        )

        continue

    t0 = time.time()

    print("Reading TSV...")

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
        },
    )

    print(
        f"Loaded: "
        f"{df.shape}"
    )

    print("Normalizing...")

    df["entity_id"] = (
        df["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["addr_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    result = df[
        [
            "entity_id",
            "name_norm",
            "addr_norm",
            "country_norm",
        ]
    ]

    print("Saving Parquet...")

    result.to_parquet(
        output_file,
        index=False,
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Shape: {result.shape}"
    )

    print(
        f"Time: "
        f"{time.time() - t0:.2f}s"
    )


print("\n" + "=" * 75)
print("CACHE BUILD COMPLETE")
print("=" * 75)

print(
    f"Total runtime: "
    f"{time.time() - START:.2f}s"
)

print("=" * 75)
print("DONE")
print("=" * 75)