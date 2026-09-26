import pandas as pd
import re
import unicodedata
from collections import defaultdict
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "resource/student_resource/dataset/train"

S1_FILE = f"{BASE}/train_source1.tsv"
S2_FILE = f"{BASE}/train_source2.tsv"
S3_FILE = f"{BASE}/train_source3.tsv"
GT_FILE = f"{BASE}/train_ground_truth.tsv"


# ============================================================
# TIMER
# ============================================================

START_TIME = time.time()


def elapsed():
    return round(time.time() - START_TIME, 2)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization.
    # Important because the dataset contains Indian-language text.
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Treat & as "and"
    value = value.replace("&", " and ")

    # Replace punctuation with spaces.
    # Keep Unicode letters/digits.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# LOAD FILE
# ============================================================

def load_source(path, name):

    print()
    print("=" * 70)
    print(f"Loading {name}")
    print("=" * 70)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
        },
        keep_default_na=True,
    )

    print(f"{name} shape: {df.shape}")
    print(f"{name} loaded in {elapsed()} seconds")

    print("\nMissing values:")

    print(df.isna().sum())

    return df


# ============================================================
# BUILD NORMALIZED COLUMNS
# ============================================================

def add_normalized_columns(df, name):

    print()
    print(f"Normalizing {name}...")

    df["name_norm"] = df["business_name"].map(normalize_text)

    df["address_norm"] = df["business_address"].map(normalize_text)

    # Country is already a clean categorical string.
    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.lower()
    )

    print(f"{name} normalization completed.")
    print(f"Elapsed: {elapsed()} seconds")

    return df


# ============================================================
# BUILD EXACT VALUE INDEX
# ============================================================

def build_index(df, column):

    index = defaultdict(list)

    values = df[column].tolist()
    ids = df["entity_id"].tolist()

    for entity_id, value in zip(ids, values):

        if value and not pd.isna(value):

            index[value].append(entity_id)

    return index


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUSINESS ENTITY RESOLUTION")
print("STEP 1 - BLOCKING ANALYSIS")
print("=" * 70)

print("\nProject paths:")
print(S1_FILE)
print(S2_FILE)
print(S3_FILE)
print(GT_FILE)


s1 = load_source(
    S1_FILE,
    "TRAIN SOURCE 1"
)

s2 = load_source(
    S2_FILE,
    "TRAIN SOURCE 2"
)

s3 = load_source(
    S3_FILE,
    "TRAIN SOURCE 3"
)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype={
        "source1_entity_id": "string",
        "matched_entity_ids": "string",
    }
)

print()
print("=" * 70)
print("GROUND TRUTH")
print("=" * 70)

print("Shape:", gt.shape)

print("\nMissing values:")
print(gt.isna().sum())


# ============================================================
# NORMALIZATION
# ============================================================

s1 = add_normalized_columns(
    s1,
    "SOURCE 1"
)

s2 = add_normalized_columns(
    s2,
    "SOURCE 2"
)

s3 = add_normalized_columns(
    s3,
    "SOURCE 3"
)


# ============================================================
# GROUND TRUTH DICTIONARY
# ============================================================

print()
print("=" * 70)
print("BUILDING GROUND TRUTH MAP")
print("=" * 70)

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

print("Ground truth entities:", len(gt_map))


# ============================================================
# SINGLETON ANALYSIS
# ============================================================

print()
print("=" * 70)
print("SINGLETON ANALYSIS")
print("=" * 70)

singleton_count = 0
matched_entity_count = 0
total_true_matches = 0
max_matches = 0

for value in gt["matched_entity_ids"]:

    if pd.isna(value) or value == "":

        singleton_count += 1

    else:

        matched_entity_count += 1

        matches = value.split(",")

        total_true_matches += len(matches)

        if len(matches) > max_matches:
            max_matches = len(matches)


print("Total Source 1 entities:", len(gt))

print(
    "Singleton entities:",
    singleton_count
)

print(
    "Entities with matches:",
    matched_entity_count
)

print(
    "Singleton percentage:",
    round(
        singleton_count / len(gt) * 100,
        4
    ),
    "%"
)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "Average matches per matched S1:",
    round(
        total_true_matches / matched_entity_count,
        4
    )
)

print(
    "Maximum matches for one S1:",
    max_matches
)


# ============================================================
# BUILD NAME INDEXES
# ============================================================

print()
print("=" * 70)
print("BUILDING EXACT NORMALIZED NAME INDEXES")
print("=" * 70)

print("Building Source 2 name index...")

s2_name_index = build_index(
    s2,
    "name_norm"
)

print(
    "Source 2 unique normalized names:",
    len(s2_name_index)
)

print("Building Source 3 name index...")

s3_name_index = build_index(
    s3,
    "name_norm"
)

print(
    "Source 3 unique normalized names:",
    len(s3_name_index)
)


# ============================================================
# EXACT NAME BLOCKING
# ============================================================

print()
print("=" * 70)
print("EXACT NORMALIZED NAME BLOCKING")
print("=" * 70)

total_true_matches = 0
found_true_matches = 0

total_candidates = 0

matched_s1_count = 0


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    truth = gt_map.get(
        s1_id,
        ""
    )

    if not truth:
        continue

    true_ids = set(
        truth.split(",")
    )

    matched_s1_count += 1

    total_true_matches += len(
        true_ids
    )

    name = row["name_norm"]

    candidates = set()

    if name:

        candidates.update(
            s2_name_index.get(
                name,
                []
            )
        )

        candidates.update(
            s3_name_index.get(
                name,
                []
            )
        )

    total_candidates += len(
        candidates
    )

    found_true_matches += len(
        true_ids.intersection(
            candidates
        )
    )


name_recall = (
    found_true_matches /
    total_true_matches
    if total_true_matches
    else 0
)

avg_candidates = (
    total_candidates /
    matched_s1_count
    if matched_s1_count
    else 0
)


print()
print("RESULTS")
print("-" * 50)

print(
    "S1 entities with matches:",
    matched_s1_count
)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "True matches retrieved:",
    found_true_matches
)

print(
    "Blocking recall:",
    round(
        name_recall * 100,
        4
    ),
    "%"
)

print(
    "Average candidates per matched S1:",
    round(
        avg_candidates,
        4
    )
)


# ============================================================
# BUILD ADDRESS INDEXES
# ============================================================

print()
print("=" * 70)
print("BUILDING EXACT NORMALIZED ADDRESS INDEXES")
print("=" * 70)

print("Building Source 2 address index...")

s2_address_index = build_index(
    s2,
    "address_norm"
)

print(
    "Source 2 unique normalized addresses:",
    len(s2_address_index)
)

print("Building Source 3 address index...")

s3_address_index = build_index(
    s3,
    "address_norm"
)

print(
    "Source 3 unique normalized addresses:",
    len(s3_address_index)
)


# ============================================================
# EXACT ADDRESS BLOCKING
# ============================================================

print()
print("=" * 70)
print("EXACT NORMALIZED ADDRESS BLOCKING")
print("=" * 70)

found_true_matches_address = 0

total_address_candidates = 0


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    truth = gt_map.get(
        s1_id,
        ""
    )

    if not truth:
        continue

    true_ids = set(
        truth.split(",")
    )

    address = row["address_norm"]

    candidates = set()

    if address:

        candidates.update(
            s2_address_index.get(
                address,
                []
            )
        )

        candidates.update(
            s3_address_index.get(
                address,
                []
            )
        )

    total_address_candidates += len(
        candidates
    )

    found_true_matches_address += len(
        true_ids.intersection(
            candidates
        )
    )


address_recall = (
    found_true_matches_address /
    total_true_matches
    if total_true_matches
    else 0
)

avg_address_candidates = (
    total_address_candidates /
    matched_s1_count
    if matched_s1_count
    else 0
)


print()
print("RESULTS")
print("-" * 50)

print(
    "True matches retrieved:",
    found_true_matches_address
)

print(
    "Blocking recall:",
    round(
        address_recall * 100,
        4
    ),
    "%"
)

print(
    "Average candidates per matched S1:",
    round(
        avg_address_candidates,
        4
    )
)


# ============================================================
# COMBINED EXACT BLOCKING
# ============================================================

print()
print("=" * 70)
print("COMBINED NAME + ADDRESS BLOCKING")
print("=" * 70)

found_combined = 0
combined_candidates = 0


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    truth = gt_map.get(
        s1_id,
        ""
    )

    if not truth:
        continue

    true_ids = set(
        truth.split(",")
    )

    candidates = set()

    name = row["name_norm"]

    address = row["address_norm"]

    if name:

        candidates.update(
            s2_name_index.get(
                name,
                []
            )
        )

        candidates.update(
            s3_name_index.get(
                name,
                []
            )
        )

    if address:

        candidates.update(
            s2_address_index.get(
                address,
                []
            )
        )

        candidates.update(
            s3_address_index.get(
                address,
                []
            )
        )

    combined_candidates += len(
        candidates
    )

    found_combined += len(
        true_ids.intersection(
            candidates
        )
    )


combined_recall = (
    found_combined /
    total_true_matches
    if total_true_matches
    else 0
)

avg_combined_candidates = (
    combined_candidates /
    matched_s1_count
    if matched_s1_count
    else 0
)


print()
print("RESULTS")
print("-" * 50)

print(
    "True matches retrieved:",
    found_combined
)

print(
    "Blocking recall:",
    round(
        combined_recall * 100,
        4
    ),
    "%"
)

print(
    "Average candidates per matched S1:",
    round(
        avg_combined_candidates,
        4
    )
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)

print(
    "Total execution time:",
    elapsed(),
    "seconds"
)

print()
print("Next step:")
print(
    "We will build high-recall token/n-gram blocking "
    "after reviewing these results."
)