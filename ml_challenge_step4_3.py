import os
import re
import difflib
from collections import Counter

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRAIN_PAIRS = "step4_train_pairs.csv"
VALID_PAIRS = "step4_valid_pairs.csv"

TRAIN_OUT = "step4_3_train_features.csv"
VALID_OUT = "step4_3_valid_features.csv"

CACHE = "normalized_cache"

S1_PATH = os.path.join(CACHE, "source1_normalized.parquet")
S2_PATH = os.path.join(CACHE, "source2_normalized.parquet")
S3_PATH = os.path.join(CACHE, "source3_normalized.parquet")


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(x):
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x)


def token_set(x):
    x = safe_text(x)
    if not x:
        return set()
    return set(x.split())


def char_sim(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def token_jaccard(a, b):
    ta = token_set(a)
    tb = token_set(b)

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):
    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / min(len(ta), len(tb))


def length_ratio(a, b):
    a = safe_text(a)
    b = safe_text(b)

    la = len(a)
    lb = len(b)

    if la == 0 and lb == 0:
        return 1.0

    if la == 0 or lb == 0:
        return 0.0

    return min(la, lb) / max(la, lb)


def prefix4(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if len(a) < 4 or len(b) < 4:
        return 0.0

    return float(a[:4] == b[:4])


def suffix4(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if len(a) < 4 or len(b) < 4:
        return 0.0

    return float(a[-4:] == b[-4:])


def shared_tokens(a, b):
    return float(len(token_set(a) & token_set(b)))


def number_set(x):
    return set(re.findall(r"\d+", safe_text(x)))


def number_overlap(a, b):
    na = number_set(a)
    nb = number_set(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / min(len(na), len(nb))


def first_number_match(a, b):
    na = re.findall(r"\d+", safe_text(a))
    nb = re.findall(r"\d+", safe_text(b))

    if not na or not nb:
        return 0.0

    return float(na[0] == nb[0])


def digit_similarity(a, b):
    a = re.sub(r"\D", "", safe_text(a))
    b = re.sub(r"\D", "", safe_text(b))

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


# ============================================================
# WEIGHTED TOKEN OVERLAP
# ============================================================

def build_weights(records):
    """
    Build lightweight IDF-style weights only from records
    participating in the pair files.

    This deliberately avoids scanning/holding all 12M+
    source records in Python.
    """

    counter = Counter()

    for text in records:
        counter.update(token_set(text))

    total = max(len(records), 1)

    weights = {}

    for token, freq in counter.items():
        weights[token] = np.log(
            (total + 1.0) / (freq + 1.0)
        )

    return weights


def weighted_overlap(a, b, weights):
    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    common = ta & tb

    if not common:
        return 0.0

    numerator = sum(
        weights.get(t, 1.0)
        for t in common
    )

    denom_a = sum(
        weights.get(t, 1.0)
        for t in ta
    )

    denom_b = sum(
        weights.get(t, 1.0)
        for t in tb
    )

    denom = min(denom_a, denom_b)

    if denom <= 0:
        return 0.0

    return numerator / denom


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 4.3 - MEMORY-SAFE 21 FEATURE GENERATION")
print("=" * 70)


# ============================================================
# LOAD PAIRS
# ============================================================

print("\nLoading pair files...")

train_pairs = pd.read_csv(TRAIN_PAIRS)
valid_pairs = pd.read_csv(VALID_PAIRS)

print("Train pairs:", train_pairs.shape)
print("Valid pairs:", valid_pairs.shape)

all_pairs = pd.concat(
    [train_pairs, valid_pairs],
    ignore_index=True
)

all_pairs["_pair_id"] = np.arange(
    len(all_pairs),
    dtype=np.int64
)


# ============================================================
# DUCKDB
# ============================================================

print("\nStarting DuckDB...")

con = duckdb.connect()

con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='5GB'")
con.execute("PRAGMA preserve_insertion_order=false")


# ============================================================
# REGISTER ONLY THE 57K PAIRS
# ============================================================

print("\nRegistering pair table...")

con.register(
    "pairs",
    all_pairs[
        [
            "_pair_id",
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "candidate_index",
        ]
    ]
)


# ============================================================
# FETCH ONLY REQUIRED RECORDS
# ============================================================

print("\nFetching only records required by pairs...")
print("No full-source Python dictionaries will be created.")


query = f"""
WITH pair_records AS (

    SELECT
        p._pair_id,
        p.source1_entity_id,
        p.candidate_entity_id,
        p.candidate_source,
        p.candidate_index,

        s1.name_norm AS s1_name,
        s1.addr_norm AS s1_addr,
        s1.country_norm AS s1_country

    FROM pairs p

    INNER JOIN read_parquet('{S1_PATH}') s1
        ON s1.entity_id = p.source1_entity_id
),

candidate_records AS (

    SELECT
        p._pair_id,
        s2.name_norm AS candidate_name,
        s2.addr_norm AS candidate_addr,
        s2.country_norm AS candidate_country

    FROM pairs p

    INNER JOIN read_parquet('{S2_PATH}') s2
        ON s2.entity_id = p.candidate_entity_id

    WHERE p.candidate_source = 0

    UNION ALL

    SELECT
        p._pair_id,
        s3.name_norm AS candidate_name,
        s3.addr_norm AS candidate_addr,
        s3.country_norm AS candidate_country

    FROM pairs p

    INNER JOIN read_parquet('{S3_PATH}') s3
        ON s3.entity_id = p.candidate_entity_id

    WHERE p.candidate_source = 1
)

SELECT
    p._pair_id,
    p.source1_entity_id,
    p.candidate_entity_id,
    p.candidate_source,
    p.candidate_index,

    p.s1_name,
    p.s1_addr,
    p.s1_country,

    c.candidate_name,
    c.candidate_addr,
    c.candidate_country

FROM pair_records p

INNER JOIN candidate_records c
    ON c._pair_id = p._pair_id

ORDER BY p._pair_id
"""


pair_data = con.execute(query).fetchdf()

print(
    "Joined pair records:",
    pair_data.shape
)


# ============================================================
# CHECK
# ============================================================

if len(pair_data) != len(all_pairs):
    print(
        "\nWARNING:"
        f" expected {len(all_pairs):,} rows,"
        f" got {len(pair_data):,} rows."
    )

    missing = len(all_pairs) - len(pair_data)

    if missing > 0:
        raise RuntimeError(
            f"{missing:,} pair records could not be joined."
        )


# ============================================================
# BUILD LIGHTWEIGHT TOKEN WEIGHTS
# ============================================================

print("\nBuilding lightweight token weights...")

weight_records = []

for row in pair_data.itertuples(index=False):

    weight_records.append(
        safe_text(row.s1_name)
    )

    weight_records.append(
        safe_text(row.s1_addr)
    )

    weight_records.append(
        safe_text(row.candidate_name)
    )

    weight_records.append(
        safe_text(row.candidate_addr)
    )


weights = build_weights(weight_records)

print(
    "Weighted vocabulary:",
    f"{len(weights):,}"
)

del weight_records


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    "name_exact",
    "name_char_sim",
    "name_token_jaccard",
    "name_token_overlap",
    "name_length_ratio",

    "addr_exact",
    "addr_char_sim",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_length_ratio",

    "name_prefix4",
    "name_suffix4",
    "name_shared_tokens",
    "name_weighted_overlap",

    "addr_prefix4",
    "addr_suffix4",
    "addr_shared_tokens",
    "addr_weighted_overlap",

    "number_overlap",
    "first_number_match",
    "digit_similarity",
]


# ============================================================
# CALCULATE FEATURES
# ============================================================

def calculate_features(df):

    rows = []

    total = len(df)

    print(
        f"\nCalculating {len(FEATURE_COLUMNS)} features "
        f"for {total:,} pairs..."
    )

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        s1_name = safe_text(row.s1_name)
        s1_addr = safe_text(row.s1_addr)

        candidate_name = safe_text(
            row.candidate_name
        )

        candidate_addr = safe_text(
            row.candidate_addr
        )

        features = [

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            float(
                s1_name == candidate_name
            ),

            char_sim(
                s1_name,
                candidate_name
            ),

            token_jaccard(
                s1_name,
                candidate_name
            ),

            token_overlap(
                s1_name,
                candidate_name
            ),

            length_ratio(
                s1_name,
                candidate_name
            ),

            # ------------------------------------------------
            # ADDRESS
            # ------------------------------------------------

            float(
                s1_addr == candidate_addr
            ),

            char_sim(
                s1_addr,
                candidate_addr
            ),

            token_jaccard(
                s1_addr,
                candidate_addr
            ),

            token_overlap(
                s1_addr,
                candidate_addr
            ),

            length_ratio(
                s1_addr,
                candidate_addr
            ),

            # ------------------------------------------------
            # EXTRA NAME
            # ------------------------------------------------

            prefix4(
                s1_name,
                candidate_name
            ),

            suffix4(
                s1_name,
                candidate_name
            ),

            shared_tokens(
                s1_name,
                candidate_name
            ),

            weighted_overlap(
                s1_name,
                candidate_name,
                weights
            ),

            # ------------------------------------------------
            # EXTRA ADDRESS
            # ------------------------------------------------

            prefix4(
                s1_addr,
                candidate_addr
            ),

            suffix4(
                s1_addr,
                candidate_addr
            ),

            shared_tokens(
                s1_addr,
                candidate_addr
            ),

            weighted_overlap(
                s1_addr,
                candidate_addr,
                weights
            ),

            # ------------------------------------------------
            # NUMERIC
            # ------------------------------------------------

            number_overlap(
                s1_addr,
                candidate_addr
            ),

            first_number_match(
                s1_addr,
                candidate_addr
            ),

            digit_similarity(
                s1_addr,
                candidate_addr
            ),
        ]

        rows.append(features)

        if (
            i % 5000 == 0
            or i == total
        ):
            print(
                f"Processed {i:,}/{total:,}"
            )

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS
    )


# ============================================================
# GENERATE ALL FEATURES
# ============================================================

feature_df = calculate_features(
    pair_data
)

feature_df["label"] = all_pairs["label"].to_numpy()


# ============================================================
# SPLIT TRAIN / VALID
# ============================================================

train_count = len(train_pairs)

train_features = feature_df.iloc[
    :train_count
].copy()

valid_features = feature_df.iloc[
    train_count:
].copy()


# ============================================================
# SAVE
# ============================================================

print("\nSaving feature files...")

train_features.to_csv(
    TRAIN_OUT,
    index=False
)

valid_features.to_csv(
    VALID_OUT,
    index=False
)

print("\nSaved:")
print(
    f"  {TRAIN_OUT} -> "
    f"{train_features.shape}"
)

print(
    f"  {VALID_OUT} -> "
    f"{valid_features.shape}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\nFeature list:")

for i, name in enumerate(FEATURE_COLUMNS):
    print(
        f"  feature_{i}_{name}"
    )

print("\n" + "=" * 70)
print("STEP 4.3 COMPLETE")
print("=" * 70)

con.close()