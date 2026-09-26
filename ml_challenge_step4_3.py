import os
import re
import difflib
import pandas as pd
import numpy as np
from collections import Counter


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
# BASIC HELPERS
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
    Build token weights only from the records actually needed
    for the current feature-generation run.

    This avoids materializing all tokens from all 12M+ records
    inside DuckDB.
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
# LOAD NORMALIZED RECORDS
# ============================================================

print("=" * 70)
print("STEP 4.3 - 21 FEATURE GENERATION")
print("=" * 70)

print("\nLoading pair files...")

train_pairs = pd.read_csv(TRAIN_PAIRS)
valid_pairs = pd.read_csv(VALID_PAIRS)

print("Train pairs:", train_pairs.shape)
print("Valid pairs:", valid_pairs.shape)


# ============================================================
# LOAD NORMALIZED PARQUET FILES
# ============================================================

print("\nLoading normalized source files...")

s1 = pd.read_parquet(S1_PATH)
s2 = pd.read_parquet(S2_PATH)
s3 = pd.read_parquet(S3_PATH)

print("Source1:", s1.shape)
print("Source2:", s2.shape)
print("Source3:", s3.shape)


# ============================================================
# CREATE CANDIDATE INDEX
# ============================================================

print("\nCreating candidate indexes...")

s2 = s2.reset_index(drop=True)
s3 = s3.reset_index(drop=True)

s2["candidate_index"] = np.arange(len(s2), dtype=np.int64)
s3["candidate_index"] = np.arange(len(s3), dtype=np.int64)


# ============================================================
# LOOKUP DICTIONARIES
# ============================================================

print("\nBuilding lookup dictionaries...")

s1_lookup = {}

for row in s1.itertuples(index=False):
    s1_lookup[row.entity_id] = row


s2_lookup = {}

for row in s2.itertuples(index=False):
    s2_lookup[row.candidate_index] = row


s3_lookup = {}

for row in s3.itertuples(index=False):
    s3_lookup[row.candidate_index] = row


print("S1 lookup:", f"{len(s1_lookup):,}")
print("S2 lookup:", f"{len(s2_lookup):,}")
print("S3 lookup:", f"{len(s3_lookup):,}")


# ============================================================
# IMPORTANT:
# ONLY BUILD TOKEN WEIGHTS FROM PAIRS
# ============================================================

print("\nBuilding lightweight token weights...")

needed_s1_ids = set(train_pairs["source1_entity_id"])
needed_s1_ids.update(valid_pairs["source1_entity_id"])

needed_s1_records = []

for entity_id in needed_s1_ids:
    row = s1_lookup.get(entity_id)

    if row is not None:
        needed_s1_records.append(
            safe_text(row.name_norm)
        )
        needed_s1_records.append(
            safe_text(row.addr_norm)
        )


needed_candidate_records = []

all_pairs = pd.concat(
    [train_pairs, valid_pairs],
    ignore_index=True
)

for row in all_pairs.itertuples(index=False):

    candidate_index = int(row.candidate_index)

    if int(row.candidate_source) == 0:
        cand = s2_lookup.get(candidate_index)
    else:
        cand = s3_lookup.get(candidate_index)

    if cand is not None:
        needed_candidate_records.append(
            safe_text(cand.name_norm)
        )
        needed_candidate_records.append(
            safe_text(cand.addr_norm)
        )


weights = build_weights(
    needed_s1_records + needed_candidate_records
)

print(
    "Weighted vocabulary:",
    f"{len(weights):,}"
)


# ============================================================
# FEATURE GENERATION
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


def generate_features(pairs, split_name):

    print("\n" + "-" * 70)
    print("Generating", split_name, "features")
    print("-" * 70)

    output = []

    total = len(pairs)

    for i, row in enumerate(
        pairs.itertuples(index=False),
        start=1
    ):

        s1_id = row.source1_entity_id

        candidate_index = int(row.candidate_index)
        candidate_source = int(row.candidate_source)

        s1_row = s1_lookup.get(s1_id)

        if s1_row is None:
            raise RuntimeError(
                f"Source1 ID not found: {s1_id}"
            )

        if candidate_source == 0:
            candidate_row = s2_lookup.get(
                candidate_index
            )
        else:
            candidate_row = s3_lookup.get(
                candidate_index
            )

        if candidate_row is None:
            raise RuntimeError(
                f"Candidate not found: "
                f"source={candidate_source}, "
                f"index={candidate_index}"
            )

        s1_name = safe_text(s1_row.name_norm)
        s1_addr = safe_text(s1_row.addr_norm)
        s1_country = safe_text(s1_row.country_norm)

        c_name = safe_text(candidate_row.name_norm)
        c_addr = safe_text(candidate_row.addr_norm)
        c_country = safe_text(candidate_row.country_norm)

        features = [

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            float(s1_name == c_name),

            char_sim(
                s1_name,
                c_name
            ),

            token_jaccard(
                s1_name,
                c_name
            ),

            token_overlap(
                s1_name,
                c_name
            ),

            length_ratio(
                s1_name,
                c_name
            ),

            # ------------------------------------------------
            # ADDRESS
            # ------------------------------------------------

            float(s1_addr == c_addr),

            char_sim(
                s1_addr,
                c_addr
            ),

            token_jaccard(
                s1_addr,
                c_addr
            ),

            token_overlap(
                s1_addr,
                c_addr
            ),

            length_ratio(
                s1_addr,
                c_addr
            ),

            # ------------------------------------------------
            # EXTRA NAME
            # ------------------------------------------------

            prefix4(
                s1_name,
                c_name
            ),

            suffix4(
                s1_name,
                c_name
            ),

            shared_tokens(
                s1_name,
                c_name
            ),

            weighted_overlap(
                s1_name,
                c_name,
                weights
            ),

            # ------------------------------------------------
            # EXTRA ADDRESS
            # ------------------------------------------------

            prefix4(
                s1_addr,
                c_addr
            ),

            suffix4(
                s1_addr,
                c_addr
            ),

            shared_tokens(
                s1_addr,
                c_addr
            ),

            weighted_overlap(
                s1_addr,
                c_addr,
                weights
            ),

            # ------------------------------------------------
            # NUMERIC
            # ------------------------------------------------

            number_overlap(
                s1_addr,
                c_addr
            ),

            first_number_match(
                s1_addr,
                c_addr
            ),

            digit_similarity(
                s1_addr,
                c_addr
            ),
        ]

        output.append(features)

        if i % 5000 == 0 or i == total:
            print(
                f"{split_name}: "
                f"{i:,}/{total:,}"
            )

    result = pd.DataFrame(
        output,
        columns=FEATURE_COLUMNS
    )

    result["label"] = pairs["label"].to_numpy()

    return result


# ============================================================
# TRAIN
# ============================================================

train_features = generate_features(
    train_pairs,
    "TRAIN"
)

print("\nTrain feature shape:", train_features.shape)


# ============================================================
# VALID
# ============================================================

valid_features = generate_features(
    valid_pairs,
    "VALID"
)

print("\nValid feature shape:", valid_features.shape)


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
print(" ", TRAIN_OUT)
print(" ", VALID_OUT)


print("\nFinal feature columns:")

for i, column in enumerate(FEATURE_COLUMNS):
    print(
        f"feature_{i}_{column}"
    )


print("\n" + "=" * 70)
print("STEP 4.3 COMPLETE")
print("=" * 70)

print(
    "Train:",
    train_features.shape
)

print(
    "Valid:",
    valid_features.shape
)