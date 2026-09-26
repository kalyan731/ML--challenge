import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter
import time


# ============================================================
# CONFIG
# ============================================================

BASE = "resource/student_resource/dataset/train"

S1_FILE = f"{BASE}/train_source1.tsv"
S2_FILE = f"{BASE}/train_source2.tsv"
S3_FILE = f"{BASE}/train_source3.tsv"
GT_FILE = f"{BASE}/train_ground_truth.tsv"

# Start with a sample.
# After this works, we will run full-scale.
SAMPLE_SIZE = 100_000

# Ignore extremely common tokens.
# We will experiment with this later.
MAX_TOKEN_FREQUENCY = 5000


START = time.time()


def elapsed():
    return round(time.time() - START, 2)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = value.replace(
        "&",
        " and "
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# TOKENIZATION
# ============================================================

def get_tokens(text):

    if not text:
        return set()

    return set(
        token
        for token in text.split()
        if len(token) >= 2
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 2 - HIGH RECALL TOKEN BLOCKING")
print("=" * 75)

print("\nLoading data...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype={
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string"
    }
)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype={
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string"
    }
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype={
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string"
    }
)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype={
        "source1_entity_id": "string",
        "matched_entity_ids": "string"
    }
)

print("Loaded in:", elapsed(), "seconds")


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_text)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

print("Normalization complete.")
print("Time:", elapsed())


# ============================================================
# GROUND TRUTH
# ============================================================

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)


# ============================================================
# SELECT SAMPLE
# ============================================================

matched_gt = gt[
    gt["matched_entity_ids"].notna()
    & (gt["matched_entity_ids"] != "")
].copy()

print("\nMatched S1 entities:", len(matched_gt))

sample_gt = matched_gt.head(SAMPLE_SIZE)

sample_ids = set(
    sample_gt["source1_entity_id"]
)

sample_s1 = s1[
    s1["entity_id"].isin(sample_ids)
].copy()

print(
    "Sample S1 entities:",
    len(sample_s1)
)


# ============================================================
# TOKEN INDEX
# ============================================================

def build_token_frequency(df):

    frequency = Counter()

    for name, address in zip(
        df["name_norm"],
        df["address_norm"]
    ):

        tokens = (
            get_tokens(name)
            |
            get_tokens(address)
        )

        frequency.update(tokens)

    return frequency


print("\nCalculating token frequencies...")

freq_s2 = build_token_frequency(s2)
freq_s3 = build_token_frequency(s3)

print(
    "Unique tokens S2:",
    len(freq_s2)
)

print(
    "Unique tokens S3:",
    len(freq_s3)
)


# ============================================================
# BUILD INVERTED INDEX
# ============================================================

def build_token_index(df, frequency):

    index = defaultdict(list)

    for entity_id, name, address in zip(
        df["entity_id"],
        df["name_norm"],
        df["address_norm"]
    ):

        tokens = (
            get_tokens(name)
            |
            get_tokens(address)
        )

        for token in tokens:

            # Ignore extremely common tokens.
            if frequency[token] <= MAX_TOKEN_FREQUENCY:

                index[token].append(entity_id)

    return index


print("\nBuilding Source 2 token index...")

s2_token_index = build_token_index(
    s2,
    freq_s2
)

print(
    "S2 token index size:",
    len(s2_token_index)
)

print("\nBuilding Source 3 token index...")

s3_token_index = build_token_index(
    s3,
    freq_s3
)

print(
    "S3 token index size:",
    len(s3_token_index)
)


# ============================================================
# ENTITY LOOKUP
# ============================================================

s2_country = dict(
    zip(
        s2["entity_id"],
        s2["country_norm"]
    )
)

s3_country = dict(
    zip(
        s3["entity_id"],
        s3["country_norm"]
    )
)


# ============================================================
# TEST TOKEN BLOCKING
# ============================================================

print()
print("=" * 75)
print("TOKEN BLOCKING TEST")
print("=" * 75)


total_true = 0
found_true = 0

total_candidates = 0

entities_evaluated = 0


for _, row in sample_s1.iterrows():

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

    total_true += len(true_ids)

    entities_evaluated += 1

    name_tokens = get_tokens(
        row["name_norm"]
    )

    address_tokens = get_tokens(
        row["address_norm"]
    )

    tokens = (
        name_tokens |
        address_tokens
    )

    candidates = set()

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    for token in tokens:

        candidates.update(
            s2_token_index.get(
                token,
                []
            )
        )

        candidates.update(
            s3_token_index.get(
                token,
                []
            )
        )

    # --------------------------------------------------------
    # Country filter
    # --------------------------------------------------------

    country = row["country_norm"]

    if country:

        candidates = {
            cid
            for cid in candidates

            if (
                (
                    cid.startswith("S2-")
                    and s2_country.get(cid) == country
                )
                or
                (
                    cid.startswith("S3-")
                    and s3_country.get(cid) == country
                )
            )
        }

    total_candidates += len(
        candidates
    )

    found_true += len(
        true_ids.intersection(
            candidates
        )
    )


# ============================================================
# RESULTS
# ============================================================

recall = (
    found_true / total_true
    if total_true
    else 0
)

avg_candidates = (
    total_candidates / entities_evaluated
    if entities_evaluated
    else 0
)


print()
print("=" * 75)
print("TOKEN BLOCKING RESULTS")
print("=" * 75)

print(
    "Entities evaluated:",
    entities_evaluated
)

print(
    "Total true matches:",
    total_true
)

print(
    "True matches retrieved:",
    found_true
)

print(
    "Blocking recall:",
    round(
        recall * 100,
        4
    ),
    "%"
)

print(
    "Average candidates:",
    round(
        avg_candidates,
        2
    )
)

print(
    "Total candidates:",
    total_candidates
)

print()
print("=" * 75)
print("STEP 2 COMPLETE")
print("=" * 75)

print(
    "Runtime:",
    elapsed(),
    "seconds"
)