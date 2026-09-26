import os
import re
import duckdb
import pandas as pd
import numpy as np
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = "normalized_cache_test"

S1_FILE = f"{CACHE_DIR}/source1_normalized.parquet"
S2_FILE = f"{CACHE_DIR}/source2_normalized.parquet"
S3_FILE = f"{CACHE_DIR}/source3_normalized.parquet"

CANDIDATE_FILE = "step4_test_candidate_pairs.csv"

OUTPUT_FILE = "step5_test_features.csv"


# ============================================================
# TEXT HELPERS
# ============================================================

def tokenize(text):
    if not text:
        return set()

    return set(
        token
        for token in str(text).split()
        if len(token) >= 2
    )


def char_similarity(a, b):
    if not a or not b:
        return 0.0

    a = str(a)
    b = str(b)

    if a == b:
        return 1.0

    # SequenceMatcher
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(a, b):
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / min(len(ta), len(tb))


def length_ratio(a, b):
    la = len(a) if a else 0
    lb = len(b) if b else 0

    if la == 0 and lb == 0:
        return 1.0

    if la == 0 or lb == 0:
        return 0.0

    return min(la, lb) / max(la, lb)


def prefix_match(a, b, n=4):
    if not a or not b:
        return 0.0

    return float(a[:n] == b[:n])


def suffix_match(a, b, n=4):
    if not a or not b:
        return 0.0

    return float(a[-n:] == b[-n:])


def shared_token_count(a, b):
    return float(len(tokenize(a) & tokenize(b)))


def extract_numbers(text):
    if not text:
        return set()

    return set(re.findall(r"\d+", str(text)))


def number_overlap(a, b):
    na = extract_numbers(a)
    nb = extract_numbers(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / min(len(na), len(nb))


def first_number_match(a, b):
    na = re.findall(r"\d+", str(a)) if a else []
    nb = re.findall(r"\d+", str(b)) if b else []

    if not na or not nb:
        return 0.0

    return float(na[0] == nb[0])


def digit_similarity(a, b):
    da = "".join(re.findall(r"\d", str(a))) if a else ""
    db = "".join(re.findall(r"\d", str(b))) if b else ""

    if not da and not db:
        return 1.0

    if not da or not db:
        return 0.0

    from difflib import SequenceMatcher
    return SequenceMatcher(None, da, db).ratio()


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("=" * 70)
print("STEP 5 - TEST FEATURE GENERATION")
print("=" * 70)

print("\nLoading candidates...")

pairs = pd.read_csv(
    CANDIDATE_FILE,
    header=None,
    names=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "candidate_index",
    ]
)
print("Candidate rows:", len(pairs))
print("Columns:", list(pairs.columns))


# ============================================================
# NORMALIZE CANDIDATE COLUMN NAMES
# ============================================================

# Expected:
# source1_entity_id
# candidate_entity_id
# candidate_source
# candidate_index

pairs["candidate_source"] = pairs["candidate_source"].astype(str)

print("\nCandidate sources:")
print(pairs["candidate_source"].value_counts())


# ============================================================
# FETCH ONLY REQUIRED RECORDS
# ============================================================

print("\nConnecting DuckDB...")

con = duckdb.connect()

con.execute("SET threads=2")
con.execute("SET memory_limit='4GB'")
con.execute("SET preserve_insertion_order=false")

con.execute(f"""
CREATE OR REPLACE VIEW s1 AS
SELECT *
FROM read_parquet('{S1_FILE}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW s2 AS
SELECT *
FROM read_parquet('{S2_FILE}')
""")

con.execute(f"""
CREATE OR REPLACE VIEW s3 AS
SELECT *
FROM read_parquet('{S3_FILE}')
""")


# ============================================================
# SPLIT REQUIRED IDS
# ============================================================

s1_ids = pairs["source1_entity_id"].astype(str).unique()

s2_ids = pairs.loc[
    pairs["candidate_source"] == "source2",
    "candidate_entity_id"
].astype(str).unique()

s3_ids = pairs.loc[
    pairs["candidate_source"] == "source3",
    "candidate_entity_id"
].astype(str).unique()

print("\nRequired records:")
print("Source1:", len(s1_ids))
print("Source2:", len(s2_ids))
print("Source3:", len(s3_ids))


# ============================================================
# FETCH SOURCE1
# ============================================================

print("\nFetching Source1 records...")

s1_df = con.execute("""
SELECT
    entity_id,
    name_norm,
    addr_norm,
    country_norm
FROM s1
WHERE entity_id IN (
    SELECT UNNEST(?)
)
""", [list(s1_ids)]).fetchdf()

s1_df = s1_df.rename(columns={
    "entity_id": "source1_entity_id",
    "name_norm": "s1_name",
    "addr_norm": "s1_addr",
    "country_norm": "s1_country"
})

print("Fetched Source1:", len(s1_df))


# ============================================================
# FETCH SOURCE2
# ============================================================

print("\nFetching Source2 records...")

s2_df = con.execute("""
SELECT
    entity_id,
    name_norm,
    addr_norm,
    country_norm
FROM s2
WHERE entity_id IN (
    SELECT UNNEST(?)
)
""", [list(s2_ids)]).fetchdf()

s2_df = s2_df.rename(columns={
    "entity_id": "candidate_entity_id",
    "name_norm": "candidate_name",
    "addr_norm": "candidate_addr",
    "country_norm": "candidate_country"
})

s2_df["candidate_source"] = "source2"

print("Fetched Source2:", len(s2_df))


# ============================================================
# FETCH SOURCE3
# ============================================================

print("\nFetching Source3 records...")

s3_df = con.execute("""
SELECT
    entity_id,
    name_norm,
    addr_norm,
    country_norm
FROM s3
WHERE entity_id IN (
    SELECT UNNEST(?)
)
""", [list(s3_ids)]).fetchdf()

s3_df = s3_df.rename(columns={
    "entity_id": "candidate_entity_id",
    "name_norm": "candidate_name",
    "addr_norm": "candidate_addr",
    "country_norm": "candidate_country"
})

s3_df["candidate_source"] = "source3"

print("Fetched Source3:", len(s3_df))


# ============================================================
# COMBINE CANDIDATE RECORDS
# ============================================================

candidate_records = pd.concat(
    [s2_df, s3_df],
    ignore_index=True
)

print("\nCandidate records:", len(candidate_records))


# ============================================================
# JOIN
# ============================================================

print("\nJoining candidates with Source1...")

df = pairs.merge(
    s1_df,
    on="source1_entity_id",
    how="left"
)

df = df.merge(
    candidate_records,
    on=["candidate_entity_id", "candidate_source"],
    how="left"
)

print("Joined rows:", len(df))

missing = df["candidate_name"].isna().sum()

if missing:
    print("WARNING: missing candidate records:", missing)


# ============================================================
# BUILD TOKEN FREQUENCY
# ============================================================

print("\nBuilding pair-level token vocabulary...")

name_counter = Counter()
addr_counter = Counter()

for x in pd.concat([
    df["s1_name"],
    df["candidate_name"]
]).fillna(""):

    name_counter.update(tokenize(x))

for x in pd.concat([
    df["s1_addr"],
    df["candidate_addr"]
]).fillna(""):

    addr_counter.update(tokenize(x))


def weighted_overlap(a, b, counter):
    ta = tokenize(a)
    tb = tokenize(b)

    shared = ta & tb

    if not shared:
        return 0.0

    total_a = sum(1.0 / counter.get(t, 1) for t in ta)
    total_b = sum(1.0 / counter.get(t, 1) for t in tb)

    shared_weight = sum(
        1.0 / counter.get(t, 1)
        for t in shared
    )

    denom = min(total_a, total_b)

    if denom == 0:
        return 0.0

    return shared_weight / denom


print(
    "Name vocabulary:",
    len(name_counter)
)

print(
    "Address vocabulary:",
    len(addr_counter)
)


# ============================================================
# FEATURE GENERATION
# ============================================================

print("\nGenerating 21 features...")

features = []

for row in df.itertuples(index=False):

    s1_name = row.s1_name or ""
    c_name = row.candidate_name or ""

    s1_addr = row.s1_addr or ""
    c_addr = row.candidate_addr or ""

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_exact = float(
        bool(s1_name) and
        bool(c_name) and
        s1_name == c_name
    )

    name_char_sim = char_similarity(
        s1_name,
        c_name
    )

    name_token_jaccard = token_jaccard(
        s1_name,
        c_name
    )

    name_token_overlap = token_overlap(
        s1_name,
        c_name
    )

    name_length_ratio = length_ratio(
        s1_name,
        c_name
    )

    name_prefix4 = prefix_match(
        s1_name,
        c_name
    )

    name_suffix4 = suffix_match(
        s1_name,
        c_name
    )

    name_shared_tokens = shared_token_count(
        s1_name,
        c_name
    )

    name_weighted_overlap = weighted_overlap(
        s1_name,
        c_name,
        name_counter
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    addr_exact = float(
        bool(s1_addr) and
        bool(c_addr) and
        s1_addr == c_addr
    )

    addr_char_sim = char_similarity(
        s1_addr,
        c_addr
    )

    addr_token_jaccard = token_jaccard(
        s1_addr,
        c_addr
    )

    addr_token_overlap = token_overlap(
        s1_addr,
        c_addr
    )

    addr_length_ratio = length_ratio(
        s1_addr,
        c_addr
    )

    addr_prefix4 = prefix_match(
        s1_addr,
        c_addr
    )

    addr_suffix4 = suffix_match(
        s1_addr,
        c_addr
    )

    addr_shared_tokens = shared_token_count(
        s1_addr,
        c_addr
    )

    addr_weighted_overlap = weighted_overlap(
        s1_addr,
        c_addr,
        addr_counter
    )

    # --------------------------------------------------------
    # NUMBER FEATURES
    # --------------------------------------------------------

    number_overlap_value = number_overlap(
        s1_addr,
        c_addr
    )

    first_number_match_value = first_number_match(
        s1_addr,
        c_addr
    )

    digit_similarity_value = digit_similarity(
        s1_addr,
        c_addr
    )

    features.append([
        name_exact,
        name_char_sim,
        name_token_jaccard,
        name_token_overlap,
        name_length_ratio,
        addr_exact,
        addr_char_sim,
        addr_token_jaccard,
        addr_token_overlap,
        addr_length_ratio,
        name_prefix4,
        name_suffix4,
        name_shared_tokens,
        name_weighted_overlap,
        addr_prefix4,
        addr_suffix4,
        addr_shared_tokens,
        addr_weighted_overlap,
        number_overlap_value,
        first_number_match_value,
        digit_similarity_value
    ])


# ============================================================
# FEATURE DATAFRAME
# ============================================================

FEATURE_COLUMNS = [
    "feature_0_name_exact",
    "feature_1_name_char_sim",
    "feature_2_name_token_jaccard",
    "feature_3_name_token_overlap",
    "feature_4_name_length_ratio",
    "feature_5_addr_exact",
    "feature_6_addr_char_sim",
    "feature_7_addr_token_jaccard",
    "feature_8_addr_token_overlap",
    "feature_9_addr_length_ratio",
    "feature_10_name_prefix4",
    "feature_11_name_suffix4",
    "feature_12_name_shared_tokens",
    "feature_13_name_weighted_overlap",
    "feature_14_addr_prefix4",
    "feature_15_addr_suffix4",
    "feature_16_addr_shared_tokens",
    "feature_17_addr_weighted_overlap",
    "feature_18_number_overlap",
    "feature_19_first_number_match",
    "feature_20_digit_similarity",
]

feature_df = pd.DataFrame(
    features,
    columns=FEATURE_COLUMNS
)


# ============================================================
# FINAL OUTPUT
# ============================================================

output_df = pd.concat(
    [
        pairs.reset_index(drop=True),
        feature_df
    ],
    axis=1
)

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print("Rows:", len(output_df))
print("Features:", len(FEATURE_COLUMNS))
print("Output:", OUTPUT_FILE)

print("\nFeature columns:")
for i, col in enumerate(FEATURE_COLUMNS):
    print(i, col)

print("\nFirst 5 rows:")
print(output_df.head())

con.close()