import os
import re
import math
import difflib
import duckdb
import pandas as pd
import numpy as np


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


def tokens(x):
    x = safe_text(x)
    if not x:
        return set()
    return set(t for t in x.split() if t)


def char_sim(a, b):
    a = safe_text(a)
    b = safe_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def token_jaccard(a, b):
    ta = tokens(a)
    tb = tokens(b)

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):
    ta = tokens(a)
    tb = tokens(b)

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
    return float(len(tokens(a) & tokens(b)))


def weighted_overlap(a, b, weights):
    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    common = ta & tb

    if not common:
        return 0.0

    numerator = sum(weights.get(t, 1.0) for t in common)

    denom = min(
        sum(weights.get(t, 1.0) for t in ta),
        sum(weights.get(t, 1.0) for t in tb),
    )

    if denom <= 0:
        return 0.0

    return numerator / denom


def numbers(x):
    x = safe_text(x)
    return set(re.findall(r"\d+", x))


def number_overlap(a, b):
    na = numbers(a)
    nb = numbers(b)

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
# LOAD PAIRS
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
# DUCKDB
# ============================================================

print("\nStarting DuckDB...")

con = duckdb.connect()

con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='5GB'")
con.execute("PRAGMA preserve_insertion_order=false")


# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("\nBuilding token frequencies for weighted overlap...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE token_frequency AS

    WITH s1_tokens AS (
        SELECT DISTINCT
            entity_id,
            token
        FROM read_parquet('{S1_PATH}') s
        CROSS JOIN UNNEST(
            string_split(
                TRIM(
                    CONCAT_WS(
                        ' ',
                        COALESCE(s.name_norm, ''),
                        COALESCE(s.addr_norm, '')
                    )
                ),
                ' '
            )
        ) AS t(token)
        WHERE token <> ''
    ),

    s2_tokens AS (
        SELECT DISTINCT
            entity_id,
            token
        FROM read_parquet('{S2_PATH}') s
        CROSS JOIN UNNEST(
            string_split(
                TRIM(
                    CONCAT_WS(
                        ' ',
                        COALESCE(s.name_norm, ''),
                        COALESCE(s.addr_norm, '')
                    )
                ),
                ' '
            )
        ) AS t(token)
        WHERE token <> ''
    ),

    s3_tokens AS (
        SELECT DISTINCT
            entity_id,
            token
        FROM read_parquet('{S3_PATH}') s
        CROSS JOIN UNNEST(
            string_split(
                TRIM(
                    CONCAT_WS(
                        ' ',
                        COALESCE(s.name_norm, ''),
                        COALESCE(s.addr_norm, '')
                    )
                ),
                ' '
            )
        ) AS t(token)
        WHERE token <> ''
    ),

    all_tokens AS (
        SELECT entity_id, token FROM s1_tokens
        UNION ALL
        SELECT entity_id, token FROM s2_tokens
        UNION ALL
        SELECT entity_id, token FROM s3_tokens
    )

    SELECT
        token,
        COUNT(*) AS freq
    FROM all_tokens
    GROUP BY token
""")

freq_df = con.execute("""
    SELECT token, freq
    FROM token_frequency
""").fetchdf()

print("Unique tokens:", f"{len(freq_df):,}")

# Same weighting idea used by the earlier model.
weights = {}

for row in freq_df.itertuples(index=False):
    weights[row.token] = math.log(
        1000000.0 / (float(row.freq) + 1.0)
    )

del freq_df


# ============================================================
# FUNCTION TO FETCH RECORDS
# ============================================================

def prepare_pairs(pairs, name):

    print("\n" + "-" * 70)
    print("Preparing", name)
    print("-" * 70)

    pairs = pairs.copy()

    pairs["_row_id"] = np.arange(len(pairs))

    # --------------------------------------------------------
    # S1 records
    # --------------------------------------------------------

    con.register("pair_input", pairs)

    s1_df = con.execute(f"""
        SELECT
            p._row_id,

            s1.entity_id AS source1_entity_id,
            s1.name_norm AS s1_name,
            s1.addr_norm AS s1_addr,
            s1.country_norm AS s1_country

        FROM pair_input p

        INNER JOIN read_parquet('{S1_PATH}') s1
            ON s1.entity_id = p.source1_entity_id
    """).fetchdf()

    # --------------------------------------------------------
    # Candidate records
    # --------------------------------------------------------

    s2_df = con.execute(f"""
        SELECT
            ROW_NUMBER() OVER () - 1 AS candidate_index,
            entity_id,
            name_norm,
            addr_norm,
            country_norm

        FROM read_parquet('{S2_PATH}')
    """).fetchdf()

    s3_df = con.execute(f"""
        SELECT
            ROW_NUMBER() OVER () - 1 AS candidate_index,
            entity_id,
            name_norm,
            addr_norm,
            country_norm

        FROM read_parquet('{S3_PATH}')
    """).fetchdf()

    # Candidate index lookup.
    s2_df = s2_df.set_index("candidate_index")
    s3_df = s3_df.set_index("candidate_index")

    rows = []

    print("Calculating 21 features...")

    for r in pairs.itertuples(index=False):

        row_id = r._row_id

        s1 = s1_df.iloc[row_id]

        s1_name = safe_text(s1.s1_name)
        s1_addr = safe_text(s1.s1_addr)
        s1_country = safe_text(s1.s1_country)

        candidate_index = int(r.candidate_index)
        candidate_source = int(r.candidate_source)

        if candidate_source == 0:
            cand = s2_df.loc[candidate_index]
        else:
            cand = s3_df.loc[candidate_index]

        c_name = safe_text(cand.name_norm)
        c_addr = safe_text(cand.addr_norm)
        c_country = safe_text(cand.country_norm)

        # ----------------------------------------------------
        # 21 FEATURES
        # ----------------------------------------------------

        feature_values = [

            # Name
            float(s1_name == c_name),
            char_sim(s1_name, c_name),
            token_jaccard(s1_name, c_name),
            token_overlap(s1_name, c_name),
            length_ratio(s1_name, c_name),

            # Address
            float(s1_addr == c_addr),
            char_sim(s1_addr, c_addr),
            token_jaccard(s1_addr, c_addr),
            token_overlap(s1_addr, c_addr),
            length_ratio(s1_addr, c_addr),

            # Extra name features
            prefix4(s1_name, c_name),
            suffix4(s1_name, c_name),
            shared_tokens(s1_name, c_name),
            weighted_overlap(s1_name, c_name, weights),

            # Extra address features
            prefix4(s1_addr, c_addr),
            suffix4(s1_addr, c_addr),
            shared_tokens(s1_addr, c_addr),
            weighted_overlap(s1_addr, c_addr, weights),

            # Numeric features
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

        rows.append(feature_values)

    feature_columns = [

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

    feature_df = pd.DataFrame(
        rows,
        columns=feature_columns
    )

    # IMPORTANT:
    # Keep the label separate from the feature matrix.
    if "label" in pairs.columns:
        feature_df["label"] = pairs["label"].values

    print(
        name,
        "features:",
        feature_df.shape
    )

    return feature_df


# ============================================================
# GENERATE FEATURES
# ============================================================

train_features = prepare_pairs(
    train_pairs,
    "TRAIN"
)

valid_features = prepare_pairs(
    valid_pairs,
    "VALID"
)


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

print("\nTrain:", train_features.shape)
print("Valid:", valid_features.shape)

print("\nFeature columns:")

for i, c in enumerate(
    [c for c in train_features.columns if c != "label"]
):
    print(f"  {i}: {c}")

print("\n" + "=" * 70)
print("21-FEATURE GENERATION COMPLETE")
print("=" * 70)

con.close()