import os
import re
import time
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\kulth\OneDrive\ML CHALLENGE"

CACHE_DIR = os.path.join(
    BASE,
    "normalized_cache",
)

S1_CACHE = os.path.join(
    CACHE_DIR,
    "source1_normalized.parquet",
)

S2_CACHE = os.path.join(
    CACHE_DIR,
    "source2_normalized.parquet",
)

S3_CACHE = os.path.join(
    CACHE_DIR,
    "source3_normalized.parquet",
)

TRAIN_PAIRS = os.path.join(
    BASE,
    "step4_train_pairs.csv",
)

VALID_PAIRS = os.path.join(
    BASE,
    "step4_valid_pairs.csv",
)

TRAIN_FEATURES_OUT = os.path.join(
    BASE,
    "step4_3_train_features.csv",
)

VALID_FEATURES_OUT = os.path.join(
    BASE,
    "step4_3_valid_features.csv",
)

MIN_TOKEN_LEN = 2


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def get_tokens(text):

    return {
        token
        for token in str(text).split()
        if len(token) >= MIN_TOKEN_LEN
    }


def token_jaccard(a, b):

    ta = get_tokens(a)
    tb = get_tokens(b)

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):

    ta = get_tokens(a)
    tb = get_tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / min(len(ta), len(tb))


def length_ratio(a, b):

    la = len(a)
    lb = len(b)

    if la == 0 and lb == 0:
        return 1.0

    if la == 0 or lb == 0:
        return 0.0

    return min(la, lb) / max(la, lb)


def extract_numbers(text):

    return set(
        re.findall(
            r"\d+",
            text,
        )
    )


def number_overlap(a, b):

    na = extract_numbers(a)
    nb = extract_numbers(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / min(len(na), len(nb))


def first_number_match(a, b):

    na = extract_numbers(a)
    nb = extract_numbers(b)

    if not na or not nb:
        return 0.0

    return float(
        next(iter(na)) in nb
    )


# ============================================================
# ADDITIONAL FEATURES
# ============================================================

def prefix_similarity(a, b, n=4):

    if not a or not b:
        return 0.0

    return float(
        a[:n] == b[:n]
    )


def suffix_similarity(a, b, n=4):

    if not a or not b:
        return 0.0

    return float(
        a[-n:] == b[-n:]
    )


def shared_token_count(a, b):

    ta = get_tokens(a)
    tb = get_tokens(b)

    return float(
        len(ta & tb)
    )


def weighted_token_overlap(a, b):

    ta = get_tokens(a)
    tb = get_tokens(b)

    if not ta or not tb:
        return 0.0

    shared = ta & tb

    # Rare-token proxy:
    # longer tokens receive slightly more weight.
    total_weight = sum(
        np.log1p(len(t))
        for t in ta
    )

    if total_weight == 0:
        return 0.0

    shared_weight = sum(
        np.log1p(len(t))
        for t in shared
    )

    return shared_weight / total_weight


def digit_string_similarity(a, b):

    da = "".join(
        re.findall(r"\d+", a)
    )

    db = "".join(
        re.findall(r"\d+", b)
    )

    if not da or not db:
        return 0.0

    return ratio(da, db) / 100.0


# ============================================================
# F0.5
# ============================================================

def f05_score(y_true, y_pred):

    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision == 0.0 and recall == 0.0:
        return 0.0, precision, recall

    score = (
        1.25
        * precision
        * recall
        / (0.25 * precision + recall)
    )

    return score, precision, recall


# ============================================================
# START
# ============================================================

start = time.time()

print("=" * 75)
print("STEP 4.3 - CACHED FEATURES + STRONGER BASELINE")
print("=" * 75)


# ============================================================
# LOAD CACHED DATA
# ============================================================

print("\nLoading normalized Parquet cache...")

cache_start = time.time()

s1 = pd.read_parquet(
    S1_CACHE,
)

print(
    f"S1 loaded: {s1.shape}"
)

s2 = pd.read_parquet(
    S2_CACHE,
)

print(
    f"S2 loaded: {s2.shape}"
)

s3 = pd.read_parquet(
    S3_CACHE,
)

print(
    f"S3 loaded: {s3.shape}"
)

print(
    f"Cache load time: "
    f"{time.time() - cache_start:.2f}s"
)


# ============================================================
# LOAD PAIRS
# ============================================================

print("\nLoading pair files...")

train_pairs = pd.read_csv(
    TRAIN_PAIRS,
)

valid_pairs = pd.read_csv(
    VALID_PAIRS,
)

print(
    f"Train pairs: {train_pairs.shape}"
)

print(
    f"Valid pairs: {valid_pairs.shape}"
)


# ============================================================
# VALIDATE SCHEMA
# ============================================================

required_pair_cols = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "candidate_index",
    "label",
}

for name, df in {
    "train": train_pairs,
    "valid": valid_pairs,
}.items():

    missing = (
        required_pair_cols
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} pair file missing: "
            f"{sorted(missing)}"
        )


# ============================================================
# DIRECT S1 LOOKUP
# ============================================================

print("\nCreating Source1 ID lookup...")

s1_id_to_idx = pd.Series(
    s1.index,
    index=s1["entity_id"],
).to_dict()

print(
    f"S1 IDs: {len(s1_id_to_idx):,}"
)


# ============================================================
# ARRAYS
# ============================================================

s1_name = s1["name_norm"].to_numpy()
s1_addr = s1["addr_norm"].to_numpy()

s2_name = s2["name_norm"].to_numpy()
s2_addr = s2["addr_norm"].to_numpy()

s3_name = s3["name_norm"].to_numpy()
s3_addr = s3["addr_norm"].to_numpy()


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    "name_exact",

    "name_char_sim",

    "name_token_jaccard",

    "name_token_overlap",

    "name_length_ratio",

    "name_prefix4",

    "name_suffix4",

    "name_shared_tokens",

    "name_weighted_overlap",

    "addr_exact",

    "addr_char_sim",

    "addr_token_jaccard",

    "addr_token_overlap",

    "addr_length_ratio",

    "addr_prefix4",

    "addr_suffix4",

    "addr_shared_tokens",

    "addr_weighted_overlap",

    "number_overlap",

    "first_number_match",

    "digit_similarity",

]


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(pairs):

    rows = []

    total = len(pairs)

    for pos, row in enumerate(
        pairs.itertuples(index=False),
        start=1,
    ):

        s1_id = row.source1_entity_id

        source = int(
            row.candidate_source
        )

        candidate_idx = int(
            row.candidate_index
        )

        s1_idx = s1_id_to_idx.get(
            s1_id
        )

        if s1_idx is None:
            raise ValueError(
                f"S1 ID not found: {s1_id}"
            )

        n1 = s1_name[s1_idx]
        a1 = s1_addr[s1_idx]

        if source == 0:

            if (
                candidate_idx < 0
                or candidate_idx >= len(s2)
            ):
                raise IndexError(
                    f"Invalid S2 index: "
                    f"{candidate_idx}"
                )

            n2 = s2_name[candidate_idx]
            a2 = s2_addr[candidate_idx]

        elif source == 1:

            if (
                candidate_idx < 0
                or candidate_idx >= len(s3)
            ):
                raise IndexError(
                    f"Invalid S3 index: "
                    f"{candidate_idx}"
                )

            n2 = s3_name[candidate_idx]
            a2 = s3_addr[candidate_idx]

        else:

            raise ValueError(
                f"Invalid candidate_source: "
                f"{source}"
            )

        row_features = [

            # NAME
            float(
                n1 == n2
                and n1 != ""
            ),

            ratio(
                n1,
                n2,
            ) / 100.0,

            token_jaccard(
                n1,
                n2,
            ),

            token_overlap(
                n1,
                n2,
            ),

            length_ratio(
                n1,
                n2,
            ),

            prefix_similarity(
                n1,
                n2,
            ),

            suffix_similarity(
                n1,
                n2,
            ),

            shared_token_count(
                n1,
                n2,
            ),

            weighted_token_overlap(
                n1,
                n2,
            ),

            # ADDRESS
            float(
                a1 == a2
                and a1 != ""
            ),

            ratio(
                a1,
                a2,
            ) / 100.0,

            token_jaccard(
                a1,
                a2,
            ),

            token_overlap(
                a1,
                a2,
            ),

            length_ratio(
                a1,
                a2,
            ),

            prefix_similarity(
                a1,
                a2,
            ),

            suffix_similarity(
                a1,
                a2,
            ),

            shared_token_count(
                a1,
                a2,
            ),

            weighted_token_overlap(
                a1,
                a2,
            ),

            # NUMBERS
            number_overlap(
                a1,
                a2,
            ),

            first_number_match(
                a1,
                a2,
            ),

            digit_string_similarity(
                a1,
                a2,
            ),
        ]

        rows.append(
            row_features
        )

        if (
            pos % 5000 == 0
            or pos == total
        ):

            print(
                f"Features: "
                f"{pos:,}/{total:,}"
            )

    return np.asarray(
        rows,
        dtype=np.float32,
    )


# ============================================================
# TRAIN FEATURES
# ============================================================

print(
    "\nBuilding training features..."
)

X_train = build_features(
    train_pairs
)

y_train = train_pairs[
    "label"
].to_numpy(
    dtype=np.int8
)

print(
    f"Training matrix: "
    f"{X_train.shape}"
)


# ============================================================
# VALID FEATURES
# ============================================================

print(
    "\nBuilding validation features..."
)

X_valid = build_features(
    valid_pairs
)

y_valid = valid_pairs[
    "label"
].to_numpy(
    dtype=np.int8
)

print(
    f"Validation matrix: "
    f"{X_valid.shape}"
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print(
    "\nFeature summary:"
)

for i, name in enumerate(
    FEATURE_NAMES
):

    print(
        f"{name:25s} "
        f"mean={X_train[:, i].mean():.4f} "
        f"min={X_train[:, i].min():.4f} "
        f"max={X_train[:, i].max():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining Logistic Regression..."
)

model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    solver="lbfgs",
)

model.fit(
    X_train,
    y_train,
)

print(
    "Model trained."
)


# ============================================================
# PROBABILITIES
# ============================================================

print(
    "\nPredicting validation..."
)

valid_prob = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print(
    "\nSearching F0.5 threshold..."
)

best_threshold = 0.0
best_score = -1.0
best_precision = 0.0
best_recall = 0.0

for threshold in np.arange(
    0.10,
    0.991,
    0.01,
):

    pred = (
        valid_prob >= threshold
    ).astype(np.int8)

    score, precision, recall = (
        f05_score(
            y_valid,
            pred,
        )
    )

    if score > best_score:

        best_score = score
        best_threshold = float(
            threshold
        )
        best_precision = precision
        best_recall = recall


# ============================================================
# ROW LEVEL RESULT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 4.3 RESULTS"
)

print(
    "=" * 75
)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision:       "
    f"{best_precision:.6f}"
)

print(
    f"Recall:          "
    f"{best_recall:.6f}"
)

print(
    f"Pair-level F0.5:  "
    f"{best_score:.6f}"
)


# ============================================================
# ENTITY-LEVEL EVALUATION
# ============================================================

print(
    "\nCalculating entity-level F0.5..."
)

valid_eval = valid_pairs.copy()

valid_eval[
    "probability"
] = valid_prob

valid_eval[
    "prediction"
] = (
    valid_eval[
        "probability"
    ]
    >= best_threshold
).astype(np.int8)


entity_scores = []

for source1_id, group in (
    valid_eval.groupby(
        "source1_entity_id",
        sort=False,
    )
):

    true_labels = group[
        "label"
    ].to_numpy(
        dtype=np.int8
    )

    predicted_labels = group[
        "prediction"
    ].to_numpy(
        dtype=np.int8
    )

    score, _, _ = f05_score(
        true_labels,
        predicted_labels,
    )

    entity_scores.append(
        score
    )


entity_f05 = float(
    np.mean(entity_scores)
)

print(
    f"Validation entities: "
    f"{len(entity_scores):,}"
)

print(
    f"Entity-level macro F0.5: "
    f"{entity_f05:.6f}"
)


# ============================================================
# COMPARE BASELINE
# ============================================================

BASELINE = 0.865432

print(
    "\nComparison with Step 4.2:"
)

print(
    f"Previous baseline: "
    f"{BASELINE:.6f}"
)

print(
    f"Step 4.3:          "
    f"{entity_f05:.6f}"
)

print(
    f"Difference:        "
    f"{entity_f05 - BASELINE:+.6f}"
)


# ============================================================
# COEFFICIENTS
# ============================================================

print(
    "\nModel coefficients:"
)

coefficients = model.coef_[0]

for name, coefficient in sorted(
    zip(
        FEATURE_NAMES,
        coefficients,
    ),
    key=lambda x: abs(x[1]),
    reverse=True,
):

    print(
        f"{name:25s}"
        f"{coefficient:+.6f}"
    )


# ============================================================
# SAVE FEATURES
# ============================================================

print(
    "\nSaving feature matrices..."
)

feature_columns = [
    f"feature_{i}_{name}"
    for i, name in enumerate(
        FEATURE_NAMES
    )
]

train_feature_df = pd.DataFrame(
    X_train,
    columns=feature_columns,
)

train_feature_df[
    "label"
] = y_train

valid_feature_df = pd.DataFrame(
    X_valid,
    columns=feature_columns,
)

valid_feature_df[
    "label"
] = y_valid

train_feature_df.to_csv(
    TRAIN_FEATURES_OUT,
    index=False,
)

valid_feature_df.to_csv(
    VALID_FEATURES_OUT,
    index=False,
)

print(
    f"Saved: {TRAIN_FEATURES_OUT}"
)

print(
    f"Saved: {VALID_FEATURES_OUT}"
)


# ============================================================
# DONE
# ============================================================

elapsed = time.time() - start

print(
    "\nTotal runtime: "
    f"{elapsed:.2f} seconds"
)

print(
    "=" * 75
)

print(
    "DONE"
)

print(
    "=" * 75
)