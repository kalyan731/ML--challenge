import os
import re
import time
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score


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

TRAIN_PAIRS = os.path.join(
    BASE,
    "step4_train_pairs.csv",
)

VALID_PAIRS = os.path.join(
    BASE,
    "step4_valid_pairs.csv",
)

MIN_TOKEN_LEN = 2


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
        flags=re.UNICODE
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x


def get_tokens(text):

    return {
        token
        for token in text.split()
        if len(token) >= MIN_TOKEN_LEN
    }


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

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
            text
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

    return float(next(iter(na)) in nb)


# ============================================================
# LOAD
# ============================================================

start = time.time()

print("=" * 75)
print("STEP 4.2 - FEATURE ENGINEERING + BASELINE ML")
print("=" * 75)

print("\nLoading Source1...")

s1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    dtype=str,
)

print("Loading Source2...")

s2 = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str,
)

print("Loading Source3...")

s3 = pd.read_csv(
    SOURCE3,
    sep="\t",
    dtype=str,
)

print("Loading training pairs...")

train_pairs = pd.read_csv(
    TRAIN_PAIRS,
)

print("Loading validation pairs...")

valid_pairs = pd.read_csv(
    VALID_PAIRS,
)


# ============================================================
# CHECK PAIR SCHEMA
# ============================================================

required_pair_cols = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "candidate_index",
    "label",
}

for name, df in {
    "train_pairs": train_pairs,
    "valid_pairs": valid_pairs,
}.items():

    missing = required_pair_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


# ============================================================
# CHECK SOURCE SCHEMA
# ============================================================

required_source_cols = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

for name, df in {
    "source1": s1,
    "source2": s2,
    "source3": s3,
}.items():

    missing = required_source_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


print("\nShapes:")
print("Source1:", s1.shape)
print("Source2:", s2.shape)
print("Source3:", s3.shape)
print("Train pairs:", train_pairs.shape)
print("Valid pairs:", valid_pairs.shape)


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing source data...")

for df in [s1, s2, s3]:

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
        .map(normalize_text)
    )


# ============================================================
# DIRECT SOURCE1 LOOKUP
# ============================================================

print("\nCreating Source1 ID lookup...")

s1_id_to_idx = pd.Series(
    s1.index,
    index=s1["entity_id"],
).to_dict()

print(
    "S1 IDs:",
    len(s1_id_to_idx)
)


# ============================================================
# CONVERT SOURCE DATA TO ARRAYS
# ============================================================

s1_name = s1["name_norm"].to_numpy()
s1_addr = s1["addr_norm"].to_numpy()
s1_country = s1["country_norm"].to_numpy()

s2_name = s2["name_norm"].to_numpy()
s2_addr = s2["addr_norm"].to_numpy()
s2_country = s2["country_norm"].to_numpy()

s3_name = s3["name_norm"].to_numpy()
s3_addr = s3["addr_norm"].to_numpy()
s3_country = s3["country_norm"].to_numpy()


# ============================================================
# FEATURE BUILDER
# ============================================================

FEATURE_NAMES = [
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
    "number_overlap",
    "first_number_match",
    "country_exact",
    "candidate_source",
]


def build_features(pairs):

    features = []

    total = len(pairs)

    for pos, row in enumerate(
        pairs.itertuples(index=False),
        start=1,
    ):

        s1_id = row.source1_entity_id

        source = int(row.candidate_source)
        candidate_idx = int(row.candidate_index)

        s1_idx = s1_id_to_idx.get(s1_id)

        if s1_idx is None:
            raise ValueError(
                f"Source1 ID not found: {s1_id}"
            )

        n1 = s1_name[s1_idx]
        a1 = s1_addr[s1_idx]
        c1 = s1_country[s1_idx]

        if source == 0:

            n2 = s2_name[candidate_idx]
            a2 = s2_addr[candidate_idx]
            c2 = s2_country[candidate_idx]

        elif source == 1:

            n2 = s3_name[candidate_idx]
            a2 = s3_addr[candidate_idx]
            c2 = s3_country[candidate_idx]

        else:

            raise ValueError(
                f"Invalid candidate_source: {source}"
            )

        row_features = [
            float(n1 == n2 and n1 != ""),
            ratio(n1, n2) / 100.0,
            token_jaccard(n1, n2),
            token_overlap(n1, n2),
            length_ratio(n1, n2),

            float(a1 == a2 and a1 != ""),
            ratio(a1, a2) / 100.0,
            token_jaccard(a1, a2),
            token_overlap(a1, a2),
            length_ratio(a1, a2),

            number_overlap(a1, a2),
            first_number_match(a1, a2),

            float(c1 == c2 and c1 != ""),

            float(source),
        ]

        features.append(row_features)

        if pos % 5000 == 0 or pos == total:

            print(
                f"Features: {pos:,}/{total:,}"
            )

    return np.asarray(
        features,
        dtype=np.float32,
    )


# ============================================================
# BUILD FEATURES
# ============================================================

print("\nBuilding training features...")

X_train = build_features(
    train_pairs
)

y_train = train_pairs["label"].to_numpy(
    dtype=np.int8
)

print("\nTraining feature matrix:")
print(X_train.shape)


print("\nBuilding validation features...")

X_valid = build_features(
    valid_pairs
)

y_valid = valid_pairs["label"].to_numpy(
    dtype=np.int8
)

print("\nValidation feature matrix:")
print(X_valid.shape)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\nFeature summary:")

for i, name in enumerate(FEATURE_NAMES):

    print(
        f"{name:25s}"
        f" mean={X_train[:, i].mean():.4f}"
        f" min={X_train[:, i].min():.4f}"
        f" max={X_train[:, i].max():.4f}"
    )


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
)

model.fit(
    X_train,
    y_train,
)

print("Model trained.")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print("\nPredicting validation probabilities...")

valid_prob = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# F0.5 FUNCTION
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
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )

    return score, precision, recall


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nSearching thresholds...")

best_threshold = None
best_score = -1.0
best_precision = 0.0
best_recall = 0.0

for threshold in np.arange(
    0.10,
    0.96,
    0.02,
):

    pred = (
        valid_prob >= threshold
    ).astype(np.int8)

    score, precision, recall = f05_score(
        y_valid,
        pred,
    )

    if score > best_score:

        best_score = score
        best_threshold = float(threshold)
        best_precision = precision
        best_recall = recall


# ============================================================
# FINAL ROW-LEVEL RESULT
# ============================================================

print("\n" + "=" * 75)
print("STEP 4.2 RESULTS")
print("=" * 75)

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Precision:       {best_precision:.6f}"
)

print(
    f"Recall:          {best_recall:.6f}"
)

print(
    f"F0.5:            {best_score:.6f}"
)

print(
    f"Positive rate:   "
    f"{np.mean(valid_prob >= best_threshold):.4f}"
)


# ============================================================
# ENTITY-LEVEL F0.5
# ============================================================

print("\nCalculating entity-level F0.5...")

valid_eval = valid_pairs.copy()

valid_eval["probability"] = valid_prob

valid_eval["prediction"] = (
    valid_eval["probability"]
    >= best_threshold
).astype(np.int8)


entity_scores = []

for source1_id, group in valid_eval.groupby(
    "source1_entity_id",
    sort=False,
):

    true_labels = group["label"].to_numpy(
        dtype=np.int8
    )

    predicted_labels = group["prediction"].to_numpy(
        dtype=np.int8
    )

    score, precision, recall = f05_score(
        true_labels,
        predicted_labels,
    )

    entity_scores.append(score)


entity_f05 = float(
    np.mean(entity_scores)
)

print(
    f"Validation entities: {len(entity_scores):,}"
)

print(
    f"Entity-level macro F0.5: "
    f"{entity_f05:.6f}"
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\nModel coefficients:")

coefficients = model.coef_[0]

for name, coefficient in sorted(
    zip(FEATURE_NAMES, coefficients),
    key=lambda x: abs(x[1]),
    reverse=True,
):

    print(
        f"{name:25s}"
        f"{coefficient:+.6f}"
    )


# ============================================================
# SAVE MODEL FEATURES
# ============================================================

feature_columns = [
    f"feature_{i}_{name}"
    for i, name in enumerate(FEATURE_NAMES)
]

train_feature_df = pd.DataFrame(
    X_train,
    columns=feature_columns,
)

train_feature_df["label"] = y_train

valid_feature_df = pd.DataFrame(
    X_valid,
    columns=feature_columns,
)

valid_feature_df["label"] = y_valid

train_feature_df.to_csv(
    os.path.join(
        BASE,
        "step4_train_features.csv",
    ),
    index=False,
)

valid_feature_df.to_csv(
    os.path.join(
        BASE,
        "step4_valid_features.csv",
    ),
    index=False,
)


# ============================================================
# DONE
# ============================================================

elapsed = time.time() - start

print("\nSaved:")
print("step4_train_features.csv")
print("step4_valid_features.csv")

print(
    f"\nTotal runtime: {elapsed:.2f} seconds"
)

print("=" * 75)
print("DONE")
print("=" * 75)