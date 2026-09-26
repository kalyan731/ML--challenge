import time
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)


START = time.time()

print("=" * 75)
print("STEP 4.4 - XGBOOST")
print("=" * 75)


# ================================================================
# FILES
# ================================================================

TRAIN_FILE = "step4_3_train_features.csv"
VALID_FILE = "step4_3_valid_features.csv"

# The feature CSV does not contain source1_entity_id.
# We recover it from the original validation pair file.
POSSIBLE_VALID_PAIR_FILES = [
    "step4_1_valid_pairs.csv",
    "step4_valid_pairs.csv",
    "valid_pairs.csv",
]


# ================================================================
# FIND VALIDATION PAIR FILE
# ================================================================

print("\nLocating validation pair file...")

valid_pair_file = None

for filename in POSSIBLE_VALID_PAIR_FILES:
    if Path(filename).exists():
        valid_pair_file = filename
        break

# If none of the expected names exist, search automatically.
if valid_pair_file is None:
    candidates = sorted(Path(".").glob("*valid*pairs*.csv"))

    if candidates:
        valid_pair_file = str(candidates[0])

if valid_pair_file is None:
    raise FileNotFoundError(
        "Could not find validation pair CSV.\n"
        "Expected something like:\n"
        "  step4_1_valid_pairs.csv\n"
        "Run: dir *valid*pairs*.csv"
    )

print(f"Validation pair file: {valid_pair_file}")


# ================================================================
# LOAD FEATURES
# ================================================================

print("\nLoading feature matrices...")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print(f"Train: {train_df.shape}")
print(f"Valid: {valid_df.shape}")


# ================================================================
# LOAD ORIGINAL VALIDATION PAIRS
# ================================================================

print("\nLoading validation pair metadata...")

valid_pairs = pd.read_csv(valid_pair_file)

print(f"Validation pairs: {valid_pairs.shape}")

if len(valid_pairs) != len(valid_df):
    raise ValueError(
        f"\nValidation row mismatch!\n"
        f"Feature rows: {len(valid_df):,}\n"
        f"Pair rows:    {len(valid_pairs):,}\n"
        f"\nThe feature CSV and pair CSV must have exactly "
        f"the same row order and row count."
    )

if "source1_entity_id" not in valid_pairs.columns:
    raise KeyError(
        "The validation pair file does not contain "
        "'source1_entity_id'.\n"
        f"Columns found: {list(valid_pairs.columns)}"
    )


# ================================================================
# FEATURES
# ================================================================

TARGET = "label"

DROP_COLS = {
    TARGET,
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "candidate_index",
}

FEATURES = [
    c for c in train_df.columns
    if c not in DROP_COLS
]

print(f"\nFeatures: {len(FEATURES)}")

for feature in FEATURES:
    print(f"  {feature}")


# ================================================================
# PREPARE MATRICES
# ================================================================

X_train = train_df[FEATURES].astype(np.float32)
y_train = train_df[TARGET].astype(np.int8)

X_valid = valid_df[FEATURES].astype(np.float32)
y_valid = valid_df[TARGET].astype(np.int8)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")

train_positive = int(y_train.sum())
train_negative = int((y_train == 0).sum())

valid_positive = int(y_valid.sum())
valid_negative = int((y_valid == 0).sum())

print(f"Train positives: {train_positive:,}")
print(f"Train negatives: {train_negative:,}")
print(f"Valid positives: {valid_positive:,}")
print(f"Valid negatives: {valid_negative:,}")

scale_pos_weight = train_negative / train_positive

print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ================================================================
# TRAIN XGBOOST
# ================================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.0,
    reg_alpha=0.05,
    reg_lambda=2.0,

    objective="binary:logistic",
    eval_metric="logloss",

    # Try GPU first.
    # XGBoost will fall back to CPU if unavailable.
    tree_method="hist",
    device="cuda",

    random_state=42,
    n_jobs=8,

    scale_pos_weight=scale_pos_weight,
)


try:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

except Exception as e:

    print("\nCUDA training failed.")
    print("Falling back to CPU...")
    print(f"Reason: {e}")

    model.set_params(
        device="cpu",
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )


print("Model trained.")


# ================================================================
# PREDICT
# ================================================================

print("\nPredicting validation...")

probs = model.predict_proba(X_valid)[:, 1]


# ================================================================
# THRESHOLD SEARCH
# ================================================================

print("\nSearching F0.5 threshold...")

best_threshold = None
best_pair_f05 = -1.0
best_precision = 0.0
best_recall = 0.0

threshold_results = []

for threshold in np.arange(0.50, 0.991, 0.01):

    pred = (
        probs >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_valid,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        pred,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_valid,
        pred,
        beta=0.5,
        zero_division=0,
    )

    threshold_results.append(
        (
            float(threshold),
            precision,
            recall,
            f05,
        )
    )

    if f05 > best_pair_f05:

        best_pair_f05 = f05
        best_threshold = float(threshold)
        best_precision = precision
        best_recall = recall


# ================================================================
# PAIR RESULTS
# ================================================================

print("\n" + "=" * 75)
print("STEP 4.4 RESULTS")
print("=" * 75)

print(f"Best threshold: {best_threshold:.2f}")
print(f"Precision:       {best_precision:.6f}")
print(f"Recall:          {best_recall:.6f}")
print(f"Pair-level F0.5:  {best_pair_f05:.6f}")


# ================================================================
# ENTITY-LEVEL EVALUATION
# ================================================================

print("\nCalculating entity-level F0.5...")

# IMPORTANT:
# source1_entity_id is recovered from the original pair CSV.
# The row order is preserved because the feature CSV was generated
# directly from the pair CSV.

valid_eval = pd.DataFrame(
    {
        "source1_entity_id":
            valid_pairs["source1_entity_id"].values,

        "label":
            valid_df["label"].values,

        "prob":
            probs,
    }
)

valid_eval["pred"] = (
    valid_eval["prob"] >= best_threshold
).astype(np.int8)


# ================================================================
# MACRO F0.5 PER SOURCE1 ENTITY
# ================================================================

entity_scores = []

for entity_id, group in valid_eval.groupby(
    "source1_entity_id",
    sort=False,
):

    y_true = group["label"].to_numpy()

    y_pred = group["pred"].to_numpy()

    score = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0,
    )

    entity_scores.append(score)


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


# ================================================================
# COMPARISON
# ================================================================

BASELINE = 0.873406

print("\n" + "=" * 75)
print("COMPARISON")
print("=" * 75)

print(
    f"Step 4.3 baseline: "
    f"{BASELINE:.6f}"
)

print(
    f"Step 4.4 XGBoost:  "
    f"{entity_f05:.6f}"
)

print(
    f"Difference:        "
    f"{entity_f05 - BASELINE:+.6f}"
)


# ================================================================
# TOP THRESHOLDS
# ================================================================

print("\nTop threshold results:")

threshold_results.sort(
    key=lambda x: x[3],
    reverse=True,
)

for threshold, precision, recall, f05 in (
    threshold_results[:10]
):

    print(
        f"threshold={threshold:.2f} | "
        f"precision={precision:.6f} | "
        f"recall={recall:.6f} | "
        f"F0.5={f05:.6f}"
    )


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 75)
print("FEATURE IMPORTANCE")
print("=" * 75)

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(
    ascending=False
)

for feature, value in importance.items():

    print(
        f"{feature:35s} "
        f"{value:.6f}"
    )


# ================================================================
# SAVE PREDICTIONS
# ================================================================

print("\nSaving validation predictions...")

valid_output = valid_df.copy()

valid_output["xgb_probability"] = probs

valid_output["xgb_prediction"] = (
    probs >= best_threshold
).astype(np.int8)

OUTPUT_FILE = (
    "step4_4_valid_predictions.csv"
)

valid_output.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"Saved: {OUTPUT_FILE}")


# ================================================================
# SAVE THRESHOLD RESULTS
# ================================================================

threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "precision",
        "recall",
        "f05",
    ],
)

threshold_df.to_csv(
    "step4_4_threshold_results.csv",
    index=False,
)

print(
    "Saved: "
    "step4_4_threshold_results.csv"
)


# ================================================================
# FINAL
# ================================================================

print(
    f"\nTotal runtime: "
    f"{time.time() - START:.2f}s"
)

print("=" * 75)
print("DONE")
print("=" * 75)