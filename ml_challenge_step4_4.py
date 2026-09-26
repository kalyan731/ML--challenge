import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "step4_3_train_features.csv"
VALID_FILE = "step4_3_valid_features.csv"
VALID_PAIRS_FILE = "step4_valid_pairs.csv"

MODEL_FILE = "step4_4_xgboost_model.json"
THRESHOLD_FILE = "step4_4_best_threshold.json"

VALID_PRED_FILE = "step4_4_valid_predictions.csv"
THRESHOLD_RESULTS_FILE = "step4_4_threshold_results.csv"


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):
    beta2 = 0.25

    if precision + recall == 0:
        return 0.0

    return (
        (1 + beta2)
        * precision
        * recall
        / (beta2 * precision + recall)
    )


def calculate_pr(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    score = f05(
        precision,
        recall
    )

    return precision, recall, score


# ============================================================
# ENTITY MACRO F0.5
# ============================================================

def entity_macro_f05(pair_df, threshold):

    scores = []

    for entity_id, group in pair_df.groupby(
        "source1_entity_id"
    ):

        y_true = group["label"].to_numpy()

        y_pred = (
            group["prediction_probability"].to_numpy()
            >= threshold
        ).astype(int)

        precision, recall, score = calculate_pr(
            y_true,
            y_pred
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 4.4 - XGBOOST - SAVE FINAL MODEL")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature matrices...")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print("Train:", train_df.shape)
print("Valid:", valid_df.shape)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    c
    for c in train_df.columns
    if c != "label"
]

print("\nFeatures:", len(feature_columns))

for c in feature_columns:
    print(" ", c)


X_train = train_df[feature_columns]
y_train = train_df["label"].astype(int)

X_valid = valid_df[feature_columns]
y_valid = valid_df["label"].astype(int)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_positive = int((y_train == 1).sum())
train_negative = int((y_train == 0).sum())

valid_positive = int((y_valid == 1).sum())
valid_negative = int((y_valid == 0).sum())

scale_pos_weight = (
    train_negative / train_positive
)

print("\nClass distribution:")
print("Train positives:", f"{train_positive:,}")
print("Train negatives:", f"{train_negative:,}")
print("Valid positives:", f"{valid_positive:,}")
print("Valid negatives:", f"{valid_negative:,}")
print(
    "scale_pos_weight:",
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

print("\nTraining XGBoost...")

model = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0.05,

    reg_lambda=2,

    objective="binary:logistic",

    eval_metric="logloss",

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
        eval_set=[
            (X_valid, y_valid)
        ],
        verbose=False,
    )

except Exception:

    print(
        "\nGPU unavailable. "
        "Training with CPU..."
    )

    model.set_params(
        device="cpu"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_valid, y_valid)
        ],
        verbose=False,
    )


print("Model trained.")


# ============================================================
# SAVE MODEL IMMEDIATELY
# ============================================================

print("\nSaving XGBoost model...")

model.save_model(
    MODEL_FILE
)

print(
    "Saved:",
    MODEL_FILE
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nPredicting validation...")

valid_probability = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# LOAD PAIR METADATA
# ============================================================

print("\nLoading validation pair metadata...")

valid_pairs = pd.read_csv(
    VALID_PAIRS_FILE
)

print(
    "Validation pairs:",
    valid_pairs.shape
)


# ============================================================
# BUILD VALIDATION RESULT
# ============================================================

result_df = valid_pairs.copy()

result_df["label"] = y_valid.to_numpy()

result_df[
    "prediction_probability"
] = valid_probability


# ============================================================
# SEARCH THRESHOLD
# ============================================================

print("\nSearching F0.5 threshold...")

threshold_rows = []

for threshold in np.arange(
    0.50,
    0.951,
    0.01
):

    predictions = (
        valid_probability >= threshold
    ).astype(int)

    precision, recall, score = calculate_pr(
        y_valid,
        predictions
    )

    entity_score = entity_macro_f05(
        result_df,
        threshold
    )

    threshold_rows.append({
        "threshold": round(
            float(threshold),
            2
        ),
        "precision": precision,
        "recall": recall,
        "pair_f05": score,
        "entity_macro_f05": entity_score,
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# BEST ENTITY-LEVEL THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["entity_macro_f05"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

best_precision = float(
    best_row["precision"]
)

best_recall = float(
    best_row["recall"]
)

best_pair_f05 = float(
    best_row["pair_f05"]
)

best_entity_f05 = float(
    best_row["entity_macro_f05"]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4.4 RESULTS")
print("=" * 70)

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
    f"Pair-level F0.5:  {best_pair_f05:.6f}"
)

print(
    f"Entity-level macro F0.5: "
    f"{best_entity_f05:.6f}"
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "pair_f05": best_pair_f05,
            "entity_macro_f05": best_entity_f05,
            "feature_count": len(feature_columns),
            "features": feature_columns,
        },
        f,
        indent=2,
    )

print(
    "\nSaved:",
    THRESHOLD_FILE
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

result_df[
    "prediction"
] = (
    result_df[
        "prediction_probability"
    ] >= best_threshold
).astype(int)


result_df.to_csv(
    VALID_PRED_FILE,
    index=False
)

threshold_df.to_csv(
    THRESHOLD_RESULTS_FILE,
    index=False
)

print(
    "Saved:",
    VALID_PRED_FILE
)

print(
    "Saved:",
    THRESHOLD_RESULTS_FILE
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

for row in importance_df.itertuples(
    index=False
):

    print(
        f"{row.feature:<35} "
        f"{row.importance:.6f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print(
    "Model:",
    MODEL_FILE
)

print(
    "Threshold:",
    f"{best_threshold:.2f}"
)

print(
    "Entity macro F0.5:",
    f"{best_entity_f05:.6f}"
)

print(
    "Features:",
    len(feature_columns)
)

print("\nNext stage:")
print("  TEST candidate generation")
print("  TEST feature generation")
print("  TEST prediction")
print("  final submission files")

print("=" * 70)