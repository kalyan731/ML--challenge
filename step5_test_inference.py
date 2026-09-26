import json
import os

import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "step5_test_features.csv"
MODEL_FILE = "step4_4_xgboost_model.json"
THRESHOLD_FILE = "step4_4_best_threshold.json"

OUTPUT_PREDICTIONS = "step5_test_predictions.csv"
OUTPUT_MATCHES = "matching_results.tsv"


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


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("STEP 5 - TEST INFERENCE")
print("=" * 70)

print("\nLoading test features...")

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

print("\nAll 21 features found.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(MODEL_FILE)

print("Model loaded:", MODEL_FILE)


# ============================================================
# LOAD THRESHOLD
# ============================================================

print("\nLoading threshold...")

with open(THRESHOLD_FILE, "r") as f:
    threshold_data = json.load(f)

print("Threshold file contents:", threshold_data)


# Support either:
# {"best_threshold": 0.81}
# or
# {"threshold": 0.81}

if "best_threshold" in threshold_data:
    threshold = float(threshold_data["best_threshold"])
elif "threshold" in threshold_data:
    threshold = float(threshold_data["threshold"])
else:
    raise ValueError(
        "Could not find threshold in step4_4_best_threshold.json"
    )

print("Using threshold:", threshold)


# ============================================================
# PREDICT
# ============================================================

print("\nRunning predictions...")

X = df[FEATURE_COLUMNS]

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

df["match_probability"] = probabilities

df["prediction"] = predictions


# ============================================================
# PREDICTION SUMMARY
# ============================================================

positive_count = int(predictions.sum())
negative_count = int((predictions == 0).sum())

print("\nPrediction summary:")
print("Total candidates:", len(df))
print("Predicted matches:", positive_count)
print("Predicted non-matches:", negative_count)

if len(df) > 0:
    print(
        "Match rate:",
        round(positive_count / len(df), 6)
    )


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

print("\nSaving predictions...")

df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print("Saved:", OUTPUT_PREDICTIONS)


# ============================================================
# BUILD MATCHING RESULTS
# ============================================================

print("\nBuilding matching_results.tsv...")

matched = df[
    df["prediction"] == 1
].copy()


# Safety: only S2/S3 candidate IDs
matched = matched[
    matched["candidate_source"].isin(
        ["source2", "source3"]
    )
]


# Remove duplicate candidate IDs per Source1
matched = matched.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)


# Sort by Source1 then probability descending
matched = matched.sort_values(
    [
        "source1_entity_id",
        "match_probability"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# INCLUDE EVERY SOURCE1 ENTITY
# ============================================================

all_s1 = pd.DataFrame({
    "source1_entity_id":
        df["source1_entity_id"].unique()
})

matches_by_s1 = (
    matched
    .groupby("source1_entity_id")["candidate_entity_id"]
    .apply(
        lambda x: ",".join(
            x.astype(str).tolist()
        )
    )
    .reset_index()
)

matches_by_s1 = matches_by_s1.rename(
    columns={
        "candidate_entity_id":
            "matched_entity_ids"
    }
)

result = all_s1.merge(
    matches_by_s1,
    on="source1_entity_id",
    how="left"
)

result["matched_entity_ids"] = (
    result["matched_entity_ids"]
    .fillna("")
)


# ============================================================
# SAVE MATCHING RESULTS
# ============================================================

result = result[
    [
        "source1_entity_id",
        "matched_entity_ids"
    ]
]

result.to_csv(
    OUTPUT_MATCHES,
    sep="\t",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

entities_with_matches = (
    result["matched_entity_ids"]
    .ne("")
    .sum()
)

entities_without_matches = (
    result["matched_entity_ids"]
    .eq("")
    .sum()
)

print("\n" + "=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print("Candidate rows:", len(df))
print("Source1 entities:", len(result))
print("Predicted matched entities:", entities_with_matches)
print("Predicted singleton/no-match entities:", entities_without_matches)

print("\nFiles created:")
print("1.", OUTPUT_PREDICTIONS)
print("2.", OUTPUT_MATCHES)

print("\nFirst 10 matching results:")
print(result.head(10).to_string(index=False))