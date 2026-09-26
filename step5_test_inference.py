import json
import os
import gc

import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "step5_test_features.csv"

TEST_SOURCE1_FILE = (
    "resource/student_resource/dataset/test/test_source1.tsv"
)

MODEL_FILE = "step4_4_xgboost_model.json"
THRESHOLD_FILE = "step4_4_best_threshold.json"

OUTPUT_PREDICTIONS = "step5_test_predictions.csv"
OUTPUT_MATCHES = "matching_results.tsv"

# Memory-safe inference batch.
BATCH_SIZE = 500_000


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


FEATURE_RENAME = {
    f"feature_{i}_{name}": name
    for i, name in enumerate(FEATURE_COLUMNS)
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 6 - MEMORY-SAFE TEST INFERENCE")
print("=" * 70)

print("Feature file:", FEATURE_FILE)
print("Test Source1:", TEST_SOURCE1_FILE)
print("Model:", MODEL_FILE)
print("Threshold:", THRESHOLD_FILE)
print("Batch size:", f"{BATCH_SIZE:,}")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    FEATURE_FILE,
    TEST_SOURCE1_FILE,
    MODEL_FILE,
    THRESHOLD_FILE,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

print("\n[1/6] Loading XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

print("\n[2/6] Loading threshold...")

with open(
    THRESHOLD_FILE,
    "r"
) as f:

    threshold_data = json.load(f)


if "best_threshold" in threshold_data:

    threshold = float(
        threshold_data["best_threshold"]
    )

elif "threshold" in threshold_data:

    threshold = float(
        threshold_data["threshold"]
    )

else:

    raise ValueError(
        f"No threshold found in {THRESHOLD_FILE}"
    )


print(
    "Using threshold:",
    threshold
)


# ============================================================
# REMOVE EXISTING OUTPUTS
# ============================================================

print("\n[3/6] Cleaning old outputs...")

if os.path.exists(
    OUTPUT_PREDICTIONS
):

    print(
        "Removing:",
        OUTPUT_PREDICTIONS
    )

    os.remove(
        OUTPUT_PREDICTIONS
    )


if os.path.exists(
    OUTPUT_MATCHES
):

    print(
        "Removing:",
        OUTPUT_MATCHES
    )

    os.remove(
        OUTPUT_MATCHES
    )


# ============================================================
# LOAD ALL SOURCE1 TEST IDS
# ============================================================
#
# IMPORTANT:
# This is the master entity list.
#
# Step 5 only contains Source1 entities having candidates.
# Therefore we MUST NOT derive the final Source1 list
# from step5_test_features.csv.
#
# ============================================================

print(
    "\n[4/6] Loading ALL Source1 test entities..."
)

test_source1 = pd.read_csv(
    TEST_SOURCE1_FILE,
    sep="\t",
    dtype={
        "entity_id": "string"
    },
    usecols=[
        "entity_id"
    ],
)


test_source1 = test_source1.rename(
    columns={
        "entity_id":
            "source1_entity_id"
    }
)


# Safety checks.

if test_source1[
    "source1_entity_id"
].isna().any():

    raise RuntimeError(
        "FAIL: test_source1.tsv contains "
        "missing entity IDs."
    )


test_source1 = test_source1.drop_duplicates(
    subset=[
        "source1_entity_id"
    ]
)


print(
    "Unique Source1 entities:",
    f"{len(test_source1):,}"
)


# ============================================================
# INFERENCE
# ============================================================

print(
    "\n[5/6] Running chunked inference..."
)

processed = 0
positive_total = 0
batch_number = 0

# Only positive predictions are retained.
positive_chunks = []


reader = pd.read_csv(
    FEATURE_FILE,
    chunksize=BATCH_SIZE,
    dtype={
        "source1_entity_id": "string",
        "candidate_entity_id": "string",
        "candidate_source": "string",
    },
)


for df in reader:

    batch_number += 1

    # --------------------------------------------------------
    # Rename feature columns
    # --------------------------------------------------------

    df.rename(
        columns=FEATURE_RENAME,
        inplace=True
    )


    # --------------------------------------------------------
    # Check all 21 features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing feature columns: "
            f"{missing_features}"
        )


    # --------------------------------------------------------
    # XGBoost input
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ]


    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(X)[:, 1]
    )


    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    )


    positive_count = int(
        predictions.sum()
    )


    # --------------------------------------------------------
    # Keep positive candidates only
    # --------------------------------------------------------

    if positive_count > 0:

        positive_df = pd.DataFrame({

            "candidate_index":
                df.loc[
                    predictions,
                    "candidate_index"
                ].to_numpy(),

            "source1_entity_id":
                df.loc[
                    predictions,
                    "source1_entity_id"
                ].to_numpy(),

            "candidate_entity_id":
                df.loc[
                    predictions,
                    "candidate_entity_id"
                ].to_numpy(),

            "candidate_source":
                df.loc[
                    predictions,
                    "candidate_source"
                ].to_numpy(),

            "match_probability":
                probabilities[
                    predictions
                ],
        })


        # ----------------------------------------------------
        # Only S2/S3 are valid
        # ----------------------------------------------------

        positive_df = positive_df[
            positive_df[
                "candidate_source"
            ].isin(
                [
                    "source2",
                    "source3"
                ]
            )
        ]


        if len(positive_df) > 0:

            positive_chunks.append(
                positive_df
            )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    processed += len(df)

    positive_total += positive_count

    print(
        f"Batch {batch_number:,} | "
        f"Rows {processed:,} | "
        f"Positive {positive_total:,}",
        flush=True
    )


    # --------------------------------------------------------
    # Free batch memory
    # --------------------------------------------------------

    del X
    del probabilities
    del predictions
    del df

    gc.collect()


# ============================================================
# COMBINE POSITIVE PREDICTIONS
# ============================================================

print(
    "\nCombining positive predictions..."
)


if positive_chunks:

    matched = pd.concat(
        positive_chunks,
        ignore_index=True
    )

else:

    matched = pd.DataFrame(
        columns=[
            "candidate_index",
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "match_probability",
        ]
    )


print(
    "Positive candidate rows:",
    f"{len(matched):,}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

matched.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print(
    "Saved:",
    OUTPUT_PREDICTIONS
)


# ============================================================
# CORRECT DUPLICATE KEY
# ============================================================
#
# candidate_entity_id is interpreted together with
# candidate_source throughout the pipeline.
#
# Therefore source2/X and source3/X are different entities.
#
# ============================================================

matched = matched.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]
)


# ============================================================
# SORT MATCHES
# ============================================================

matched = matched.sort_values(
    [
        "source1_entity_id",
        "match_probability",
    ],
    ascending=[
        True,
        False,
    ],
)


# ============================================================
# BUILD MATCHED ENTITY STRINGS
# ============================================================

print(
    "\nBuilding Source1 match lists..."
)


matches_by_s1 = (
    matched
    .groupby(
        "source1_entity_id",
        sort=False
    )[
        "candidate_entity_id"
    ]
    .apply(
        lambda x:
            ",".join(
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


# ============================================================
# LEFT JOIN ON ALL SOURCE1 ENTITIES
# ============================================================

print(
    "Joining against ALL Source1 test entities..."
)


result = test_source1.merge(
    matches_by_s1,
    on="source1_entity_id",
    how="left",
)


# Entities with no predicted match become empty strings.

result["matched_entity_ids"] = (
    result[
        "matched_entity_ids"
    ]
    .fillna("")
)


result = result[
    [
        "source1_entity_id",
        "matched_entity_ids",
    ]
]


# ============================================================
# FINAL VALIDATION BEFORE WRITE
# ============================================================

print(
    "\nValidating final result..."
)


# No duplicate Source1 IDs.

duplicate_s1 = int(
    result[
        "source1_entity_id"
    ]
    .duplicated()
    .sum()
)


if duplicate_s1 != 0:

    raise RuntimeError(
        "FAIL: duplicate Source1 IDs "
        "in final result."
    )


# Every Source1 ID must be present.

expected_s1_count = len(
    test_source1
)

actual_s1_count = len(
    result
)


if actual_s1_count != expected_s1_count:

    raise RuntimeError(
        "FAIL: final Source1 count mismatch. "
        f"Expected {expected_s1_count:,}, "
        f"got {actual_s1_count:,}."
    )


# No missing Source1 IDs.

missing_s1 = int(
    result[
        "source1_entity_id"
    ]
    .isna()
    .sum()
)


if missing_s1 != 0:

    raise RuntimeError(
        "FAIL: missing Source1 IDs "
        "in final result."
    )


# ============================================================
# WRITE FINAL TSV
# ============================================================

print(
    "\n[6/6] Writing matching_results.tsv..."
)


result.to_csv(
    OUTPUT_MATCHES,
    sep="\t",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

entities_total = len(
    result
)

entities_with_matches = int(
    result[
        "matched_entity_ids"
    ]
    .ne("")
    .sum()
)

entities_without_matches = int(
    result[
        "matched_entity_ids"
    ]
    .eq("")
    .sum()
)


print()
print("=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print(
    "Candidate rows processed:",
    f"{processed:,}"
)

print(
    "Positive candidate predictions:",
    f"{positive_total:,}"
)

print(
    "Unique Source1 entities:",
    f"{entities_total:,}"
)

print(
    "Entities with matches:",
    f"{entities_with_matches:,}"
)

print(
    "Entities without matches:",
    f"{entities_without_matches:,}"
)

print(
    "Duplicate Source1 IDs:",
    duplicate_s1
)

print()
print(
    "Predictions file:",
    OUTPUT_PREDICTIONS
)

print(
    "Final submission:",
    OUTPUT_MATCHES
)

print()
print(
    "ALL FINAL BASIC CHECKS PASSED."
)

print("=" * 70)