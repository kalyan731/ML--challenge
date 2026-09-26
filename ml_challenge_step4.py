import pandas as pd
import re
import time
import math
import random
from collections import defaultdict, Counter


# ============================================================
# CONFIG
# ============================================================

BASE = "resource/student_resource/dataset/train"

SOURCE1 = f"{BASE}/train_source1.tsv"
SOURCE2 = f"{BASE}/train_source2.tsv"
SOURCE3 = f"{BASE}/train_source3.tsv"
GROUND_TRUTH = f"{BASE}/train_ground_truth.tsv"

# FIRST BENCHMARK
SAMPLE_SIZE = 5_000

# Hard negatives per Source1
NEGATIVES_PER_ENTITY = 8

# Number of rare tokens used for candidate retrieval
MAX_QUERY_TOKENS = 4

# Ignore extremely common tokens
TOKEN_FREQ_LIMIT = 5000

# Minimum token length
MIN_TOKEN_LEN = 2

RANDOM_SEED = 42

VALIDATION_FRACTION = 0.20

CHECKPOINT_EVERY = 500

OUTPUT_TRAIN = "step4_train_pairs.csv"
OUTPUT_VALID = "step4_valid_pairs.csv"


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
# START
# ============================================================

start = time.time()

print("=" * 75)
print("STEP 4.1 - FAST TRAINING PAIR GENERATION")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

print("\nLoading Source1...")

s1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    dtype=str
)

print("Loading Source2...")

s2 = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str
)

print("Loading Source3...")

s3 = pd.read_csv(
    SOURCE3,
    sep="\t",
    dtype=str
)

print("Loading Ground Truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

print("\nShapes:")
print("Source1:", s1.shape)
print("Source2:", s2.shape)
print("Source3:", s3.shape)
print("Ground Truth:", gt.shape)


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in [s1, s2, s3]:

    # Bug 1 fix: guard against missing columns rather than assuming names
    name_col = "business_name" if "business_name" in df.columns else df.columns[0]
    addr_col = "business_address" if "business_address" in df.columns else ""

    df["name_norm"] = (
        df[name_col]
        .fillna("")
        .map(normalize_text)
    )

    df["addr_norm"] = (
        df[addr_col].fillna("").map(normalize_text)
        if addr_col
        else ""
    )


# ============================================================
# GROUND TRUTH
# ============================================================

print("\nPreparing ground truth...")

gt_map = {}

for entity_id, matches in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    if pd.isna(matches) or str(matches).strip() == "":

        gt_map[entity_id] = set()

    else:

        gt_map[entity_id] = {
            x.strip()
            for x in str(matches).split(",")
            if x.strip()
        }


# ============================================================
# SOURCE ID LOOKUPS
# ============================================================

print("\nCreating direct ID lookups...")

s2_id_to_idx = {
    entity_id: idx
    for idx, entity_id
    in enumerate(s2["entity_id"])
}

s3_id_to_idx = {
    entity_id: idx
    for idx, entity_id
    in enumerate(s3["entity_id"])
}

print(
    "S2 ID lookup:",
    len(s2_id_to_idx)
)

print(
    "S3 ID lookup:",
    len(s3_id_to_idx)
)


# ============================================================
# SELECT SAMPLE
# ============================================================

print("\nSelecting matched Source1 sample...")

sample_ids = []

for entity_id in s1["entity_id"]:

    if gt_map.get(entity_id):

        sample_ids.append(entity_id)

        if len(sample_ids) >= SAMPLE_SIZE:
            break


print(
    "Selected:",
    len(sample_ids)
)


# ============================================================
# ENTITY-LEVEL SPLIT
# ============================================================

print("\nCreating train/validation split...")

rng = random.Random(
    RANDOM_SEED
)

shuffled = sample_ids.copy()

rng.shuffle(shuffled)

split = int(
    len(shuffled)
    * (1 - VALIDATION_FRACTION)
)

train_entities = set(
    shuffled[:split]
)

valid_entities = set(
    shuffled[split:]
)

print(
    "Train entities:",
    len(train_entities)
)

print(
    "Validation entities:",
    len(valid_entities)
)


# ============================================================
# SOURCE ARRAYS
# ============================================================

s1_ids = s1["entity_id"].to_numpy()
# Bug 2 fix: fillna("") on s1 country so it matches index keys built from s2/s3
s1_country = s1["country"].fillna("").to_numpy()
s1_name = s1["name_norm"].to_numpy()
s1_addr = s1["addr_norm"].to_numpy()

s2_ids = s2["entity_id"].to_numpy()
s2_country = s2["country"].fillna("").to_numpy()
s2_name = s2["name_norm"].to_numpy()
s2_addr = s2["addr_norm"].to_numpy()

s3_ids = s3["entity_id"].to_numpy()
s3_country = s3["country"].fillna("").to_numpy()
s3_name = s3["name_norm"].to_numpy()
s3_addr = s3["addr_norm"].to_numpy()


# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("\nBuilding token frequency...")

token_frequency = Counter()

# Bug 5 fix: include s1 tokens so s1-only tokens don't get
# artificially high weights and waste MAX_QUERY_TOKENS slots
for name, addr in zip(
    s1_name,
    s1_addr
):

    token_frequency.update(
        get_tokens(
            name + " " + addr
        )
    )


for name, addr in zip(
    s2_name,
    s2_addr
):

    token_frequency.update(
        get_tokens(
            name + " " + addr
        )
    )


for name, addr in zip(
    s3_name,
    s3_addr
):

    token_frequency.update(
        get_tokens(
            name + " " + addr
        )
    )


print(
    "Unique tokens:",
    len(token_frequency)
)


# ============================================================
# TOKEN WEIGHTS
# ============================================================

print("\nCalculating token weights...")

token_weight = {}

for token, freq in token_frequency.items():

    if freq <= TOKEN_FREQ_LIMIT:

        token_weight[token] = math.log(
            1_000_000 / (freq + 1)
        )


print(
    "Usable tokens:",
    len(token_weight)
)


# ============================================================
# INVERTED INDEX
# ============================================================

print("\nBuilding compact token index...")

index = defaultdict(list)


# ----------------------------
# Source2
# ----------------------------

for idx in range(len(s2)):

    country = s2_country[idx]

    tokens = get_tokens(
        s2_name[idx]
        + " "
        + s2_addr[idx]
    )

    for token in tokens:

        if token in token_weight:

            index[
                (country, token)
            ].append(
                (0, idx)
            )


# ----------------------------
# Source3
# ----------------------------

for idx in range(len(s3)):

    country = s3_country[idx]

    tokens = get_tokens(
        s3_name[idx]
        + " "
        + s3_addr[idx]
    )

    for token in tokens:

        if token in token_weight:

            index[
                (country, token)
            ].append(
                (1, idx)
            )


print(
    "Index entries:",
    len(index)
)


# ============================================================
# SOURCE1 LOOKUP
# ============================================================

s1_lookup = {
    entity_id: idx
    for idx, entity_id
    in enumerate(s1["entity_id"])
}


# ============================================================
# GENERATE PAIRS
# ============================================================

print("\nGenerating pairs...")

train_rows = []
valid_rows = []

processed = 0

positive_count = 0
negative_count = 0

total_candidates = 0

generation_start = time.time()


for entity_id in sample_ids:

    s1_idx = s1_lookup[
        entity_id
    ]

    # Bug 2 fix: use pre-built numpy arrays (fillna("") applied)
    # instead of s1.at[] which returns NaN as float → str gives "nan"
    country = s1_country[s1_idx]

    name = s1_name[s1_idx]

    addr = s1_addr[s1_idx]

    true_matches = gt_map[
        entity_id
    ]

    # ========================================================
    # POSITIVES
    # ========================================================

    destination = (
        train_rows
        if entity_id in train_entities
        else valid_rows
    )

    for matched_id in true_matches:

        if matched_id in s2_id_to_idx:

            source = 0
            candidate_idx = s2_id_to_idx[
                matched_id
            ]

        elif matched_id in s3_id_to_idx:

            source = 1
            candidate_idx = s3_id_to_idx[
                matched_id
            ]

        else:

            continue

        destination.append({

            "source1_entity_id":
                entity_id,

            "candidate_entity_id":
                matched_id,

            "candidate_source":
                source,

            "candidate_index":
                candidate_idx,

            "label":
                1

        })

        positive_count += 1

    # ========================================================
    # QUERY TOKENS
    #
    # Pick rarest informative tokens.
    # ========================================================

    source_tokens = (
        get_tokens(name)
        |
        get_tokens(addr)
    )

    informative_tokens = [
        token
        for token in source_tokens
        if token in token_weight
    ]

    informative_tokens.sort(
        # Bug 4 fix: secondary sort by token string for determinism
        # when two tokens share the same frequency
        key=lambda x:
            (token_frequency[x], x)
    )

    query_tokens = informative_tokens[
        :MAX_QUERY_TOKENS
    ]

    # ========================================================
    # RETRIEVE HARD NEGATIVES
    # ========================================================

    candidates = defaultdict(float)

    for token in query_tokens:

        postings = index.get(
            (
                country,
                token
            )
        )

        if postings is None:
            continue

        weight = token_weight[
            token
        ]

        for source, candidate_idx in postings:

            if source == 0:

                candidate_id = s2_ids[
                    candidate_idx
                ]

            else:

                candidate_id = s3_ids[
                    candidate_idx
                ]

            # Never select true matches
            if candidate_id in true_matches:
                continue

            candidates[
                (
                    source,
                    candidate_idx
                )
            ] += weight


    total_candidates += len(
        candidates
    )

    # ========================================================
    # SORT HARD NEGATIVES
    # ========================================================

    hard_negatives = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    hard_negatives = hard_negatives[
        :NEGATIVES_PER_ENTITY
    ]

    # ========================================================
    # ADD NEGATIVES
    # ========================================================

    for (
        (source, candidate_idx),
        score
    ) in hard_negatives:

        if source == 0:

            candidate_id = s2_ids[
                candidate_idx
            ]

        else:

            candidate_id = s3_ids[
                candidate_idx
            ]

        destination.append({

            "source1_entity_id":
                entity_id,

            "candidate_entity_id":
                candidate_id,

            "candidate_source":
                source,

            "candidate_index":
                candidate_idx,

            "label":
                0

        })

        negative_count += 1

    processed += 1

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if processed % CHECKPOINT_EVERY == 0:

        elapsed = (
            time.time()
            - generation_start
        )

        print(
            f"Processed "
            f"{processed:,}/"
            f"{len(sample_ids):,}"
            f" | train rows "
            f"{len(train_rows):,}"
            f" | valid rows "
            f"{len(valid_rows):,}"
            f" | elapsed "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint
        pd.DataFrame(
            train_rows
        ).to_csv(
            OUTPUT_TRAIN,
            index=False
        )

        pd.DataFrame(
            valid_rows
        ).to_csv(
            OUTPUT_VALID,
            index=False
        )


# ============================================================
# FINAL DATAFRAMES
# ============================================================

print("\nCreating final DataFrames...")

train_pairs = pd.DataFrame(
    train_rows
)

valid_pairs = pd.DataFrame(
    valid_rows
)


# ============================================================
# SHUFFLE
# ============================================================

if len(train_pairs) > 0:

    train_pairs = train_pairs.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

if len(valid_pairs) > 0:

    valid_pairs = valid_pairs.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)


# ============================================================
# FINAL SAVE
# ============================================================

train_pairs.to_csv(
    OUTPUT_TRAIN,
    index=False
)

valid_pairs.to_csv(
    OUTPUT_VALID,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("STEP 4.1 RESULTS")
print("=" * 75)

print(
    "Processed:",
    processed
)

print(
    "Train entities:",
    len(train_entities)
)

print(
    "Validation entities:",
    len(valid_entities)
)

print()

print(
    "Training pairs:",
    len(train_pairs)
)

print(
    "Validation pairs:",
    len(valid_pairs)
)

print()

if len(train_pairs):

    print(
        "Training positives:",
        int(
            (train_pairs["label"] == 1).sum()
        )
    )

    print(
        "Training negatives:",
        int(
            (train_pairs["label"] == 0).sum()
        )
    )


if len(valid_pairs):

    print(
        "Validation positives:",
        int(
            (valid_pairs["label"] == 1).sum()
        )
    )

    print(
        "Validation negatives:",
        int(
            (valid_pairs["label"] == 0).sum()
        )
    )

print()

print(
    "Average retrieved candidates/entity:",
    round(
        total_candidates / max(processed, 1),
        2
    )
)

print()

print(
    "Saved:",
    OUTPUT_TRAIN
)

print(
    "Saved:",
    OUTPUT_VALID
)

print()

print(
    "Runtime:",
    round(
        time.time() - start,
        2
    ),
    "seconds"
)

print("=" * 75)
print("DONE")
print("=" * 75)
