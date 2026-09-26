import pandas as pd
import numpy as np
import re
import time
import math
import heapq
from collections import defaultdict, Counter


# ============================================================
# CONFIG
# ============================================================

BASE = "resource/student_resource/dataset/train"

SOURCE1 = f"{BASE}/train_source1.tsv"
SOURCE2 = f"{BASE}/train_source2.tsv"
SOURCE3 = f"{BASE}/train_source3.tsv"
GROUND_TRUTH = f"{BASE}/train_ground_truth.tsv"

# IMPORTANT: benchmark first
SAMPLE_SIZE = 10_000

# Keep only this many after the cheap first-stage ranking
RERANK_K = 300

TOP_K = [25, 50, 100, 200, 500]

TOKEN_FREQ_LIMIT = 5000


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
        x
        for x in text.split()
        if len(x) >= 2
    }


def get_numbers(text):

    return set(
        re.findall(r"\d+", text)
    )


# ============================================================
# LOAD
# ============================================================

total_start = time.time()

print("=" * 70)
print("STEP 3 - FAST TWO-STAGE CANDIDATE RANKING")
print("=" * 70)

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

print("\nLoaded:")
print("Source1:", s1.shape)
print("Source2:", s2.shape)
print("Source3:", s3.shape)
print("Ground Truth:", gt.shape)


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

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

        gt_map[entity_id] = set(
            x.strip()
            for x in str(matches).split(",")
            if x.strip()
        )


# ============================================================
# SOURCE1 LOOKUP
# ============================================================

print("\nPreparing Source1 lookup...")

s1_lookup = {
    entity_id: i
    for i, entity_id
    in enumerate(s1["entity_id"])
}


# ============================================================
# SELECT 10K MATCHED ENTITIES
# ============================================================

sample_ids = []

for entity_id in s1["entity_id"]:

    if gt_map.get(entity_id):

        sample_ids.append(entity_id)

        if len(sample_ids) >= SAMPLE_SIZE:
            break


print(
    "Benchmark sample:",
    len(sample_ids)
)


# ============================================================
# SOURCE ARRAYS
# ============================================================

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

print("\nBuilding token frequencies...")

token_freq = Counter()

for name, addr in zip(
    s2_name,
    s2_addr
):

    token_freq.update(
        get_tokens(
            name + " " + addr
        )
    )


for name, addr in zip(
    s3_name,
    s3_addr
):

    token_freq.update(
        get_tokens(
            name + " " + addr
        )
    )


print(
    "Unique tokens:",
    len(token_freq)
)


# ============================================================
# TOKEN WEIGHTS
# ============================================================

print("\nBuilding token weights...")

token_weight = {}

for token, freq in token_freq.items():

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

print("\nBuilding country-token index...")

index = defaultdict(list)

# Source2
for idx in range(len(s2)):

    country = s2_country[idx]

    tokens = get_tokens(
        s2_name[idx] + " " + s2_addr[idx]
    )

    for token in tokens:

        if token in token_weight:

            index[
                (country, token)
            ].append(
                (0, idx)
            )


# Source3
for idx in range(len(s3)):

    country = s3_country[idx]

    tokens = get_tokens(
        s3_name[idx] + " " + s3_addr[idx]
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
# EXACT NAME INDEX
# ============================================================

print("\nBuilding exact name index...")

exact_name = defaultdict(list)

for idx in range(len(s2)):

    name = s2_name[idx]

    if name:

        exact_name[
            (
                s2_country[idx],
                name
            )
        ].append(
            (0, idx)
        )


for idx in range(len(s3)):

    name = s3_name[idx]

    if name:

        exact_name[
            (
                s3_country[idx],
                name
            )
        ].append(
            (1, idx)
        )


# ============================================================
# EXACT ADDRESS INDEX
# ============================================================

print("Building exact address index...")

exact_addr = defaultdict(list)

for idx in range(len(s2)):

    addr = s2_addr[idx]

    if addr:

        exact_addr[
            (
                s2_country[idx],
                addr
            )
        ].append(
            (0, idx)
        )


for idx in range(len(s3)):

    addr = s3_addr[idx]

    if addr:

        exact_addr[
            (
                s3_country[idx],
                addr
            )
        ].append(
            (1, idx)
        )


# ============================================================
# RANKING
# ============================================================

print("\nStarting two-stage ranking...")

ranking_start = time.time()

recall_counts = {
    k: 0
    for k in TOP_K
}

total_true_matches = 0

candidate_counts = []

processed = 0


for entity_id in sample_ids:

    s1_idx = s1_lookup[
        entity_id
    ]

    country = str(
        s1.at[
            s1_idx,
            "country"
        ]
    )

    name = s1.at[
        s1_idx,
        "name_norm"
    ]

    addr = s1.at[
        s1_idx,
        "addr_norm"
    ]

    source_tokens = get_tokens(
        name + " " + addr
    )

    # ========================================================
    # STAGE 1
    # Fast weighted token accumulation
    # ========================================================

    scores = defaultdict(float)

    for token in source_tokens:

        weight = token_weight.get(
            token
        )

        if weight is None:
            continue

        postings = index.get(
            (
                country,
                token
            )
        )

        if postings is None:
            continue

        for source, idx in postings:

            scores[
                (source, idx)
            ] += weight

    # Exact name gets a strong bonus
    if name:

        for source, idx in exact_name.get(
            (country, name),
            []
        ):

            scores[
                (source, idx)
            ] += 30.0

    # Exact address gets a strong bonus
    if addr:

        for source, idx in exact_addr.get(
            (country, addr),
            []
        ):

            scores[
                (source, idx)
            ] += 20.0

    candidate_counts.append(
        len(scores)
    )

    if not scores:
        processed += 1
        continue

    # ========================================================
    # Keep ONLY top 300
    # ========================================================

    top_candidates = heapq.nlargest(
        RERANK_K,
        scores.items(),
        key=lambda x: x[1]
    )

    # ========================================================
    # STAGE 2
    # Detailed scoring only on top 300
    # ========================================================

    detailed = []

    source_name_tokens = source_tokens

    source_addr_tokens = get_tokens(
        addr
    )

    source_numbers = get_numbers(
        addr
    )

    for (source, idx), fast_score in top_candidates:

        if source == 0:

            c_name = s2_name[idx]
            c_addr = s2_addr[idx]

        else:

            c_name = s3_name[idx]
            c_addr = s3_addr[idx]

        c_name_tokens = get_tokens(
            c_name
        )

        c_addr_tokens = get_tokens(
            c_addr
        )

        # ----------------------------
        # Name Jaccard
        # ----------------------------

        if source_name_tokens and c_name_tokens:

            name_intersection = len(
                source_name_tokens
                & c_name_tokens
            )

            name_union = len(
                source_name_tokens
                | c_name_tokens
            )

            name_jaccard = (
                name_intersection
                / name_union
            )

            name_overlap = (
                name_intersection
                / min(
                    len(source_name_tokens),
                    len(c_name_tokens)
                )
            )

        else:

            name_jaccard = 0.0
            name_overlap = 0.0

        # ----------------------------
        # Address Jaccard
        # ----------------------------

        if source_addr_tokens and c_addr_tokens:

            addr_intersection = len(
                source_addr_tokens
                & c_addr_tokens
            )

            addr_union = len(
                source_addr_tokens
                | c_addr_tokens
            )

            addr_jaccard = (
                addr_intersection
                / addr_union
            )

        else:

            addr_jaccard = 0.0

        # ----------------------------
        # Number overlap
        # ----------------------------

        c_numbers = get_numbers(
            c_addr
        )

        if source_numbers and c_numbers:

            number_match = len(
                source_numbers
                & c_numbers
            ) / min(
                len(source_numbers),
                len(c_numbers)
            )

        else:

            number_match = 0.0

        # ----------------------------
        # Exact
        # ----------------------------

        exact_name_flag = int(
            bool(name)
            and name == c_name
        )

        exact_addr_flag = int(
            bool(addr)
            and addr == c_addr
        )

        # ====================================================
        # FINAL RANK SCORE
        # ====================================================

        final_score = (
            fast_score
            + 25.0 * exact_name_flag
            + 15.0 * exact_addr_flag
            + 8.0 * name_jaccard
            + 5.0 * name_overlap
            + 4.0 * addr_jaccard
            + 4.0 * number_match
        )

        detailed.append(
            (
                final_score,
                source,
                idx
            )
        )

    # ========================================================
    # Final ranking
    # ========================================================

    detailed.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # ========================================================
    # Ground truth
    # ========================================================

    true_matches = gt_map[
        entity_id
    ]

    total_true_matches += len(
        true_matches
    )

    # ========================================================
    # Recall@K
    # ========================================================

    for k in TOP_K:

        found = 0

        limit = min(
            k,
            len(detailed)
        )

        for j in range(limit):

            _, source, idx = detailed[j]

            if source == 0:

                candidate_id = s2_ids[idx]

            else:

                candidate_id = s3_ids[idx]

            if candidate_id in true_matches:

                found += 1

        recall_counts[k] += found

    processed += 1

    # ========================================================
    # Progress
    # ========================================================

    if processed % 500 == 0:

        elapsed = (
            time.time()
            - ranking_start
        )

        print(
            f"Processed "
            f"{processed:,}/"
            f"{len(sample_ids):,}"
            f" | avg candidates "
            f"{np.mean(candidate_counts):,.1f}"
            f" | elapsed "
            f"{elapsed:.1f}s"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3 - FINAL BENCHMARK")
print("=" * 70)

print(
    "Processed:",
    processed
)

print(
    "True matches:",
    total_true_matches
)

print(
    "Mean candidates:",
    round(
        np.mean(candidate_counts),
        2
    )
)

print(
    "Median candidates:",
    round(
        np.median(candidate_counts),
        2
    )
)

print("\nRecall:")

for k in TOP_K:

    recall = (
        recall_counts[k]
        / total_true_matches
        * 100
    )

    print(
        f"Recall@{k:<3} = "
        f"{recall:.4f}%"
    )

print("\nRuntime:")

print(
    f"{time.time() - total_start:.2f} seconds"
)

print("=" * 70)
print("DONE")
print("=" * 70)