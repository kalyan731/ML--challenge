import duckdb
import random
import time

# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 5_000
NEGATIVES_PER_ENTITY = 8
MAX_QUERY_TOKENS = 4
TOKEN_FREQ_LIMIT = 5_000
MIN_TOKEN_LEN = 2

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

S1 = "normalized_cache/source1_normalized.parquet"
S2 = "normalized_cache/source2_normalized.parquet"
S3 = "normalized_cache/source3_normalized.parquet"
GT = "resource/student_resource/dataset/train/train_ground_truth.tsv"

OUTPUT_TRAIN = "step4_train_pairs.csv"
OUTPUT_VALID = "step4_valid_pairs.csv"

# ============================================================
# START
# ============================================================

start = time.time()

print("Starting DuckDB pair generation...")

con = duckdb.connect()
con.execute("PRAGMA threads=2")
con.execute("PRAGMA memory_limit='6GB'")
con.execute("PRAGMA temp_directory='duckdb_tmp'")

# ============================================================
# SOURCE TABLES
# ============================================================

print("Preparing source tables...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s2 AS
SELECT
    row_number() OVER () - 1 AS candidate_index,
    entity_id,
    name_norm,
    addr_norm,
    country_norm
FROM read_parquet('{S2}')
""")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s3 AS
SELECT
    row_number() OVER () - 1 AS candidate_index,
    entity_id,
    name_norm,
    addr_norm,
    country_norm
FROM read_parquet('{S3}')
""")

# ============================================================
# SAMPLE 5,000 MATCHED SOURCE1 ENTITIES
# ============================================================

print("Selecting sample...")

sample_rows = con.execute(f"""
SELECT
    s.entity_id
FROM read_parquet('{S1}') s
JOIN read_csv(
    '{GT}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id':'VARCHAR',
        'matched_entity_ids':'VARCHAR'
    }}
) g
ON s.entity_id = g.source1_entity_id
WHERE g.matched_entity_ids IS NOT NULL
  AND trim(g.matched_entity_ids) <> ''
LIMIT {SAMPLE_SIZE}
""").fetchall()

sample_ids = [x[0] for x in sample_rows]

print("Selected:", len(sample_ids))

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

rng = random.Random(RANDOM_SEED)
rng.shuffle(sample_ids)

split = int(len(sample_ids) * (1 - VALIDATION_FRACTION))

train_ids = sample_ids[:split]
valid_ids = sample_ids[split:]

print("Train entities:", len(train_ids))
print("Validation entities:", len(valid_ids))

# Store split
con.execute("""
CREATE OR REPLACE TEMP TABLE selected_entities (
    source1_entity_id VARCHAR,
    split VARCHAR
)
""")

con.executemany(
    "INSERT INTO selected_entities VALUES (?, ?)",
    [(x, "train") for x in train_ids]
    + [(x, "valid") for x in valid_ids]
)

# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE gt AS
SELECT
    source1_entity_id,
    matched_entity_ids
FROM read_csv(
    '{GT}',
    delim='\\t',
    header=true,
    columns={{
        'source1_entity_id':'VARCHAR',
        'matched_entity_ids':'VARCHAR'
    }}
)
WHERE source1_entity_id IN (
    SELECT source1_entity_id
    FROM selected_entities
)
""")

# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("Building token frequency...")

con.execute("""
CREATE OR REPLACE TEMP TABLE token_frequency AS

SELECT
    token,
    COUNT(DISTINCT entity_id) AS freq
FROM (

    SELECT
        entity_id,
        country_norm,
        token
    FROM s2,
    UNNEST(
        string_split(
            trim(coalesce(name_norm, '') || ' ' || coalesce(addr_norm, '')),
            ' '
        )
    ) AS t(token)

    UNION ALL

    SELECT
        entity_id,
        country_norm,
        token
    FROM s3,
    UNNEST(
        string_split(
            trim(coalesce(name_norm, '') || ' ' || coalesce(addr_norm, '')),
            ' '
        )
    ) AS t(token)

)
WHERE length(token) >= ?
  AND trim(token) <> ''
GROUP BY token
HAVING COUNT(DISTINCT entity_id) <= ?
""", [MIN_TOKEN_LEN, TOKEN_FREQ_LIMIT])

token_count = con.execute(
    "SELECT COUNT(*) FROM token_frequency"
).fetchone()[0]

print("Usable tokens:", token_count)

# ============================================================
# SOURCE1 QUERY TOKENS
# ============================================================

print("Building query tokens...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE query_tokens AS

SELECT
    s.entity_id AS source1_entity_id,
    s.country_norm,
    t.token,
    tf.freq
FROM read_parquet('{S1}') s
JOIN selected_entities e
  ON s.entity_id = e.source1_entity_id

CROSS JOIN UNNEST(
    string_split(
        trim(coalesce(s.name_norm, '') || ' ' || coalesce(s.addr_norm, '')),
        ' '
    )
) AS t(token)

JOIN token_frequency tf
  ON t.token = tf.token

WHERE length(t.token) >= {MIN_TOKEN_LEN}
  AND trim(t.token) <> ''
""")

# Keep the 4 rarest tokens per Source1
con.execute("""
CREATE OR REPLACE TEMP TABLE top_query_tokens AS

SELECT
    source1_entity_id,
    country_norm,
    token,
    freq
FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY source1_entity_id
            ORDER BY freq ASC, token ASC
        ) AS rn
    FROM query_tokens
)
WHERE rn <= ?
""", [MAX_QUERY_TOKENS])

# ============================================================
# TRUE MATCHES
# ============================================================

print("Preparing positive pairs...")

con.execute("""
CREATE OR REPLACE TEMP TABLE positive_pairs AS

SELECT
    g.source1_entity_id,
    trim(x.matched_id) AS candidate_entity_id,
    CASE
        WHEN trim(x.matched_id) IN (
            SELECT entity_id FROM s2
        ) THEN 0
        ELSE 1
    END AS candidate_source,
    CASE
        WHEN trim(x.matched_id) IN (
            SELECT entity_id FROM s2
        )
        THEN (
            SELECT candidate_index
            FROM s2
            WHERE entity_id = trim(x.matched_id)
            LIMIT 1
        )
        ELSE (
            SELECT candidate_index
            FROM s3
            WHERE entity_id = trim(x.matched_id)
            LIMIT 1
        )
    END AS candidate_index,
    1 AS label,
    e.split
FROM gt g
JOIN selected_entities e
  ON g.source1_entity_id = e.source1_entity_id
CROSS JOIN UNNEST(
    string_split(g.matched_entity_ids, ',')
) AS x(matched_id)
WHERE trim(x.matched_id) <> ''
""")

positive_count = con.execute(
    "SELECT COUNT(*) FROM positive_pairs"
).fetchone()[0]

print("Positive pairs:", positive_count)

# ============================================================
# HARD NEGATIVE CANDIDATES
# ============================================================

print("Generating hard negatives...")

con.execute("""
CREATE OR REPLACE TEMP TABLE candidate_scores AS

SELECT
    q.source1_entity_id,
    q.country_norm,
    q.token,

    0 AS candidate_source,
    s.candidate_index,
    s.entity_id AS candidate_entity_id,

    1.0 / (1.0 + q.freq) AS token_weight

FROM top_query_tokens q
JOIN s2 s
  ON s.country_norm = q.country_norm
JOIN token_frequency tf
  ON tf.token = q.token

WHERE q.token IN (
    SELECT token
    FROM top_query_tokens
    WHERE source1_entity_id = q.source1_entity_id
)

UNION ALL

SELECT
    q.source1_entity_id,
    q.country_norm,
    q.token,

    1 AS candidate_source,
    s.candidate_index,
    s.entity_id AS candidate_entity_id,

    1.0 / (1.0 + q.freq) AS token_weight

FROM top_query_tokens q
JOIN s3 s
  ON s.country_norm = q.country_norm
JOIN token_frequency tf
  ON tf.token = q.token
""")

# ============================================================
# REMOVE TRUE MATCHES + RANK
# ============================================================

con.execute("""
CREATE OR REPLACE TEMP TABLE ranked_negatives AS

SELECT
    c.source1_entity_id,
    c.candidate_entity_id,
    c.candidate_source,
    c.candidate_index,
    0 AS label,
    e.split,

    SUM(c.token_weight) AS score

FROM candidate_scores c

JOIN selected_entities e
  ON c.source1_entity_id = e.source1_entity_id

WHERE NOT EXISTS (
    SELECT 1
    FROM positive_pairs p
    WHERE p.source1_entity_id = c.source1_entity_id
      AND p.candidate_entity_id = c.candidate_entity_id
)

GROUP BY
    c.source1_entity_id,
    c.candidate_entity_id,
    c.candidate_source,
    c.candidate_index,
    e.split
""")

# Top 8 negatives per Source1
con.execute("""
CREATE OR REPLACE TEMP TABLE negative_pairs AS

SELECT
    source1_entity_id,
    candidate_entity_id,
    candidate_source,
    candidate_index,
    label,
    split
FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY source1_entity_id
            ORDER BY score DESC, candidate_entity_id ASC
        ) AS rn
    FROM ranked_negatives
)
WHERE rn <= ?
""", [NEGATIVES_PER_ENTITY])

negative_count = con.execute(
    "SELECT COUNT(*) FROM negative_pairs"
).fetchone()[0]

print("Negative pairs:", negative_count)

# ============================================================
# WRITE TRAIN
# ============================================================

print("Writing training pairs...")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        candidate_entity_id,
        candidate_source,
        candidate_index,
        label
    FROM (
        SELECT * FROM positive_pairs
        UNION ALL
        SELECT * FROM negative_pairs
    )
    WHERE split = 'train'
    ORDER BY random()
)
TO '{OUTPUT_TRAIN}'
WITH (HEADER, DELIMITER ',')
""")

# ============================================================
# WRITE VALID
# ============================================================

print("Writing validation pairs...")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        candidate_entity_id,
        candidate_source,
        candidate_index,
        label
    FROM (
        SELECT * FROM positive_pairs
        UNION ALL
        SELECT * FROM negative_pairs
    )
    WHERE split = 'valid'
    ORDER BY random()
)
TO '{OUTPUT_VALID}'
WITH (HEADER, DELIMITER ',')
""")

# ============================================================
# SUMMARY
# ============================================================

train_count = con.execute(
    f"SELECT COUNT(*) FROM read_csv('{OUTPUT_TRAIN}')"
).fetchone()[0]

valid_count = con.execute(
    f"SELECT COUNT(*) FROM read_csv('{OUTPUT_VALID}')"
).fetchone()[0]

print()
print("========================================")
print("PAIR GENERATION COMPLETE")
print("========================================")
print("Train rows:", train_count)
print("Valid rows:", valid_count)
print("Positive pairs:", positive_count)
print("Negative pairs:", negative_count)
print("Elapsed:", round(time.time() - start, 2), "seconds")