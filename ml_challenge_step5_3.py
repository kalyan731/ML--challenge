import time
from pathlib import Path

import duckdb


START = time.time()

# ================================================================
# CONFIG
# ================================================================

CACHE_DIR = Path("normalized_cache_test")

S1_FILE = (CACHE_DIR / "source1_normalized.parquet").resolve()
S2_FILE = (CACHE_DIR / "source2_normalized.parquet").resolve()
S3_FILE = (CACHE_DIR / "source3_normalized.parquet").resolve()

OUTPUT_FILE = Path(
    "test_candidate_pairs.tsv"
).resolve()

DB_FILE = Path(
    "step5_3_blocking.duckdb"
).resolve()

# Ignore extremely common tokens.
MAX_TOKEN_FREQ = 5000

# Token must contain at least 2 characters.
MIN_TOKEN_LEN = 2


# ================================================================
# START
# ================================================================

print("=" * 75)
print("STEP 5.3 - DUCKDB MEMORY-SAFE BLOCKING")
print("=" * 75)

print("\nNo normalization will be performed.")
print("Using existing normalized Parquet cache.")

print("\nS1:", S1_FILE)
print("S2:", S2_FILE)
print("S3:", S3_FILE)


# ================================================================
# CONNECT
# ================================================================

print("\nOpening DuckDB...")

con = duckdb.connect(
    str(DB_FILE)
)

# Keep memory under control for the 16 GB machine.
con.execute(
    "SET memory_limit='6GB'"
)

con.execute(
    "SET threads=4"
)

con.execute(
    "SET preserve_insertion_order=false"
)

print("DuckDB configured.")


# ================================================================
# VERIFY FILES
# ================================================================

for path in [
    S1_FILE,
    S2_FILE,
    S3_FILE,
]:

    if not path.exists():
        raise FileNotFoundError(path)


# ================================================================
# CREATE SOURCE VIEWS
# ================================================================

print("\nCreating Parquet views...")

con.execute(
    f"""
    CREATE OR REPLACE VIEW s1 AS
    SELECT
        row_number() OVER () - 1 AS source1_index,
        entity_id AS source1_entity_id,
        name_norm,
        addr_norm,
        country_norm
    FROM read_parquet('{S1_FILE.as_posix()}')
    """
)

con.execute(
    f"""
    CREATE OR REPLACE VIEW s2 AS
    SELECT
        row_number() OVER () - 1 AS candidate_index,
        entity_id AS candidate_entity_id,
        name_norm,
        addr_norm,
        country_norm
    FROM read_parquet('{S2_FILE.as_posix()}')
    """
)

con.execute(
    f"""
    CREATE OR REPLACE VIEW s3 AS
    SELECT
        row_number() OVER () - 1 AS candidate_index,
        entity_id AS candidate_entity_id,
        name_norm,
        addr_norm,
        country_norm
    FROM read_parquet('{S3_FILE.as_posix()}')
    """
)

print("Views created.")


# ================================================================
# BUILD TOKEN TABLES
# ================================================================

print("\nBuilding S2 token table...")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE s2_tokens AS
    SELECT
        candidate_index,
        candidate_entity_id,
        country_norm,
        token
    FROM s2,
    LATERAL unnest(
        string_split(
            trim(
                regexp_replace(
                    concat(
                        coalesce(name_norm, ''),
                        ' ',
                        coalesce(addr_norm, '')
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ),
            ' '
        )
    ) AS t(token)
    WHERE length(token) >= {MIN_TOKEN_LEN}
    """
)

print("S2 tokens ready.")


print("\nBuilding S3 token table...")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE s3_tokens AS
    SELECT
        candidate_index,
        candidate_entity_id,
        country_norm,
        token
    FROM s3,
    LATERAL unnest(
        string_split(
            trim(
                regexp_replace(
                    concat(
                        coalesce(name_norm, ''),
                        ' ',
                        coalesce(addr_norm, '')
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ),
            ' '
        )
    ) AS t(token)
    WHERE length(token) >= {MIN_TOKEN_LEN}
    """
)

print("S3 tokens ready.")


# ================================================================
# TOKEN FREQUENCY
# ================================================================

print("\nCalculating token frequencies...")

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE token_frequency AS

    SELECT
        token,
        COUNT(*) AS frequency

    FROM
    (
        SELECT token
        FROM s2_tokens

        UNION ALL

        SELECT token
        FROM s3_tokens
    )

    GROUP BY token
    """
)

token_count = con.execute(
    """
    SELECT COUNT(*)
    FROM token_frequency
    """
).fetchone()[0]

print(
    f"Unique tokens: {token_count:,}"
)


# ================================================================
# USABLE TOKENS
# ================================================================

print("\nFiltering common tokens...")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE usable_tokens AS
    SELECT
        token,
        frequency
    FROM token_frequency
    WHERE frequency <= {MAX_TOKEN_FREQ}
      AND length(token) >= {MIN_TOKEN_LEN}
    """
)

usable_count = con.execute(
    """
    SELECT COUNT(*)
    FROM usable_tokens
    """
).fetchone()[0]

print(
    f"Usable tokens: {usable_count:,}"
)


# ================================================================
# INDEX TABLES
# ================================================================

print("\nFiltering S2 tokens...")

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE s2_index AS
    SELECT
        t.candidate_index,
        t.candidate_entity_id,
        t.country_norm,
        t.token
    FROM s2_tokens t
    INNER JOIN usable_tokens u
        ON t.token = u.token
    """
)

print("S2 index ready.")


print("\nFiltering S3 tokens...")

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE s3_index AS
    SELECT
        t.candidate_index,
        t.candidate_entity_id,
        t.country_norm,
        t.token
    FROM s3_tokens t
    INNER JOIN usable_tokens u
        ON t.token = u.token
    """
)

print("S3 index ready.")


# ================================================================
# QUERY TOKEN TABLE
# ================================================================

print("\nBuilding Source1 query tokens...")

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE s1_tokens AS

    SELECT
        source1_index,
        source1_entity_id,
        country_norm,
        token

    FROM
    (
        SELECT
            source1_index,
            source1_entity_id,
            country_norm,
            token

        FROM s1,

        LATERAL unnest(
            string_split(
                trim(
                    regexp_replace(
                        concat(
                            coalesce(name_norm, ''),
                            ' ',
                            coalesce(addr_norm, '')
                        ),
                        '\\s+',
                        ' ',
                        'g'
                    )
                ),
                ' '
            )
        ) AS t(token)
    )

    WHERE length(token) >= 2
    """
)

print("Source1 query tokens ready.")


# ================================================================
# RANK TOKENS PER SOURCE1
# ================================================================

print("\nSelecting up to 4 rarest query tokens per Source1...")

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE s1_query_tokens AS

    SELECT
        source1_index,
        source1_entity_id,
        country_norm,
        token

    FROM
    (
        SELECT
            t.source1_index,
            t.source1_entity_id,
            t.country_norm,
            t.token,
            u.frequency,

            ROW_NUMBER() OVER (
                PARTITION BY t.source1_index
                ORDER BY
                    u.frequency ASC,
                    t.token ASC
            ) AS rn

        FROM s1_tokens t

        INNER JOIN usable_tokens u
            ON t.token = u.token
    )

    WHERE rn <= 4
    """
)


# ================================================================
# CANDIDATE GENERATION
# ================================================================

print("\nGenerating candidates...")

# We write directly from DuckDB to a TSV.
# Candidates are country-matched and share at least one
# rare/informative token.

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()


query = f"""
COPY
(
    SELECT DISTINCT

        q.source1_entity_id,

        c.candidate_entity_id,

        c.candidate_source,

        c.candidate_index

    FROM s1_query_tokens q

    INNER JOIN
    (
        SELECT
            candidate_index,
            candidate_entity_id,
            country_norm,
            token,
            0 AS candidate_source

        FROM s2_index

        UNION ALL

        SELECT
            candidate_index,
            candidate_entity_id,
            country_norm,
            token,
            1 AS candidate_source

        FROM s3_index
    ) c

        ON q.country_norm = c.country_norm
       AND q.token = c.token

)

TO '{OUTPUT_FILE.as_posix()}'
(
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER TRUE
)
"""

con.execute(query)


# ================================================================
# RESULTS
# ================================================================

row_count = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_csv(
        '{OUTPUT_FILE.as_posix()}',
        delim='\\t',
        header=true
    )
    """
).fetchone()[0]


s1_count = con.execute(
    """
    SELECT COUNT(*)
    FROM s1
    """
).fetchone()[0]


print("\n" + "=" * 75)
print("STEP 5.3 COMPLETE")
print("=" * 75)

print(
    f"Source1 entities: "
    f"{s1_count:,}"
)

print(
    f"Candidate pairs: "
    f"{row_count:,}"
)

if s1_count > 0:
    print(
        f"Average candidates/S1: "
        f"{row_count / s1_count:.2f}"
    )

print(
    f"Output: "
    f"{OUTPUT_FILE}"
)

print(
    f"Runtime: "
    f"{time.time() - START:.2f}s"
)

print("=" * 75)
print("DONE")
print("=" * 75)

con.close()