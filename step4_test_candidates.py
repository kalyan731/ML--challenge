#!/usr/bin/env python3

"""
STEP 4 - TEST CANDIDATE GENERATION

Generates final candidate pairs for Source 1 test entities against
Source 2 and Source 3.

Design goals:
- Memory-safe: DuckDB memory capped at 5 GB
- Batch processing of Source 1
- Strong token blocking
- Country-aware matching without hard-coding countries
- Exact name/address candidates always retained
- Top-K candidates per source
- Output contains only final candidate pairs
"""

import os
import time
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "."

S1_CACHE = "./normalized_cache_test/source1_normalized.parquet"
S2_CACHE = "./normalized_cache_test/source2_normalized.parquet"
S3_CACHE = "./normalized_cache_test/source3_normalized.parquet"

OUTPUT = "./step4_test_candidate_pairs.csv"

DB_FILE = "./test_candidate_index.duckdb"
TEMP_DIR = "./duckdb_test_tmp"

# Keep RAM comfortably below the 8 GB target.
DUCKDB_MEMORY = "5GB"
DUCKDB_THREADS = 2

# Batch size for Source 1.
BATCH_SIZE = 10_000

# Ignore extremely common tokens.
MAX_TOKEN_FREQ = 5_000

# Maximum candidates retained from each source per Source 1 entity.
TOP_K_PER_SOURCE = 40


# ============================================================
# HELPERS
# ============================================================

def log(message):
    print(message, flush=True)


def remove_old_output():
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    if os.path.exists(TEMP_DIR):
        # DuckDB temporary files can safely be removed from a
        # previous completed/interrupted run.
        import shutil
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    log("=" * 70)
    log("STEP 4 - TEST CANDIDATE GENERATION")
    log("=" * 70)

    remove_old_output()

    # --------------------------------------------------------
    # CONNECT DUCKDB
    # --------------------------------------------------------

    log("\nOpening DuckDB...")

    con = duckdb.connect(DB_FILE)

    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    required_files = [
        S1_CACHE,
        S2_CACHE,
        S3_CACHE,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    log("Input files found.")

    # --------------------------------------------------------
    # CREATE SOURCE VIEWS
    # --------------------------------------------------------

    log("\nCreating Parquet views...")

    con.execute(f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT *
        FROM read_parquet('{S1_CACHE}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW source2 AS
        SELECT *
        FROM read_parquet('{S2_CACHE}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW source3 AS
        SELECT *
        FROM read_parquet('{S3_CACHE}')
    """)

    # --------------------------------------------------------
    # COUNT SOURCE 1
    # --------------------------------------------------------

    total_s1 = con.execute("""
        SELECT COUNT(*)
        FROM source1
    """).fetchone()[0]

    log(f"Test Source1 entities: {total_s1:,}")
    log(f"Batch size: {BATCH_SIZE:,}")
    log(f"Expected batches: {(total_s1 + BATCH_SIZE - 1) // BATCH_SIZE:,}")

    # --------------------------------------------------------
    # BUILD TOKEN INDEX
    # --------------------------------------------------------

    log("\nBuilding token index...")

    con.execute("""
        CREATE TABLE token_index AS

        SELECT DISTINCT
            token,
            entity_id AS candidate_entity_id,
            'source2' AS candidate_source,
            country_norm
        FROM source2
        CROSS JOIN UNNEST(name_tokens) AS t(token)

        UNION

        SELECT DISTINCT
            token,
            entity_id AS candidate_entity_id,
            'source3' AS candidate_source,
            country_norm
        FROM source3
        CROSS JOIN UNNEST(name_tokens) AS t(token)
    """)

    token_count = con.execute("""
        SELECT COUNT(*)
        FROM token_index
    """).fetchone()[0]

    log(f"Token index rows: {token_count:,}")

    # --------------------------------------------------------
    # TOKEN FREQUENCY
    # --------------------------------------------------------

    log("Calculating token frequencies...")

    con.execute("""
        CREATE TABLE token_frequency AS
        SELECT
            token,
            COUNT(*) AS frequency
        FROM token_index
        GROUP BY token
    """)

    usable_tokens = con.execute(f"""
        SELECT COUNT(*)
        FROM token_frequency
        WHERE frequency <= {MAX_TOKEN_FREQ}
    """).fetchone()[0]

    log(
        f"Usable tokens (frequency <= {MAX_TOKEN_FREQ:,}): "
        f"{usable_tokens:,}"
    )

    # --------------------------------------------------------
    # SOURCE 1 ROW NUMBERS
    # --------------------------------------------------------

    log("\nPreparing Source1 batches...")

    con.execute("""
        CREATE VIEW source1_numbered AS

        SELECT
            *,
            ROW_NUMBER() OVER () AS s1_row_number

        FROM source1
    """)

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    total_candidate_rows = 0
    total_batches = (total_s1 + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(0, total_s1, BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_s1
        )

        batch_no = batch_start // BATCH_SIZE + 1

        batch_time = time.time()

        log(
            f"\nBatch {batch_no:,}/{total_batches:,} "
            f"Source1 rows "
            f"{batch_start + 1:,}-{batch_end:,}"
        )

        # ----------------------------------------------------
        # SOURCE 1 BATCH
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS s1_batch")

        con.execute(f"""
            CREATE TEMP TABLE s1_batch AS

            SELECT *
            FROM source1_numbered
            WHERE s1_row_number > {batch_start}
              AND s1_row_number <= {batch_end}
        """)

        # ----------------------------------------------------
        # TOKEN CANDIDATES
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS batch_candidates")

        con.execute(f"""
            CREATE TEMP TABLE batch_candidates AS

            WITH shared_tokens AS (

                SELECT
                    s.entity_id AS source1_entity_id,
                    ti.candidate_entity_id,
                    ti.candidate_source,

                    COUNT(DISTINCT t.token) AS shared_token_count,

                    CASE
                        WHEN
                            s.country_norm IS NULL
                            OR ti.country_norm IS NULL
                            OR s.country_norm = ''
                            OR ti.country_norm = ''
                            OR s.country_norm = ti.country_norm
                        THEN 1
                        ELSE 0
                    END AS country_compatible

                FROM s1_batch s

                CROSS JOIN UNNEST(s.name_tokens) AS t(token)

                INNER JOIN token_frequency tf
                    ON tf.token = t.token
                   AND tf.frequency <= {MAX_TOKEN_FREQ}

                INNER JOIN token_index ti
                    ON ti.token = t.token

                GROUP BY
                    s.entity_id,
                    ti.candidate_entity_id,
                    ti.candidate_source,
                    s.country_norm,
                    ti.country_norm
            ),

            ranked AS (

                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            source1_entity_id,
                            candidate_source

                        ORDER BY
                            country_compatible DESC,
                            shared_token_count DESC,
                            candidate_entity_id
                    ) AS rn

                FROM shared_tokens

                WHERE country_compatible = 1
            )

            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source

            FROM ranked

            WHERE rn <= {TOP_K_PER_SOURCE}
        """)

        # ----------------------------------------------------
        # EXACT NAME CANDIDATES
        # ----------------------------------------------------

        con.execute("""
            INSERT INTO batch_candidates

            SELECT DISTINCT
                s.entity_id AS source1_entity_id,
                x.entity_id AS candidate_entity_id,
                'source2' AS candidate_source

            FROM s1_batch s
            INNER JOIN source2 x
                ON s.name_norm <> ''
               AND s.name_norm = x.name_norm

            WHERE
                s.country_norm IS NULL
                OR x.country_norm IS NULL
                OR s.country_norm = ''
                OR x.country_norm = ''
                OR s.country_norm = x.country_norm
        """)

        con.execute("""
            INSERT INTO batch_candidates

            SELECT DISTINCT
                s.entity_id AS source1_entity_id,
                x.entity_id AS candidate_entity_id,
                'source3' AS candidate_source

            FROM s1_batch s
            INNER JOIN source3 x
                ON s.name_norm <> ''
               AND s.name_norm = x.name_norm

            WHERE
                s.country_norm IS NULL
                OR x.country_norm IS NULL
                OR s.country_norm = ''
                OR x.country_norm = ''
                OR s.country_norm = x.country_norm
        """)

        # ----------------------------------------------------
        # EXACT ADDRESS CANDIDATES
        # ----------------------------------------------------

        con.execute("""
            INSERT INTO batch_candidates

            SELECT DISTINCT
                s.entity_id AS source1_entity_id,
                x.entity_id AS candidate_entity_id,
                'source2' AS candidate_source

            FROM s1_batch s
            INNER JOIN source2 x
                ON s.address_norm <> ''
               AND s.address_norm = x.address_norm

            WHERE
                s.country_norm IS NULL
                OR x.country_norm IS NULL
                OR s.country_norm = ''
                OR x.country_norm = ''
                OR s.country_norm = x.country_norm
        """)

        con.execute("""
            INSERT INTO batch_candidates

            SELECT DISTINCT
                s.entity_id AS source1_entity_id,
                x.entity_id AS candidate_entity_id,
                'source3' AS candidate_source

            FROM s1_batch s
            INNER JOIN source3 x
                ON s.address_norm <> ''
               AND s.address_norm = x.address_norm

            WHERE
                s.country_norm IS NULL
                OR x.country_norm IS NULL
                OR s.country_norm = ''
                OR x.country_norm = ''
                OR s.country_norm = x.country_norm
        """)

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        con.execute("""
            CREATE TEMP TABLE batch_output AS

            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                candidate_source

            FROM batch_candidates
        """)

        batch_rows = con.execute("""
            SELECT COUNT(*)
            FROM batch_output
        """).fetchone()[0]

        total_candidate_rows += batch_rows

        # ----------------------------------------------------
        # WRITE OUTPUT
        # ----------------------------------------------------

        if batch_no == 1:

            con.execute(f"""
                COPY batch_output
                TO '{OUTPUT}'
                (
                    HEADER TRUE,
                    DELIMITER ','
                )
            """)

        else:

            con.execute(f"""
                COPY batch_output
                TO '{OUTPUT}'
                (
                    HEADER FALSE,
                    DELIMITER ',',
                    APPEND TRUE
                )
            """)

        elapsed = time.time() - batch_time

        log(
            f"  Candidate rows: {batch_rows:,}"
        )

        log(
            f"  Total candidate rows: "
            f"{total_candidate_rows:,}"
        )

        log(
            f"  Batch time: {elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    log("\n" + "=" * 70)
    log("CANDIDATE GENERATION COMPLETE")
    log("=" * 70)

    output_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{OUTPUT}',
            header=true,
            delim=','
        )
    """).fetchone()[0]

    unique_s1 = con.execute(f"""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM read_csv(
            '{OUTPUT}',
            header=true,
            delim=','
        )
    """).fetchone()[0]

    source2_candidates = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{OUTPUT}',
            header=true,
            delim=','
        )
        WHERE candidate_source = 'source2'
    """).fetchone()[0]

    source3_candidates = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{OUTPUT}',
            header=true,
            delim=','
        )
        WHERE candidate_source = 'source3'
    """).fetchone()[0]

    log(f"Source1 entities:       {total_s1:,}")
    log(f"Source1 with candidates:{unique_s1:,}")
    log(f"Candidate rows:         {output_rows:,}")
    log(f"Source2 candidates:     {source2_candidates:,}")
    log(f"Source3 candidates:     {source3_candidates:,}")

    coverage = (
        unique_s1 / total_s1 * 100
        if total_s1
        else 0
    )

    log(f"Source1 candidate coverage: {coverage:.2f}%")

    total_time = time.time() - start_time

    log(f"Total time: {total_time / 60:.2f} minutes")
    log(f"Output: {OUTPUT}")

    con.close()

    log("\nDone.")


if __name__ == "__main__":
    main()