import os
import time
import duckdb

# ============================================================
# STEP 4.5 - TEST CANDIDATE GENERATION
# Memory-safe, batched DuckDB token blocking
# ============================================================

BASE = "."

S1_CACHE = f"{BASE}/normalized_cache_test/source1_normalized.parquet"
S2_CACHE = f"{BASE}/normalized_cache_test/source2_normalized.parquet"
S3_CACHE = f"{BASE}/normalized_cache_test/source3_normalized.parquet"

OUTPUT = f"{BASE}/step4_test_candidate_pairs.csv"
DB_FILE = f"{BASE}/test_candidate_index.duckdb"
TEMP_DIR = f"{BASE}/duckdb_test_tmp"

BATCH_SIZE = 10000

# Maximum token frequency allowed in candidate index.
# Very common tokens create enormous candidate sets.
MAX_TOKEN_FREQ = 5000

# Keep the strongest candidates from each source for each Source1 row.
TOP_K_PER_SOURCE = 40


def main():

    start = time.time()

    print("=" * 70)
    print("STEP 4.5 - TEST CANDIDATE GENERATION")
    print("=" * 70)

    os.makedirs(TEMP_DIR, exist_ok=True)

    print("\nInput caches:")
    print(S1_CACHE)
    print(S2_CACHE)
    print(S3_CACHE)

    print("\nOpening DuckDB...")

    con = duckdb.connect(DB_FILE)

    # Keep RAM below EC2 limit.
    con.execute("SET memory_limit='5GB'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute("SET threads=2")

    # --------------------------------------------------------
    # Load candidate sources
    # --------------------------------------------------------

    print("\nRegistering Source2 and Source3...")

    con.execute(f"""
        CREATE OR REPLACE VIEW source2 AS
        SELECT
            entity_id,
            name_norm,
            addr_norm,
            country_norm
        FROM read_parquet('{S2_CACHE}')
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW source3 AS
        SELECT
            entity_id,
            name_norm,
            addr_norm,
            country_norm
        FROM read_parquet('{S3_CACHE}')
    """)

    # --------------------------------------------------------
    # Build token index
    # --------------------------------------------------------

    print("\nBuilding token index...")
    print("This is disk-backed and may take some time.")

    con.execute("DROP TABLE IF EXISTS token_index")

    con.execute("""
        CREATE TABLE token_index AS

        SELECT DISTINCT
            token,
            candidate_entity_id,
            candidate_source,
            candidate_index,
            country_norm

        FROM (

            SELECT
                token,
                entity_id AS candidate_entity_id,
                'source2' AS candidate_source,
                entity_id AS candidate_index,
                country_norm

            FROM source2,
            LATERAL unnest(string_split(
                lower(coalesce(name_norm, '') || ' ' ||
                      coalesce(addr_norm, '')),
                '\\s+'
            )) AS t(token)

            WHERE length(token) >= 2

            UNION ALL

            SELECT
                token,
                entity_id AS candidate_entity_id,
                'source3' AS candidate_source,
                entity_id AS candidate_index,
                country_norm

            FROM source3,
            LATERAL unnest(string_split(
                lower(coalesce(name_norm, '') || ' ' ||
                      coalesce(addr_norm, '')),
                '\\s+'
            ) AS t(token)

            WHERE length(token) >= 2
        )
    """)

    print("Token index created.")

    # --------------------------------------------------------
    # Token frequencies
    # --------------------------------------------------------

    print("\nCalculating token frequencies...")

    con.execute("DROP TABLE IF EXISTS token_frequency")

    con.execute("""
        CREATE TABLE token_frequency AS
        SELECT
            token,
            COUNT(*) AS freq
        FROM token_index
        GROUP BY token
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_token_frequency
        ON token_frequency(token)
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_token_index
        ON token_index(token)
    """)

    freq_count = con.execute("""
        SELECT COUNT(*)
        FROM token_frequency
        WHERE freq <= ?
    """, [MAX_TOKEN_FREQ]).fetchone()[0]

    print(
        f"Usable tokens (frequency <= {MAX_TOKEN_FREQ:,}): "
        f"{freq_count:,}"
    )

    # --------------------------------------------------------
    # Source1
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT
            row_number() OVER () AS s1_row_number,
            entity_id,
            name_norm,
            addr_norm,
            country_norm
        FROM read_parquet('{S1_CACHE}')
    """)

    total_s1 = con.execute("""
        SELECT COUNT(*)
        FROM source1
    """).fetchone()[0]

    print(f"\nTest Source1 entities: {total_s1:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Top candidates/source: {TOP_K_PER_SOURCE}")

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if os.path.exists(OUTPUT):
        print(f"\nRemoving existing output: {OUTPUT}")
        os.remove(OUTPUT)

    first_batch = True

    # --------------------------------------------------------
    # Batch loop
    # --------------------------------------------------------

    for batch_start in range(0, total_s1, BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_s1
        )

        batch_no = batch_start // BATCH_SIZE + 1
        total_batches = (total_s1 + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\nBatch {batch_no}/{total_batches} "
            f"rows {batch_start:,} - {batch_end:,}"
        )

        con.execute("DROP TABLE IF EXISTS s1_batch")

        con.execute("""
            CREATE TEMP TABLE s1_batch AS
            SELECT
                s1_row_number,
                entity_id,
                name_norm,
                addr_norm,
                country_norm
            FROM source1
            WHERE s1_row_number > ?
              AND s1_row_number <= ?
        """, [batch_start, batch_end])

        # ----------------------------------------------------
        # Candidate scoring by shared tokens
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS batch_candidates")

        con.execute(f"""
            CREATE TEMP TABLE batch_candidates AS

            WITH s1_tokens AS (

                SELECT DISTINCT
                    s1_row_number,
                    entity_id,
                    country_norm,
                    token

                FROM s1_batch,

                LATERAL unnest(string_split(
                    lower(
                        coalesce(name_norm, '') || ' ' ||
                        coalesce(addr_norm, '')
                    ),
                    '\\s+'
                ) AS t(token)

                WHERE length(token) >= 2
            ),

            usable_tokens AS (

                SELECT
                    s.token,
                    s.s1_row_number,
                    s.entity_id,
                    s.country_norm

                FROM s1_tokens s

                INNER JOIN token_frequency f
                    ON s.token = f.token

                WHERE f.freq <= {MAX_TOKEN_FREQ}
            ),

            scored AS (

                SELECT
                    u.s1_row_number,
                    u.entity_id AS source1_entity_id,

                    ti.candidate_entity_id,
                    ti.candidate_source,
                    ti.candidate_index,

                    COUNT(DISTINCT u.token) AS shared_token_count

                FROM usable_tokens u

                INNER JOIN token_index ti
                    ON u.token = ti.token
                   AND (
                        u.country_norm = ti.country_norm
                        OR u.country_norm IS NULL
                        OR ti.country_norm IS NULL
                   )

                GROUP BY
                    u.s1_row_number,
                    u.entity_id,
                    ti.candidate_entity_id,
                    ti.candidate_source,
                    ti.candidate_index
            ),

            ranked AS (

                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            s1_row_number,
                            candidate_source

                        ORDER BY
                            shared_token_count DESC,
                            candidate_entity_id
                    ) AS rn

                FROM scored
            )

            SELECT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                candidate_index

            FROM ranked

            WHERE rn <= {TOP_K_PER_SOURCE}
        """)

        # ----------------------------------------------------
        # Add exact normalized name candidates.
        # These are extremely valuable and should never be
        # discarded by TOP_K.
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS exact_name_candidates")

        con.execute("""
            CREATE TEMP TABLE exact_name_candidates AS

            SELECT DISTINCT
                b.entity_id AS source1_entity_id,
                s.entity_id AS candidate_entity_id,
                'source2' AS candidate_source,
                s.entity_id AS candidate_index

            FROM s1_batch b

            INNER JOIN source2 s
                ON b.name_norm <> ''
               AND b.name_norm = s.name_norm
               AND (
                    b.country_norm = s.country_norm
                    OR b.country_norm IS NULL
                    OR s.country_norm IS NULL
               )

            UNION

            SELECT DISTINCT
                b.entity_id AS source1_entity_id,
                s.entity_id AS candidate_entity_id,
                'source3' AS candidate_source,
                s.entity_id AS candidate_index

            FROM s1_batch b

            INNER JOIN source3 s
                ON b.name_norm <> ''
               AND b.name_norm = s.name_norm
               AND (
                    b.country_norm = s.country_norm
                    OR b.country_norm IS NULL
                    OR s.country_norm IS NULL
               )
        """)

        # ----------------------------------------------------
        # Add exact normalized address candidates.
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS exact_addr_candidates")

        con.execute("""
            CREATE TEMP TABLE exact_addr_candidates AS

            SELECT DISTINCT
                b.entity_id AS source1_entity_id,
                s.entity_id AS candidate_entity_id,
                'source2' AS candidate_source,
                s.entity_id AS candidate_index

            FROM s1_batch b

            INNER JOIN source2 s
                ON b.addr_norm <> ''
               AND b.addr_norm = s.addr_norm
               AND (
                    b.country_norm = s.country_norm
                    OR b.country_norm IS NULL
                    OR s.country_norm IS NULL
               )

            UNION

            SELECT DISTINCT
                b.entity_id AS source1_entity_id,
                s.entity_id AS candidate_entity_id,
                'source3' AS candidate_source,
                s.entity_id AS candidate_index

            FROM s1_batch b

            INNER JOIN source3 s
                ON b.addr_norm <> ''
               AND b.addr_norm = s.addr_norm
               AND (
                    b.country_norm = s.country_norm
                    OR b.country_norm IS NULL
                    OR s.country_norm IS NULL
               )
        """)

        # ----------------------------------------------------
        # Combine candidates and remove duplicates.
        # ----------------------------------------------------

        con.execute("DROP TABLE IF EXISTS batch_output")

        con.execute("""
            CREATE TEMP TABLE batch_output AS

            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                candidate_source,
                candidate_index

            FROM (

                SELECT *
                FROM batch_candidates

                UNION ALL

                SELECT *
                FROM exact_name_candidates

                UNION ALL

                SELECT *
                FROM exact_addr_candidates
            )
        """)

        batch_count = con.execute("""
            SELECT COUNT(*)
            FROM batch_output
        """).fetchone()[0]

        print(f"Candidates in batch: {batch_count:,}")

        # ----------------------------------------------------
        # Append to CSV
        # ----------------------------------------------------

        if first_batch:

            con.execute(f"""
                COPY (
                    SELECT
                        source1_entity_id,
                        candidate_entity_id,
                        candidate_source,
                        candidate_index

                    FROM batch_output

                    ORDER BY
                        source1_entity_id,
                        candidate_source,
                        candidate_entity_id
                )

                TO '{OUTPUT}'
                (
                    FORMAT CSV,
                    HEADER TRUE
                )
            """)

            first_batch = False

        else:

            con.execute(f"""
                COPY (
                    SELECT
                        source1_entity_id,
                        candidate_entity_id,
                        candidate_source,
                        candidate_index

                    FROM batch_output

                    ORDER BY
                        source1_entity_id,
                        candidate_source,
                        candidate_entity_id
                )

                TO '{OUTPUT}'
                (
                    FORMAT CSV,
                    HEADER FALSE,
                    APPEND TRUE
                )
            """)

        elapsed = time.time() - start

        print(
            f"Elapsed: {elapsed / 60:.1f} min"
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(f"Output: {OUTPUT}")

    if os.path.exists(OUTPUT):
        size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        print(f"Output size: {size_mb:.1f} MB")

    con.close()

    print(
        f"Total runtime: {(time.time() - start) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()