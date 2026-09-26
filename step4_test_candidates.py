#!/usr/bin/env python3

"""
STEP 4 - TEST CANDIDATE GENERATION (optimized + robust)

Generates final candidate pairs for Source 1 test entities against
Source 2 and Source 3.

Fixes applied:
  * Valid DuckDB UNNEST syntax (comma-join / LATERAL, not CROSS JOIN UNNEST)
  * Deterministic pagination: ORDER BY entity_id on LIMIT/OFFSET batches
  * Guard against empty batch output dir before final CSV write
  * Per-source token frequency semantics made explicit (global cap documented)
  * Faster final stats (read Parquet batches, not re-parse CSV)
  * Batch counters reused instead of re-scanning CSV

Key features:
  * Auto-detects token column name/shape in source1/2/3
  * preserve_insertion_order = false  (fixes OOM)
  * Batch by LIMIT/OFFSET (no ROW_NUMBER() OVER () pagination)
  * Token frequency pre-filtered via HAVING
  * Country compatibility pushed into WHERE of the shared-token CTE
  * Per-batch Parquet output; single final CSV write
"""

import os
import shutil
import time
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

S1_CACHE = "/home/ec2-user/ML--challenge/normalized_cache_test/source1_normalized.parquet"
S2_CACHE = "/home/ec2-user/ML--challenge/normalized_cache_test/source2_normalized.parquet"
S3_CACHE = "/home/ec2-user/ML--challenge/normalized_cache_test/source3_normalized.parquet"

OUTPUT = "./step4_test_candidate_pairs.csv"
BATCH_OUTPUT_DIR = "./step4_batches"

DB_FILE = "./test_candidate_index.duckdb"
TEMP_DIR = "./duckdb_test_tmp"

DUCKDB_MEMORY = "5GB"
DUCKDB_THREADS = 2

BATCH_SIZE = 10_000
MAX_TOKEN_FREQ = 5_000          # global cap across source2 + source3
TOP_K_PER_SOURCE = 40

# Candidate column names for the token list, in priority order.
TOKEN_COL_CANDIDATES = (
    "name_tokens",
    "tokens",
    "token_list",
    "name_token_list",
)


# ============================================================
# HELPERS
# ============================================================

def log(msg):
    print(msg, flush=True)


def prepare_dirs():
    for p in (OUTPUT, DB_FILE):
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(BATCH_OUTPUT_DIR):
        shutil.rmtree(BATCH_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def describe_parquet(con, path):
    """Return {column_name: column_type} for a Parquet file."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def resolve_token_expr(cols, label):
    """
    Figure out how to expand tokens from a source's columns.

    Returns (token_col_name_or_None, sql_expr_template).

    The returned template uses '{alias}' as a placeholder for the table
    alias, and is designed to be used in a *table function* position:

        FROM source2 s, <expr> AS t(token)

    which is valid DuckDB syntax. It is NOT valid inside
    `CROSS JOIN <expr>` in older DuckDB versions, so we use the
    comma-join form everywhere.
    """
    for cand in TOKEN_COL_CANDIDATES:
        if cand in cols:
            ctype = cols[cand].upper()
            is_list = ("[" in ctype) or ("LIST" in ctype)
            if is_list:
                return cand, f"UNNEST({{alias}}.{cand})"
            else:
                return cand, f"UNNEST(string_split({{alias}}.{cand}, ' '))"

    if "name_norm" in cols:
        log(f"  [{label}] No token column found; falling back to string_split(name_norm).")
        return None, "UNNEST(string_split(coalesce({alias}.name_norm, ''), ' '))"

    raise RuntimeError(
        f"[{label}] Could not find a usable token column and no name_norm "
        f"column exists either. Columns: {list(cols.keys())}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    log("=" * 70)
    log("STEP 4 - TEST CANDIDATE GENERATION (optimized)")
    log("=" * 70)

    prepare_dirs()

    # --------------------------------------------------------
    # CONNECT + TUNE
    # --------------------------------------------------------

    log("\nOpening DuckDB...")
    con = duckdb.connect(DB_FILE)

    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # Critical for avoiding OOM on token index build.
    # NOTE: this disables stable row order for unordered scans, so all
    # paginated reads below MUST include an explicit ORDER BY.
    con.execute("SET preserve_insertion_order = false")

    for path in (S1_CACHE, S2_CACHE, S3_CACHE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # --------------------------------------------------------
    # SCHEMA INSPECTION
    # --------------------------------------------------------

    log("\nInspecting schemas...")
    s1_cols = describe_parquet(con, S1_CACHE)
    s2_cols = describe_parquet(con, S2_CACHE)
    s3_cols = describe_parquet(con, S3_CACHE)

    for label, cols in (("source1", s1_cols), ("source2", s2_cols), ("source3", s3_cols)):
        log(f"\n  --- {label} ---")
        for k, v in cols.items():
            log(f"    {k:25s} {v}")

    # Resolve token expressions per source.
    s1_tok_col, s1_tok_tmpl = resolve_token_expr(s1_cols, "source1")
    s2_tok_col, s2_tok_tmpl = resolve_token_expr(s2_cols, "source2")
    s3_tok_col, s3_tok_tmpl = resolve_token_expr(s3_cols, "source3")

    log("\nToken expressions resolved:")
    log(f"  source1: {s1_tok_col or '<from name_norm>'} -> {s1_tok_tmpl}")
    log(f"  source2: {s2_tok_col or '<from name_norm>'} -> {s2_tok_tmpl}")
    log(f"  source3: {s3_tok_col or '<from name_norm>'} -> {s3_tok_tmpl}")

    # Sanity: required common columns
    for label, cols in (("source1", s1_cols), ("source2", s2_cols), ("source3", s3_cols)):
        for required in ("entity_id", "country_norm", "name_norm", "addr_norm"):
            if required not in cols:
                raise RuntimeError(
                    f"[{label}] Missing required column '{required}'. "
                    f"Columns: {list(cols.keys())}"
                )

    # --------------------------------------------------------
    # VIEWS
    # --------------------------------------------------------

    log("\nCreating Parquet views...")
    con.execute(f"CREATE OR REPLACE VIEW source1 AS SELECT * FROM read_parquet('{S1_CACHE}')")
    con.execute(f"CREATE OR REPLACE VIEW source2 AS SELECT * FROM read_parquet('{S2_CACHE}')")
    con.execute(f"CREATE OR REPLACE VIEW source3 AS SELECT * FROM read_parquet('{S3_CACHE}')")

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    total_s1 = con.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    log(f"\nTest Source1 entities: {total_s1:,}")
    log(f"Batch size:            {BATCH_SIZE:,}")
    log(f"Expected batches:      {(total_s1 + BATCH_SIZE - 1) // BATCH_SIZE:,}")

    # --------------------------------------------------------
    # TOKEN FREQUENCY  (global across S2 + S3)
    # --------------------------------------------------------

    log("\nCalculating token frequencies...")

    s2_unnest = s2_tok_tmpl.format(alias="s")
    s3_unnest = s3_tok_tmpl.format(alias="s")

    con.execute("DROP TABLE IF EXISTS token_frequency")
    con.execute(f"""
        CREATE TABLE token_frequency AS
        SELECT
            token,
            COUNT(*) AS frequency
        FROM (
            SELECT t.token
            FROM source2 s, {s2_unnest} AS t(token)

            UNION ALL

            SELECT t.token
            FROM source3 s, {s3_unnest} AS t(token)
        )
        WHERE token IS NOT NULL
          AND token <> ''
        GROUP BY token
        HAVING COUNT(*) <= {MAX_TOKEN_FREQ}
    """)

    usable_tokens = con.execute(
        "SELECT COUNT(*) FROM token_frequency"
    ).fetchone()[0]

    log(f"  usable tokens (global freq <= {MAX_TOKEN_FREQ:,}): {usable_tokens:,}")

    # --------------------------------------------------------
    # FILTERED TOKEN INDEX
    # --------------------------------------------------------

    log("Building filtered token index...")

    # Only usable tokens are retained. No ART index is created:
    # DuckDB hash-joins this fine, and the ART build was the source
    # of the reported memory-cap failure.
    con.execute("DROP TABLE IF EXISTS token_index")
    con.execute(f"""
        CREATE TABLE token_index AS
        SELECT DISTINCT
            t.token,
            s.entity_id      AS candidate_entity_id,
            'source2'        AS candidate_source,
            s.country_norm   AS country_norm
        FROM source2 s, {s2_unnest} AS t(token)
        INNER JOIN token_frequency tf
            ON tf.token = t.token
        WHERE t.token IS NOT NULL
          AND t.token <> ''

        UNION ALL

        SELECT DISTINCT
            t.token,
            s.entity_id      AS candidate_entity_id,
            'source3'        AS candidate_source,
            s.country_norm   AS country_norm
        FROM source3 s, {s3_unnest} AS t(token)
        INNER JOIN token_frequency tf
            ON tf.token = t.token
        WHERE t.token IS NOT NULL
          AND t.token <> ''
    """)

    dedup_count = con.execute(
        "SELECT COUNT(*) FROM token_index"
    ).fetchone()[0]

    log(f"  filtered token index rows: {dedup_count:,}")

    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    total_candidate_rows = 0
    total_batches = (total_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    s1_unnest = s1_tok_tmpl.format(alias="s")

    for batch_start in range(0, total_s1, BATCH_SIZE):
        batch_no = batch_start // BATCH_SIZE + 1
        batch_time = time.time()

        log(f"\nBatch {batch_no:,}/{total_batches:,}  offset={batch_start:,}")

        # ---- source1 batch (deterministic order!) ----
        con.execute("DROP TABLE IF EXISTS s1_batch")
        con.execute(f"""
            CREATE TEMP TABLE s1_batch AS
            SELECT *
            FROM source1
            ORDER BY entity_id
            LIMIT {BATCH_SIZE} OFFSET {batch_start}
        """)

        # ---- token-shared candidates ----
        con.execute("DROP TABLE IF EXISTS batch_candidates")
        con.execute(f"""
            CREATE TEMP TABLE batch_candidates AS

            WITH shared_tokens AS (
                SELECT
                    s.entity_id              AS source1_entity_id,
                    ti.candidate_entity_id,
                    ti.candidate_source,
                    COUNT(*)                 AS shared_token_count
                FROM s1_batch s, {s1_unnest} AS t(token)
                INNER JOIN token_frequency tf
                    ON tf.token = t.token
                INNER JOIN token_index ti
                    ON ti.token = t.token
                WHERE
                    s.country_norm IS NULL
                    OR ti.country_norm IS NULL
                    OR s.country_norm = ''
                    OR ti.country_norm = ''
                    OR s.country_norm = ti.country_norm
                GROUP BY
                    s.entity_id,
                    ti.candidate_entity_id,
                    ti.candidate_source
            ),

            ranked AS (
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id, candidate_source
                        ORDER BY shared_token_count DESC, candidate_entity_id
                    ) AS rn
                FROM shared_tokens
            )

            SELECT source1_entity_id, candidate_entity_id, candidate_source
            FROM ranked
            WHERE rn <= {TOP_K_PER_SOURCE}
        """)

        # ---- exact name + address candidates ----
        con.execute("""
            INSERT INTO batch_candidates

            WITH exact_matches AS (
                SELECT DISTINCT
                    s.entity_id AS source1_entity_id,
                    x.entity_id AS candidate_entity_id,
                    'source2'   AS candidate_source
                FROM s1_batch s
                JOIN source2 x
                  ON s.name_norm <> '' AND s.name_norm = x.name_norm
                WHERE
                    s.country_norm IS NULL OR x.country_norm IS NULL
                    OR s.country_norm = '' OR x.country_norm = ''
                    OR s.country_norm = x.country_norm

                UNION ALL

                SELECT DISTINCT
                    s.entity_id AS source1_entity_id,
                    x.entity_id AS candidate_entity_id,
                    'source2'   AS candidate_source
                FROM s1_batch s
                JOIN source2 x
                  ON s.addr_norm <> '' AND s.addr_norm = x.addr_norm
                WHERE
                    s.country_norm IS NULL OR x.country_norm IS NULL
                    OR s.country_norm = '' OR x.country_norm = ''
                    OR s.country_norm = x.country_norm

                UNION ALL

                SELECT DISTINCT
                    s.entity_id AS source1_entity_id,
                    x.entity_id AS candidate_entity_id,
                    'source3'   AS candidate_source
                FROM s1_batch s
                JOIN source3 x
                  ON s.name_norm <> '' AND s.name_norm = x.name_norm
                WHERE
                    s.country_norm IS NULL OR x.country_norm IS NULL
                    OR s.country_norm = '' OR x.country_norm = ''
                    OR s.country_norm = x.country_norm

                UNION ALL

                SELECT DISTINCT
                    s.entity_id AS source1_entity_id,
                    x.entity_id AS candidate_entity_id,
                    'source3'   AS candidate_source
                FROM s1_batch s
                JOIN source3 x
                  ON s.addr_norm <> '' AND s.addr_norm = x.addr_norm
                WHERE
                    s.country_norm IS NULL OR x.country_norm IS NULL
                    OR s.country_norm = '' OR x.country_norm = ''
                    OR s.country_norm = x.country_norm
            )
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                candidate_source
            FROM exact_matches
        """)

        # ---- write per-batch parquet ----
        batch_file = f"{BATCH_OUTPUT_DIR}/batch_{batch_no:05d}.parquet"
        con.execute(f"""
            COPY (
                SELECT DISTINCT
                    source1_entity_id,
                    candidate_entity_id,
                    candidate_source
                FROM batch_candidates
            )
            TO '{batch_file}' (FORMAT PARQUET)
        """)

        batch_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{batch_file}')"
        ).fetchone()[0]
        total_candidate_rows += batch_rows

        log(f"  Candidate rows:       {batch_rows:,}")
        log(f"  Total candidate rows: {total_candidate_rows:,}")
        log(f"  Batch time:           {time.time() - batch_time:.1f}s")

    # --------------------------------------------------------
    # FINAL WRITE
    # --------------------------------------------------------

    if total_candidate_rows == 0:
        log("\nNo candidate rows produced; writing empty CSV with header.")
        con.execute(f"""
            COPY (
                SELECT
                    CAST(NULL AS VARCHAR) AS source1_entity_id,
                    CAST(NULL AS VARCHAR) AS candidate_entity_id,
                    CAST(NULL AS VARCHAR) AS candidate_source
                WHERE FALSE
            )
            TO '{OUTPUT}' (HEADER TRUE, DELIMITER ',')
        """)
    else:
        log("\nWriting final CSV...")
        con.execute(f"""
            COPY (
                SELECT source1_entity_id, candidate_entity_id, candidate_source
                FROM read_parquet('{BATCH_OUTPUT_DIR}/batch_*.parquet')
            )
            TO '{OUTPUT}' (HEADER TRUE, DELIMITER ',')
        """)

    # --------------------------------------------------------
    # STATISTICS  (read from Parquet batches, not the CSV)
    # --------------------------------------------------------

    log("\n" + "=" * 70)
    log("CANDIDATE GENERATION COMPLETE")
    log("=" * 70)

    if total_candidate_rows > 0:
        stats = con.execute(f"""
            SELECT
                COUNT(*)                                              AS output_rows,
                COUNT(DISTINCT source1_entity_id)                     AS unique_s1,
                COUNT(*) FILTER (WHERE candidate_source = 'source2')  AS s2_cand,
                COUNT(*) FILTER (WHERE candidate_source = 'source3')  AS s3_cand
            FROM read_parquet('{BATCH_OUTPUT_DIR}/batch_*.parquet')
        """).fetchone()
    else:
        stats = (0, 0, 0, 0)

    output_rows, unique_s1, s2_cand, s3_cand = stats

    log(f"Source1 entities:           {total_s1:,}")
    log(f"Source1 with candidates:    {unique_s1:,}")
    log(f"Candidate rows:             {output_rows:,}")
    log(f"Source2 candidates:         {s2_cand:,}")
    log(f"Source3 candidates:         {s3_cand:,}")

    coverage = (unique_s1 / total_s1 * 100) if total_s1 else 0
    log(f"Source1 candidate coverage: {coverage:.2f}%")

    log(f"Total time: {((time.time() - start_time) / 60):.2f} minutes")
    log(f"Output: {OUTPUT}")

    con.close()
    log("\nDone.")


if __name__ == "__main__":
    main()