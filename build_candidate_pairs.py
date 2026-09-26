#!/usr/bin/env python3
"""Build output/candidate_pairs.tsv — chunked, memory-safe."""

import os
import duckdb

CANDIDATE_FILE = "step4_test_candidate_pairs.csv"
TEST_SOURCE1_FILE = "resource/student_resource/dataset/test/test_source1.tsv"
OUTPUT_FILE = "output/candidate_pairs.tsv"
TEMP_DIR = "./duckdb_cp_tmp"

CHUNK_SIZE = 200_000  # Source1 entities per chunk

os.makedirs("output", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("=" * 70, flush=True)
print("BUILDING candidate_pairs.tsv (chunked)", flush=True)
print("=" * 70, flush=True)

con = duckdb.connect()
con.execute("SET threads=2")
con.execute("SET memory_limit='4GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")
con.execute("SET preserve_insertion_order=false")
con.execute("PRAGMA max_temp_directory_size='30GB'")

# ============================================================
# Load source1 IDs
# ============================================================
print("\n[1/5] Loading Source1 test entities...", flush=True)

con.execute(f"""
    CREATE OR REPLACE TABLE s1 AS
    SELECT DISTINCT entity_id AS source1_entity_id
    FROM read_csv('{TEST_SOURCE1_FILE}', delim='\t', header=true, quote='',
        columns={{'entity_id':'VARCHAR','business_name':'VARCHAR',
                  'business_address':'VARCHAR','country':'VARCHAR'}})
""")

n_s1 = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
print(f"  Source1 entities: {n_s1:,}", flush=True)

# ============================================================
# Assign chunk_id to each Source1
# ============================================================
print("\n[2/5] Assigning chunks...", flush=True)

con.execute(f"""
    CREATE OR REPLACE TABLE s1_chunked AS
    SELECT
        source1_entity_id,
        (ROW_NUMBER() OVER () - 1) // {CHUNK_SIZE} AS chunk_id
    FROM s1
""")

n_chunks = con.execute("SELECT MAX(chunk_id) + 1 FROM s1_chunked").fetchone()[0]
print(f"  Chunks: {n_chunks}", flush=True)

# ============================================================
# Load candidates as a persistent table (so it's materialized once)
# ============================================================
print("\n[3/5] Loading candidate pairs into table...", flush=True)

con.execute(f"""
    CREATE OR REPLACE TABLE cand AS
    SELECT
        CAST(source1_entity_id   AS VARCHAR) AS source1_entity_id,
        CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
    FROM read_csv('{CANDIDATE_FILE}', header=true, quote='')
""")

n_cand = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
print(f"  Candidate rows: {n_cand:,}", flush=True)

# ============================================================
# Write output incrementally
# ============================================================
print(f"\n[4/5] Processing {n_chunks} chunks...", flush=True)

# Header first
with open(OUTPUT_FILE, "w") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")

for chunk_id in range(n_chunks):
    chunk_file = f"{TEMP_DIR}/chunk_{chunk_id:04d}.tsv"

    con.execute(f"""
        COPY (
            WITH my_s1 AS (
                SELECT source1_entity_id
                FROM s1_chunked
                WHERE chunk_id = {chunk_id}
            ),
            agg AS (
                SELECT
                    c.source1_entity_id,
                    string_agg(c.candidate_entity_id, ',') AS candidate_entity_ids
                FROM cand c
                WHERE c.source1_entity_id IN (SELECT source1_entity_id FROM my_s1)
                GROUP BY c.source1_entity_id
            )
            SELECT
                s.source1_entity_id,
                COALESCE(a.candidate_entity_ids, '') AS candidate_entity_ids
            FROM my_s1 s
            LEFT JOIN agg a ON a.source1_entity_id = s.source1_entity_id
        ) TO '{chunk_file}' (HEADER false, DELIMITER '\t')
    """)

    # Append to final file (skip any header DuckDB might add — we set HEADER false)
    with open(chunk_file, "r") as fin:
        # Read first line — if it's our data (starts with "S1-"), include it
        # if it's a header line, skip it. Since we set HEADER false, no header.
        pass
    with open(chunk_file, "rb") as fin, open(OUTPUT_FILE, "ab") as fout:
        while True:
            block = fin.read(1024 * 1024)
            if not block:
                break
            fout.write(block)

    os.remove(chunk_file)
    print(f"  chunk {chunk_id+1}/{n_chunks} done", flush=True)

# ============================================================
# Verify
# ============================================================
print(f"\n[5/5] Verifying {OUTPUT_FILE}...", flush=True)

row_count = con.execute(
    f"SELECT COUNT(*) FROM read_csv('{OUTPUT_FILE}', delim='\t', header=true, quote='')"
).fetchone()[0]

print(f"  Rows written: {row_count:,}", flush=True)
print(f"  Expected:     {n_s1:,}", flush=True)
print(f"  Match: {row_count == n_s1}", flush=True)

con.close()
print("\nDONE", flush=True)