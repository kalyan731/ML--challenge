#!/usr/bin/env python3
"""
STEP 5 - PASS 4 with pre-split base (no re-sorting)
Runtime: ~25 min
"""

import os
import shutil
import time
import duckdb
from rapidfuzz import fuzz

MEMORY_LIMIT = "5GB"
THREADS = 2
TEMP_DIR = "./duckdb_step5_tmp"
MAX_TEMP_GB = 20
CHUNK_ROWS = 5_000_000

BASE_PARQUET = "step5_base.parquet"
NAME_W_PARQUET = "step5_name_w.parquet"
ADDR_W_PARQUET = "step5_addr_w.parquet"
PARTS_DIR = "step5_base_parts"
OUTPUT_FILE = "step5_test_features.csv"
OUTPUT_CHUNK_DIR = "step5_output_chunks"
EXPECTED_ROWS = 94_237_965

os.makedirs(TEMP_DIR, exist_ok=True)
if os.path.exists(OUTPUT_CHUNK_DIR):
    shutil.rmtree(OUTPUT_CHUNK_DIR)
os.makedirs(OUTPUT_CHUNK_DIR, exist_ok=True)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

start = time.time()
print("=" * 70, flush=True)
print("STEP 5 - PASS 4 with pre-split base", flush=True)
print("=" * 70, flush=True)

for f in (BASE_PARQUET, NAME_W_PARQUET, ADDR_W_PARQUET):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing: {f}")

con = duckdb.connect()
con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
con.execute(f"SET threads={THREADS}")
con.execute(f"SET temp_directory='{TEMP_DIR}'")
con.execute("SET preserve_insertion_order=false")
con.execute(f"PRAGMA max_temp_directory_size='{MAX_TEMP_GB}GB'")

base_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{BASE_PARQUET}')").fetchone()[0]
num_chunks = (base_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
print(f"Base rows: {base_rows:,}  →  {num_chunks} chunks", flush=True)


# ============================================================
# STEP X - Pre-split base into physical files (ONCE)
# ============================================================
print(f"\n[X] Splitting base into {num_chunks} physical files...", flush=True)
t0 = time.time()

if os.path.exists(PARTS_DIR):
    shutil.rmtree(PARTS_DIR)
os.makedirs(PARTS_DIR, exist_ok=True)

# Use file_row_number to avoid ORDER BY, works because candidate_index
# was written in order when pass 1 built the base parquet.
for i in range(num_chunks):
    offset = i * CHUNK_ROWS
    part_file = f"{PARTS_DIR}/part_{i:03d}.parquet"
    t1 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{BASE_PARQUET}')
            LIMIT {CHUNK_ROWS} OFFSET {offset}
        ) TO '{part_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"    split {i+1}/{num_chunks}: {time.time()-t1:.1f}s", flush=True)

print(f"  Split done in {time.time()-t0:.1f}s", flush=True)


# ============================================================
# UDFs
# ============================================================
def _fuzz(a, b):
    if not a or not b:
        return 0.0
    a, b = str(a), str(b)
    if a == b:
        return 1.0
    return fuzz.ratio(a, b) / 100.0


def _digits(s):
    if not s:
        return ""
    return "".join(c for c in str(s) if c.isdigit())


def _dsim(a, b):
    da = _digits(a)
    db = _digits(b)
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    return fuzz.ratio(da, db) / 100.0


con.create_function("fuzz_ratio_norm", _fuzz, ["VARCHAR", "VARCHAR"], "DOUBLE")
con.create_function("digit_sim_norm", _dsim, ["VARCHAR", "VARCHAR"], "DOUBLE")


# ============================================================
# STEP C - Features per partition (reads only its own file)
# ============================================================
print(f"\n[C] Generating 21 features for {num_chunks} partitions...", flush=True)

chunk_csvs = []
for i in range(num_chunks):
    part_file = f"{PARTS_DIR}/part_{i:03d}.parquet"
    out_csv = f"{OUTPUT_CHUNK_DIR}/part_{i:03d}.csv"
    chunk_csvs.append(out_csv)

    t0 = time.time()
    con.execute(f"""
        COPY (
            WITH chunk AS (
                SELECT
                    b.candidate_index, b.source1_entity_id, b.candidate_entity_id,
                    b.candidate_source, b.s1_name, b.s1_addr, b.c_name, b.c_addr,
                    nw.total_a AS nw_a, nw.total_b AS nw_b, nw.shared_w AS nw_s,
                    aw.total_a AS aw_a, aw.total_b AS aw_b, aw.shared_w AS aw_s
                FROM read_parquet('{part_file}') b
                LEFT JOIN read_parquet('{NAME_W_PARQUET}') nw USING (candidate_index)
                LEFT JOIN read_parquet('{ADDR_W_PARQUET}') aw USING (candidate_index)
            ),
            tok AS (
                SELECT *,
                    list_filter(string_split(s1_name, ' '), x -> length(x) >= 2) AS ta_n,
                    list_filter(string_split(c_name,  ' '), x -> length(x) >= 2) AS tb_n,
                    list_filter(string_split(s1_addr, ' '), x -> length(x) >= 2) AS ta_a,
                    list_filter(string_split(c_addr,  ' '), x -> length(x) >= 2) AS tb_a
                FROM chunk
            )
            SELECT
                t.candidate_index, t.source1_entity_id, t.candidate_entity_id, t.candidate_source,

                CAST(CASE WHEN t.s1_name<>'' AND t.s1_name=t.c_name THEN 1.0 ELSE 0.0 END AS FLOAT) AS feature_0_name_exact,
                CAST(fuzz_ratio_norm(t.s1_name, t.c_name) AS FLOAT) AS feature_1_name_char_sim,

                CAST(CASE WHEN len(t.ta_n)=0 AND len(t.tb_n)=0 THEN 1.0
                          WHEN len(t.ta_n)=0 OR  len(t.tb_n)=0 THEN 0.0
                          ELSE len(list_intersect(t.ta_n,t.tb_n))::DOUBLE
                               / len(list_distinct(list_concat(t.ta_n,t.tb_n)))::DOUBLE
                     END AS FLOAT) AS feature_2_name_token_jaccard,

                CAST(CASE WHEN len(t.ta_n)=0 OR len(t.tb_n)=0 THEN 0.0
                          ELSE len(list_intersect(t.ta_n,t.tb_n))::DOUBLE
                               / least(len(t.ta_n),len(t.tb_n))::DOUBLE
                     END AS FLOAT) AS feature_3_name_token_overlap,

                CAST(CASE WHEN length(t.s1_name)=0 AND length(t.c_name)=0 THEN 1.0
                          WHEN length(t.s1_name)=0 OR  length(t.c_name)=0 THEN 0.0
                          ELSE least(length(t.s1_name),length(t.c_name))::DOUBLE
                               / greatest(length(t.s1_name),length(t.c_name))::DOUBLE
                     END AS FLOAT) AS feature_4_name_length_ratio,

                CAST(CASE WHEN t.s1_addr<>'' AND t.s1_addr=t.c_addr THEN 1.0 ELSE 0.0 END AS FLOAT) AS feature_5_addr_exact,
                CAST(fuzz_ratio_norm(t.s1_addr, t.c_addr) AS FLOAT) AS feature_6_addr_char_sim,

                CAST(CASE WHEN len(t.ta_a)=0 AND len(t.tb_a)=0 THEN 1.0
                          WHEN len(t.ta_a)=0 OR  len(t.tb_a)=0 THEN 0.0
                          ELSE len(list_intersect(t.ta_a,t.tb_a))::DOUBLE
                               / len(list_distinct(list_concat(t.ta_a,t.tb_a)))::DOUBLE
                     END AS FLOAT) AS feature_7_addr_token_jaccard,

                CAST(CASE WHEN len(t.ta_a)=0 OR len(t.tb_a)=0 THEN 0.0
                          ELSE len(list_intersect(t.ta_a,t.tb_a))::DOUBLE
                               / least(len(t.ta_a),len(t.tb_a))::DOUBLE
                     END AS FLOAT) AS feature_8_addr_token_overlap,

                CAST(CASE WHEN length(t.s1_addr)=0 AND length(t.c_addr)=0 THEN 1.0
                          WHEN length(t.s1_addr)=0 OR  length(t.c_addr)=0 THEN 0.0
                          ELSE least(length(t.s1_addr),length(t.c_addr))::DOUBLE
                               / greatest(length(t.s1_addr),length(t.c_addr))::DOUBLE
                     END AS FLOAT) AS feature_9_addr_length_ratio,

                CAST(CASE WHEN length(t.s1_name)=0 OR length(t.c_name)=0 THEN 0.0
                          WHEN substr(t.s1_name,1,4)=substr(t.c_name,1,4) THEN 1.0
                          ELSE 0.0 END AS FLOAT) AS feature_10_name_prefix4,

                CAST(CASE WHEN length(t.s1_name)=0 OR length(t.c_name)=0 THEN 0.0
                          WHEN right(t.s1_name,4)=right(t.c_name,4) THEN 1.0
                          ELSE 0.0 END AS FLOAT) AS feature_11_name_suffix4,

                CAST(len(list_intersect(t.ta_n,t.tb_n)) AS FLOAT) AS feature_12_name_shared_tokens,

                CAST(CASE WHEN len(t.ta_n)=0 OR len(t.tb_n)=0 THEN 0.0
                          ELSE coalesce(t.nw_s,0.0)
                               / nullif(least(coalesce(t.nw_a,0.0), coalesce(t.nw_b,0.0)),0.0)
                     END AS FLOAT) AS feature_13_name_weighted_overlap,

                CAST(CASE WHEN length(t.s1_addr)=0 OR length(t.c_addr)=0 THEN 0.0
                          WHEN substr(t.s1_addr,1,4)=substr(t.c_addr,1,4) THEN 1.0
                          ELSE 0.0 END AS FLOAT) AS feature_14_addr_prefix4,

                CAST(CASE WHEN length(t.s1_addr)=0 OR length(t.c_addr)=0 THEN 0.0
                          WHEN right(t.s1_addr,4)=right(t.c_addr,4) THEN 1.0
                          ELSE 0.0 END AS FLOAT) AS feature_15_addr_suffix4,

                CAST(len(list_intersect(t.ta_a,t.tb_a)) AS FLOAT) AS feature_16_addr_shared_tokens,

                CAST(CASE WHEN len(t.ta_a)=0 OR len(t.tb_a)=0 THEN 0.0
                          ELSE coalesce(t.aw_s,0.0)
                               / nullif(least(coalesce(t.aw_a,0.0), coalesce(t.aw_b,0.0)),0.0)
                     END AS FLOAT) AS feature_17_addr_weighted_overlap,

                CAST(CASE WHEN len(regexp_extract_all(t.s1_addr,'\\d+'))=0
                           OR len(regexp_extract_all(t.c_addr,'\\d+'))=0 THEN 0.0
                          ELSE len(list_intersect(
                                regexp_extract_all(t.s1_addr,'\\d+'),
                                regexp_extract_all(t.c_addr,'\\d+')))::DOUBLE
                               / least(
                                len(regexp_extract_all(t.s1_addr,'\\d+')),
                                len(regexp_extract_all(t.c_addr,'\\d+')))::DOUBLE
                     END AS FLOAT) AS feature_18_number_overlap,

                CAST(CASE WHEN regexp_extract(t.s1_addr,'\\d+',0)=''
                           OR regexp_extract(t.c_addr,'\\d+',0)='' THEN 0.0
                          WHEN regexp_extract(t.s1_addr,'\\d+',0)
                             = regexp_extract(t.c_addr,'\\d+',0) THEN 1.0
                          ELSE 0.0 END AS FLOAT) AS feature_19_first_number_match,

                CAST(digit_sim_norm(t.s1_addr, t.c_addr) AS FLOAT) AS feature_20_digit_similarity
            FROM tok t
        ) TO '{out_csv}' (HEADER, DELIMITER ',')
    """)

    print(f"  chunk {i+1}/{num_chunks}: {time.time()-t0:.1f}s", flush=True)


# ============================================================
# CONCATENATE
# ============================================================
print("\nConcatenating into final CSV...", flush=True)
t0 = time.time()

with open(OUTPUT_FILE, "wb") as out:
    for i, ch in enumerate(chunk_csvs):
        with open(ch, "rb") as inp:
            data = inp.read()
            if i == 0:
                out.write(data)
            else:
                idx = data.find(b"\n")
                out.write(data[idx + 1:])

size_gb = os.path.getsize(OUTPUT_FILE) / 1e9
print(f"  Final CSV: {size_gb:.2f} GB  ({time.time()-t0:.1f}s)", flush=True)


# ============================================================
# VERIFY
# ============================================================
print("\nVerifying...", flush=True)
row_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{OUTPUT_FILE}', header=true)").fetchone()[0]
col_count = con.execute(f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM read_csv('{OUTPUT_FILE}', header=true))").fetchone()[0]

print(f"  CSV rows    : {row_count:,}", flush=True)
print(f"  CSV columns : {col_count}", flush=True)
print(f"  Expected    : {EXPECTED_ROWS:,}", flush=True)
if row_count == EXPECTED_ROWS:
    print("  Row count MATCHES.", flush=True)
else:
    print("  WARNING: row count mismatch!", flush=True)

# Keep parts for recovery, only delete chunk CSVs
for f in chunk_csvs:
    if os.path.exists(f):
        os.remove(f)
if os.path.isdir(OUTPUT_CHUNK_DIR) and not os.listdir(OUTPUT_CHUNK_DIR):
    os.rmdir(OUTPUT_CHUNK_DIR)

elapsed = time.time() - start
print()
print("=" * 70, flush=True)
print("PASS 4 COMPLETE", flush=True)
print("=" * 70, flush=True)
print(f"Output  : {OUTPUT_FILE}", flush=True)
print(f"Rows    : {row_count:,}", flush=True)
print(f"Elapsed : {elapsed / 60:.2f} minutes", flush=True)
print(f"Keep for recovery: {PARTS_DIR}/", flush=True)
print(f"  rm -rf {PARTS_DIR}  # after verifying", flush=True)
print("=" * 70, flush=True)

con.close()