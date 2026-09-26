#!/usr/bin/env python3
"""
STEP 5 - 4-PASS CHUNKED FEATURE GENERATION
Output: step5_test_features.csv
Rows:   94,237,965
"""

import os
import time
import duckdb
from rapidfuzz import fuzz

CACHE_DIR = "normalized_cache_test"
S1_FILE = f"{CACHE_DIR}/source1_normalized.parquet"
S2_FILE = f"{CACHE_DIR}/source2_normalized.parquet"
S3_FILE = f"{CACHE_DIR}/source3_normalized.parquet"
CANDIDATE_FILE = "step4_test_candidate_pairs.csv"

BASE_PARQUET = "step5_base.parquet"
NAME_W_PARQUET = "step5_name_w.parquet"
ADDR_W_PARQUET = "step5_addr_w.parquet"
OUTPUT_FILE = "step5_test_features.csv"

MEMORY_LIMIT = "5GB"
THREADS = 2
TEMP_DIR = "./duckdb_step5_tmp"
MAX_TEMP_GB = 20
CHUNK_ROWS = 5_000_000
EXPECTED_ROWS = 94_237_965

os.makedirs(TEMP_DIR, exist_ok=True)
start_time = time.time()

print("=" * 70, flush=True)
print("STEP 5 - 4-PASS CHUNKED FEATURE GENERATION", flush=True)
print("=" * 70, flush=True)
print(f"Candidates : {CANDIDATE_FILE}", flush=True)
print(f"Output     : {OUTPUT_FILE}", flush=True)
print(f"Expected   : {EXPECTED_ROWS:,} rows", flush=True)
print(f"Memory     : {MEMORY_LIMIT}", flush=True)
print(f"Threads    : {THREADS}", flush=True)
print(f"Chunk size : {CHUNK_ROWS:,}", flush=True)
print("=" * 70, flush=True)

for path in (S1_FILE, S2_FILE, S3_FILE, CANDIDATE_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

# Remove derived files (KEEP step5_base.parquet if it exists)
for f in (NAME_W_PARQUET, ADDR_W_PARQUET, OUTPUT_FILE):
    if os.path.exists(f):
        print(f"Removing existing {f}", flush=True)
        os.remove(f)

# Remove stale chunk files
for f in os.listdir("."):
    if f.startswith("step5_name_w_chunk_") or f.startswith("step5_addr_w_chunk_"):
        os.remove(f)


def connect():
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    c.execute(f"SET threads={THREADS}")
    c.execute(f"SET temp_directory='{TEMP_DIR}'")
    c.execute("SET preserve_insertion_order=false")
    c.execute(f"PRAGMA max_temp_directory_size='{MAX_TEMP_GB}GB'")
    return c


con = connect()

# ============================================================
# VIEWS
# ============================================================
print("\nCreating views...", flush=True)

con.execute(f"""
    CREATE OR REPLACE VIEW source1 AS
    SELECT CAST(entity_id AS VARCHAR) AS entity_id,
           CAST(name_norm AS VARCHAR) AS name_norm,
           CAST(addr_norm AS VARCHAR) AS addr_norm,
           CAST(country_norm AS VARCHAR) AS country_norm
    FROM read_parquet('{S1_FILE}')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW source2 AS
    SELECT CAST(entity_id AS VARCHAR) AS entity_id,
           CAST(name_norm AS VARCHAR) AS name_norm,
           CAST(addr_norm AS VARCHAR) AS addr_norm,
           CAST(country_norm AS VARCHAR) AS country_norm
    FROM read_parquet('{S2_FILE}')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW source3 AS
    SELECT CAST(entity_id AS VARCHAR) AS entity_id,
           CAST(name_norm AS VARCHAR) AS name_norm,
           CAST(addr_norm AS VARCHAR) AS addr_norm,
           CAST(country_norm AS VARCHAR) AS country_norm
    FROM read_parquet('{S3_FILE}')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW candidates AS
    SELECT CAST(source1_entity_id   AS VARCHAR) AS source1_entity_id,
           CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
           CAST(candidate_source    AS VARCHAR) AS candidate_source,
           (ROW_NUMBER() OVER ()) - 1           AS candidate_index
    FROM read_csv('{CANDIDATE_FILE}', header=true, delim=',',
        columns={{'source1_entity_id':'VARCHAR',
                  'candidate_entity_id':'VARCHAR',
                  'candidate_source':'VARCHAR'}})
""")
con.execute("""
    CREATE OR REPLACE VIEW candidate_lookup AS
    SELECT entity_id, name_norm, addr_norm, 'source2' AS candidate_source FROM source2
    UNION ALL
    SELECT entity_id, name_norm, addr_norm, 'source3' AS candidate_source FROM source3
""")


# ============================================================
# PASS 1 - Base parquet (skip if exists)
# ============================================================
if os.path.exists(BASE_PARQUET):
    print("\n[PASS 1/4] Base parquet already exists, skipping.", flush=True)
    base_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{BASE_PARQUET}')").fetchone()[0]
    print(f"  Base rows: {base_rows:,}", flush=True)
else:
    print("\n[PASS 1/4] Building base parquet...", flush=True)
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT c.candidate_index, c.source1_entity_id, c.candidate_entity_id,
                   c.candidate_source,
                   coalesce(s1.name_norm,'') AS s1_name,
                   coalesce(s1.addr_norm,'') AS s1_addr,
                   coalesce(cand.name_norm,'') AS c_name,
                   coalesce(cand.addr_norm,'') AS c_addr
            FROM candidates c
            LEFT JOIN source1 s1 ON s1.entity_id = c.source1_entity_id
            LEFT JOIN candidate_lookup cand
                ON cand.entity_id = c.candidate_entity_id
               AND cand.candidate_source = c.candidate_source
        ) TO '{BASE_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    base_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{BASE_PARQUET}')").fetchone()[0]
    print(f"  Base rows: {base_rows:,}  ({time.time()-t0:.1f}s)", flush=True)


# ============================================================
# TOKEN WEIGHT TABLES
# ============================================================
print("\nBuilding token weight tables...", flush=True)
t0 = time.time()
con.execute("""
    CREATE OR REPLACE TABLE name_token_weight AS
    WITH freq AS (
        SELECT token, COUNT(*) AS freq FROM (
            SELECT UNNEST(string_split(coalesce(name_norm,''), ' ')) AS token FROM source1
            UNION ALL
            SELECT UNNEST(string_split(coalesce(name_norm,''), ' ')) FROM source2
            UNION ALL
            SELECT UNNEST(string_split(coalesce(name_norm,''), ' ')) FROM source3
        ) WHERE length(token) >= 2 GROUP BY token
    )
    SELECT token, 1.0 / log2(freq + 2.0) AS w FROM freq
""")
con.execute("""
    CREATE OR REPLACE TABLE addr_token_weight AS
    WITH freq AS (
        SELECT token, COUNT(*) AS freq FROM (
            SELECT UNNEST(string_split(coalesce(addr_norm,''), ' ')) AS token FROM source1
            UNION ALL
            SELECT UNNEST(string_split(coalesce(addr_norm,''), ' ')) FROM source2
            UNION ALL
            SELECT UNNEST(string_split(coalesce(addr_norm,''), ' ')) FROM source3
        ) WHERE length(token) >= 2 GROUP BY token
    )
    SELECT token, 1.0 / log2(freq + 2.0) AS w FROM freq
""")
nv = con.execute("SELECT COUNT(*) FROM name_token_weight").fetchone()[0]
av = con.execute("SELECT COUNT(*) FROM addr_token_weight").fetchone()[0]
print(f"  Name vocab: {nv:,}   Addr vocab: {av:,}  ({time.time()-t0:.1f}s)", flush=True)


# ============================================================
# PASS 2 - Name weighted overlap (CHUNKED)
# ============================================================
print("\n[PASS 2/4] Name weighted overlap (chunked)...", flush=True)

num_chunks = (base_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
print(f"  {num_chunks} chunks of {CHUNK_ROWS:,} rows", flush=True)

chunk_files = []
for chunk_no in range(num_chunks):
    offset = chunk_no * CHUNK_ROWS
    chunk_file = f"step5_name_w_chunk_{chunk_no:03d}.parquet"
    chunk_files.append(chunk_file)
    if os.path.exists(chunk_file):
        os.remove(chunk_file)
    t0 = time.time()
    con.execute(f"""
        COPY (
            WITH chunk AS (
                SELECT candidate_index, s1_name, c_name
                FROM read_parquet('{BASE_PARQUET}')
                ORDER BY candidate_index
                LIMIT {CHUNK_ROWS} OFFSET {offset}
            ),
            tok AS (
                SELECT candidate_index,
                       list_filter(string_split(s1_name,' '), x -> length(x) >= 2) AS ta,
                       list_filter(string_split(c_name, ' '), x -> length(x) >= 2) AS tb
                FROM chunk
            ),
            sums AS (
                SELECT candidate_index, role, SUM(w.w) AS total_w
                FROM (
                    SELECT candidate_index, 'a' AS role, UNNEST(ta) AS token FROM tok
                    UNION ALL
                    SELECT candidate_index, 'b' AS role, UNNEST(tb) AS token FROM tok
                ) x
                JOIN name_token_weight w ON w.token = x.token
                GROUP BY candidate_index, role
            ),
            shared AS (
                SELECT t.candidate_index, SUM(w.w) AS shared_w
                FROM tok t, UNNEST(list_intersect(t.ta, t.tb)) AS u(token)
                JOIN name_token_weight w ON w.token = u.token
                GROUP BY t.candidate_index
            )
            SELECT s.candidate_index,
                   MAX(CASE WHEN s.role='a' THEN s.total_w END) AS total_a,
                   MAX(CASE WHEN s.role='b' THEN s.total_w END) AS total_b,
                   coalesce(sh.shared_w, 0.0) AS shared_w
            FROM sums s
            LEFT JOIN shared sh USING (candidate_index)
            GROUP BY s.candidate_index, sh.shared_w
        ) TO '{chunk_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"    chunk {chunk_no+1}/{num_chunks}: {time.time()-t0:.1f}s", flush=True)

print("  Merging name_w chunks...", flush=True)
con.execute(f"""
    COPY (SELECT * FROM read_parquet({chunk_files}))
    TO '{NAME_W_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
for f in chunk_files:
    if os.path.exists(f):
        os.remove(f)
nw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{NAME_W_PARQUET}')").fetchone()[0]
print(f"  name_w rows: {nw_rows:,}  ({time.time()-start_time:.1f}s)", flush=True)


# ============================================================
# PASS 3 - Address weighted overlap (CHUNKED)
# ============================================================
print("\n[PASS 3/4] Address weighted overlap (chunked)...", flush=True)

chunk_files = []
for chunk_no in range(num_chunks):
    offset = chunk_no * CHUNK_ROWS
    chunk_file = f"step5_addr_w_chunk_{chunk_no:03d}.parquet"
    chunk_files.append(chunk_file)
    if os.path.exists(chunk_file):
        os.remove(chunk_file)
    t0 = time.time()
    con.execute(f"""
        COPY (
            WITH chunk AS (
                SELECT candidate_index, s1_addr, c_addr
                FROM read_parquet('{BASE_PARQUET}')
                ORDER BY candidate_index
                LIMIT {CHUNK_ROWS} OFFSET {offset}
            ),
            tok AS (
                SELECT candidate_index,
                       list_filter(string_split(s1_addr,' '), x -> length(x) >= 2) AS ta,
                       list_filter(string_split(c_addr, ' '), x -> length(x) >= 2) AS tb
                FROM chunk
            ),
            sums AS (
                SELECT candidate_index, role, SUM(w.w) AS total_w
                FROM (
                    SELECT candidate_index, 'a' AS role, UNNEST(ta) AS token FROM tok
                    UNION ALL
                    SELECT candidate_index, 'b' AS role, UNNEST(tb) AS token FROM tok
                ) x
                JOIN addr_token_weight w ON w.token = x.token
                GROUP BY candidate_index, role
            ),
            shared AS (
                SELECT t.candidate_index, SUM(w.w) AS shared_w
                FROM tok t, UNNEST(list_intersect(t.ta, t.tb)) AS u(token)
                JOIN addr_token_weight w ON w.token = u.token
                GROUP BY t.candidate_index
            )
            SELECT s.candidate_index,
                   MAX(CASE WHEN s.role='a' THEN s.total_w END) AS total_a,
                   MAX(CASE WHEN s.role='b' THEN s.total_w END) AS total_b,
                   coalesce(sh.shared_w, 0.0) AS shared_w
            FROM sums s
            LEFT JOIN shared sh USING (candidate_index)
            GROUP BY s.candidate_index, sh.shared_w
        ) TO '{chunk_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"    chunk {chunk_no+1}/{num_chunks}: {time.time()-t0:.1f}s", flush=True)

print("  Merging addr_w chunks...", flush=True)
con.execute(f"""
    COPY (SELECT * FROM read_parquet({chunk_files}))
    TO '{ADDR_W_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
for f in chunk_files:
    if os.path.exists(f):
        os.remove(f)
aw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{ADDR_W_PARQUET}')").fetchone()[0]
print(f"  addr_w rows: {aw_rows:,}  ({time.time()-start_time:.1f}s)", flush=True)


# ============================================================
# UDFs
# ============================================================
print("\nRegistering rapidfuzz UDFs...", flush=True)

def _fuzz(a, b):
    if not a or not b: return 0.0
    a, b = str(a), str(b)
    if a == b: return 1.0
    return fuzz.ratio(a, b) / 100.0

def _digits(s):
    return "".join(c for c in str(s) if c.isdigit()) if s else ""

def _dsim(a, b):
    da, db = _digits(a), _digits(b)
    if not da and not db: return 1.0
    if not da or not db: return 0.0
    return fuzz.ratio(da, db) / 100.0

con.create_function("fuzz_ratio_norm", _fuzz, ["VARCHAR","VARCHAR"], "DOUBLE")
con.create_function("digit_sim_norm", _dsim, ["VARCHAR","VARCHAR"], "DOUBLE")


# ============================================================
# PASS 4 - Final features -> CSV
# ============================================================
print("\n[PASS 4/4] Final features -> CSV...", flush=True)
print("(this is the long step, ~15-20 min)", flush=True)

FEATURE_SQL = f"""
WITH tok AS (
    SELECT
        b.candidate_index, b.source1_entity_id, b.candidate_entity_id,
        b.candidate_source, b.s1_name, b.s1_addr, b.c_name, b.c_addr,
        list_filter(string_split(b.s1_name,' '), x -> length(x) >= 2) AS ta_name,
        list_filter(string_split(b.c_name, ' '), x -> length(x) >= 2) AS tb_name,
        list_filter(string_split(b.s1_addr,' '), x -> length(x) >= 2) AS ta_addr,
        list_filter(string_split(b.c_addr, ' '), x -> length(x) >= 2) AS tb_addr,
        nw.total_a AS nw_total_a, nw.total_b AS nw_total_b, nw.shared_w AS nw_shared_w,
        aw.total_a AS aw_total_a, aw.total_b AS aw_total_b, aw.shared_w AS aw_shared_w
    FROM read_parquet('{BASE_PARQUET}') b
    LEFT JOIN read_parquet('{NAME_W_PARQUET}') nw USING (candidate_index)
    LEFT JOIN read_parquet('{ADDR_W_PARQUET}') aw USING (candidate_index)
)
SELECT
    t.candidate_index, t.source1_entity_id, t.candidate_entity_id, t.candidate_source,
    CAST(CASE WHEN t.s1_name<>'' AND t.s1_name=t.c_name THEN 1.0 ELSE 0.0 END AS FLOAT) AS feature_0_name_exact,
    CAST(fuzz_ratio_norm(t.s1_name, t.c_name) AS FLOAT) AS feature_1_name_char_sim,
    CAST(CASE WHEN len(t.ta_name)=0 AND len(t.tb_name)=0 THEN 1.0
              WHEN len(t.ta_name)=0 OR len(t.tb_name)=0 THEN 0.0
              ELSE len(list_intersect(t.ta_name,t.tb_name))::DOUBLE
                   / len(list_distinct(list_concat(t.ta_name,t.tb_name)))::DOUBLE
         END AS FLOAT) AS feature_2_name_token_jaccard,
    CAST(CASE WHEN len(t.ta_name)=0 OR len(t.tb_name)=0 THEN 0.0
              ELSE len(list_intersect(t.ta_name,t.tb_name))::DOUBLE
                   / least(len(t.ta_name),len(t.tb_name))::DOUBLE
         END AS FLOAT) AS feature_3_name_token_overlap,
    CAST(CASE WHEN length(t.s1_name)=0 AND length(t.c_name)=0 THEN 1.0
              WHEN length(t.s1_name)=0 OR length(t.c_name)=0 THEN 0.0
              ELSE least(length(t.s1_name),length(t.c_name))::DOUBLE
                   / greatest(length(t.s1_name),length(t.c_name))::DOUBLE
         END AS FLOAT) AS feature_4_name_length_ratio,
    CAST(CASE WHEN t.s1_addr<>'' AND t.s1_addr=t.c_addr THEN 1.0 ELSE 0.0 END AS FLOAT) AS feature_5_addr_exact,
    CAST(fuzz_ratio_norm(t.s1_addr, t.c_addr) AS FLOAT) AS feature_6_addr_char_sim,
    CAST(CASE WHEN len(t.ta_addr)=0 AND len(t.tb_addr)=0 THEN 1.0
              WHEN len(t.ta_addr)=0 OR len(t.tb_addr)=0 THEN 0.0
              ELSE len(list_intersect(t.ta_addr,t.tb_addr))::DOUBLE
                   / len(list_distinct(list_concat(t.ta_addr,t.tb_addr)))::DOUBLE
         END AS FLOAT) AS feature_7_addr_token_jaccard,
    CAST(CASE WHEN len(t.ta_addr)=0 OR len(t.tb_addr)=0 THEN 0.0
              ELSE len(list_intersect(t.ta_addr,t.tb_addr))::DOUBLE
                   / least(len(t.ta_addr),len(t.tb_addr))::DOUBLE
         END AS FLOAT) AS feature_8_addr_token_overlap,
    CAST(CASE WHEN length(t.s1_addr)=0 AND length(t.c_addr)=0 THEN 1.0
              WHEN length(t.s1_addr)=0 OR length(t.c_addr)=0 THEN 0.0
              ELSE least(length(t.s1_addr),length(t.c_addr))::DOUBLE
                   / greatest(length(t.s1_addr),length(t.c_addr))::DOUBLE
         END AS FLOAT) AS feature_9_addr_length_ratio,
    CAST(CASE WHEN length(t.s1_name)=0 OR length(t.c_name)=0 THEN 0.0
              WHEN substr(t.s1_name,1,4)=substr(t.c_name,1,4) THEN 1.0
              ELSE 0.0 END AS FLOAT) AS feature_10_name_prefix4,
    CAST(CASE WHEN length(t.s1_name)=0 OR length(t.c_name)=0 THEN 0.0
              WHEN right(t.s1_name,4)=right(t.c_name,4) THEN 1.0
              ELSE 0.0 END AS FLOAT) AS feature_11_name_suffix4,
    CAST(len(list_intersect(t.ta_name,t.tb_name)) AS FLOAT) AS feature_12_name_shared_tokens,
    CAST(CASE WHEN len(t.ta_name)=0 OR len(t.tb_name)=0 THEN 0.0
              ELSE coalesce(t.nw_shared_w,0.0)
                   / nullif(least(coalesce(t.nw_total_a,0.0), coalesce(t.nw_total_b,0.0)),0.0)
         END AS FLOAT) AS feature_13_name_weighted_overlap,
    CAST(CASE WHEN length(t.s1_addr)=0 OR length(t.c_addr)=0 THEN 0.0
              WHEN substr(t.s1_addr,1,4)=substr(t.c_addr,1,4) THEN 1.0
              ELSE 0.0 END AS FLOAT) AS feature_14_addr_prefix4,
    CAST(CASE WHEN length(t.s1_addr)=0 OR length(t.c_addr)=0 THEN 0.0
              WHEN right(t.s1_addr,4)=right(t.c_addr,4) THEN 1.0
              ELSE 0.0 END AS FLOAT) AS feature_15_addr_suffix4,
    CAST(len(list_intersect(t.ta_addr,t.tb_addr)) AS FLOAT) AS feature_16_addr_shared_tokens,
    CAST(CASE WHEN len(t.ta_addr)=0 OR len(t.tb_addr)=0 THEN 0.0
              ELSE coalesce(t.aw_shared_w,0.0)
                   / nullif(least(coalesce(t.aw_total_a,0.0), coalesce(t.aw_total_b,0.0)),0.0)
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
"""

con.execute(f"COPY ({FEATURE_SQL}) TO '{OUTPUT_FILE}' (HEADER, DELIMITER ',')")

print("\nVerifying output...", flush=True)
row_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{OUTPUT_FILE}', header=true)").fetchone()[0]
col_count = con.execute(f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM read_csv('{OUTPUT_FILE}', header=true))").fetchone()[0]

print(f"CSV rows    : {row_count:,}", flush=True)
print(f"CSV columns : {col_count}", flush=True)
print(f"Expected    : {EXPECTED_ROWS:,}", flush=True)
if row_count == EXPECTED_ROWS:
    print("Row count MATCHES.", flush=True)
else:
    print("WARNING: row count mismatch!", flush=True)

elapsed = time.time() - start_time
print()
print("=" * 70, flush=True)
print("STEP 5 COMPLETE", flush=True)
print("=" * 70, flush=True)
print(f"Output  : {OUTPUT_FILE}", flush=True)
print(f"Rows    : {row_count:,}", flush=True)
print(f"Elapsed : {elapsed / 60:.2f} minutes", flush=True)
if elapsed > 0:
    print(f"Rate    : {row_count / elapsed:,.0f} rows/sec", flush=True)
print("=" * 70, flush=True)

con.close()