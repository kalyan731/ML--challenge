import pandas as pd
from pathlib import Path

BASE = Path("resource/student_resource/dataset/test")

FILES = {
    "source1": BASE / "test_source1.tsv",
    "source2": BASE / "test_source2.tsv",
    "source3": BASE / "test_source3.tsv",
}

print("=" * 70)
print("STEP 5.1 - TEST DATA VERIFICATION")
print("=" * 70)

for name, path in FILES.items():

    print(f"\nLoading {name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing/empty values:")
    for col in df.columns:
        empty = (df[col].str.strip() == "").sum()
        print(f"  {col}: {empty:,}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)