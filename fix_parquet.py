"""
Fix parquet files for cross-version pyarrow compatibility.
Re-saves all parquet files using write_statistics=False and version='2.6'
to avoid the 'Repetition level histogram size mismatch' error.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import glob

print(f"Using pyarrow {pa.__version__}")

parquet_files = []
for pattern in ["data/**/*.parquet", "models/**/*.parquet"]:
    parquet_files.extend(glob.glob(pattern, recursive=True))

print(f"\nFound {len(parquet_files)} parquet files to fix:\n")

for filepath in parquet_files:
    try:
        print(f"  Fixing: {filepath} ... ", end="")
        df = pd.read_parquet(filepath)
        # Re-save with maximum compatibility settings
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            filepath,
            version='2.6',
            write_statistics=False,
            use_dictionary=True,
            compression='snappy',
        )
        print(f"OK ({len(df):,} rows)")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nDone! All parquet files re-saved for compatibility.")
