"""
One-off: look at the PleIAs/Portuguese-PD dataset without downloading it.

    pip install huggingface_hub pyarrow
    python scripts/inspect_pleias.py

Prints the parquet files, their sizes, the column names, the number of rows, and a
few metadata fields of the first rows (reads only the file footers + one row group,
not the full 3 GB).
"""
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "PleIAs/Portuguese-PD"

api = HfApi()
info = api.dataset_info(REPO, files_metadata=True)
files = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(".parquet")]
print(f"{len(files)} parquet files, {sum(s for _, s in files) / 1e9:.2f} GB total")
for name, size in files:
    print(f"  {size / 1e6:8.1f} MB  {name}")

fs = HfFileSystem()
for name, _ in files[:2]:
    print(f"\n=== {name}")
    with fs.open(f"datasets/{REPO}/{name}", "rb") as f:
        pf = pq.ParquetFile(f)
        print("rows:", pf.metadata.num_rows, "| row groups:", pf.num_row_groups)
        print("columns:", pf.schema_arrow.names)
        table = pf.read_row_group(0)
        cols = [c for c in table.column_names if c not in ("text", "complete_text", "content")]
        for i in range(min(3, table.num_rows)):
            row = {c: str(table[c][i])[:80] for c in cols}
            text_col = next((c for c in ("text", "complete_text", "content") if c in table.column_names), None)
            if text_col:
                row["<text preview>"] = str(table[text_col][i])[:200].replace("\n", " ")
                row["<text chars>"] = len(str(table[text_col][i]))
            print(f"  row {i}: {row}")
