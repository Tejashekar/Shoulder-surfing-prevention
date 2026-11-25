"""Create schema-only result CSVs. This writes headers, never sample rows."""
import csv
import os
from experiments.schemas import SCHEMAS

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(BASE, exist_ok=True)
for name, fields in SCHEMAS.items():
    path = os.path.join(BASE, name)
    rewrite = not os.path.exists(path) or os.path.getsize(path) == 0
    if not rewrite:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, [])
            has_records = next(reader, None) is not None
        if existing != fields and not has_records:
            rewrite = True
        elif existing != fields:
            print(f"Kept existing {name}; its schema differs and it contains rows. Migrate it manually before appending.")
    if rewrite:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)
print(f"Ensured result CSV headers in {BASE}; no measurements were created.")
