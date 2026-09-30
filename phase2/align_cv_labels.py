"""
Populate car_catalog.cv_label_canon — the collapsed rarity-lookup key.

Idempotent and non-destructive: adds the column if missing, then backfills it from cv_label using
the shared match_key() (phase1/match_key.py). Does NOT touch cv_label, tiers, or anything ingest /
classify_vehicles / apply_tiers own — safe to run repeatedly without a re-ingest.

Run:
    cd phase2
    ..\.venv\Scripts\python.exe align_cv_labels.py
"""
import os
import sys

import psycopg2

# Shared canonicalizer lives in phase1.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
from match_key import match_key  # noqa: E402

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)


def main() -> None:
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # 1. Schema: add the column + index if they don't exist yet.
    cur.execute("ALTER TABLE car_catalog ADD COLUMN IF NOT EXISTS cv_label_canon VARCHAR(300)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_cv_canon ON car_catalog (cv_label_canon)")

    # 2. Compute the key per distinct cv_label (thousands of rows share ~2.4k labels).
    cur.execute("SELECT DISTINCT cv_label FROM car_catalog WHERE cv_label IS NOT NULL")
    labels = [r[0] for r in cur.fetchall()]

    updates = [(match_key(lbl), lbl) for lbl in labels]
    updates = [(k, lbl) for k, lbl in updates if k]  # drop any that canonicalize to nothing

    cur.executemany(
        "UPDATE car_catalog SET cv_label_canon = %s WHERE cv_label = %s",
        updates,
    )

    conn.commit()

    # 3. Report.
    cur.execute("SELECT count(*) FROM car_catalog WHERE cv_label_canon IS NOT NULL")
    filled = cur.fetchone()[0]
    cur.execute("SELECT count(DISTINCT cv_label_canon) FROM car_catalog")
    distinct_keys = cur.fetchone()[0]

    print(f"Distinct cv_labels processed : {len(labels)}")
    print(f"Rows with cv_label_canon set : {filled}")
    print(f"Distinct canonical keys      : {distinct_keys}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
