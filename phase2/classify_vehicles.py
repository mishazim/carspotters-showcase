"""
Post-ingest vehicle-type classification.

NHTSA's vPIC API lumps a make's entire land-vehicle lineup under one make name,
so makes like Suzuki arrive with motorcycles/ATVs/scooters mixed in with cars.
This script:

  1. Ensures the `models.vehicle_type` column exists (car | motorcycle).
  2. Drops generic NHTSA class-bucket placeholders that aren't real models.
  3. Tags each non-car Suzuki model as 'motorcycle' so the app can treat bike
     scans as an "easter egg" rather than a normal catalog find.

Idempotent — safe to re-run after every `ingest.py`.

Usage:
    python classify_vehicles.py
    python classify_vehicles.py --dry-run
"""
import argparse
import os

import psycopg2

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)

# Suzuki passenger cars/SUVs/trucks sold in the US (confirmed via Wikipedia
# "List of Suzuki automobiles" + auto-brochures.com US lineup). Everything else
# under the Suzuki make is a motorcycle / scooter / ATV / UTV.
SUZUKI_CARS = {
    "Aerio", "Equator", "Esteem", "Forenza", "Forsa", "Grand Vitara",
    "Grand Vitara XL-7", "Kizashi", "Reno", "Samurai", "Sidekick",
    "Sidekick Sport", "Swift", "SX4", "Verona", "Vitara", "X-90", "XL7",
}

# Generic NHTSA classification buckets, not real consumer models — delete.
SUZUKI_AMBIGUOUS_DROP = {
    "Business", "Family", "Leisure", "Mini-Leisure", "Touring", "Enduro",
    "Off Road Play", "Moto-Cross", "Low Speed Vehicle",
}


def run(args):
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # 1. Ensure column exists (no-op if schema already has it).
    cur.execute(
        "ALTER TABLE models ADD COLUMN IF NOT EXISTS "
        "vehicle_type VARCHAR(20) NOT NULL DEFAULT 'car'"
    )

    cur.execute("SELECT id FROM makes WHERE name = 'Suzuki'")
    row = cur.fetchone()
    if row is None:
        print("No Suzuki make found — nothing to classify.")
        conn.rollback()
        return
    suzuki_id = row[0]

    # 2. Drop ambiguous placeholder models (catalog rows first — no cascade FK).
    cur.execute(
        "SELECT id, name FROM models WHERE make_id = %s AND name = ANY(%s)",
        (suzuki_id, list(SUZUKI_AMBIGUOUS_DROP)),
    )
    drop = cur.fetchall()
    drop_ids = [r[0] for r in drop]
    if drop_ids:
        cur.execute("DELETE FROM car_catalog WHERE model_id = ANY(%s)", (drop_ids,))
        catalog_deleted = cur.rowcount
        cur.execute("DELETE FROM models WHERE id = ANY(%s)", (drop_ids,))
        models_deleted = cur.rowcount
    else:
        catalog_deleted = models_deleted = 0

    # 3. Tag Suzuki non-car models as motorcycle (cars stay default 'car').
    cur.execute(
        "UPDATE models SET vehicle_type = 'motorcycle' "
        "WHERE make_id = %s AND name <> ALL(%s)",
        (suzuki_id, list(SUZUKI_CARS)),
    )
    tagged_moto = cur.rowcount

    # Report
    cur.execute(
        "SELECT vehicle_type, count(*) FROM models WHERE make_id = %s "
        "GROUP BY vehicle_type ORDER BY 1",
        (suzuki_id,),
    )
    breakdown = cur.fetchall()

    print(f"Dropped ambiguous models : {models_deleted} "
          f"({catalog_deleted} catalog rows)")
    print(f"Tagged as motorcycle     : {tagged_moto}")
    print("Suzuki model breakdown   :")
    for vt, n in breakdown:
        print(f"    {vt:12} {n}")

    if args.dry_run:
        print("\n[dry-run] rolling back.")
        conn.rollback()
    else:
        conn.commit()
        print("\nCommitted.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
