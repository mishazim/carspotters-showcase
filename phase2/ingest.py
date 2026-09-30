"""
Phase 2 data ingest: pulls make/model data from the free NHTSA vPIC API
and loads it into the car_catalog database.

Steps:
  1. Fetch all makes from NHTSA
  2. Filter to passenger-vehicle makes that appear in our Phase 1 CV model
     (plus a curated set of well-known consumer makes)
  3. For each make, fetch its models
  4. Insert makes, models, and catalog rows (years 2000-2024) with initial
     rarity tiers assigned by tiers.py

Usage:
    python ingest.py
    python ingest.py --years 2010 2020   # restrict year range
    python ingest.py --dry-run           # print counts, don't write to DB
"""
import argparse
import time
import os
import sys
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras
from tqdm import tqdm

from tiers import assign_tier

NHTSA_BASE = "https://vpic.nhtsa.dot.gov/api/vehicles"
DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)

# Curated set of makes to ingest — covers what users are likely to encounter.
# Keeps the initial ingest fast and the catalog focused.
MAKES_ALLOWLIST = {
    # Mass market US/Japan/Korea
    "Acura", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ford",
    "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia", "Lexus", "Lincoln",
    "Mazda", "Mercury", "Mitsubishi", "Nissan", "Pontiac", "Ram", "Saturn",
    "Scion", "Subaru", "Suzuki", "Toyota",
    # European
    "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Fiat",
    "Jaguar", "Land Rover", "Lotus", "Maserati", "Mercedes-Benz", "MINI",
    "Porsche", "Rolls-Royce", "Saab", "Volkswagen", "Volvo",
    # Exotic / ultra-rare
    "Bugatti", "Ferrari", "Hennessey", "Koenigsegg", "Lamborghini",
    "McLaren", "Pagani", "Rimac", "Spyker",
    # EV newcomers
    "Rivian", "Lucid", "Tesla",
}


def get(client: httpx.Client, path: str, retries: int = 3) -> list:
    url = f"{NHTSA_BASE}/{path}"
    for attempt in range(retries):
        try:
            r = client.get(url, timeout=30)
            r.raise_for_status()
            return r.json().get("Results", [])
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] failed: {url} — {e}")
                return []
            time.sleep(1.5 ** attempt)
    return []


def fetch_all_makes(client: httpx.Client) -> list[dict]:
    print("Fetching all makes from NHTSA...")
    results = get(client, "GetAllMakes?format=json")
    # NHTSA returns make names in inconsistent casing (mostly uppercase, e.g.
    # "TOYOTA"). Match the allowlist case-insensitively and store our preferred
    # title-case spelling rather than NHTSA's raw value.
    display_by_upper = {name.upper(): name for name in MAKES_ALLOWLIST}
    makes = [
        {"nhtsa_make_id": r["Make_ID"], "name": display_by_upper[r["Make_Name"].upper()]}
        for r in results
        if r["Make_Name"].upper() in display_by_upper
    ]
    print(f"  {len(results):,} total makes -> {len(makes)} after allowlist filter")
    return makes


def fetch_models_for_make(client: httpx.Client, nhtsa_make_id: int) -> list[str]:
    results = get(client, f"GetModelsForMakeId/{nhtsa_make_id}?format=json")
    time.sleep(0.2)  # polite rate limiting
    return [r["Model_Name"] for r in results if r.get("Model_Name")]


def ingest(args):
    years = list(range(args.years[0], args.years[1] + 1))
    print(f"Year range: {years[0]}–{years[-1]}  ({len(years)} years)")

    with httpx.Client() as client:
        makes = fetch_all_makes(client)

        if args.dry_run:
            for m in makes:
                print(f"  {m['name']}")
            print(f"\nDry run: would ingest {len(makes)} makes × {len(years)} years")
            return

        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = False
        cur = conn.cursor()

        total_catalog = 0

        for make in tqdm(makes, desc="makes"):
            make_name = make["name"]
            nhtsa_id  = make["nhtsa_make_id"]

            # Upsert make
            cur.execute("""
                INSERT INTO makes (name, nhtsa_make_id)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET nhtsa_make_id = EXCLUDED.nhtsa_make_id
                RETURNING id
            """, (make_name, nhtsa_id))
            make_id = cur.fetchone()[0]

            # Fetch models
            model_names = fetch_models_for_make(client, nhtsa_id)
            if not model_names:
                continue

            # Upsert models
            model_ids = {}
            for model_name in model_names:
                cur.execute("""
                    INSERT INTO models (make_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (make_id, name) DO NOTHING
                    RETURNING id
                """, (make_id, model_name))
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "SELECT id FROM models WHERE make_id=%s AND name=%s",
                        (make_id, model_name)
                    )
                    row = cur.fetchone()
                model_ids[model_name] = row[0]

            # Build catalog rows — one per (model, year)
            tier, source = assign_tier(make_name, production_count=None)
            cv_label = make_name  # placeholder; refined when we add year/model detail
            catalog_rows = [
                (make_id, model_ids[mn], yr, tier, source, f"{make_name} {mn}")
                for mn in model_names
                for yr in years
            ]

            psycopg2.extras.execute_values(cur, """
                INSERT INTO car_catalog
                    (make_id, model_id, year, rarity_tier, rarity_source, cv_label)
                VALUES %s
                ON CONFLICT (make_id, model_id, year) DO NOTHING
            """, catalog_rows)

            total_catalog += len(catalog_rows)

        conn.commit()
        cur.close()
        conn.close()

    print(f"\nIngest complete.")
    print(f"  Makes   : {len(makes)}")
    print(f"  Catalog : {total_catalog:,} rows ({total_catalog // len(years)} models × {len(years)} years)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs=2, type=int, default=[2000, 2024],
                        metavar=("FROM", "TO"), help="Year range to ingest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be ingested without writing to DB")
    ingest(parser.parse_args())
