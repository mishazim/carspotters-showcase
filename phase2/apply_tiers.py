"""
Re-apply make-based rarity tiers to the existing catalog.

Use this after editing MAKE_TIER_DEFAULTS in tiers.py — it re-derives the
rule-based tier for every catalog row whose production_count is still unknown,
WITHOUT re-running ingest.py (which would re-add dropped models and reset
vehicle_type tags). Rows with a manual production_count are left untouched.

Idempotent — safe to re-run.

Usage:
    python apply_tiers.py
    python apply_tiers.py --dry-run
"""
import argparse
import os

import psycopg2

from tiers import tier_from_make

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)


def run(args):
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM makes ORDER BY name")
    makes = cur.fetchall()

    changed = 0
    for make_id, name in makes:
        tier = tier_from_make(name)
        cur.execute(
            """
            UPDATE car_catalog
               SET rarity_tier = %s, rarity_source = 'rule_based'
             WHERE make_id = %s
               AND production_count IS NULL
               AND rarity_tier <> %s
            """,
            (tier, make_id, tier),
        )
        changed += cur.rowcount

    cur.execute(
        "SELECT rarity_tier, count(*) FROM car_catalog GROUP BY rarity_tier ORDER BY 2 DESC"
    )
    dist = cur.fetchall()

    print(f"Catalog rows re-tiered: {changed}")
    print("Tier distribution:")
    for tier, n in dist:
        print(f"    {tier:8} {n:>7,}")

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
