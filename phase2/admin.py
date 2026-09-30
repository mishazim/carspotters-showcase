"""
CarSpotters catalog admin CLI.

Commands:
    python admin.py stats                        # DB summary
    python admin.py list --make Toyota           # list catalog entries
    python admin.py set-tier --make Ferrari --model "488" --tier orange
    python admin.py set-count --make Ferrari --model "488" --year 2016 --count 3000
    python admin.py unknown                      # show all 'unknown' tier entries
"""
import argparse
import os
import sys

import psycopg2
import psycopg2.extras

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)

TIER_COLOURS = {
    "grey": "\033[90m", "green": "\033[32m", "blue": "\033[34m",
    "purple": "\033[35m", "orange": "\033[33m", "mythic": "\033[93m",
    "unknown": "\033[31m",
}
RESET = "\033[0m"

VALID_TIERS = {"grey", "green", "blue", "purple", "orange", "mythic", "unknown"}


def coloured(tier: str) -> str:
    return f"{TIER_COLOURS.get(tier, '')}{tier}{RESET}"


def connect():
    try:
        return psycopg2.connect(DB_DSN)
    except Exception as e:
        print(f"Cannot connect to database: {e}")
        print(f"Is the Docker container running?  (cd phase2 && docker compose up -d)")
        sys.exit(1)


def cmd_stats(args):
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM makes")
    n_makes = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM models")
    n_models = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM car_catalog")
    n_catalog = cur.fetchone()[0]

    print(f"\nDatabase summary")
    print(f"  Makes   : {n_makes:,}")
    print(f"  Models  : {n_models:,}")
    print(f"  Catalog : {n_catalog:,} entries")

    cur.execute("""
        SELECT rarity_tier, COUNT(*) AS n
        FROM car_catalog
        GROUP BY rarity_tier
        ORDER BY n DESC
    """)
    print(f"\nTier breakdown:")
    for tier, count in cur.fetchall():
        print(f"  {coloured(tier):30s} {count:>8,}")

    cur.close()
    conn.close()


def cmd_list(args):
    conn = connect()
    cur = conn.cursor()
    query = """
        SELECT mk.name, md.name, cc.year, cc.rarity_tier, cc.production_count
        FROM car_catalog cc
        JOIN makes  mk ON mk.id = cc.make_id
        JOIN models md ON md.id = cc.model_id
        WHERE 1=1
    """
    params = []
    if args.make:
        query += " AND LOWER(mk.name) = LOWER(%s)"
        params.append(args.make)
    if args.model:
        query += " AND LOWER(md.name) = LOWER(%s)"
        params.append(args.model)
    if args.tier:
        query += " AND cc.rarity_tier = %s"
        params.append(args.tier)
    if args.year:
        query += " AND cc.year = %s"
        params.append(args.year)
    query += " ORDER BY mk.name, md.name, cc.year LIMIT %s"
    params.append(args.limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    print(f"\n{'Make':<25} {'Model':<30} {'Year':<6} {'Tier':<12} {'Count'}")
    print("-" * 85)
    for make, model, year, tier, count in rows:
        count_str = f"{count:,}" if count else "—"
        print(f"{make:<25} {model:<30} {year:<6} {coloured(tier):<20} {count_str}")
    print(f"\n{len(rows)} rows shown (limit {args.limit})")
    cur.close()
    conn.close()


def cmd_set_tier(args):
    if args.tier not in VALID_TIERS:
        print(f"Invalid tier '{args.tier}'. Valid: {', '.join(sorted(VALID_TIERS))}")
        sys.exit(1)
    conn = connect()
    cur = conn.cursor()
    query = """
        UPDATE car_catalog cc
        SET rarity_tier = %s, rarity_source = 'manual'
        FROM makes mk, models md
        WHERE cc.make_id = mk.id AND cc.model_id = md.id
          AND LOWER(mk.name) = LOWER(%s)
          AND LOWER(md.name) = LOWER(%s)
    """
    params = [args.tier, args.make, args.model]
    if args.year:
        query += " AND cc.year = %s"
        params.append(args.year)
    cur.execute(query, params)
    n = cur.rowcount
    conn.commit()
    print(f"Updated {n} row(s) -> {coloured(args.tier)}")
    cur.close()
    conn.close()


def cmd_set_count(args):
    from tiers import tier_from_count
    tier = tier_from_count(args.count)
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE car_catalog cc
        SET production_count = %s, rarity_tier = %s, rarity_source = 'manual'
        FROM makes mk, models md
        WHERE cc.make_id = mk.id AND cc.model_id = md.id
          AND LOWER(mk.name) = LOWER(%s)
          AND LOWER(md.name) = LOWER(%s)
          AND cc.year = %s
    """, (args.count, tier, args.make, args.model, args.year))
    n = cur.rowcount
    conn.commit()
    print(f"Set production_count={args.count:,} -> tier {coloured(tier)} ({n} row(s))")
    cur.close()
    conn.close()


def cmd_unknown(args):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT mk.name, COUNT(DISTINCT md.id) AS models
        FROM car_catalog cc
        JOIN makes  mk ON mk.id = cc.make_id
        JOIN models md ON md.id = cc.model_id
        WHERE cc.rarity_tier = 'unknown'
        GROUP BY mk.name
        ORDER BY models DESC
        LIMIT 30
    """)
    rows = cur.fetchall()
    print(f"\nMakes with 'unknown' tier entries:")
    for make, n_models in rows:
        print(f"  {make:<30} {n_models} models")
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(prog="admin")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats")

    p_list = sub.add_parser("list")
    p_list.add_argument("--make");  p_list.add_argument("--model")
    p_list.add_argument("--tier");  p_list.add_argument("--year", type=int)
    p_list.add_argument("--limit", type=int, default=50)

    p_tier = sub.add_parser("set-tier")
    p_tier.add_argument("--make", required=True); p_tier.add_argument("--model", required=True)
    p_tier.add_argument("--tier", required=True); p_tier.add_argument("--year", type=int)

    p_count = sub.add_parser("set-count")
    p_count.add_argument("--make", required=True);  p_count.add_argument("--model", required=True)
    p_count.add_argument("--year", type=int, required=True)
    p_count.add_argument("--count", type=int, required=True)

    sub.add_parser("unknown")

    args = parser.parse_args()
    {"stats": cmd_stats, "list": cmd_list, "set-tier": cmd_set_tier,
     "set-count": cmd_set_count, "unknown": cmd_unknown}[args.command](args)


if __name__ == "__main__":
    main()
