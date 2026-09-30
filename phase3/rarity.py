"""
Map a CV prediction ("Make Model") to a rarity tier using the Phase 2 catalog.

Strategy:
  1. Exact match on car_catalog.cv_label_canon (collapsed rarity-lookup key) -> real catalog hit.
  2. Fall back to the make's modal tier -> model not in catalog but make is.
  3. Otherwise 'unknown'.
"""
import os
import sys

from db import get_conn

# Shared canonicalizer (phase1/match_key.py) — must reduce the incoming prediction the same way
# align_cv_labels.py reduced cv_label_canon, or exact matches silently miss.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
from match_key import match_key  # noqa: E402


def _known_makes(cur) -> list[str]:
    cur.execute("SELECT name FROM makes ORDER BY length(name) DESC")
    return [r[0] for r in cur.fetchall()]


def _split_make_model(label: str, makes: list[str]) -> tuple[str | None, str]:
    low = label.lower()
    for m in makes:  # longest make names first -> "Land Rover" before "Land"
        if low.startswith(m.lower() + " ") or low == m.lower():
            return m, label[len(m):].strip()
    return None, label


def lookup_rarity(label: str) -> dict:
    with get_conn() as conn:
        cur = conn.cursor()

        # 1. Exact match on the collapsed canonical key.
        key = match_key(label)
        row = None
        if key:
            cur.execute(
                """
                SELECT rarity_tier, count(*)
                  FROM car_catalog
                 WHERE cv_label_canon = %s
                 GROUP BY rarity_tier
                 ORDER BY 2 DESC
                 LIMIT 1
                """,
                (key,),
            )
            row = cur.fetchone()
        makes = _known_makes(cur)
        make, model = _split_make_model(label, makes)

        if row:
            return {
                "make": make,
                "model": model or label,
                "rarity_tier": row[0],
                "matched_by": "catalog",
            }

        # 2. Make-level default tier.
        if make:
            cur.execute(
                """
                SELECT c.rarity_tier, count(*)
                  FROM car_catalog c
                  JOIN makes m ON m.id = c.make_id
                 WHERE m.name = %s
                 GROUP BY 1
                 ORDER BY 2 DESC
                 LIMIT 1
                """,
                (make,),
            )
            r = cur.fetchone()
            return {
                "make": make,
                "model": model,
                "rarity_tier": r[0] if r else "unknown",
                "matched_by": "make_default",
            }

        # 3. Nothing.
        return {"make": None, "model": label, "rarity_tier": "unknown", "matched_by": "none"}
