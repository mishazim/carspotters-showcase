"""
Diagnostic: how many of the Phase 1 CV classes resolve to a real catalog
rarity vs. fall back to make-default vs. miss entirely?

This mirrors the exact matching logic in phase3/rarity.py (lowercased exact
cv_label match -> longest-make-prefix split -> make-default), but loads the
lookup sets ONCE instead of opening a connection per class, so it runs in a
blink. Use it to size the cv_label alignment work and to track progress.

Run:
    cd phase2
    ..\.venv\Scripts\python.exe measure_cv_alignment.py

Writes a residual report of unmatched classes to notes\cv_alignment_residual.csv
"""
import csv
import os
import sys

import psycopg2

# Shared canonicalizer (phase1/match_key.py) — same key rarity.py matches on.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
from match_key import match_key  # noqa: E402

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES_TXT = os.path.join(HERE, "..", "data", "splits", "classes.txt")
RESIDUAL_CSV = os.path.join(HERE, "..", "notes", "cv_alignment_residual.csv")


def load_classes() -> list[str]:
    with open(CLASSES_TXT, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def split_make(label: str, makes: list[str]) -> tuple[str | None, str]:
    """Longest make-name first -> 'Land Rover' before 'Land'. Mirrors rarity.py."""
    low = label.lower()
    for m in makes:
        ml = m.lower()
        if low.startswith(ml + " ") or low == ml:
            return m, label[len(m):].strip()
    return None, label


def main() -> None:
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Distinct canonical keys present in the catalog (exact-match set, mirrors rarity.py step 1).
    cur.execute("SELECT DISTINCT cv_label_canon FROM car_catalog WHERE cv_label_canon IS NOT NULL")
    cv_label_set = {r[0] for r in cur.fetchall()}

    # Makes that actually have catalog rows -> eligible for make-default.
    cur.execute(
        """
        SELECT DISTINCT m.name
          FROM makes m JOIN car_catalog c ON c.make_id = m.id
        """
    )
    makes_with_catalog = {r[0] for r in cur.fetchall()}
    # Longest first for the greedy prefix split.
    makes_sorted = sorted(makes_with_catalog, key=len, reverse=True)

    cur.close()
    conn.close()

    classes = load_classes()

    catalog, make_default, none = [], [], []
    for label in classes:
        key = match_key(label)
        if key and key in cv_label_set:
            catalog.append(label)
            continue
        make, model = split_make(label, makes_sorted)
        if make and make in makes_with_catalog:
            make_default.append((label, make, model))
        else:
            none.append((label, make, model))

    total = len(classes)

    def pct(n: int) -> str:
        return f"{n:5d}  ({100*n/total:5.1f}%)"

    print(f"\nCV classes total: {total}\n")
    print(f"  catalog  (exact cv_label hit) : {pct(len(catalog))}")
    print(f"  make_default (make only)      : {pct(len(make_default))}")
    print(f"  none (no make in catalog)     : {pct(len(none))}\n")

    # Which makes drive the make-default bucket?
    from collections import Counter
    md_by_make = Counter(m for _, m, _ in make_default)
    print("Top makes falling back to make-default:")
    for mk, n in md_by_make.most_common(15):
        print(f"    {mk:20s} {n:4d}")

    none_makes = Counter(lbl.split()[0] for lbl, _, _ in none)
    print("\nClasses with NO catalog make (first token):")
    for mk, n in none_makes.most_common(15):
        print(f"    {mk:20s} {n:4d}")

    # Residual report for the manual/alias pass.
    os.makedirs(os.path.dirname(RESIDUAL_CSV), exist_ok=True)
    with open(RESIDUAL_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cv_class", "resolved_as", "split_make", "split_model"])
        for lbl, mk, mdl in make_default:
            w.writerow([lbl, "make_default", mk, mdl])
        for lbl, mk, mdl in none:
            w.writerow([lbl, "none", mk or "", mdl])
    print(f"\nWrote residual report ({len(make_default)+len(none)} rows) -> {RESIDUAL_CSV}")


if __name__ == "__main__":
    main()
