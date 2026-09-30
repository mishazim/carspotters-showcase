"""
Rarity-lookup match key.

`canonical_label()` (normalize_labels.py) unifies make spelling, separators, casing and the
model year, but deliberately keeps body-style suffixes (Sedan / Coupe / SUV / ...) because the
training classes need them. For *rarity lookup* we want the opposite: collapse those body-style
and drivetrain variants so a CV prediction like "BMW X5 Suv" or "Audi A4 Avant Wagon" lines up
with the plain catalog model ("BMW X5", "Audi A4").

`match_key()` is the single source of truth for that collapse. It is applied identically to:
  - every catalog cv_label at population time  (phase2/align_cv_labels.py -> cv_label_canon)
  - the incoming CV prediction at request time (phase3/rarity.py)

so both sides reduce to the same lowercase string. It does NOT attempt the trim<->series
granularity rollup (e.g. NHTSA "328i" -> "BMW 3 Series"); that is a deferred, per-make effort.
"""
import re

from normalize_labels import canonical_label, split_make

# Trailing body-style / body-configuration / drivetrain tokens to strip. Multi-word entries are
# checked before single words. Order within a length group does not matter — stripping loops until
# stable, so "A4 Avant Wagon" peels "wagon" then "avant".
_STRIP_MULTIWORD = [
    "crew cab", "extended cab", "regular cab", "quad cab", "king cab", "double cab",
    "mega cab", "super cab", "access cab", "club cab",
]
_STRIP_SINGLE = [
    # body styles
    "convertible", "coupe", "sedan", "wagon", "hatchback", "cabriolet", "roadster",
    "spyder", "spider", "van", "minivan", "suv", "fastback", "liftback", "estate",
    "touring", "avant", "supercrew", "supercab", "crewcab",
    # drivetrain / layout that NHTSA usually omits from the base model name
    "quattro", "xdrive", "4matic", "awd", "4wd", "fwd", "rwd", "2wd", "4x4",
]


def _strip_suffixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for tok in _STRIP_MULTIWORD:
            if key.endswith(" " + tok):
                key = key[: -len(tok) - 1].strip()
                changed = True
        for tok in _STRIP_SINGLE:
            if key.endswith(" " + tok):
                key = key[: -len(tok) - 1].strip()
                changed = True
    return key


# ---------------------------------------------------------------------------
# Per-make trim<->series rollup.
#
# Some makes are stored at a different granularity than the CV model predicts.
# The rollup collapses both sides to one shared key. It is applied identically
# to the catalog cv_label and the incoming prediction, so a synthesized key
# (e.g. "3 series") need not literally exist in the catalog — the trim row that
# carries the rarity tier reduces to the same key. Only makes with a regular,
# catalog-backed pattern are handled here; messy/pre-2000 makes are left alone.
# ---------------------------------------------------------------------------

_LEXUS_SERIES = {"ct", "es", "gs", "gx", "hs", "is", "lc", "lfa", "ls",
                 "lx", "nx", "rc", "rz", "rx", "sc", "tx", "ux"}

# Mercedes body classes, longest first so "cls" wins over "cl"/"c".
_MB_CLASSES = ["cls", "clk", "cla", "glc", "gla", "glb", "gle", "glk", "gls",
               "slc", "slk", "sls", "cl", "gl", "sl", "ml", "eq",
               "a", "b", "c", "e", "g", "m", "r", "s"]


def _rollup(make: str | None, model: str) -> str:
    """Collapse a trim/spacing variant to the catalog's model granularity."""
    m = model
    if not m:
        return m

    if make == "Lexus":
        mm = re.match(r"^([a-z]{2})\s*\d{3}h?$", m)  # es350 / es 350 / rx400h
        if mm and mm.group(1) in _LEXUS_SERIES:
            return mm.group(1)

    elif make == "BMW":
        mm = re.match(r"^(\d)\d{2}", m)              # 328i / 325e / 550i -> N series
        if mm:
            return f"{mm.group(1)} series"
        mm = re.match(r"^(\d)\s*series", m)          # "3 series" / "3series" -> N series
        if mm:
            return f"{mm.group(1)} series"

    elif make == "Mercedes-Benz":
        mm = re.match(r"^(\d{3})d$", m)              # 190d / 300d -> 190 / 300 (bare numeric)
        if mm:
            return mm.group(1)
        mm = re.match(r"^([a-z]{1,3})\d+[a-z]?$", m)  # c300 / ml350 / gla45 -> "<class> class"
        if mm:
            letters = mm.group(1)
            for cls in _MB_CLASSES:
                if letters.startswith(cls):
                    return f"{cls} class"

    elif make == "Mazda":
        m = re.sub(r"\bmiata\b", "", m).strip()      # drop marketing suffix
        m = re.sub(r"^mazda(\d)", r"\1", m)          # mazda3 -> 3
        m = re.sub(r"^(cx|mx|rx)\s*(\d)", r"\1 \2", m)  # cx5 -> cx 5
        m = re.sub(r"^b\d{4}$", "b series", m)       # b2200 -> b series

    return m


def match_key(label: str | None) -> str | None:
    """
    Collapse a "Make Model ..." label to a lowercase rarity-lookup key.

    Returns None only for empty/None input. Never strips the whole model to nothing: if the model
    portion is entirely body-style tokens, the un-stripped canonical form is kept so we still have
    a key to match on.
    """
    canon = canonical_label(label, drop_year=True)
    if not canon:
        return None
    key = canon.lower()
    # "3series" / "5 series" spacing seen in VMMRdb-derived labels.
    key = re.sub(r"(\d)series\b", r"\1 series", key)
    stripped = _strip_suffixes(key)
    # Guard: don't let a model made entirely of body-style words collapse to just the make.
    first_tok = key.split(" ", 1)[0]
    if stripped == first_tok and key != first_tok:
        stripped = key

    # Per-make trim<->series rollup (Lexus / BMW / Mercedes / Mazda). Applied to both sides.
    make_canon, remainder = split_make(stripped)
    if make_canon is not None:
        rolled = _rollup(make_canon, remainder)
        return f"{make_canon.lower()} {rolled}".strip()
    return stripped


if __name__ == "__main__":
    cases = [
        ("BMW X5 Suv", "bmw x5"),
        ("Audi A4 Avant Wagon", "audi a4"),
        ("Acura Tsx Sedan", "acura tsx"),
        ("Mercedes-Benz C Class Sedan", "mercedes-benz c class"),
        ("BMW 3series", "bmw 3 series"),
        ("Ford F 150 Crew Cab", "ford f 150"),
        ("Audi A6 Quattro", "audi a6"),
        ("Chevrolet Corvette", "chevrolet corvette"),   # no suffix -> unchanged
        ("Tesla Model S 2015", "tesla model s"),
        # --- rollup cases ---
        ("Lexus Es350", "lexus es"),
        ("Lexus Rx400h", "lexus rx"),
        ("Lexus ES", "lexus es"),                        # catalog side, unchanged
        ("BMW 328i", "bmw 3 series"),                    # catalog trim -> series
        ("BMW 3 Series Sedan", "bmw 3 series"),          # CV coarse -> series
        ("BMW 530xi Wagon", "bmw 5 series"),
        ("Mercedes-Benz C300", "mercedes-benz c class"),
        ("Mercedes-Benz Ml350", "mercedes-benz ml class"),
        ("Mercedes-Benz Cls550", "mercedes-benz cls class"),
        ("Mercedes-Benz C-Class", "mercedes-benz c class"),  # catalog side, unchanged
        ("Mazda 3 Hatchback", "mazda 3"),
        ("Mazda Mazda3", "mazda 3"),                     # catalog side
        ("Mazda Cx5", "mazda cx 5"),
        ("Mazda B2200", "mazda b series"),
        ("Mazda Mx5 Miata", "mazda mx 5"),
    ]
    ok = True
    for raw, want in cases:
        got = match_key(raw)
        flag = "OK " if got == want else "XX "
        if got != want:
            ok = False
        print(f"{flag}{raw:35s} -> {got!r:30s} (want {want!r})")
        assert match_key(got) == got, f"not idempotent: {raw} -> {got} -> {match_key(got)}"
    print("\nAll cases passed." if ok else "\nSOME CASES FAILED.")
