"""
Cross-dataset label canonicalization.

Different sources spell the same vehicle differently:
    VMMRdb folder : "mercedes_benz_e_class_2010"   (lowercase, underscores, trailing year)
    Stanford Cars : "Mercedes-Benz E-Class Sedan 2010"  (Title Case, hyphens, body-style suffix)
    Phase 2 catalog cv_label : "Mercedes-Benz E Class"

`canonical_label()` maps any of these to one deterministic, idempotent "Make Model" string so the
same physical car collapses to a single training class regardless of which dataset it came from.

Notes:
- Body-style suffixes (Sedan / Coupe / SUV / ...) are intentionally NOT stripped here — that stays
  the job of phase1/evaluate_merged.py + phase1/inference.py, which merge body-style variants at
  eval/inference time. This module only unifies make spelling, separators, casing and the year.
- Make identification uses longest-prefix matching against a known-makes list (so multi-word makes
  like "Alfa Romeo" / "Land Rover" split correctly instead of a naive first-token split).
"""
import re

# Canonical make spellings. Mirrors phase2/ingest.py MAKES_ALLOWLIST plus common historic/import
# makes that appear in VMMRdb but not the Phase 2 catalog (kept so we don't discard training data;
# they simply won't resolve a rarity tier until added to the catalog).
CANONICAL_MAKES = [
    # Mass market US / Japan / Korea
    "Acura", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ford", "GMC", "Honda",
    "Hyundai", "Infiniti", "Jeep", "Kia", "Lexus", "Lincoln", "Mazda", "Mercury", "Mitsubishi",
    "Nissan", "Pontiac", "Ram", "Saturn", "Scion", "Subaru", "Suzuki", "Toyota",
    # European
    "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Fiat", "Jaguar", "Land Rover",
    "Lotus", "Maserati", "Mercedes-Benz", "MINI", "Porsche", "Rolls-Royce", "Saab",
    "Volkswagen", "Volvo",
    # Exotic / ultra-rare
    "Bugatti", "Ferrari", "Hennessey", "Koenigsegg", "Lamborghini", "McLaren", "Pagani",
    "Rimac", "Spyker",
    # EV newcomers
    "Rivian", "Lucid", "Tesla",
    # Historic / import makes common in VMMRdb (no catalog tier yet)
    "AM General", "Daewoo", "Eagle", "Geo", "Hummer", "Isuzu", "Oldsmobile", "Plymouth",
    "Panoz", "Smart", "Genesis", "Polestar",
]

# Alias -> canonical. Keys are matched after lowercasing + separator normalization (see _norm_sep).
MAKE_ALIASES = {
    "mercedes benz": "Mercedes-Benz",
    "mercedes": "Mercedes-Benz",
    "mercedesbenz": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
    "chevy": "Chevrolet",
    "vw": "Volkswagen",
    "volkswagon": "Volkswagen",
    "landrover": "Land Rover",
    "range rover": "Land Rover",   # Range Rover is a Land Rover model line
    "rangerover": "Land Rover",
    "alfaromeo": "Alfa Romeo",
    "astonmartin": "Aston Martin",
    "rolls royce": "Rolls-Royce",
    "rollsroyce": "Rolls-Royce",
    "mini cooper": "MINI",
    "mercedes-benz": "Mercedes-Benz",
    "am general": "AM General",
    "amgeneral": "AM General",
    "mini": "MINI",
    "bmw": "BMW",
    "gmc": "GMC",
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _norm_sep(s: str) -> str:
    """Lowercase; underscores/hyphens -> spaces; collapse whitespace."""
    s = s.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


# Build the lookup of make-alias -> canonical, longest first so "alfa romeo" beats "alfa".
_MAKE_LOOKUP: list[tuple[str, str]] = []
for _canon in CANONICAL_MAKES:
    _MAKE_LOOKUP.append((_norm_sep(_canon), _canon))
for _alias, _canon in MAKE_ALIASES.items():
    _MAKE_LOOKUP.append((_norm_sep(_alias), _canon))
_MAKE_LOOKUP.sort(key=lambda kv: len(kv[0]), reverse=True)

# Model tokens to force upper-case for readability (matching correctness does not depend on these,
# since normalization is applied identically to every source).
_FORCE_UPPER = {"gt", "gts", "gti", "gtr", "rs", "rsx", "sv", "srt", "ss", "amg", "svr", "sti"}


def _title_model(model: str) -> str:
    out = []
    for tok in model.split():
        if tok in _FORCE_UPPER:
            out.append(tok.upper())
        else:
            out.append(tok.capitalize())  # single chars like "r" -> "R"
    return " ".join(out)


def split_make(raw: str) -> tuple[str | None, str]:
    """Return (canonical_make, remainder). canonical_make is None if no known make matches."""
    norm = _norm_sep(raw)
    for alias_norm, canon in _MAKE_LOOKUP:
        if norm == alias_norm or norm.startswith(alias_norm + " "):
            remainder = norm[len(alias_norm):].strip()
            return canon, remainder
    return None, norm


def canonical_label(raw: str, *, drop_year: bool = True, require_known_make: bool = False) -> str | None:
    """
    Canonicalize a raw label to "Make Model".

    drop_year          : strip a trailing 4-digit model year (default True).
    require_known_make : if True, return None when the make is unrecognized; if False, fall back
                         to treating the first token as the make (keeps otherwise-usable data).
    """
    if raw is None:
        return None
    s = _norm_sep(raw)
    if drop_year:
        s = _YEAR_RE.sub("", s).strip()
        s = re.sub(r"\s+", " ", s)
    if not s:
        return None

    make, remainder = split_make(s)
    if make is None:
        if require_known_make:
            return None
        # Fallback: first token is the make.
        parts = s.split(" ", 1)
        make = parts[0].capitalize()
        remainder = parts[1] if len(parts) > 1 else ""

    model = _title_model(remainder)
    return f"{make} {model}".strip()


if __name__ == "__main__":
    # Idempotency + cross-source collision self-test.
    cases = [
        "honda_accord_2003",
        "Honda Accord",
        "mercedes_benz_e_class_2010",
        "Mercedes-Benz E-Class Sedan 2010",
        "alfa_romeo_147_2005",
        "land_rover_range_rover_2015",
        "range_rover_2015",
        "bmw_3_series_2007",
        "BMW 3 Series Sedan 2012",
        "acura_integra_type_r_2001",
        "Acura Integra Type R 2001",
        "tesla_model_s_2015",
        "chevy_corvette_2019",
        "mazda_cx_5_2018",
        "vw_gti_2016",
        "ford_mustang_gt_2020",
    ]
    print(f"{'RAW':40s} -> CANONICAL")
    for c in cases:
        out = canonical_label(c)
        # idempotency check
        assert canonical_label(out) == out, f"NOT idempotent: {c} -> {out} -> {canonical_label(out)}"
        print(f"{c:40s} -> {out}")
