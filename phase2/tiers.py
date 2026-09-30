"""
Rarity tier assignment logic.

Tier     Colour   Production threshold (annual units)
------   ------   ------------------------------------
grey     Grey     > 500,000
green    Green    50,001 – 500,000
blue     Blue     5,001 – 50,000
purple   Purple   501 – 5,000
orange   Orange   1 – 500
mythic   Gold     ≤ 10  (or 1-of-1 customs / prototypes)
"""

# Makes that default to a known tier when production count is unavailable.
# Keyed by lowercase make name, value is the fallback tier.
MAKE_TIER_DEFAULTS: dict[str, str] = {
    # Mass-market — grey/green
    "toyota": "grey", "ford": "grey", "chevrolet": "grey", "honda": "grey",
    "nissan": "grey", "hyundai": "green", "kia": "green", "jeep": "green",
    "ram": "green", "gmc": "green", "subaru": "green", "volkswagen": "green",
    "chrysler": "green", "dodge": "green", "buick": "green",

    # Premium / near-luxury — blue
    "bmw": "blue", "mercedes-benz": "blue", "audi": "blue", "lexus": "blue",
    "acura": "blue", "infiniti": "blue", "cadillac": "blue", "lincoln": "blue",
    "volvo": "blue", "genesis": "blue", "alfa romeo": "blue",

    # Performance / exotic — purple
    "porsche": "purple", "land rover": "blue", "jaguar": "blue",
    "maserati": "purple", "bentley": "purple", "rolls-royce": "purple",
    "aston martin": "purple", "lotus": "purple",

    # Ultra-exotic — orange
    "ferrari": "orange", "lamborghini": "orange", "mclaren": "orange",
    "pagani": "orange", "koenigsegg": "orange", "rimac": "orange",

    # Mythic
    "bugatti": "mythic", "hennessey": "mythic",

    # Discontinued / niche / EV-newcomer makes. Banded uncommon(green)–rare(blue):
    # green for makes still common on the road, blue for genuinely scarce ones.
    "pontiac": "green", "saturn": "green", "mercury": "green",   # discontinued GM/Ford, large legacy fleets
    "mazda": "green", "mitsubishi": "green",                     # alive, mid-volume
    "mini": "green", "tesla": "green",                           # alive, common (tesla/mazda floored up from grey)
    "suzuki": "blue",                                            # cars left US ~2013, low volume, fleet shrinking
    "saab": "blue", "fiat": "blue",                             # discontinued / tiny US volume, scarce now
    "rivian": "blue", "lucid": "blue",                          # new EVs, still uncommon on the road
    "spyker": "blue",                                           # ultra-exotic, capped at the rare ceiling
}


def tier_from_count(count: int) -> str:
    if count > 500_000:
        return "grey"
    if count > 50_000:
        return "green"
    if count > 5_000:
        return "blue"
    if count > 500:
        return "purple"
    if count > 10:
        return "orange"
    return "mythic"


def tier_from_make(make_name: str) -> str:
    """Fallback tier when production count is unknown."""
    return MAKE_TIER_DEFAULTS.get(make_name.lower(), "unknown")


def assign_tier(make_name: str, production_count: int | None) -> tuple[str, str]:
    """
    Returns (rarity_tier, rarity_source).
    Prefers production_count when available, falls back to make-based rule.
    """
    if production_count is not None:
        return tier_from_count(production_count), "rule_based"
    tier = tier_from_make(make_name)
    return tier, "rule_based"
