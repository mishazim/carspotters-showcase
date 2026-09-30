"""
Wikimedia Commons image harvester — the SAFE, licensable data source.

Everything on Wikimedia Commons is freely licensed (CC0 / public-domain / CC-BY / CC-BY-SA);
Commons does not host fair-use media the way English Wikipedia does. That makes it the one source
we can harvest for a *shipped* product — provided we (a) keep only recognizably-free licenses and
(b) record the license + attribution per image so we can honor CC-BY / CC-BY-SA downstream.

This module:
  - queries the Commons API for a given "{year} {make} {model}"
  - keeps only images whose license we can positively identify as free
  - records full provenance (source page, license, author/credit, attribution-required) to a manifest
  - dedups within a run via perceptual hash, enforces a minimum resolution
  - rate-limits politely and sends a descriptive User-Agent (required by Wikimedia policy)

A full corpus run is gated behind an explicit CLI flag; by default it does a small proof harvest.

Usage:
    cd phase1
    ..\.venv\Scripts\python.exe -m harvest.wikimedia --proof
    ..\.venv\Scripts\python.exe -m harvest.wikimedia --make Ford --model Mustang --year 1967 --n 20
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
from dataclasses import dataclass, asdict

import imagehash
import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks for a descriptive UA identifying the tool + a contact. Generic/absent UAs get
# throttled or blocked. See https://meta.wikimedia.org/wiki/User-Agent_policy.
USER_AGENT = (
    "CarSpotters-Harvester/0.1 "
    "(https://github.com/mishazim/carspotters-showcase)"
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "data", "raw", "commons")

# License value prefixes we accept as free-and-attributable. Commons license codes look like
# "cc0", "cc-by-2.0", "cc-by-sa-4.0", "pd", "publicdomain". We keep the ones we can attribute
# correctly and skip anything we can't positively classify (conservative = safe to ship).
_FREE_PREFIXES = ("cc0", "cc-by", "pd", "publicdomain", "public domain")

_MIN_SIDE = 400          # drop tiny thumbnails/icons
_THUMB_WIDTH = 1280      # normalized download width (Commons renders a thumb at this size)
_REQ_PAUSE = 1.0         # seconds between API/image requests — be a polite guest


@dataclass
class ImageRecord:
    make: str
    model: str
    year: int
    query: str
    commons_title: str
    source_page_url: str
    image_url: str
    license: str
    license_url: str
    artist: str
    credit: str
    attribution_required: str
    width: int
    height: int
    local_path: str


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _meta(extmeta: dict, key: str) -> str:
    """Pull a value out of the Commons extmetadata blob, tolerating missing keys/HTML."""
    v = (extmeta.get(key) or {}).get("value", "")
    return str(v).replace("\n", " ").strip()


def _license_code(extmeta: dict) -> str:
    # extmetadata.License is the machine code ("cc-by-sa-4.0"); LicenseShortName is human ("CC BY-SA 4.0").
    return (_meta(extmeta, "License") or _meta(extmeta, "LicenseShortName")).lower()


def _is_free(license_code: str) -> bool:
    lc = license_code.lower()
    return any(lc.startswith(p) for p in _FREE_PREFIXES)


def search_commons(sess: requests.Session, query: str, limit: int) -> list[dict]:
    """Return raw imageinfo dicts for File-namespace results matching `query`."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,               # File:
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": _THUMB_WIDTH,
        "iiextmetadatafilter": "License|LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired",
        "maxlag": 5,                     # back off if the cluster is busy
    }
    r = sess.get(API, params=params, timeout=30)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for p in pages.values():
        ii = p.get("imageinfo")
        if ii:
            out.append({"title": p.get("title", ""), **ii[0]})
    return out


def harvest_vehicle(
    sess: requests.Session,
    make: str,
    model: str,
    year: int,
    out_root: str,
    n: int,
) -> list[ImageRecord]:
    query = f"{year} {make} {model}"
    raw = search_commons(sess, query, limit=max(n * 3, 15))  # over-fetch; many get filtered
    time.sleep(_REQ_PAUSE)

    veh_dir = os.path.join(out_root, f"{make}_{model}_{year}".replace(" ", "-").lower())
    os.makedirs(veh_dir, exist_ok=True)

    seen_hashes: set[imagehash.ImageHash] = set()
    kept: list[ImageRecord] = []

    for item in raw:
        if len(kept) >= n:
            break
        if not str(item.get("mime", "")).startswith("image/"):
            continue
        ext = item.get("extmetadata") or {}
        lic = _license_code(ext)
        if not _is_free(lic):
            continue  # can't positively classify as free -> skip (safe)

        dl_url = item.get("thumburl") or item.get("url")
        if not dl_url:
            continue
        try:
            resp = sess.get(dl_url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            continue
        finally:
            time.sleep(_REQ_PAUSE)

        if min(img.size) < _MIN_SIDE:
            continue
        ph = imagehash.phash(img)
        if any(ph - s <= 4 for s in seen_hashes):
            continue  # near-duplicate
        seen_hashes.add(ph)

        fname = f"{len(kept):03d}.jpg"
        local_path = os.path.join(veh_dir, fname)
        img.save(local_path, "JPEG", quality=90)

        kept.append(ImageRecord(
            make=make, model=model, year=year, query=query,
            commons_title=item.get("title", ""),
            source_page_url=item.get("descriptionurl", ""),
            image_url=item.get("url", ""),
            license=lic,
            license_url=_meta(ext, "LicenseUrl"),
            artist=_meta(ext, "Artist"),
            credit=_meta(ext, "Credit"),
            attribution_required=_meta(ext, "AttributionRequired") or "true",
            width=int(item.get("width", img.size[0])),
            height=int(item.get("height", img.size[1])),
            local_path=os.path.relpath(local_path, out_root),
        ))
    return kept


def write_manifest(records: list[ImageRecord], out_root: str) -> str:
    """Append/write the provenance manifest — one row per kept image, license + attribution intact."""
    path = os.path.join(out_root, "manifest.csv")
    exists = os.path.exists(path)
    fields = list(asdict(records[0]).keys()) if records else []
    with open(path, "a", newline="", encoding="utf-8") as fh:
        if records:
            w = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                w.writeheader()
            for r in records:
                w.writerow(asdict(r))
    return path


# A tiny, deliberately mixed proof set: an iconic classic (lots of free photos), an obscure
# pre-2000 model (the hard, sparse case), and a modern car (sanity).
_PROOF_VEHICLES = [
    ("Ford", "Mustang", 1967),
    ("AMC", "Gremlin", 1974),
    ("Toyota", "Camry", 2015),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest freely-licensed car images from Wikimedia Commons.")
    ap.add_argument("--proof", action="store_true", help="run the small mixed proof set")
    ap.add_argument("--make"); ap.add_argument("--model"); ap.add_argument("--year", type=int)
    ap.add_argument("--n", type=int, default=8, help="target kept images per vehicle")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    sess = _session()

    if args.proof:
        vehicles = _PROOF_VEHICLES
    elif args.make and args.model and args.year:
        vehicles = [(args.make, args.model, args.year)]
    else:
        ap.error("pass --proof, or --make/--model/--year")

    total = 0
    for make, model, year in vehicles:
        recs = harvest_vehicle(sess, make, model, year, out_root, args.n)
        write_manifest(recs, out_root)
        total += len(recs)
        lic_summary = ", ".join(sorted({r.license for r in recs})) or "(none kept)"
        print(f"{year} {make} {model:12s} -> {len(recs):2d} images  [{lic_summary}]")

    print(f"\nKept {total} freely-licensed images -> {out_root}")
    print(f"Provenance + attribution: {os.path.join(out_root, 'manifest.csv')}")


if __name__ == "__main__":
    main()
