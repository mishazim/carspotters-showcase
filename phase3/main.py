"""
CarSpotters Phase 3 API.

Endpoints:
  POST /auth/register, /auth/login, GET /auth/me
  POST /identify            -> run CV model, return card or "unrecognized"
  POST /collection          -> add an identified car to the user's CarDex
  GET  /collection          -> list the user's spotted cars
  POST /unknown             -> submit a user label for an unrecognized car (flywheel)
  GET  /photos/{name}       -> serve stored photos (static mount)
"""
import io
import uuid
from pathlib import Path

import imagehash
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import auth
from db import get_conn
from inference import get_predictor
from rarity import lookup_rarity

# Cars below this confidence are treated as "not in the dataset" -> flywheel.
CONF_THRESHOLD = 0.35

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CarSpotters API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev only; lock down for production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")


# ----------------------------- helpers -------------------------------------
def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    try:
        payload = auth.decode_token(authorization[7:])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    return {"id": int(payload["sub"]), "username": payload["username"]}


def _save_image(data: bytes) -> tuple[str, str]:
    """Persist upload, return (filename, perceptual-hash)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    name = f"{uuid.uuid4().hex}.jpg"
    img.save(UPLOAD_DIR / name, "JPEG", quality=90)
    return name, str(imagehash.phash(img))


# ----------------------------- auth ----------------------------------------
class Credentials(BaseModel):
    username: str
    password: str


@app.post("/auth/register")
def register(creds: Credentials):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username = %s", (creds.username,))
        if cur.fetchone():
            raise HTTPException(409, "Username already taken")
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (creds.username, auth.hash_pw(creds.password)),
        )
        uid = cur.fetchone()[0]
    return {"token": auth.make_token(uid, creds.username), "username": creds.username}


@app.post("/auth/login")
def login(creds: Credentials):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, password_hash FROM users WHERE username = %s", (creds.username,)
        )
        row = cur.fetchone()
    if not row or not auth.verify_pw(creds.password, row[1]):
        raise HTTPException(401, "Bad username or password")
    return {"token": auth.make_token(row[0], creds.username), "username": creds.username}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT username, xp, level FROM users WHERE id = %s", (user["id"],))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return {"username": row[0], "xp": row[1], "level": row[2]}


# ----------------------------- identify ------------------------------------
@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read image")

    preds = get_predictor().predict(img, topk=3)
    label, conf = preds[0]
    name, phash = _save_image(data)
    guesses = [{"label": l, "confidence": round(c, 4)} for l, c in preds]

    if conf < CONF_THRESHOLD:
        # Not in the dataset (or too uncertain) -> hand off to the label flow.
        return {
            "recognized": False,
            "confidence": round(conf, 4),
            "photo_token": name,
            "top_guesses": guesses,
        }

    rar = lookup_rarity(label)
    return {
        "recognized": True,
        "confidence": round(conf, 4),
        "label": label,
        "photo_token": name,
        "top_guesses": guesses,
        **rar,
    }


# ----------------------------- collection ----------------------------------
class CollectionItem(BaseModel):
    photo_token: str
    make: str | None = None
    model: str | None = None
    year: int | None = None
    rarity_tier: str | None = None
    confidence: float | None = None
    lat: float | None = None
    lng: float | None = None


def _phash_of(token: str) -> str | None:
    path = UPLOAD_DIR / token
    if not path.exists():
        raise HTTPException(400, "Unknown photo_token (was it uploaded via /identify?)")
    return str(imagehash.phash(Image.open(path).convert("RGB")))


@app.post("/collection")
def add_to_collection(item: CollectionItem, user: dict = Depends(current_user)):
    phash = _phash_of(item.photo_token)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM user_collection WHERE user_id = %s AND phash = %s",
            (user["id"], phash),
        )
        if cur.fetchone():
            raise HTTPException(409, "You've already collected this exact photo")
        cur.execute(
            """
            INSERT INTO user_collection
                (user_id, make, model, year, rarity_tier, confidence, photo_path, phash, lat, lng)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (user["id"], item.make, item.model, item.year, item.rarity_tier,
             item.confidence, item.photo_token, phash, item.lat, item.lng),
        )
        new_id = cur.fetchone()[0]
        cur.execute("UPDATE users SET xp = xp + 10 WHERE id = %s", (user["id"],))
        cur.execute(
            """INSERT INTO sighting_log (user_id, make, model, rarity_tier, lat, lng)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (user["id"], item.make, item.model, item.rarity_tier, item.lat, item.lng),
        )
    return {"id": new_id, "status": "added"}


@app.get("/collection")
def get_collection(user: dict = Depends(current_user)):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, make, model, year, rarity_tier, confidence, photo_path, spotted_at
              FROM user_collection
             WHERE user_id = %s
             ORDER BY spotted_at DESC
            """,
            (user["id"],),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["spotted_at"] = r["spotted_at"].isoformat()
    return {"cars": rows}


# ----------------------------- unknown (flywheel) --------------------------
class UnknownLabel(BaseModel):
    photo_token: str
    predicted_label: str | None = None
    predicted_confidence: float | None = None
    make: str
    model: str
    year: int | None = None


@app.post("/unknown")
def submit_unknown(label: UnknownLabel, user: dict = Depends(current_user)):
    phash = _phash_of(label.photo_token)
    # Trust the human label: derive rarity from their make/model via the catalog.
    rar = lookup_rarity(f"{label.make} {label.model}".strip())
    tier = rar["rarity_tier"]

    with get_conn() as conn:
        cur = conn.cursor()
        # 1. Training data (the flywheel).
        cur.execute(
            """
            INSERT INTO unknown_submissions
                (user_id, photo_path, phash, predicted_label, predicted_confidence,
                 user_make, user_model, user_year)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (user["id"], label.photo_token, phash, label.predicted_label,
             label.predicted_confidence, label.make, label.model, label.year),
        )
        sub_id = cur.fetchone()[0]

        # 2. Also add it to the user's CarDex (dedupe on the same photo).
        cur.execute(
            "SELECT id FROM user_collection WHERE user_id = %s AND phash = %s",
            (user["id"], phash),
        )
        existing = cur.fetchone()
        if existing:
            collection_id, added = existing[0], False
        else:
            cur.execute(
                """
                INSERT INTO user_collection
                    (user_id, make, model, year, rarity_tier, confidence, photo_path, phash)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (user["id"], label.make, label.model, label.year, tier,
                 None, label.photo_token, phash),
            )
            collection_id, added = cur.fetchone()[0], True
            cur.execute("UPDATE users SET xp = xp + 10 WHERE id = %s", (user["id"],))
            cur.execute(
                """INSERT INTO sighting_log (user_id, make, model, rarity_tier)
                   VALUES (%s,%s,%s,%s)""",
                (user["id"], label.make, label.model, tier),
            )

    return {
        "id": sub_id,
        "collection_id": collection_id,
        "added": added,
        "make": label.make,
        "model": label.model,
        "rarity_tier": tier,
        "status": "submitted",
        "message": (
            f"Added {label.make} {label.model} to your CarDex (+10 XP) and thanks "
            "for helping train the model!"
            if added else
            "You've already collected this exact photo — thanks for the label!"
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
