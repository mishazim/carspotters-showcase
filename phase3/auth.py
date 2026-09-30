"""
Minimal JWT auth. Passwords hashed with stdlib PBKDF2 (no native bcrypt dep),
tokens signed with PyJWT (HS256).
"""
import base64
import hashlib
import hmac
import os
import time

import jwt

SECRET = os.getenv("CARSPOTTERS_SECRET", "dev-secret-change-me")
ALGO = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 days
_ITERATIONS = 200_000


def hash_pw(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"{base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_pw(password: str, stored: str) -> bool:
    try:
        salt_b64, dk_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except (ValueError, base64.binascii.Error):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return hmac.compare_digest(actual, expected)


def make_token(user_id: int, username: str) -> str:
    now = int(time.time())
    payload = {"sub": str(user_id), "username": username, "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=[ALGO])
