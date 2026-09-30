"""Tiny psycopg2 connection helper shared by the Phase 3 backend."""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://carspotters:carspotters@localhost:5432/carspotters",
)


@contextmanager
def get_conn():
    """Commit-on-success / rollback-on-error connection context manager."""
    conn = psycopg2.connect(DB_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
