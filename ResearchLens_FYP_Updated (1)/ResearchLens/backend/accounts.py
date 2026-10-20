"""Local password accounts and owner-scoped research sessions."""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from backend.config import DATA_DIR

DB_PATH = DATA_DIR / "accounts.db"


@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT UNIQUE, salt TEXT, digest TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT REFERENCES users(id), name TEXT, created REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS attempts (username TEXT PRIMARY KEY, failures INTEGER, last REAL)")
            yield conn
    finally:
        conn.close()


def _username(name):
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,40}", name):
        raise ValueError("Username: 3-40 letters, digits or underscores.")
    return name


def _digest(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()


def register(username, password):
    username = _username(username)
    if not 12 <= len(password) <= 256:
        raise ValueError("Use a password with 12-256 characters.")
    uid, salt = str(uuid.uuid4()), secrets.token_hex(16)
    digest = _digest(password, salt)
    try:
        with connection() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?)", (uid, username, salt, digest))
    except sqlite3.IntegrityError:
        raise ValueError("That username is unavailable.") from None
    return uid


def authenticate(username, password):
    username = _username(username)
    if len(password) > 256:
        return None
    with connection() as c:
        attempt = c.execute("SELECT * FROM attempts WHERE username=?", (username,)).fetchone()
        if attempt and attempt['failures'] >= 5 and time.time() - attempt['last'] < 300:
            raise ValueError("Too many attempts. Try again in five minutes.")
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        # An unknown username still incurs password derivation work.
        actual = _digest(password, row['salt'] if row else '00' * 16)
        if row and hmac.compare_digest(actual, row['digest']):
            c.execute("DELETE FROM attempts WHERE username=?", (username,))
            return row['id']
        failures = attempt['failures'] + 1 if attempt and time.time() - attempt['last'] < 300 else 1
        c.execute("INSERT OR REPLACE INTO attempts VALUES (?,?,?)", (username, failures, time.time()))
    return None


def create_session(owner):
    sid = str(uuid.uuid4())
    with connection() as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, owner, "New research", time.time()))
    return sid


def list_sessions(owner):
    with connection() as c:
        return [dict(r) for r in c.execute("SELECT * FROM sessions WHERE owner=? ORDER BY created DESC", (owner,))]


def require_owner(owner, session_id):
    with connection() as c:
        row = c.execute("SELECT id FROM sessions WHERE owner=? AND id=?", (owner, session_id)).fetchone()
    if not row:
        raise PermissionError("Research session unavailable.")


def rename_session(owner, session_id, name):
    require_owner(owner, session_id)
    with connection() as c:
        c.execute("UPDATE sessions SET name=? WHERE owner=? AND id=?", (name.strip()[:80] or "Research", owner, session_id))
