"""Persist only complete, owner-checked turns, including their references."""
from contextlib import contextmanager
import json
import sqlite3
from backend.accounts import require_owner
from backend.config import DATA_DIR

DB_PATH = DATA_DIR / 'turns.db'


@contextmanager
def _connect():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute('CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, payload TEXT)')
    try:
        with c:
            yield c
    finally:
        c.close()


def load_turns(owner, session_id):
    require_owner(owner, session_id)
    with _connect() as c:
        rows = c.execute('SELECT payload FROM turns WHERE session=? ORDER BY id', (session_id,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def save_turn(owner, session_id, turn):
    require_owner(owner, session_id)
    with _connect() as c:
        c.execute('INSERT INTO turns (session, payload) VALUES (?,?)', (session_id, json.dumps(turn)))
