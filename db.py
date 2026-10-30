import sqlite3
import numpy as np
from config import DB_FILE

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, judul TEXT NOT NULL,
    kategori TEXT NOT NULL, diperbarui TEXT NOT NULL, text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""

def connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn

def get_meta(conn, key):
    row = conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
    return row['value'] if row else None

def set_meta(conn, key, value):
    conn.execute('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)', (key, value))

def to_blob(vector):
    return np.asarray(vector, dtype=np.float32).tobytes()

def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)