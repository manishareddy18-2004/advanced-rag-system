import sqlite3
from contextlib import contextmanager

from app.config import SQLITE_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS indexed_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                chunks_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def save_query(question, answer):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_history (question, answer)
            VALUES (?, ?)
            """,
            (question, answer)
        )


def save_document(title, source, chunks_count):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO indexed_documents
                (title, source, chunks_count)
            VALUES (?, ?, ?)
            """,
            (title, source, chunks_count)
        )


def get_history(limit=20):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, created_at
            FROM query_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        return [dict(row) for row in rows]


def get_documents():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, source, chunks_count, created_at
            FROM indexed_documents
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]