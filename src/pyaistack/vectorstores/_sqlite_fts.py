"""Transactional FTS5 migration. Call while holding the store write transaction."""

import sqlite3

from ..exceptions import VectorStoreError

SCHEMA_VERSION = "1"


def ensure_fts(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT value FROM pyaistack_metadata WHERE key='fts_schema_version'"
    ).fetchone()
    if row:
        if row[0] != SCHEMA_VERSION:
            raise VectorStoreError("unsupported FTS schema version; upgrade PyAIStack")
        return
    connection.execute(
        "CREATE VIRTUAL TABLE pyaistack_fts USING fts5(text, "
        "content='pyaistack_vectors', content_rowid='rowid', tokenize='unicode61')"
    )
    connection.execute("""CREATE TRIGGER pyaistack_fts_insert AFTER INSERT ON pyaistack_vectors
        BEGIN INSERT INTO pyaistack_fts(rowid, text) VALUES (new.rowid, new.text); END""")
    connection.execute("""CREATE TRIGGER pyaistack_fts_delete AFTER DELETE ON pyaistack_vectors
        BEGIN INSERT INTO pyaistack_fts(pyaistack_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text); END""")
    connection.execute("""CREATE TRIGGER pyaistack_fts_update AFTER UPDATE ON pyaistack_vectors
        BEGIN INSERT INTO pyaistack_fts(pyaistack_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text);
        INSERT INTO pyaistack_fts(rowid, text) VALUES (new.rowid, new.text); END""")
    connection.execute("INSERT INTO pyaistack_fts(pyaistack_fts) VALUES ('rebuild')")
    connection.execute(
        "INSERT INTO pyaistack_metadata(key,value) VALUES ('fts_schema_version', ?)",
        (SCHEMA_VERSION,),
    )
