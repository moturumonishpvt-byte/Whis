"""Persistent SQLite storage for WHIS memory records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from app.memory.exceptions import StorageError
from app.memory.models import MemoryCategory, MemoryRecord

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = Path("data/whis_memory.db")


class MemoryStorage:
    """Manages SQLite storage for memory records, embeddings, and metadata."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            self.db_path: Path = DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path)

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None or str(self.db_path) != ":memory:":
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            if str(self.db_path) == ":memory:":
                self._conn = conn
            return conn
        return self._conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        embedding TEXT,
                        metadata TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
                )
        except Exception as exc:
            raise StorageError(f"Failed to initialize SQLite storage: {exc}") from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row into a MemoryRecord."""
        embedding_val = None
        if row["embedding"]:
            try:
                embedding_val = json.loads(row["embedding"])
            except Exception:
                embedding_val = None

        metadata_val: Dict[str, Any] = {}
        if row["metadata"]:
            try:
                metadata_val = json.loads(row["metadata"])
            except Exception:
                metadata_val = {}

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            category=MemoryCategory.from_str(row["category"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=embedding_val,
            metadata=metadata_val,
        )

    def save(self, record: MemoryRecord) -> None:
        """Insert or replace a memory record in the database."""
        conn = self._get_connection()
        try:
            emb_json = json.dumps(record.embedding) if record.embedding is not None else None
            meta_json = json.dumps(record.metadata) if record.metadata else None

            with conn:
                conn.execute(
                    """
                    INSERT INTO memories (id, content, category, created_at, updated_at, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        category = excluded.category,
                        updated_at = excluded.updated_at,
                        embedding = excluded.embedding,
                        metadata = excluded.metadata
                    """,
                    (
                        record.id,
                        record.content,
                        record.category.value,
                        record.created_at,
                        record.updated_at,
                        emb_json,
                        meta_json,
                    ),
                )
        except Exception as exc:
            raise StorageError(f"Failed to save memory '{record.id}': {exc}", memory_id=record.id) from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a single memory record by its identifier."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except Exception as exc:
            raise StorageError(f"Failed to fetch memory '{memory_id}': {exc}", memory_id=memory_id) from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def update(self, record: MemoryRecord) -> None:
        """Update an existing memory record."""
        record.touch()
        self.save(record)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory record by ID. Returns True if deleted, False if not found."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0
        except Exception as exc:
            raise StorageError(f"Failed to delete memory '{memory_id}': {exc}", memory_id=memory_id) from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def list_all(
        self,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> List[MemoryRecord]:
        """List all memory records, optionally filtered by category."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if category is not None:
                cat_enum = MemoryCategory.from_str(category)
                cursor.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY created_at ASC",
                    (cat_enum.value,),
                )
            else:
                cursor.execute("SELECT * FROM memories ORDER BY created_at ASC")
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        except Exception as exc:
            raise StorageError(f"Failed to list memories: {exc}") from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def find_by_content(
        self,
        content: str,
        category: Optional[Union[str, MemoryCategory]] = None,
    ) -> Optional[MemoryRecord]:
        """Check for existing memory with identical stripped content (for duplicate detection)."""
        stripped = content.strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if category is not None:
                cat_enum = MemoryCategory.from_str(category)
                cursor.execute(
                    "SELECT * FROM memories WHERE content = ? AND category = ? LIMIT 1",
                    (stripped, cat_enum.value),
                )
            else:
                cursor.execute(
                    "SELECT * FROM memories WHERE content = ? LIMIT 1",
                    (stripped,),
                )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except Exception as exc:
            raise StorageError(f"Failed to search memory by content: {exc}") from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def count(self, category: Optional[Union[str, MemoryCategory]] = None) -> int:
        """Return total number of stored memories, optionally filtered by category."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if category is not None:
                cat_enum = MemoryCategory.from_str(category)
                cursor.execute("SELECT COUNT(*) FROM memories WHERE category = ?", (cat_enum.value,))
            else:
                cursor.execute("SELECT COUNT(*) FROM memories")
            val = cursor.fetchone()[0]
            return int(val)
        except Exception as exc:
            raise StorageError(f"Failed to count memories: {exc}") from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def clear(self, category: Optional[Union[str, MemoryCategory]] = None) -> int:
        """Clear memory records from database. Returns number of records deleted."""
        conn = self._get_connection()
        try:
            with conn:
                if category is not None:
                    cat_enum = MemoryCategory.from_str(category)
                    cursor = conn.execute("DELETE FROM memories WHERE category = ?", (cat_enum.value,))
                else:
                    cursor = conn.execute("DELETE FROM memories")
                return cursor.rowcount
        except Exception as exc:
            raise StorageError(f"Failed to clear memories: {exc}") from exc
        finally:
            if str(self.db_path) != ":memory:":
                conn.close()

    def close(self) -> None:
        """Close connection if in-memory."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
