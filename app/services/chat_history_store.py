from abc import ABC, abstractmethod
import sqlite3
from pathlib import Path
from typing import Dict, List


class ChatHistoryStore(ABC):
    """Abstract chat history store so the persistence layer can be swapped later."""

    @abstractmethod
    def get_messages(self, session_id: str, limit: int | None = None) -> List[Dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        limit: int | None = None,
    ) -> None:
        raise NotImplementedError


class InMemoryChatHistoryStore(ChatHistoryStore):
    """In-memory history store for local development and testing."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def get_messages(self, session_id: str, limit: int | None = None) -> List[Dict[str, str]]:
        messages = self._store.get(session_id, [])
        if limit is None:
            return list(messages)
        return list(messages[-limit:])

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        limit: int | None = None,
    ) -> None:
        messages = self._store.setdefault(session_id, [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

        if limit is not None and len(messages) > limit:
            self._store[session_id] = messages[-limit:]


class SQLiteChatHistoryStore(ChatHistoryStore):
    """SQLite-backed chat history store for durable local or embedded persistence."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id ON chat_messages(session_id, id)"
            )
            connection.commit()

    def get_messages(self, session_id: str, limit: int | None = None) -> List[Dict[str, str]]:
        query = "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC"
        params: tuple[str, ...] = (session_id,)

        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        limit: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_message),
                    (session_id, "assistant", assistant_message),
                ],
            )

            if limit is not None:
                row = connection.execute(
                    "SELECT id FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (session_id, limit - 1),
                ).fetchone()

                if row is not None:
                    connection.execute(
                        "DELETE FROM chat_messages WHERE session_id = ? AND id < ?",
                        (session_id, row["id"]),
                    )

            connection.commit()