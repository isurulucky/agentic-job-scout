import json
import sqlite3
import uuid

from typing import Any


class NavExperienceStore:
    """Stores and retrieves browser navigation experiences using SQLite."""

    MAX_RECORDS_PER_FRAMEWORK = 25

    def __init__(self, db_path: str = "nav_experiences.db") -> None:
        self.db_path = db_path
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS navigation_experiences (
                    uuid TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    successful INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    experiences TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_framework_results
                ON navigation_experiences (
                    framework,
                    result_count DESC
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS page_states (
                    uuid TEXT PRIMARY KEY,
                    experience_uuid TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    page_url TEXT,
                    page_title TEXT,
                    page_state TEXT,
                    FOREIGN KEY (experience_uuid)
                        REFERENCES navigation_experiences(uuid)
                        ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_page_states
                ON page_states (
                    experience_uuid,
                    step_index
                )
                """
            )

    def store(
        self,
        framework: str,
        result_count: int,
        experiences: Any,
    ) -> str:
        """
        Store an experience.

        Retains at most 25 records per framework, removing the oldest
        records when the limit is exceeded.

        Returns the stored UUID.
        """

        experience_uuid = str(uuid.uuid4())
        experiences_json = json.dumps(experiences, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO navigation_experiences (
                    uuid,
                    framework,
                    successful,
                    result_count,
                    experiences
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experience_uuid,
                    framework,
                    1,
                    result_count,
                    experiences_json,
                ),
            )

            # Keep only the 25 most recently stored records
            # for this framework.
            conn.execute(
                """
                DELETE FROM navigation_experiences
                WHERE framework = ?
                  AND uuid NOT IN (
                      SELECT uuid
                      FROM navigation_experiences
                      WHERE framework = ?
                      ORDER BY created_at DESC, rowid DESC
                      LIMIT ?
                  )
                """,
                (
                    framework,
                    framework,
                    self.MAX_RECORDS_PER_FRAMEWORK,
                ),
            )

        return experience_uuid

    def retrieve(
        self,
        framework: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve experiences for a framework, ordered by result_count.

        Args:
            framework: 'Workday', 'Greenhouse', or 'custom-<domain>'.
            limit: Maximum number of experiences to return.

        Returns:
            A list of dictionaries containing the experience metadata
            and decoded JSON experience.
        """
        if limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT uuid, framework, successful, result_count,
                       experiences, created_at
                FROM navigation_experiences
                WHERE framework = ?
                  AND successful = 1
                ORDER BY result_count DESC, created_at DESC
                LIMIT ?
                """,
                (framework, limit),
            ).fetchall()

        return [
            {
                "uuid": row["uuid"],
                "framework": row["framework"],
                "successful": bool(row["successful"]),
                "result_count": row["result_count"],
                "experiences": json.loads(row["experiences"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    def delete(self, experience_uuid: str) -> bool:
        """
        Delete a stored navigation experience by UUID.

        Returns:
            True if a record was deleted, otherwise False.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM navigation_experiences
                WHERE uuid = ?
                """,
                (experience_uuid,),
            )

            return cursor.rowcount > 0
        
    def delete_by_target_urls(self, target_urls: list[str]) -> int:
        """
        Delete stored experiences associated with the given target URLs.

        Returns:
            Number of deleted records.
        """
        if not target_urls:
            return 0

        deleted_count = 0

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT uuid, experiences
                FROM navigation_experiences
                """
            ).fetchall()

            for row in rows:
                experiences = json.loads(row["experiences"])

                # Support both a single experience and a list of experiences.
                if isinstance(experiences, dict):
                    experiences = [experiences]

                if any(
                    experience.get("target_url") in target_urls
                    for experience in experiences
                ):
                    cursor = conn.execute(
                        """
                        DELETE FROM navigation_experiences
                        WHERE uuid = ?
                        """,
                        (row["uuid"],),
                    )
                    deleted_count += cursor.rowcount

        return deleted_count

    def store_page_states(
        self,
        experience_uuid: str,
        page_states: list[dict[str, Any]],
    ) -> int:
        """
        Store page states associated with a navigation experience.

        Returns:
            Number of inserted page states.
        """
        if not page_states:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO page_states (
                    uuid,
                    experience_uuid,
                    step_index,
                    page_url,
                    page_title,
                    page_state
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        experience_uuid,
                        page_state["step_index"],
                        page_state.get("page_url"),
                        page_state.get("page_title"),
                        json.dumps(
                            page_state.get("page_state", {}),
                            ensure_ascii=False,
                        ),
                    )
                    for page_state in page_states
                ],
            )

        return len(page_states)

def delete_page_states(self, experience_uuid: str) -> int:
    """
    Delete all page states associated with an experience.

    Returns:
        Number of deleted page states.
    """
    with self._connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM page_states
            WHERE experience_uuid = ?
            """,
            (experience_uuid,),
        )

        return cursor.rowcount