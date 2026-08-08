import json
from pathlib import Path
import aiosqlite

from ai_inference_optimization_platform.logging.logger import logger

DB_PATH = Path("data/metrics.db")


class MetricsDatabase:
    """SQLite-based persistent metrics storage for telemetry and analytics."""

    @staticmethod
    async def initialize() -> None:
        """Initializes the SQLite database and creates the metrics table if not exists."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    cache_status TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    metadata TEXT
                )
            """
            )
            await db.commit()
        logger.info("Metrics database initialized and table ready.")

    @staticmethod
    async def save_metric(
        timestamp: str,
        prompt_hash: str,
        provider: str,
        model_name: str,
        cache_status: str,
        total_latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Asynchronously saves an inference execution metric."""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    """
                    INSERT INTO inference_metrics 
                    (timestamp, prompt_hash, provider, model_name, cache_status, total_latency_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        prompt_hash,
                        provider,
                        model_name,
                        cache_status,
                        total_latency_ms,
                        json.dumps(metadata) if metadata else "{}",
                    ),
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to save metric to database: {e}")

    @staticmethod
    async def get_history(limit: int = 50) -> list[dict]:
        """Retrieves past inference metrics for observability."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM inference_metrics ORDER BY id DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]