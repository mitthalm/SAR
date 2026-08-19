"""
AETHER SAR — SQLite Database & Local Image Output Storage Helper
Handles persistent logging of /colorize and /classify runs, saving generated
output images to backend/outputs/ and storing metadata in backend/history.db.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "history.db"
OUTPUTS_DIR = BASE_DIR / "outputs"


def init_db():
    """Ensure outputs directory exists and initialize SQLite database schema."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                inference_time_ms REAL NOT NULL,
                psnr REAL,
                ssim REAL
            )
            """
        )
        conn.commit()


def save_history_entry(
    mode: str,
    filename: str,
    image_bytes: bytes,
    inference_time_ms: float,
    psnr: float | None = None,
    ssim: float | None = None,
) -> str:
    """
    Saves the output PNG image bytes to backend/outputs/{record_id}.png
    and inserts a metadata record into history.db. Returns the record_id.
    """
    init_db()
    record_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    out_filename = f"{record_id}.png"
    out_file_path = OUTPUTS_DIR / out_filename

    # Save PNG file to disk
    with open(out_file_path, "wb") as f:
        f.write(image_bytes)

    # Insert DB record
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO history (id, mode, timestamp, filename, image_path, inference_time_ms, psnr, ssim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                mode,
                now_iso,
                filename,
                str(out_file_path),
                inference_time_ms,
                psnr,
                ssim,
            ),
        )
        conn.commit()

    return record_id


def get_history_records(limit: int = 20) -> list[dict]:
    """Retrieves the last N records from history.db, ordered by timestamp descending."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, mode, timestamp, filename, image_path, inference_time_ms, psnr, ssim
            FROM history
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_history_record_by_id(record_id: str) -> dict | None:
    """Retrieves a single history record by ID."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_history_record(record_id: str) -> bool:
    """Deletes a history record from SQLite and removes its output image file from disk."""
    init_db()
    record = get_history_record_by_id(record_id)
    if not record:
        return False

    # Remove image file from disk if exists
    img_path = Path(record["image_path"])
    if img_path.exists():
        try:
            os.remove(img_path)
        except OSError:
            pass

    # Delete record from database
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ?", (record_id,))
        conn.commit()

    return True
