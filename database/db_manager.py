"""
BlindAid — Database Manager
=============================
Provides MySQL (primary) + SQLite (fallback) persistence.

Thread-safe via a background writer thread: callers enqueue writes
and never block on I/O. Reads are synchronous with LRU caching.

Usage:
    db = DBManager()
    db.start()
    db.insert_detection(session_id, frame_id, det, urgency, zone)
    db.stop()
"""

import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Try MySQL; fall back to SQLite silently ──────────────────────────────────

try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("[DB] mysql-connector-python not installed. Using SQLite fallback.")


# ── Constants ─────────────────────────────────────────────────────────────────

SQLITE_PATH = Path(__file__).parent.parent / "data" / "blindaid.db"
WRITE_QUEUE_MAX = 5000          # buffer up to 5k pending writes
BATCH_FLUSH_INTERVAL = 2.0     # flush writes every 2 seconds
BATCH_SIZE = 50                 # or when batch reaches 50 items


# ── DBManager ─────────────────────────────────────────────────────────────────

class DBManager:
    """
    Thread-safe database manager.

    All writes are non-blocking: they go into an internal queue and are
    flushed to the database by a background thread in batches.

    Reads are synchronous but cached (see QueryCache).
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        db_cfg = self.config.get("database", {})

        self._use_mysql    = MYSQL_AVAILABLE and not db_cfg.get("use_sqlite_fallback", False)
        self._mysql_cfg    = db_cfg
        self._pool         = None
        self._sqlite_conn  = None
        self._write_queue  = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._stop_event   = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._started      = False

        # Session tracking
        self.session_id    = str(uuid.uuid4())
        self._frame_count  = 0
        self._session_start = datetime.utcnow()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Connect to DB and start background writer thread."""
        if self._started:
            return True

        ok = self._connect()
        if ok:
            self._ensure_schema()
            self._register_session()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="DB-Writer"
            )
            self._writer_thread.start()
            self._started = True
            logger.info(f"[DB] Started. Backend={'MySQL' if self._use_mysql else 'SQLite'}. "
                        f"Session: {self.session_id[:8]}...")
        return ok

    def stop(self):
        """Flush remaining writes and close connection."""
        if not self._started:
            return
        self._stop_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        self._update_session_end()
        self._close()
        self._started = False
        logger.info("[DB] Stopped and connection closed.")

    # ── Public Write API (non-blocking) ──────────────────────────────────────

    def insert_detection(self, frame_id: int, label: str, confidence: float,
                         zone: str, urgency: str, bbox: tuple,
                         center_x: float, center_y: float, area_fraction: float):
        """Queue a detection event for async DB write."""
        payload = {
            "op": "detection",
            "session_id": self.session_id,
            "frame_id": frame_id,
            "label": label,
            "confidence": round(confidence, 4),
            "zone": zone,
            "urgency": urgency,
            "bbox_json": json.dumps({
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "cx": round(center_x, 4), "cy": round(center_y, 4),
                "area_fraction": round(area_fraction, 4)
            }),
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds"),
        }
        self._enqueue(payload)

    def insert_ocr(self, frame_id: int, raw_text: str, cleaned_text: str,
                   confidence: float, zone: str, semantic_tag: Optional[str] = None,
                   word_dynamics: Optional[List] = None):
        """Queue an OCR event for async DB write."""
        payload = {
            "op": "ocr",
            "session_id": self.session_id,
            "frame_id": frame_id,
            "raw_text": raw_text[:1000],
            "cleaned_text": (cleaned_text or "")[:512],
            "confidence": round(confidence, 4),
            "zone": zone,
            "semantic_tag": semantic_tag,
            "word_dynamics_json": json.dumps(word_dynamics or []),
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds"),
        }
        self._enqueue(payload)

    def insert_training_sample(self, image_path: str, label: Optional[str],
                                annotation: Optional[dict], source: str,
                                split: str = "train", phase: str = "supervised"):
        """Queue a training sample record."""
        payload = {
            "op": "training_sample",
            "image_path": str(image_path),
            "label": label,
            "annotation_json": json.dumps(annotation or {}),
            "source": source,
            "split": split,
            "phase": phase,
        }
        self._enqueue(payload)

    def insert_training_run(self, run_id: str, phase: str, epoch: int,
                             metrics: Dict[str, float], notes: str = ""):
        """Queue a training metrics row."""
        payload = {
            "op": "training_run",
            "run_id": run_id,
            "phase": phase,
            "epoch": epoch,
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "loss": metrics.get("loss"),
            "notes": notes,
        }
        self._enqueue(payload)

    def insert_rl_episode(self, episode: int, step: int, state: List[float],
                           action: int, reward: float, next_state: List[float],
                           done: bool, q_values: Optional[List[float]] = None):
        """Queue an RL transition for replay buffer persistence."""
        payload = {
            "op": "rl_episode",
            "episode": episode,
            "step": step,
            "state_json": json.dumps(state),
            "action": action,
            "reward": round(reward, 4),
            "next_state_json": json.dumps(next_state),
            "done": int(done),
            "q_values_json": json.dumps(q_values or []),
        }
        self._enqueue(payload)

    def update_session_fps(self, cam_fps: float, inf_fps: float, total_frames: int):
        """Update rolling FPS stats for current session (non-blocking)."""
        payload = {
            "op": "session_fps",
            "session_id": self.session_id,
            "cam_fps": cam_fps,
            "inf_fps": inf_fps,
            "total_frames": total_frames,
        }
        self._enqueue(payload)

    # ── Public Read API (synchronous) ─────────────────────────────────────────

    def query_recent_detections(self, limit: int = 20) -> List[Dict]:
        """Return the N most recent detection events for this session."""
        sql = """
            SELECT label, confidence, zone, urgency, timestamp
            FROM detection_events
            WHERE session_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        return self._read(sql, (self.session_id, limit))

    def query_session_stats(self) -> Dict:
        """Return aggregate stats for current session."""
        sql = """
            SELECT COUNT(*) as total_detections,
                   AVG(confidence) as avg_confidence,
                   SUM(urgency='CRITICAL') as critical_count
            FROM detection_events
            WHERE session_id = %s
        """
        rows = self._read(sql, (self.session_id,))
        return rows[0] if rows else {}

    # ── Internal: queue and writer ────────────────────────────────────────────

    def _enqueue(self, payload: dict):
        try:
            self._write_queue.put_nowait(payload)
        except queue.Full:
            logger.warning("[DB] Write queue full — dropping event (system overloaded).")

    def _writer_loop(self):
        """Background thread: drain write queue in batches."""
        batch: List[dict] = []
        last_flush = time.time()

        while not self._stop_event.is_set() or not self._write_queue.empty():
            # Collect items up to BATCH_SIZE or until timeout
            try:
                item = self._write_queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                pass

            elapsed = time.time() - last_flush
            if len(batch) >= BATCH_SIZE or (elapsed >= BATCH_FLUSH_INTERVAL and batch):
                self._flush_batch(batch)
                batch = []
                last_flush = time.time()

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[dict]):
        """Write a batch of queued operations to the database."""
        if not batch:
            return
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for item in batch:
                op = item.get("op")
                if op == "detection":
                    self._exec_insert_detection(cursor, item)
                elif op == "ocr":
                    self._exec_insert_ocr(cursor, item)
                elif op == "training_sample":
                    self._exec_insert_training_sample(cursor, item)
                elif op == "training_run":
                    self._exec_insert_training_run(cursor, item)
                elif op == "rl_episode":
                    self._exec_insert_rl_episode(cursor, item)
                elif op == "session_fps":
                    self._exec_update_session(cursor, item)

            conn.commit()
            cursor.close()
            if not self._use_mysql:
                conn.close()
        except Exception as e:
            logger.error(f"[DB] Batch flush error: {e}", exc_info=False)

    # ── SQL Executors ─────────────────────────────────────────────────────────

    def _exec_insert_detection(self, cursor, d):
        sql = """
            INSERT INTO detection_events
                (session_id, timestamp, frame_id, label, confidence, zone, urgency, bbox_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            d["session_id"], d["timestamp"], d["frame_id"],
            d["label"], d["confidence"], d["zone"], d["urgency"], d["bbox_json"]
        ))

    def _exec_insert_ocr(self, cursor, d):
        sql = """
            INSERT INTO ocr_events
                (session_id, timestamp, frame_id, raw_text, cleaned_text,
                 confidence, zone, semantic_tag, word_dynamics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            d["session_id"], d["timestamp"], d["frame_id"],
            d["raw_text"], d["cleaned_text"], d["confidence"],
            d["zone"], d.get("semantic_tag"), d.get("word_dynamics_json", "[]")
        ))

    def _exec_insert_training_sample(self, cursor, d):
        sql = """
            INSERT INTO training_samples
                (image_path, label, annotation_json, source, split, phase)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            d["image_path"], d.get("label"), d["annotation_json"],
            d["source"], d["split"], d["phase"]
        ))

    def _exec_insert_training_run(self, cursor, d):
        sql = """
            INSERT INTO training_runs
                (run_id, phase, epoch, map50, map50_95, precision, recall, loss, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            d["run_id"], d["phase"], d["epoch"],
            d.get("map50"), d.get("map50_95"), d.get("precision"),
            d.get("recall"), d.get("loss"), d.get("notes", "")
        ))

    def _exec_insert_rl_episode(self, cursor, d):
        sql = """
            INSERT INTO rl_episodes
                (episode, step, state_json, action, reward, next_state_json, done, q_values_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            d["episode"], d["step"], d["state_json"],
            d["action"], d["reward"], d["next_state_json"],
            d["done"], d.get("q_values_json", "[]")
        ))

    def _exec_update_session(self, cursor, d):
        if self._use_mysql:
            sql = """
                UPDATE session_log
                SET cam_fps_avg=%s, inf_fps_avg=%s, total_frames=%s
                WHERE session_id=%s
            """
        else:
            sql = """
                UPDATE session_log
                SET cam_fps_avg=?, inf_fps_avg=?, total_frames=?
                WHERE session_id=?
            """
        cursor.execute(sql, (
            d["cam_fps"], d["inf_fps"], d["total_frames"], d["session_id"]
        ))

    # ── Connection Management ─────────────────────────────────────────────────

    def _connect(self) -> bool:
        if self._use_mysql:
            return self._connect_mysql()
        else:
            return self._connect_sqlite()

    def _connect_mysql(self) -> bool:
        try:
            cfg = self._mysql_cfg
            pool_cfg = {
                "pool_name": "blindaid_pool",
                "pool_size": 5,
                "host":     cfg.get("host", "127.0.0.1"),
                "port":     cfg.get("port", 3306),
                "user":     cfg.get("user", "root"),
                "password": cfg.get("password", ""),
                "database": cfg.get("db_name", "blindaid"),
                "autocommit": False,
                "connect_timeout": 5,
            }
            self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_cfg)
            # Test connection
            conn = self._pool.get_connection()
            conn.close()
            logger.info("[DB] MySQL connection pool created.")
            return True
        except Exception as e:
            logger.warning(f"[DB] MySQL unavailable ({e}). Falling back to SQLite.")
            self._use_mysql = False
            return self._connect_sqlite()

    def _connect_sqlite(self) -> bool:
        try:
            SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # SQLite connections are per-thread (check_same_thread=False for writer thread)
            self._sqlite_path = str(SQLITE_PATH)
            # Test open
            conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
            conn.close()
            logger.info(f"[DB] SQLite connected: {self._sqlite_path}")
            return True
        except Exception as e:
            logger.error(f"[DB] SQLite connection failed: {e}")
            return False

    def _get_connection(self):
        if self._use_mysql:
            return self._pool.get_connection()
        else:
            conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    def _close(self):
        if self._use_mysql and self._pool:
            pass  # Pool cleans up automatically
        # SQLite connections are per-call, nothing to close globally

    def _read(self, sql: str, params: tuple) -> List[Dict]:
        """Synchronous read with automatic %s → ? conversion for SQLite."""
        try:
            if not self._use_mysql:
                sql = sql.replace("%s", "?")
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            if not self._use_mysql:
                conn.close()
            return rows
        except Exception as e:
            logger.error(f"[DB] Read error: {e}")
            return []

    # ── Schema Setup ──────────────────────────────────────────────────────────

    def _ensure_schema(self):
        """Create tables if they don't exist (SQLite auto-setup; MySQL uses schema.sql)."""
        if self._use_mysql:
            # MySQL: tables should be created via schema.sql
            # But we try to create them gracefully if missing
            self._ensure_mysql_tables()
        else:
            self._ensure_sqlite_tables()

    def _ensure_sqlite_tables(self):
        """Create SQLite tables (simplified schema matching MySQL)."""
        ddl = """
        CREATE TABLE IF NOT EXISTS session_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_frames INTEGER DEFAULT 0,
            cam_fps_avg REAL DEFAULT 0,
            inf_fps_avg REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            zone TEXT NOT NULL,
            urgency TEXT NOT NULL,
            bbox_json TEXT
        );
        CREATE TABLE IF NOT EXISTS ocr_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT,
            confidence REAL NOT NULL,
            zone TEXT NOT NULL,
            semantic_tag TEXT,
            word_dynamics_json TEXT
        );
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            label TEXT,
            annotation_json TEXT,
            source TEXT NOT NULL,
            split TEXT DEFAULT 'train',
            phase TEXT DEFAULT 'supervised',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            map50 REAL,
            map50_95 REAL,
            precision REAL,
            recall REAL,
            loss REAL,
            notes TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rl_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode INTEGER NOT NULL,
            step INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            action INTEGER NOT NULL,
            reward REAL NOT NULL,
            next_state_json TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0,
            q_values_json TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            label_hint TEXT,
            centroid_json TEXT NOT NULL,
            member_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_det_session ON detection_events(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_ocr_session ON ocr_events(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_rl_episode  ON rl_episodes(episode, step);
        """
        conn = self._get_connection()
        conn.executescript(ddl)
        conn.commit()
        conn.close()
        logger.info("[DB] SQLite schema ensured.")

    def _ensure_mysql_tables(self):
        """Best-effort: run schema.sql against MySQL if tables missing."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.warning("[DB] schema.sql not found. Tables may need manual creation.")
            return
        try:
            conn = self._pool.get_connection()
            cursor = conn.cursor()
            sql_text = schema_path.read_text(encoding="utf-8")
            # Execute statement by statement (ignore CREATE DATABASE)
            for stmt in sql_text.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.upper().startswith(("CREATE DATABASE", "USE ")):
                    try:
                        cursor.execute(stmt)
                    except Exception:
                        pass  # Table may already exist
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("[DB] MySQL schema applied.")
        except Exception as e:
            logger.warning(f"[DB] MySQL schema setup warning: {e}")

    def _register_session(self):
        """Insert a new session_log row."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if self._use_mysql:
                sql = "INSERT INTO session_log (session_id, start_time) VALUES (%s, %s)"
            else:
                sql = "INSERT INTO session_log (session_id, start_time) VALUES (?, ?)"
            cursor.execute(sql, (self.session_id, self._session_start.isoformat()))
            conn.commit()
            cursor.close()
            if not self._use_mysql:
                conn.close()
        except Exception as e:
            logger.warning(f"[DB] Session registration failed: {e}")

    def _update_session_end(self):
        """Mark session as ended."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            ph = "%s" if self._use_mysql else "?"
            sql = f"UPDATE session_log SET end_time={ph} WHERE session_id={ph}"
            cursor.execute(sql, (datetime.utcnow().isoformat(), self.session_id))
            conn.commit()
            cursor.close()
            if not self._use_mysql:
                conn.close()
        except Exception as e:
            logger.warning(f"[DB] Session end update failed: {e}")
