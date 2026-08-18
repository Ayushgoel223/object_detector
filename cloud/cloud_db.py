"""
BlindAid — Cloud Database (Supabase)
======================================
Supabase is a FREE PostgreSQL-based cloud database.
Free tier: 500MB storage, 2GB bandwidth/month — more than enough.

Why Supabase instead of just local MySQL:
  - Works when laptop is OFF (cloud-hosted)
  - GitHub Actions training can write metrics/detections to it
  - When laptop turns ON, it syncs latest data from cloud
  - Free at supabase.com

Tables mirrored to Supabase (from local schema.sql):
  - training_runs      → cloud receives training metrics from GitHub Actions
  - cluster_centroids  → cloud stores Phase 2 clustering results
  - detection_events   → optional: sync session highlights
  - system_health      → CI/CD health logs

Sync strategy:
  LOCAL (MySQL/SQLite) ←→ CLOUD (Supabase)
    Laptop ON : writes locally + queues cloud sync
    Laptop OFF: GitHub Actions writes directly to cloud
    Laptop ON again: pulls latest cloud data to local

Usage:
  python cloud/cloud_db.py --action log_training --phase supervised
  python cloud/cloud_db.py --action sync_clusters
  python cloud/cloud_db.py --action pull          # Pull cloud → local
  python cloud/cloud_db.py --action status
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[CloudDB] %(message)s")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Try Supabase client ───────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("supabase not installed. Run: pip install supabase")


class CloudDB:
    """
    Thin wrapper around Supabase for BlindAid cloud persistence.
    Falls back to local JSON file if Supabase unavailable.
    """

    FALLBACK_LOG = ROOT / "data" / "cloud_log_fallback.jsonl"

    def __init__(self):
        self.url     = os.environ.get("SUPABASE_URL", "")
        self.key     = os.environ.get("SUPABASE_KEY", "")
        self.client: Optional["Client"] = None
        self._fallback = not SUPABASE_AVAILABLE or not self.url or not self.key

        if not self._fallback:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("[CloudDB] Connected to Supabase.")
            except Exception as e:
                logger.warning(f"[CloudDB] Supabase connect failed: {e}. Using local fallback.")
                self._fallback = True
        else:
            logger.info("[CloudDB] Using local JSON fallback (Supabase not configured).")

    # ── Training Run Logging ──────────────────────────────────────────────────

    def log_training_run(self, phase: str, epoch: int, metrics: Dict,
                          run_id: str = "", notes: str = "") -> bool:
        """Log a training epoch result to cloud."""
        record = {
            "run_id":    run_id or f"ci-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            "phase":     phase,
            "epoch":     epoch,
            "map50":     metrics.get("map50"),
            "map50_95":  metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall":    metrics.get("recall"),
            "loss":      metrics.get("loss"),
            "reward":    metrics.get("reward"),     # RL reward
            "notes":     notes,
            "timestamp": datetime.utcnow().isoformat(),
            "source":    "github_actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        }
        return self._insert("training_runs", record)

    # ── Cluster Centroids Sync ────────────────────────────────────────────────

    def sync_centroids(self, centroids: List[Dict]) -> bool:
        """Upload cluster centroids from Phase 2 to cloud."""
        if not centroids:
            return True
        # Upsert on cluster_id
        for c in centroids:
            c["updated_at"] = datetime.utcnow().isoformat()
        return self._upsert("cluster_centroids", centroids, on_conflict="cluster_id")

    def fetch_centroids(self) -> List[Dict]:
        """Fetch latest cluster centroids from cloud."""
        return self._select("cluster_centroids", order="cluster_id")

    # ── Model Performance Tracking ────────────────────────────────────────────

    def get_best_metrics(self, phase: str) -> Optional[Dict]:
        """Get the best recorded metrics for a training phase."""
        if self._fallback:
            return None
        try:
            result = (self.client.table("training_runs")
                      .select("*")
                      .eq("phase", phase)
                      .order("map50", desc=True)
                      .limit(1)
                      .execute())
            return result.data[0] if result.data else None
        except Exception as e:
            logger.debug(f"get_best_metrics error: {e}")
            return None

    def get_training_history(self, phase: str, limit: int = 50) -> List[Dict]:
        """Get recent training history for plotting."""
        return self._select(
            "training_runs",
            filters={"phase": phase},
            order="timestamp",
            limit=limit,
        )

    # ── System Health Logging ─────────────────────────────────────────────────

    def log_health(self, cpu_pct: float, ram_pct: float, cam_fps: float,
                    inf_fps: float, latency_ms: float) -> bool:
        record = {
            "cpu_pct":    round(cpu_pct, 1),
            "ram_pct":    round(ram_pct, 1),
            "cam_fps":    round(cam_fps, 1),
            "inf_fps":    round(inf_fps, 1),
            "latency_ms": round(latency_ms, 1),
            "timestamp":  datetime.utcnow().isoformat(),
        }
        return self._insert("system_health", record)

    # ── Knowledge Sync ────────────────────────────────────────────────────────

    def upsert_knowledge(self, key: str, value: Any, confidence: float = 1.0) -> bool:
        record = {
            "key":        key,
            "value_json": json.dumps(value),
            "confidence": confidence,
            "updated_at": datetime.utcnow().isoformat(),
        }
        return self._upsert("semantic_knowledge", [record], on_conflict="key")

    def get_knowledge(self, key: str) -> Optional[Any]:
        rows = self._select("semantic_knowledge", filters={"key": key}, limit=1)
        if rows:
            return json.loads(rows[0]["value_json"])
        return None

    # ── Session Stats ─────────────────────────────────────────────────────────

    def log_session(self, session_id: str, total_frames: int,
                     cam_fps: float, inf_fps: float) -> bool:
        record = {
            "session_id":   session_id,
            "total_frames": total_frames,
            "cam_fps_avg":  round(cam_fps, 1),
            "inf_fps_avg":  round(inf_fps, 1),
            "end_time":     datetime.utcnow().isoformat(),
            "source":       "local_laptop",
        }
        return self._insert("session_log_cloud", record)

    # ── Pull Cloud → Local ────────────────────────────────────────────────────

    def pull_to_local(self, local_db_manager=None) -> Dict[str, int]:
        """
        Pull cloud data into local MySQL/SQLite.
        Called when laptop starts up after being off.
        Returns {table: rows_synced}
        """
        synced = {}

        # Pull training runs
        runs = self.get_training_history("supervised", limit=100)
        runs += self.get_training_history("rl", limit=100)
        if runs and local_db_manager:
            for r in runs:
                try:
                    local_db_manager.insert_training_run(
                        run_id=r.get("run_id", "cloud"),
                        phase=r.get("phase", "supervised"),
                        epoch=r.get("epoch", 0),
                        metrics={k: r.get(k) for k in ["map50","map50_95","precision","recall","loss"]},
                        notes=r.get("notes","cloud sync"),
                    )
                except Exception:
                    pass
        synced["training_runs"] = len(runs)

        logger.info(f"[CloudDB] Pulled from cloud: {synced}")
        return synced

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _insert(self, table: str, record: dict) -> bool:
        if self._fallback:
            return self._fallback_write(table, record)
        try:
            self.client.table(table).insert(record).execute()
            return True
        except Exception as e:
            logger.debug(f"[CloudDB] Insert {table} failed: {e}")
            return self._fallback_write(table, record)

    def _upsert(self, table: str, records: List[dict],
                 on_conflict: str = "id") -> bool:
        if self._fallback:
            for r in records:
                self._fallback_write(table, r)
            return True
        try:
            self.client.table(table).upsert(records, on_conflict=on_conflict).execute()
            return True
        except Exception as e:
            logger.debug(f"[CloudDB] Upsert {table} failed: {e}")
            return False

    def _select(self, table: str, filters: dict = None, order: str = None,
                 limit: int = 100) -> List[dict]:
        if self._fallback:
            return []
        try:
            q = self.client.table(table).select("*")
            if filters:
                for k, v in filters.items():
                    q = q.eq(k, v)
            if order:
                q = q.order(order)
            if limit:
                q = q.limit(limit)
            result = q.execute()
            return result.data or []
        except Exception as e:
            logger.debug(f"[CloudDB] Select {table} failed: {e}")
            return []

    def _fallback_write(self, table: str, record: dict) -> bool:
        """Write to local JSONL file when Supabase unavailable."""
        try:
            self.FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {"table": table, "data": record}
            with open(self.FALLBACK_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return True
        except Exception:
            return False

    def status(self):
        print("\n=== BlindAid Cloud DB Status ===")
        print(f"Supabase URL : {self.url[:30]}..." if self.url else "Supabase URL : not set")
        print(f"Connected    : {'✓' if not self._fallback else '✗ (fallback mode)'}")
        if self._fallback:
            print(f"Fallback log : {self.FALLBACK_LOG}")
            if self.FALLBACK_LOG.exists():
                lines = sum(1 for _ in open(self.FALLBACK_LOG))
                print(f"Queued rows  : {lines}")

        if not self._fallback:
            for phase in ["supervised", "rl"]:
                best = self.get_best_metrics(phase)
                if best:
                    print(f"Best {phase:12}: mAP50={best.get('map50','?'):.3f}  ({best.get('timestamp','?')[:10]})")


# ── SQL schema additions for Supabase ─────────────────────────────────────────

SUPABASE_SQL = """
-- Run this once in Supabase SQL Editor (Dashboard → SQL Editor)

CREATE TABLE IF NOT EXISTS training_runs (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT,
    phase       TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    map50       FLOAT,
    map50_95    FLOAT,
    precision   FLOAT,
    recall      FLOAT,
    loss        FLOAT,
    reward      FLOAT,
    notes       TEXT,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    source      TEXT DEFAULT 'local'
);

CREATE TABLE IF NOT EXISTS cluster_centroids (
    id            BIGSERIAL PRIMARY KEY,
    cluster_id    INTEGER NOT NULL UNIQUE,
    label_hint    TEXT,
    centroid_json TEXT NOT NULL,
    member_count  INTEGER DEFAULT 0,
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS semantic_knowledge (
    id          BIGSERIAL PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,
    value_json  TEXT NOT NULL,
    confidence  FLOAT DEFAULT 1.0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_health (
    id          BIGSERIAL PRIMARY KEY,
    cpu_pct     FLOAT,
    ram_pct     FLOAT,
    cam_fps     FLOAT,
    inf_fps     FLOAT,
    latency_ms  FLOAT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_log_cloud (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL,
    total_frames INTEGER,
    cam_fps_avg  FLOAT,
    inf_fps_avg  FLOAT,
    end_time     TIMESTAMPTZ,
    source       TEXT DEFAULT 'local_laptop'
);

-- Enable Row Level Security (optional but recommended)
-- ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlindAid Cloud Database")
    parser.add_argument("--action", choices=["status", "log_training", "sync_clusters", "pull", "print_sql"],
                        default="status")
    parser.add_argument("--phase", default="supervised")
    args = parser.parse_args()

    # Load .env
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    db = CloudDB()

    if args.action == "status":
        db.status()

    elif args.action == "log_training":
        # Test log
        ok = db.log_training_run(
            phase=args.phase,
            epoch=0,
            metrics={"map50": 0.0, "loss": 99.0},
            notes="test from CLI",
        )
        print(f"Logged: {'✓' if ok else '✗'}")

    elif args.action == "sync_clusters":
        # Load local cluster centroids and push to cloud
        features_dir = ROOT / "training" / "data" / "features"
        print(f"Cluster sync from {features_dir} → cloud (not yet implemented in CLI)")

    elif args.action == "pull":
        synced = db.pull_to_local()
        print(f"Pulled: {synced}")

    elif args.action == "print_sql":
        print(SUPABASE_SQL)
