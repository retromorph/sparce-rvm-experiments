"""Minimal auditable SQLite snapshot ledger with non-destructive rollback.

Demo-grade persistence. Production requires authentication, ACLs, encrypted data,
immutable/WORM evidence store, replication, telemetry, and tenancy isolation.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .policy import CorrectivePolicy


def _utc():
    return datetime.now(timezone.utc).isoformat()


class SnapshotLedger:
    def __init__(self, path):
        self.path = str(Path(path)) if str(path) != ':memory:' else ':memory:'
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS snapshots (
            version INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            parent INTEGER,
            payload TEXT NOT NULL
        )""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            snapshot_version INTEGER NOT NULL REFERENCES snapshots(version),
            event_type TEXT NOT NULL,
            details TEXT NOT NULL
        )""")
        self.connection.commit()

    def commit(self, policy, *, reason, evidence_ids=()):
        """Commit an already independently validated candidate policy (or a plain snapshot dict)."""
        state = policy if isinstance(policy, dict) else policy.to_dict()
        payload = json.dumps(state, separators=(',', ':'), sort_keys=True)
        with self.connection:
            prior = self.connection.execute("SELECT max(version) FROM snapshots").fetchone()[0]
            cursor = self.connection.execute(
                "INSERT INTO snapshots(created_at,reason,parent,payload) VALUES(?,?,?,?)",
                (_utc(), reason, prior, payload))
            version = cursor.lastrowid
            self.connection.execute(
                "INSERT INTO events(created_at,snapshot_version,event_type,details) VALUES(?,?,?,?)",
                (_utc(), version, 'commit', json.dumps({'evidence_ids': list(evidence_ids)})))
        return version

    def load(self, version=None):
        if version is None:
            row = self.connection.execute(
                "SELECT version,payload FROM snapshots ORDER BY version DESC LIMIT 1").fetchone()
        else:
            row = self.connection.execute(
                "SELECT version,payload FROM snapshots WHERE version=?", (int(version),)).fetchone()
        if row is None:
            raise LookupError("No snapshot for this version")
        state = json.loads(row[1])
        if "model" in state and "scope" in state:
            return row[0], CorrectivePolicy.from_dict(state)
        return row[0], state  # generic snapshot payload (full joint posterior of every scope partition)

    def rollback(self, to_version, *, reason="rollback"):
        """Rollback appends a snapshot; history is never rewritten."""
        _, restored = self.load(to_version)
        return self.commit(restored, reason=f'{reason}:restore:{to_version}')

    def payload_bytes(self, version):
        row = self.connection.execute("SELECT length(payload) FROM snapshots WHERE version=?", (int(version),)).fetchone()
        return int(row[0]) if row else 0

    def events(self):
        rows = self.connection.execute(
            "SELECT snapshot_version,event_type,details FROM events ORDER BY event_id").fetchall()
        return [{'version': row[0], 'type': row[1], **json.loads(row[2])} for row in rows]

    def close(self):
        self.connection.close()
