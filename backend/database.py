"""
SARAL — Database Module
SQLite-backed persistence for users, reports, and rewards.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "saral.db")


def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist and seed default users."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    NOT NULL UNIQUE,
            password    TEXT    NOT NULL,
            role        TEXT    NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'authority')),
            city        TEXT    DEFAULT '',
            karma_points INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            plate_number   TEXT    DEFAULT '',
            manual_plate   TEXT    DEFAULT '',
            violation_type TEXT    NOT NULL,
            confidence     REAL    DEFAULT 0.0,
            media_url      TEXT    DEFAULT '',
            annotated_url  TEXT    DEFAULT '',
            location       TEXT    DEFAULT '',
            description    TEXT    DEFAULT '',
            status         TEXT    NOT NULL DEFAULT 'Under Review'
                           CHECK(status IN ('Under Review', 'Approved', 'Rejected', 'Auto-Rejected')),
            helmet_detected TEXT   DEFAULT '',
            fine_amount    TEXT    DEFAULT '',
            source         TEXT    DEFAULT 'helmet_plate'
                           CHECK(source IN ('helmet_plate', 'parking')),
            vehicle_type   TEXT    DEFAULT '',
            created_at     TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS rewards_redeemed (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            reward_id  TEXT    NOT NULL,
            title      TEXT    NOT NULL,
            cost       INTEGER NOT NULL,
            redeemed_at TEXT   DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Seed default users if table is empty
    if cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO users (name, email, password, role, city, karma_points) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("Aarav Kumar", "aarav@saral.in", "citizen123", "user", "Bengaluru", 0),
                ("Insp. T. Prasad", "prasad@authority.in", "authority123", "authority", "Bengaluru", 0),
            ],
        )

    conn.commit()

    # Migration: add annotated_url column if missing (for existing DBs)
    try:
        cursor.execute("SELECT annotated_url FROM reports LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE reports ADD COLUMN annotated_url TEXT DEFAULT ''")
        conn.commit()

    # Migration: add manual_plate column if missing (for existing DBs)
    try:
        cursor.execute("SELECT manual_plate FROM reports LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE reports ADD COLUMN manual_plate TEXT DEFAULT ''")
        conn.commit()

    # Migration: add fine_amount column if missing (for existing DBs)
    try:
        cursor.execute("SELECT fine_amount FROM reports LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE reports ADD COLUMN fine_amount TEXT DEFAULT ''")
        conn.commit()

    # Migration: add source column if missing (for existing DBs)
    try:
        cursor.execute("SELECT source FROM reports LIMIT 1")
    except sqlite3.OperationalError:
        # SQLite does not support CHECK constraints in ALTER TABLE;
        # the constraint is enforced by application logic for migrated rows.
        cursor.execute("ALTER TABLE reports ADD COLUMN source TEXT DEFAULT 'helmet_plate'")
        conn.commit()

    # Migration: add vehicle_type column if missing (for existing DBs)
    try:
        cursor.execute("SELECT vehicle_type FROM reports LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE reports ADD COLUMN vehicle_type TEXT DEFAULT ''")
        conn.commit()

    conn.close()


# ─── User Operations ───

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(name: str, email: str, password: str, role: str = "user", city: str = "") -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO users (name, email, password, role, city) VALUES (?, ?, ?, ?, ?)",
        (name, email, password, role, city),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return get_user_by_id(user_id)


def update_user_karma(user_id: int, points_delta: int) -> int:
    """Add points_delta to user's karma. Returns new balance."""
    conn = get_db()
    conn.execute("UPDATE users SET karma_points = karma_points + ? WHERE id = ?", (points_delta, user_id))
    conn.commit()
    row = conn.execute("SELECT karma_points FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row["karma_points"] if row else 0


def update_user_profile(user_id: int, name: str, email: str, city: str, password: str = None):
    conn = get_db()
    if password:
        conn.execute("UPDATE users SET name = ?, email = ?, city = ?, password = ? WHERE id = ?", (name, email, city, password, user_id))
    else:
        conn.execute("UPDATE users SET name = ?, email = ?, city = ? WHERE id = ?", (name, email, city, user_id))
    conn.commit()
    conn.close()


# ─── Report Operations ───

def create_report(
    user_id: int,
    plate_number: str,
    violation_type: str,
    confidence: float,
    media_url: str = "",
    annotated_url: str = "",
    location: str = "",
    description: str = "",
    helmet_detected: str = "",
    status: str = "Under Review",
    manual_plate: str = "",
    fine_amount: str = "",
    source: str = "helmet_plate",
    vehicle_type: str = "",
) -> Dict[str, Any]:
    """
    Insert a new report row and return it as a dict.

    Works for both helmet/plate reports (source='helmet_plate') and
    parking-violation reports (source='parking').
    """
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO reports
               (user_id, plate_number, violation_type, confidence,
                media_url, annotated_url, location, description,
                helmet_detected, status, manual_plate, fine_amount, source, vehicle_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, plate_number, violation_type, confidence,
            media_url, annotated_url, location, description,
            helmet_detected, status, manual_plate, fine_amount, source, vehicle_type,
        ),
    )
    conn.commit()
    report_id = cursor.lastrowid
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row)


def create_parking_violation_report(
    user_id: int,
    plate_number: str,
    violation_type: str,
    fine_amount: str,
    confidence: float,
    vehicle_type: str = "",
    media_url: str = "",
    annotated_url: str = "",
    location: str = "",
    description: str = "",
) -> Dict[str, Any]:
    """
    Convenience wrapper that stores a single parking violation using the
    same `reports` table as helmet violations.

    Differences from a helmet report:
      • source       = 'parking'      (distinguishes it in queries)
      • fine_amount  = the fine string from model2 (e.g. '₹500')
      • vehicle_type = vehicle label from YOLO (e.g. 'Car', 'Motorcycle')
      • status       = 'Under Review' (consistent with the rest of the system)
      • helmet_detected is left blank (not applicable)
      • manual_plate    is left blank (OCR plate is stored in plate_number)
    """
    return create_report(
        user_id=user_id,
        plate_number=plate_number,
        violation_type=violation_type,
        confidence=confidence,
        media_url=media_url,
        annotated_url=annotated_url,
        location=location,
        description=description,
        helmet_detected="",
        status="Under Review",
        manual_plate="",
        fine_amount=fine_amount,
        source="parking",
        vehicle_type=vehicle_type,
    )


def get_parking_violation_reports(
    user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return all parking-violation reports, optionally filtered by user.
    Results are ordered newest-first.
    """
    conn = get_db()
    if user_id is not None:
        rows = conn.execute(
            "SELECT * FROM reports WHERE source = 'parking' AND user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports WHERE source = 'parking' ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_reports_by_user(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_reports() -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """SELECT r.id, r.user_id, r.plate_number, r.manual_plate,
                  r.violation_type, r.confidence, r.media_url, r.annotated_url,
                  r.location, r.description, r.status, r.helmet_detected,
                  r.fine_amount, r.source, r.vehicle_type, r.created_at,
                  u.name  AS reporter_name,
                  u.email AS reporter_email
           FROM reports r
           JOIN users u ON r.user_id = u.id
           WHERE r.status = 'Under Review'
           ORDER BY r.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_reports() -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        """SELECT r.id, r.user_id, r.plate_number, r.manual_plate,
                  r.violation_type, r.confidence, r.media_url, r.annotated_url,
                  r.location, r.description, r.status, r.helmet_detected,
                  r.fine_amount, r.source, r.vehicle_type, r.created_at,
                  u.name AS reporter_name
           FROM reports r
           JOIN users u ON r.user_id = u.id
           ORDER BY r.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_report_status(report_id: int, status: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    conn.execute("UPDATE reports SET status = ? WHERE id = ?", (status, report_id))
    conn.commit()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_report_manual_plate(report_id: int, manual_plate: str) -> Optional[Dict[str, Any]]:
    """Update the citizen-reported manual plate on an existing report."""
    conn = get_db()
    conn.execute("UPDATE reports SET manual_plate = ? WHERE id = ?", (manual_plate, report_id))
    conn.commit()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_report_details(report_id: int, **fields) -> Optional[Dict[str, Any]]:
    """Update arbitrary report fields (location, description, violation_type, manual_plate)."""
    allowed = {"location", "description", "violation_type", "manual_plate"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        row = get_report_by_id(report_id)
        return row
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [report_id]
    conn = get_db()
    conn.execute(f"UPDATE reports SET {set_clause} WHERE id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_report_stats(user_id: int) -> Dict[str, int]:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM reports WHERE user_id = ?", (user_id,)).fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM reports WHERE user_id = ? AND status = 'Approved'", (user_id,)).fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM reports WHERE user_id = ? AND status = 'Under Review'", (user_id,)).fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM reports WHERE user_id = ? AND status = 'Rejected'", (user_id,)).fetchone()[0]
    conn.close()
    return {"total": total, "approved": approved, "pending": pending, "rejected": rejected}


def get_authority_stats() -> Dict[str, int]:
    conn = get_db()
    pending = conn.execute("SELECT COUNT(*) FROM reports WHERE status = 'Under Review'").fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM reports WHERE status = 'Approved'").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM reports WHERE status = 'Rejected'").fetchone()[0]
    # Auto-Rejected reports are system-rejected, not counted in authority stats
    total = pending + approved + rejected
    action_pct = round((approved + rejected) / total * 100) if total > 0 else 0
    conn.close()
    return {"pending": pending, "approved": approved, "rejected": rejected, "total": total, "action_pct": action_pct}


# ─── Rewards Operations ───

def get_redeemed_rewards(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM rewards_redeemed WHERE user_id = ? ORDER BY redeemed_at DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_redeemed_reward(user_id: int, reward_id: str, title: str, cost: int) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO rewards_redeemed (user_id, reward_id, title, cost) VALUES (?, ?, ?, ?)",
        (user_id, reward_id, title, cost),
    )
    conn.commit()
    row_id = cursor.lastrowid
    row = conn.execute("SELECT * FROM rewards_redeemed WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


# Initialize DB on import
init_db()
