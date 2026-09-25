"""
database/database.py â€” SQLite Database Connection & Initialization
===================================================================

Day 16 â€” Database & Investigation History

Manages SQLite connection lifecycle, schema initialization, and foreign
key integrity for PhishDec.

Default storage location:
    data/phishdec.db
"""

import os
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Project root calculation
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "phishdec.db")


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Return the absolute path to the SQLite database file."""
    if custom_path:
        return os.path.abspath(custom_path)
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = get_db_path(db_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    conn = sqlite3.connect(path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Check if already WAL to avoid instant lock exceptions
    try:
        current_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if current_mode.lower() != "wal":
            conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
        
    return conn

def initialize_database(db_path: Optional[str] = None) -> None:
    """
    Create all four Day 16 relational tables and performance indexes.

    Schema:
        1. analyses: Main investigation metadata & fused assessment
        2. agent_results: Individual results from the 6 AI agents
        3. risk_factors: Granular risk factors extracted during analysis
        4. evidence: Concrete evidence tokens and indicators
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # 1. analyses Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_verdict TEXT NOT NULL,
                    risk_score REAL,
                    risk_level TEXT,
                    confidence REAL,
                    usable_agent_count INTEGER NOT NULL,
                    total_agent_count INTEGER NOT NULL,
                    coverage REAL NOT NULL,
                    consensus_available INTEGER NOT NULL,
                    consensus_satisfied INTEGER NOT NULL,
                    conflict_detected INTEGER NOT NULL,
                    limited_evidence INTEGER NOT NULL,
                    evidence_state TEXT,
                    decision_basis_json TEXT,
                    risk_calibration_json TEXT,
                    execution_time_ms REAL,
                    notes TEXT,
                    tags TEXT
                )
                """
            )

            # 2. agent_results Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    display_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    signal_available INTEGER NOT NULL,
                    prediction TEXT,
                    risk_score REAL,
                    raw_risk_score REAL,
                    risk_level TEXT,
                    confidence REAL,
                    reason TEXT,
                    calibration_curve TEXT,
                    operational_meaning TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
                )
                """
            )

            # 3. risk_factors Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    agent_name TEXT,
                    factor TEXT NOT NULL,
                    severity TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
                )
                """
            )

            # 4. evidence Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    agent_name TEXT,
                    evidence_type TEXT,
                    description TEXT NOT NULL,
                    severity TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
                )
                """
            )

            # Indexes for high-performance history querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_analysis_id ON analyses (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_url ON analyses (url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_final_verdict ON analyses (final_verdict)")
            # Day 27: Add missing indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_risk_level ON analyses (risk_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_domain ON analyses (domain)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_results_analysis_id ON agent_results (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_factors_analysis_id ON risk_factors (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_analysis_id ON evidence (analysis_id)")

            # 5. alerts Table (Day 24)
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT,
                    reference_id TEXT,
                    dedup_key TEXT,
                    status TEXT NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    resolved_at TEXT
                )
                '''
            )

            # 6. notifications Table (Day 24)
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notification_id TEXT UNIQUE NOT NULL,
                    alert_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    read_at TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (alert_id) ON DELETE CASCADE
                )
                '''
            )
            
            # Indexes for Day 24
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_dedup_key ON alerts (dedup_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user_role_status ON notifications (user_role, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications (created_at DESC)")

            # 7. users Table (Day 25)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'analyst', 'viewer')),
                    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'disabled')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TEXT
                )
                """
            )

            # 8. sessions Table (Day 25)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
                """
            )

            # 9. auth_audit_logs Table (Day 25)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    ip_address TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Indexes for Day 25
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions (session_token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_audit_created ON auth_audit_logs (created_at DESC)")

            # Seed default accounts if users table is empty
            user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if user_count == 0:
                from werkzeug.security import generate_password_hash
                import datetime
                import uuid

                now = datetime.datetime.utcnow().isoformat()
                default_users = [
                    ("USR-ADMIN01", "admin", "admin@phishdec.local", generate_password_hash("admin123"), "admin"),
                    ("USR-ANALYST01", "analyst01", "analyst01@phishdec.local", generate_password_hash("analyst123"), "analyst"),
                    ("USR-VIEWER01", "viewer01", "viewer01@phishdec.local", generate_password_hash("viewer123"), "viewer")
                ]
                for uid, uname, email, phash, role in default_users:
                    conn.execute(
                        """
                        INSERT INTO users (user_id, username, email, password_hash, role, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                        """,
                        (uid, uname, email, phash, role, now, now)
                    )
                logger.info("Default Day 25 users seeded successfully.")


            


        logger.info("Database initialized successfully at: %s", get_db_path(db_path))
    finally:
        conn.close()


