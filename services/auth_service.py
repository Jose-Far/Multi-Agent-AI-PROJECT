import sqlite3
import uuid
import datetime
import secrets
import logging
from typing import Dict, Any, Optional, Tuple, List
from werkzeug.security import generate_password_hash, check_password_hash
from database.database import get_connection

logger = logging.getLogger(__name__)

# RBAC Permissions Definitions
ROLE_PERMISSIONS = {
    "super_admin": {
        "analysis:create", "analysis:read", "analysis:delete",
        "investigations:read", "investigations:write",
        "reports:read", "reports:create",
        "alerts:read", "alerts:write",
        "users:read", "users:write", "users:admin_write",
        "system:read", "system:write", "monitoring:read", "logs:read", "settings:write"
    },
    "admin": {
        "analysis:create", "analysis:read", "analysis:delete",
        "investigations:read", "investigations:write",
        "reports:read", "reports:create",
        "alerts:read", "alerts:write",
        "users:read", "users:write",
        "system:read", "monitoring:read"
    },
    "analyst": {
        "analysis:create", "analysis:read",
        "investigations:read", "investigations:write",
        "reports:read", "reports:create",
        "alerts:read",
        "notes:write"
    },
    "viewer": {
        "analysis:read",
        "investigations:read",
        "reports:read",
        "alerts:read"
    }
}

class AuthService:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.MAX_FAILED_ATTEMPTS = 5
        self.LOCKOUT_MINUTES = 15
        self.SESSION_DURATION_HOURS = 24

    def _get_connection(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def log_audit_event(self, event_type: str, user_id: Optional[str], username: Optional[str], ip_address: Optional[str], details: str, conn: Optional[sqlite3.Connection] = None):
        now = datetime.datetime.utcnow().isoformat()
        if conn:
            try:
                conn.execute(
                    """
                    INSERT INTO auth_audit_logs (event_type, user_id, username, ip_address, details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event_type, user_id, username, ip_address, details, now)
                )
            except Exception as e:
                logger.error(f"Failed to record auth audit log with existing conn: {e}")
        else:
            c = self._get_connection()
            try:
                with c:
                    c.execute(
                        """
                        INSERT INTO auth_audit_logs (event_type, user_id, username, ip_address, details, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (event_type, user_id, username, ip_address, details, now)
                    )
            except Exception as e:
                logger.error(f"Failed to record auth audit log: {e}")
            finally:
                c.close()

    def authenticate(self, username_or_email: str, password: str, ip_address: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        username_or_email = (username_or_email or "").strip().lower()
        if not username_or_email or not password:
            return False, "Username/email and password are required.", None

        now = datetime.datetime.utcnow()
        now_iso = now.isoformat()

        trigger_alert = None

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                user = conn.execute(
                    "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
                    (username_or_email, username_or_email)
                ).fetchone()

                if not user:
                    self.log_audit_event("LOGIN_FAILURE", None, username_or_email, ip_address, "Account not found", conn=conn)
                    return False, "Invalid username or password.", None

                user = dict(user)

                # 1. Check account status
                if user["status"] != "active":
                    self.log_audit_event("LOGIN_FAILURE", user["user_id"], user["username"], ip_address, "Account is disabled", conn=conn)
                    return False, "Account is disabled. Please contact an administrator.", None

                # 2. Check lockout
                if user.get("locked_until"):
                    try:
                        lock_until_dt = datetime.datetime.fromisoformat(user["locked_until"])
                        if now < lock_until_dt:
                            remaining = int((lock_until_dt - now).total_seconds() / 60) + 1
                            self.log_audit_event("LOGIN_FAILURE", user["user_id"], user["username"], ip_address, f"Account locked ({remaining}m remaining)", conn=conn)
                            return False, f"Account is temporarily locked. Try again in {remaining} minutes.", None
                    except Exception:
                        pass

                # 3. Check password
                if not check_password_hash(user["password_hash"], password):
                    new_failed = (user.get("failed_login_attempts") or 0) + 1
                    locked_until = None

                    if new_failed >= self.MAX_FAILED_ATTEMPTS:
                        lock_dt = now + datetime.timedelta(minutes=self.LOCKOUT_MINUTES)
                        locked_until = lock_dt.isoformat()
                        conn.execute(
                            "UPDATE users SET failed_login_attempts = ?, locked_until = ? WHERE user_id = ?",
                            (new_failed, locked_until, user["user_id"])
                        )
                        self.log_audit_event("ACCOUNT_LOCKED", user["user_id"], user["username"], ip_address, f"Locked after {new_failed} failed attempts", conn=conn)
                        
                        trigger_alert = {
                            "event_type": "security_brute_force",
                            "severity": "high",
                            "title": f"Brute Force Detected: {user['username']}",
                            "message": f"Account '{user['username']}' locked after {new_failed} failed login attempts from IP: {ip_address or 'unknown'}.",
                            "source": "auth_service",
                            "dedup_key": f"auth_lockout:{user['user_id']}"
                        }
                        return False, f"Account locked due to multiple failed login attempts. Try again in {self.LOCKOUT_MINUTES} minutes.", None
                    else:
                        conn.execute(
                            "UPDATE users SET failed_login_attempts = ? WHERE user_id = ?",
                            (new_failed, user["user_id"])
                        )
                        self.log_audit_event("LOGIN_FAILURE", user["user_id"], user["username"], ip_address, f"Invalid password ({new_failed}/{self.MAX_FAILED_ATTEMPTS})", conn=conn)
                        return False, "Invalid username or password.", None

                # 4. Successful login
                token = secrets.token_hex(32)
                expires_at = (now + datetime.timedelta(hours=self.SESSION_DURATION_HOURS)).isoformat()

                conn.execute(
                    "UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = ? WHERE user_id = ?",
                    (now_iso, user["user_id"])
                )
                conn.execute(
                    "INSERT INTO sessions (session_token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (token, user["user_id"], now_iso, expires_at)
                )

                secure_pw_hash = generate_password_hash(password)[:20] + "..."
                self.log_audit_event("LOGIN_SUCCESS", user["user_id"], user["username"], ip_address, f"Login successful. Authenticated with secured credential hash: {secure_pw_hash}", conn=conn)

                safe_user = {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                    "status": user["status"],
                    "permissions": list(ROLE_PERMISSIONS.get(user["role"], set()))
                }
                return True, token, safe_user
        except Exception as e:
            logger.error(f"AuthService authenticate error: {e}")
            return False, "An error occurred during authentication.", None
        finally:
            conn.close()
            # If a security alert was triggered, execute it outside the connection transaction
            if trigger_alert:
                try:
                    from services.alert_engine import AlertEngine
                    AlertEngine(self.db_path).process_event(trigger_alert)
                except Exception as alert_err:
                    logger.error(f"Failed to trigger security alert: {alert_err}")

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None

        now_iso = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                row = conn.execute(
                    """
                    SELECT s.session_token, s.expires_at, u.user_id, u.username, u.email, u.role, u.status, u.last_seen
                    FROM sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.session_token = ? AND s.expires_at > ? AND u.status = 'active'
                    """,
                    (token, now_iso)
                ).fetchone()

                if not row:
                    return None

                user_data = dict(row)
                user_data["permissions"] = list(ROLE_PERMISSIONS.get(user_data["role"], set()))
                
                # Throttle DB writes: update last_seen only every 10 seconds
                last_seen_str = user_data.get("last_seen")
                should_update = True
                if last_seen_str:
                    try:
                        # SQLite dates can sometimes have Z at the end or not
                        clean_str = last_seen_str.replace("Z", "+00:00")
                        last_seen_dt = datetime.datetime.fromisoformat(clean_str)
                        # Remove timezone info for comparison if it exists
                        last_seen_dt = last_seen_dt.replace(tzinfo=None)
                        if (datetime.datetime.utcnow() - last_seen_dt).total_seconds() < 10:
                            should_update = False
                    except:
                        pass
                        
                if should_update:
                    conn.execute("UPDATE users SET last_seen = ? WHERE user_id = ?", (now_iso, user_data["user_id"]))
                
                return user_data
        except Exception as e:
            logger.error(f"AuthService validate_session error: {e}")
            return None
        finally:
            conn.close()

    def terminate_session(self, token: str, ip_address: Optional[str] = None) -> bool:
        if not token:
            return False

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                session = conn.execute("SELECT user_id FROM sessions WHERE session_token = ?", (token,)).fetchone()
                if session:
                    user = conn.execute("SELECT username FROM users WHERE user_id = ?", (session["user_id"],)).fetchone()
                    username = user["username"] if user else None
                    conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))
                    self.log_audit_event("LOGOUT", session["user_id"], username, ip_address, "User logged out", conn=conn)
                    return True
                return False
        except Exception as e:
            logger.error(f"AuthService terminate_session error: {e}")
            return False
        finally:
            conn.close()

    def delete_user(self, user_id: str, admin_id: Optional[str] = None) -> Tuple[bool, str]:
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if not user:
                    return False, "User not found."

                # Get the caller's role
                caller_role = "unknown"
                if admin_id:
                    c = conn.execute("SELECT role FROM users WHERE user_id = ?", (admin_id,)).fetchone()
                    if c: caller_role = c["role"]

                # Super Admin enforcement
                if caller_role != "super_admin":
                    return False, "Only Super Admins can permanently delete user accounts."

                # Prevent self-deletion
                if user_id == admin_id:
                    return False, "You cannot delete your own account."

                # Safeguard: Prevent deleting the only super_admin
                if user["role"] == "super_admin":
                    role_count = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'super_admin' AND status = 'active'").fetchone()[0]
                    if role_count <= 1:
                        return False, "Cannot delete the only active super_admin account."

                # Delete user
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                
                # Terminate sessions
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

                # Audit Log
                self.log_audit_event("USER_DELETED", admin_id, None, "system", f"Deleted user {user['username']} ({user_id})", conn=conn)

            return True, "User deleted successfully."
        except Exception as e:
            return False, f"Database error: {e}"
        finally:
            conn.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                # User is considered online if last_seen is within the last 30 seconds
                # We use Julian Day conversion to calculate seconds difference
                query = """
                SELECT user_id, username, email, role, status, created_at, last_login,
                CASE 
                    WHEN last_seen IS NOT NULL AND (julianday('now') - julianday(last_seen)) * 86400 < 30 THEN 1 
                    ELSE 0 
                END AS is_online
                FROM users 
                ORDER BY created_at ASC
                """
                rows = conn.execute(query).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"AuthService get_all_users error: {e}")
            return []
        finally:
            conn.close()

    def create_user(self, username: str, email: str, password: str, role: str, creator_id: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        username = (username or "").strip()
        email = (email or "").strip().lower()
        role = (role or "").strip().lower()

        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long.", None
        if not email or "@" not in email:
            return False, "A valid email address is required.", None
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long.", None
        if role not in ("super_admin", "admin", "analyst", "viewer"):
            return False, "Invalid role.", None

        import uuid
        now = datetime.datetime.utcnow().isoformat()
        user_id = f"USR-{uuid.uuid4().hex[:8].upper()}"
        phash = generate_password_hash(password)

        conn = self._get_connection()
        try:
            with conn:
                if creator_id:
                    c = conn.execute("SELECT role FROM users WHERE user_id = ?", (creator_id,)).fetchone()
                    creator_role = c[0] if c else "unknown"
                    if creator_role == "admin" and role in ("super_admin", "admin"):
                        return False, "Admins cannot create Super Admins or other Admins.", None
                conn.execute(
                    """
                    INSERT INTO users (user_id, username, email, password_hash, role, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (user_id, username, email, phash, role, now, now)
                )
                self.log_audit_event("USER_CREATED", user_id, username, None, f"Created by {creator_id or 'system'} with role {role}", conn=conn)
            return True, "User created successfully.", {"user_id": user_id, "username": username, "email": email, "role": role, "status": "active"}
        except sqlite3.IntegrityError:
            return False, "Username or email already exists.", None
        except Exception as e:
            logger.error(f"AuthService create_user error: {e}")
            return False, str(e), None
        finally:
            conn.close()

    def update_user_role(self, user_id: str, new_role: str, admin_id: Optional[str] = None) -> Tuple[bool, str]:
        if new_role not in ("super_admin", "admin", "analyst", "viewer"):
            return False, "Invalid role."

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if not user:
                    return False, "User not found."

                # Get the caller's role
                caller_role = "unknown"
                if admin_id:
                    c = conn.execute("SELECT role FROM users WHERE user_id = ?", (admin_id,)).fetchone()
                    if c: caller_role = c["role"]

                # Restrictions for normal Admins
                if caller_role == "admin":
                    if user["role"] in ("super_admin", "admin") and user_id != admin_id:
                        return False, "Admins cannot modify other Admins or Super Admins."
                    if new_role in ("super_admin", "admin"):
                        return False, "Admins cannot promote users to Admin or Super Admin."

                # Safeguard: Prevent demoting the only super admin or admin
                if user["role"] in ("super_admin", "admin") and new_role not in ("super_admin", "admin"):
                    role_count = conn.execute("SELECT COUNT(*) FROM users WHERE role = ? AND status = 'active'", (user["role"],)).fetchone()[0]
                    if role_count <= 1:
                        return False, f"Cannot change role of the last active {user['role']}."

                conn.execute("UPDATE users SET role = ?, updated_at = ? WHERE user_id = ?", (new_role, datetime.datetime.utcnow().isoformat(), user_id))
                self.log_audit_event("ROLE_CHANGED", user_id, user["username"], None, f"Changed from {user['role']} to {new_role} by {admin_id}", conn=conn)
                return True, "User role updated successfully."
        except Exception as e:
            logger.error(f"AuthService update_user_role error: {e}")
            return False, str(e)
        finally:
            conn.close()

    def toggle_user_status(self, user_id: str, new_status: str, admin_id: Optional[str] = None) -> Tuple[bool, str]:
        if new_status not in ("active", "disabled"):
            return False, "Invalid status."

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if not user:
                    return False, "User not found."

                # Get the caller's role
                caller_role = "unknown"
                if admin_id:
                    c = conn.execute("SELECT role FROM users WHERE user_id = ?", (admin_id,)).fetchone()
                    if c: caller_role = c["role"]

                # Restrictions for normal Admins
                if caller_role == "admin" and user["role"] in ("super_admin", "admin") and user_id != admin_id:
                    return False, "Admins cannot disable other Admins or Super Admins."

                # Safeguard: Prevent disabling the only admin or super_admin
                if user["role"] in ("super_admin", "admin") and new_status == "disabled":
                    role_count = conn.execute("SELECT COUNT(*) FROM users WHERE role = ? AND status = 'active'", (user["role"],)).fetchone()[0]
                    if role_count <= 1:
                        return False, f"Cannot disable the only active {user['role']} account."

                conn.execute("UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?", (new_status, datetime.datetime.utcnow().isoformat(), user_id))
                
                # Invalidate active sessions if disabling
                if new_status == "disabled":
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

                event = "USER_DISABLED" if new_status == "disabled" else "USER_ENABLED"
                self.log_audit_event(event, user_id, user["username"], None, f"Status changed to {new_status} by {admin_id}", conn=conn)
                return True, f"User {new_status} successfully."
        except Exception as e:
            logger.error(f"AuthService toggle_user_status error: {e}")
            return False, str(e)
        finally:
            conn.close()

    def reset_password(self, user_id: str, new_password: str, admin_id: Optional[str] = None) -> Tuple[bool, str]:
        if not new_password or len(new_password) < 6:
            return False, "Password must be at least 6 characters long."

        phash = generate_password_hash(new_password)
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                user = conn.execute("SELECT username FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if not user:
                    return False, "User not found."

                conn.execute(
                    "UPDATE users SET password_hash = ?, failed_login_attempts = 0, locked_until = NULL, updated_at = ? WHERE user_id = ?",
                    (phash, datetime.datetime.utcnow().isoformat(), user_id)
                )
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

                self.log_audit_event("PASSWORD_RESET", user_id, user["username"], None, f"Password reset by {admin_id or 'user'}", conn=conn)
                return True, "Password reset successfully."
        except Exception as e:
            logger.error(f"AuthService reset_password error: {e}")
            return False, str(e)
        finally:
            conn.close()

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                rows = conn.execute(
                    "SELECT * FROM auth_audit_logs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"AuthService get_audit_logs error: {e}")
            return []
        finally:
            conn.close()
