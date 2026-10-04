"""
backend/services/auth_service.py
Step 23 — Authentication, Role-Based Access Control, Session Management & Audit Logging Service.

Provides:
- Secure PBKDF2-HMAC-SHA256 password hashing with individual salts
- Thread-safe persistent SQLite storage for users, sessions, and audit logs
- Token-based session tracking with instant server-side revocation
- Brute-force protection and login rate-limiting
- Granular RBAC permissions enforcement and camera-level isolation
- Last-administrator deletion/disablement protection
- Structured append-only audit trail
"""
import hashlib
import hmac
import json
import logging
import math
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.schemas.auth import (
    ROLE_PERMISSIONS,
    AuditActionOutcome,
    AuditLogEntry,
    AuditLogEvent,
    AuditLogListResponse,
    Permission,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
    UserRole,
    UserStatus,
    UserUpdateRequest,
)
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.auth_service")


class AuthService:
    """
    Central authentication, user management, RBAC, and security audit repository.
    Stores security tables in data/auth.db with SQLite WAL mode.
    """

    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = project_root / settings.AUTH_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # Rate limiting state: key = (username, ip_address) -> list of failed epoch timestamps
        self._failed_attempts: Dict[Tuple[str, str], List[float]] = {}
        self._rate_limit_lock = threading.Lock()

        self._init_database()
        self._bootstrap_admin_if_empty()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with WAL and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_database(self) -> None:
        """Initializes tables for users, active sessions, and audit logging."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Users table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT,
                        full_name TEXT,
                        password_hash TEXT NOT NULL,
                        password_salt TEXT NOT NULL,
                        role TEXT NOT NULL,
                        status TEXT NOT NULL,
                        permitted_cameras_json TEXT,
                        created_at REAL NOT NULL,
                        last_login REAL
                    )
                    """
                )
                # Sessions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        is_revoked INTEGER NOT NULL DEFAULT 0,
                        ip_address TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                    )
                    """
                )
                # Audit logs table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        log_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        actor_id TEXT,
                        actor_username TEXT,
                        resource_type TEXT,
                        resource_id TEXT,
                        action_outcome TEXT NOT NULL,
                        ip_address TEXT,
                        correlation_id TEXT,
                        details_json TEXT
                    )
                    """
                )
                # Performance & Security Indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(timestamp DESC);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_logs(event_type);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor_username);")
                conn.commit()

    # --------------------------------------------------------------------------
    # Cryptographic Password Hashing & Verification
    # --------------------------------------------------------------------------
    @staticmethod
    def hash_password(password: str) -> Tuple[str, str]:
        """
        Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations and a 16-byte random salt.
        Returns (hash_hex, salt_hex).
        """
        salt = secrets.token_bytes(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return pwd_hash.hex(), salt.hex()

    @staticmethod
    def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
        """
        Verifies password against stored salt and hash using constant-time comparison.
        """
        try:
            salt = bytes.fromhex(salt_hex)
            expected_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()
            return hmac.compare_digest(expected_hash, hash_hex)
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # Bootstrap Initial Administrator
    # --------------------------------------------------------------------------
    def _bootstrap_admin_if_empty(self) -> None:
        """Seeds initial Administrator account if the users table is completely empty."""
        settings = get_settings()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM users")
                if cursor.fetchone()["count"] == 0:
                    admin_id = f"usr_{uuid.uuid4().hex[:10]}"
                    username = settings.AUTH_DEFAULT_ADMIN_USERNAME
                    password = settings.AUTH_DEFAULT_ADMIN_PASSWORD
                    email = settings.AUTH_DEFAULT_ADMIN_EMAIL
                    h, s = self.hash_password(password)
                    now = time.time()

                    cursor.execute(
                        """
                        INSERT INTO users (
                            user_id, username, email, full_name, password_hash, password_salt,
                            role, status, permitted_cameras_json, created_at, last_login
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            admin_id,
                            username,
                            email,
                            "System Administrator",
                            h,
                            s,
                            UserRole.ADMIN.value,
                            UserStatus.ACTIVE.value,
                            None,
                            now,
                            None,
                        ),
                    )
                    conn.commit()

                    self.log_audit(
                        event_type=AuditLogEvent.USER_CREATE.value,
                        actor_id="system",
                        actor_username="system",
                        resource_type="user",
                        resource_id=admin_id,
                        action_outcome=AuditActionOutcome.SUCCESS,
                        details={"username": username, "role": UserRole.ADMIN.value, "initial_seed": True},
                    )
                    logger.info(f"Initialized default system administrator '{username}'.")

                    if settings.ENVIRONMENT.lower() == "production":
                        if password == "IntelliWatch2026!":
                            logger.warning(
                                "CRITICAL SECURITY WARNING: System administrator initialized with default password in production! "
                                "Immediately update password via POST /api/v1/auth/change-password or configure AUTH_DEFAULT_ADMIN_PASSWORD in .env"
                            )
                        if settings.AUTH_SECRET_KEY == "intelliwatch-control-room-super-secret-key-2026-cctv":
                            logger.warning(
                                "CRITICAL SECURITY WARNING: Default development AUTH_SECRET_KEY active in production! "
                                "Please define a unique random AUTH_SECRET_KEY in your .env file."
                            )

    # --------------------------------------------------------------------------
    # Rate Limiting & Brute Force Defense
    # --------------------------------------------------------------------------
    def check_rate_limit(self, username: str, ip_address: str) -> None:
        """
        Checks if login attempts exceed threshold within lockout window.
        Raises ValueError if rate limit exceeded.
        """
        settings = get_settings()
        max_attempts = settings.AUTH_RATE_LIMIT_ATTEMPTS
        lockout_sec = settings.AUTH_RATE_LIMIT_LOCKOUT_SECONDS
        now = time.time()
        key = (username.strip().lower(), ip_address or "unknown")

        with self._rate_limit_lock:
            attempts = self._failed_attempts.get(key, [])
            # Filter attempts within lockout window
            recent = [t for t in attempts if (now - t) < lockout_sec]
            self._failed_attempts[key] = recent

            if len(recent) >= max_attempts:
                remaining = int(lockout_sec - (now - recent[0]))
                raise ValueError(
                    f"Too many failed login attempts. Account temporarily locked. Try again in {max(1, remaining)} seconds."
                )

    def record_failed_attempt(self, username: str, ip_address: str) -> None:
        """Records a failed login attempt for rate limiting."""
        now = time.time()
        key = (username.strip().lower(), ip_address or "unknown")
        with self._rate_limit_lock:
            if key not in self._failed_attempts:
                self._failed_attempts[key] = []
            self._failed_attempts[key].append(now)

    def clear_failed_attempts(self, username: str, ip_address: str) -> None:
        """Clears failed attempt history upon successful login."""
        key = (username.strip().lower(), ip_address or "unknown")
        with self._rate_limit_lock:
            self._failed_attempts.pop(key, None)

    # --------------------------------------------------------------------------
    # Authentication & Session Invalidation
    # --------------------------------------------------------------------------
    def authenticate_user(
        self, username: str, password: str, ip_address: Optional[str] = None
    ) -> Tuple[UserResponse, str, int]:
        """
        Authenticates user with username & password.
        Returns (UserResponse, token, expires_in_seconds).
        """
        clean_user = username.strip()
        self.check_rate_limit(clean_user, ip_address or "unknown")

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (clean_user,))
                row = cursor.fetchone()

                if not row or not self.verify_password(password, row["password_hash"], row["password_salt"]):
                    self.record_failed_attempt(clean_user, ip_address or "unknown")
                    self.log_audit(
                        event_type=AuditLogEvent.AUTH_LOGIN_FAILURE.value,
                        actor_username=clean_user,
                        action_outcome=AuditActionOutcome.FAILURE,
                        ip_address=ip_address,
                        details={"reason": "Invalid credentials"},
                    )
                    raise ValueError("Invalid username or password.")

                if row["status"] != UserStatus.ACTIVE.value:
                    self.log_audit(
                        event_type=AuditLogEvent.AUTH_LOGIN_FAILURE.value,
                        actor_id=row["user_id"],
                        actor_username=row["username"],
                        action_outcome=AuditActionOutcome.DENIED,
                        ip_address=ip_address,
                        details={"reason": "Account disabled"},
                    )
                    raise ValueError("Account is disabled. Please contact an administrator.")

                # Authentication successful: clear failed attempts
                self.clear_failed_attempts(clean_user, ip_address or "unknown")

                # Update last login timestamp
                now = time.time()
                cursor.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (now, row["user_id"]))
                conn.commit()

                # Create session token
                settings = get_settings()
                expires_in = settings.AUTH_TOKEN_EXPIRE_MINUTES * 60
                token = secrets.token_urlsafe(36)
                expires_at = now + expires_in

                cursor.execute(
                    """
                    INSERT INTO sessions (token, user_id, created_at, expires_at, is_revoked, ip_address)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (token, row["user_id"], now, expires_at, ip_address),
                )
                conn.commit()

                user_model = self._row_to_user_model(row)
                user_model.last_login = now

                self.log_audit(
                    event_type=AuditLogEvent.AUTH_LOGIN_SUCCESS.value,
                    actor_id=row["user_id"],
                    actor_username=row["username"],
                    action_outcome=AuditActionOutcome.SUCCESS,
                    ip_address=ip_address,
                    details={"role": row["role"]},
                )

                return user_model, token, expires_in

    def validate_token(self, token: str) -> Optional[UserResponse]:
        """
        Validates a bearer token. Checks expiration and revocation.
        Returns UserResponse if valid, None otherwise.
        """
        if not token:
            return None

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                cursor.execute(
                    """
                    SELECT s.expires_at, s.is_revoked, u.*
                    FROM sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.token = ?
                    """,
                    (token,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                if row["is_revoked"] == 1 or row["expires_at"] < now:
                    return None

                if row["status"] != UserStatus.ACTIVE.value:
                    return None

                return self._row_to_user_model(row)

    def revoke_session(self, token: str, actor_username: Optional[str] = None) -> None:
        """Invalidates a single session token (Logout)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM sessions WHERE token = ?", (token,))
                row = cursor.fetchone()
                if row:
                    cursor.execute("UPDATE sessions SET is_revoked = 1 WHERE token = ?", (token,))
                    conn.commit()
                    self.log_audit(
                        event_type=AuditLogEvent.AUTH_LOGOUT.value,
                        actor_id=row["user_id"],
                        actor_username=actor_username,
                        action_outcome=AuditActionOutcome.SUCCESS,
                        details={"token_revoked": True},
                    )

    def revoke_user_sessions(self, user_id: str, reason: str = "security") -> None:
        """Revokes all active sessions for a specific user (on role change, disable, or password change)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE sessions SET is_revoked = 1 WHERE user_id = ? AND is_revoked = 0", (user_id,))
                conn.commit()
                self.log_audit(
                    event_type=AuditLogEvent.AUTH_SESSION_REVOKED.value,
                    actor_id=user_id,
                    action_outcome=AuditActionOutcome.SUCCESS,
                    details={"reason": reason},
                )

    # --------------------------------------------------------------------------
    # User Management & Administration
    # --------------------------------------------------------------------------
    def create_user(self, req: UserCreateRequest, actor: Optional[UserResponse] = None) -> UserResponse:
        """Creates a new user account with administrative authorization."""
        clean_user = req.username.strip()
        h, s = self.hash_password(req.password)
        now = time.time()
        user_id = f"usr_{uuid.uuid4().hex[:10]}"
        cam_json = json.dumps(req.permitted_cameras) if req.permitted_cameras is not None else None

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (clean_user,))
                if cursor.fetchone():
                    raise ValueError(f"Username '{clean_user}' is already taken.")

                cursor.execute(
                    """
                    INSERT INTO users (
                        user_id, username, email, full_name, password_hash, password_salt,
                        role, status, permitted_cameras_json, created_at, last_login
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        clean_user,
                        req.email,
                        req.full_name,
                        h,
                        s,
                        req.role.value,
                        UserStatus.ACTIVE.value,
                        cam_json,
                        now,
                        None,
                    ),
                )
                conn.commit()

        self.log_audit(
            event_type=AuditLogEvent.USER_CREATE.value,
            actor_id=actor.user_id if actor else "admin",
            actor_username=actor.username if actor else "admin",
            resource_type="user",
            resource_id=user_id,
            action_outcome=AuditActionOutcome.SUCCESS,
            details={"username": clean_user, "role": req.role.value},
        )
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Retrieves a user profile by ID without secret hashes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                return self._row_to_user_model(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[UserResponse]:
        """Retrieves a user profile by username."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
                row = cursor.fetchone()
                return self._row_to_user_model(row) if row else None

    def list_users(self) -> UserListResponse:
        """Returns all system user records."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users ORDER BY created_at ASC")
                rows = cursor.fetchall()
                items = [self._row_to_user_model(r) for r in rows]
                return UserListResponse(items=items, total=len(items))

    def update_user(self, user_id: str, req: UserUpdateRequest, actor: Optional[UserResponse] = None) -> UserResponse:
        """Updates user profile, role, status, or camera permissions."""
        with self._lock:
            existing = self.get_user(user_id)
            if not existing:
                raise ValueError(f"User '{user_id}' not found.")

            # Last Administrator Protection: Check if changing role or status of the only active admin
            if existing.role == UserRole.ADMIN:
                if (req.role and req.role != UserRole.ADMIN) or (req.status and req.status == UserStatus.DISABLED):
                    self._ensure_not_last_admin(user_id)

            updates: List[str] = []
            params: List[Any] = []

            if req.email is not None:
                updates.append("email = ?")
                params.append(req.email)
            if req.full_name is not None:
                updates.append("full_name = ?")
                params.append(req.full_name)
            if req.role is not None:
                updates.append("role = ?")
                params.append(req.role.value)
            if req.status is not None:
                updates.append("status = ?")
                params.append(req.status.value)
            if req.permitted_cameras is not None:
                updates.append("permitted_cameras_json = ?")
                params.append(json.dumps(req.permitted_cameras))

            if updates:
                params.append(user_id)
                with self._get_connection() as conn:
                    conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", params)
                    conn.commit()

            # Revoke existing sessions if permissions or status changed
            if req.role is not None or (req.status is not None and req.status == UserStatus.DISABLED):
                self.revoke_user_sessions(user_id, reason="Role or status updated")

        self.log_audit(
            event_type=AuditLogEvent.USER_UPDATE.value,
            actor_id=actor.user_id if actor else "admin",
            actor_username=actor.username if actor else "admin",
            resource_type="user",
            resource_id=user_id,
            action_outcome=AuditActionOutcome.SUCCESS,
            details=req.model_dump(exclude_unset=True),
        )
        return self.get_user(user_id)

    def change_password(self, user_id: str, current_pw: str, new_password: str) -> bool:
        """Allows an authenticated user to change their own password."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT password_hash, password_salt, username FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if not row or not self.verify_password(current_pw, row["password_hash"], row["password_salt"]):
                    raise ValueError("Current password verification failed.")

                h, s = self.hash_password(new_password)
                cursor.execute(
                    "UPDATE users SET password_hash = ?, password_salt = ? WHERE user_id = ?",
                    (h, s, user_id),
                )
                conn.commit()

            # Revoke all existing sessions for this user
            self.revoke_user_sessions(user_id, reason="Password changed")

        self.log_audit(
            event_type=AuditLogEvent.AUTH_PASSWORD_CHANGE.value,
            actor_id=user_id,
            actor_username=row["username"] if row else None,
            resource_type="user",
            resource_id=user_id,
            action_outcome=AuditActionOutcome.SUCCESS,
        )
        return True

    def reset_password(self, user_id: str, new_password: str, actor: Optional[UserResponse] = None) -> bool:
        """Administrative password reset."""
        with self._lock:
            h, s = self.hash_password(new_password)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"User '{user_id}' not found.")

                cursor.execute(
                    "UPDATE users SET password_hash = ?, password_salt = ? WHERE user_id = ?",
                    (h, s, user_id),
                )
                conn.commit()

            # Revoke existing sessions for safety
            self.revoke_user_sessions(user_id, reason="Administrative password reset")

        self.log_audit(
            event_type=AuditLogEvent.USER_PASSWORD_RESET.value,
            actor_id=actor.user_id if actor else "admin",
            actor_username=actor.username if actor else "admin",
            resource_type="user",
            resource_id=user_id,
            action_outcome=AuditActionOutcome.SUCCESS,
        )
        return True

    def delete_user(self, user_id: str, actor: Optional[UserResponse] = None) -> bool:
        """Deletes user account permanently, subject to last-administrator protection."""
        with self._lock:
            existing = self.get_user(user_id)
            if not existing:
                raise ValueError(f"User '{user_id}' not found.")

            if existing.role == UserRole.ADMIN:
                self._ensure_not_last_admin(user_id)

            self.revoke_user_sessions(user_id, reason="Account deleted")

            with self._get_connection() as conn:
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()

        self.log_audit(
            event_type=AuditLogEvent.USER_DELETE.value,
            actor_id=actor.user_id if actor else "admin",
            actor_username=actor.username if actor else "admin",
            resource_type="user",
            resource_id=user_id,
            action_outcome=AuditActionOutcome.SUCCESS,
            details={"deleted_username": existing.username},
        )
        return True

    def _ensure_not_last_admin(self, user_id_to_check: str) -> None:
        """Verifies that at least one other active Administrator remains."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM users
                WHERE role = ? AND status = ? AND user_id != ?
                """,
                (UserRole.ADMIN.value, UserStatus.ACTIVE.value, user_id_to_check),
            )
            count = cursor.fetchone()["count"]
            if count == 0:
                raise ValueError("Operation rejected: Cannot remove, disable, or demote the last active Administrator.")

    # --------------------------------------------------------------------------
    # RBAC Permission Checking Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def user_has_permission(user: UserResponse, permission: Permission) -> bool:
        """Returns True if the user role includes the requested permission."""
        perms = ROLE_PERMISSIONS.get(user.role, set())
        return permission in perms

    @staticmethod
    def user_can_access_camera(user: UserResponse, camera_id: str) -> bool:
        """
        Returns True if user has camera permission.
        Admins and users with permitted_cameras=None have unrestricted camera access.
        """
        if user.role == UserRole.ADMIN or user.permitted_cameras is None:
            return True
        return camera_id in user.permitted_cameras

    # --------------------------------------------------------------------------
    # Audit Logging System
    # --------------------------------------------------------------------------
    def log_audit(
        self,
        event_type: str,
        actor_id: Optional[str] = None,
        actor_username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action_outcome: AuditActionOutcome = AuditActionOutcome.SUCCESS,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """
        Appends an immutable audit log record to data/auth.db.
        Redacts sensitive fields (passwords, tokens).
        """
        log_id = f"aud_{uuid.uuid4().hex[:12]}"
        now = time.time()
        det = dict(details or {})

        # Redact any passwords or secret tokens from details
        for key in list(det.keys()):
            if any(s in key.lower() for s in ["password", "token", "secret", "salt"]):
                det[key] = "[REDACTED]"

        outcome_val = action_outcome.value if hasattr(action_outcome, "value") else str(action_outcome)
        det_json = json.dumps(det)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_logs (
                        log_id, event_type, timestamp, actor_id, actor_username,
                        resource_type, resource_id, action_outcome, ip_address,
                        correlation_id, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_id,
                        event_type,
                        now,
                        actor_id,
                        actor_username,
                        resource_type,
                        resource_id,
                        outcome_val,
                        ip_address,
                        correlation_id,
                        det_json,
                    ),
                )
                conn.commit()

        return AuditLogEntry(
            log_id=log_id,
            event_type=event_type,
            timestamp=now,
            actor_id=actor_id,
            actor_username=actor_username,
            resource_type=resource_type,
            resource_id=resource_id,
            action_outcome=action_outcome,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=det,
        )

    def list_audit_logs(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        outcome: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> AuditLogListResponse:
        """Lists audit log entries with multi-column filtering and pagination (newest first)."""
        page = max(1, page)
        page_size = max(1, min(200, page_size))
        offset = (page - 1) * page_size

        where_clauses: List[str] = []
        params: List[Any] = []

        if event_type:
            where_clauses.append("event_type = ?")
            params.append(event_type.strip())
        if actor:
            where_clauses.append("actor_username LIKE ?")
            params.append(f"%{actor.strip()}%")
        if outcome:
            where_clauses.append("action_outcome = ?")
            params.append(outcome.strip().upper())
        if start_time is not None:
            where_clauses.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            where_clauses.append("timestamp <= ?")
            params.append(float(end_time))

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) as count FROM audit_logs {where_str}", params)
                total = cursor.fetchone()["count"]

                query = f"""
                    SELECT * FROM audit_logs {where_str}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query, params + [page_size, offset])
                rows = cursor.fetchall()

                items: List[AuditLogEntry] = []
                for r in rows:
                    details = {}
                    if r["details_json"]:
                        try:
                            details = json.loads(r["details_json"])
                        except Exception:
                            details = {}

                    items.append(
                        AuditLogEntry(
                            log_id=r["log_id"],
                            event_type=r["event_type"],
                            timestamp=r["timestamp"],
                            actor_id=r["actor_id"],
                            actor_username=r["actor_username"],
                            resource_type=r["resource_type"],
                            resource_id=r["resource_id"],
                            action_outcome=AuditActionOutcome(r["action_outcome"]),
                            ip_address=r["ip_address"],
                            correlation_id=r["correlation_id"],
                            details=details,
                        )
                    )

                total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

                return AuditLogListResponse(
                    items=items,
                    total=total,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                )

    def reset(self) -> None:
        """Clears all records for test isolation and re-seeds admin."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM sessions")
                conn.execute("DELETE FROM audit_logs")
                conn.execute("DELETE FROM users")
                conn.commit()
            with self._rate_limit_lock:
                self._failed_attempts.clear()
            self._bootstrap_admin_if_empty()

    @staticmethod
    def _row_to_user_model(row: sqlite3.Row) -> UserResponse:
        """Constructs UserResponse with computed permissions."""
        role_enum = UserRole(row["role"])
        perms = [p.value for p in ROLE_PERMISSIONS.get(role_enum, set())]
        cam_list = None
        if row["permitted_cameras_json"]:
            try:
                cam_list = json.loads(row["permitted_cameras_json"])
            except Exception:
                cam_list = None

        return UserResponse(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            role=role_enum,
            status=UserStatus(row["status"]),
            permitted_cameras=cam_list,
            created_at=row["created_at"],
            last_login=row["last_login"],
            permissions=perms,
        )


_auth_service_instance: Optional[AuthService] = None
_auth_lock = threading.Lock()


def get_auth_service() -> AuthService:
    """Returns the singleton instance of AuthService."""
    global _auth_service_instance
    if _auth_service_instance is None:
        with _auth_lock:
            if _auth_service_instance is None:
                _auth_service_instance = AuthService()
    return _auth_service_instance
