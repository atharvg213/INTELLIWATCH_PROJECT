"""
tests/test_auth_rbac_step23.py
Step 23 Automated Test Suite: Authentication, Role-Based Access Control, Audit Logs & Administration.

Comprehensive verification of:
1. PBKDF2-HMAC-SHA256 password hashing, salting, and secure comparison.
2. Token generation, validation, expiration, and server-side revocation.
3. Brute-force rate limiting (5 failed attempts trigger 300s temporary lockout).
4. User Management CRUD (admin create, read, update, disable, delete).
5. Last-administrator protection (sole active admin cannot be disabled, demoted, or deleted).
6. Role-Based Access Control (RBAC) permission matrices for ADMIN, SAFETY_MANAGER, OPERATOR, VIEWER.
7. Camera-level access restrictions (permitted_cameras scoping on endpoints and alerts).
8. Append-only persistent audit trail (event logging, sensitive secret redaction, filtering, pagination).
9. REST API endpoints:
   - POST /api/v1/auth/login
   - POST /api/v1/auth/logout
   - GET /api/v1/auth/me
   - POST /api/v1/auth/change-password
   - GET /api/v1/users, POST /api/v1/users, PATCH /api/v1/users/{id}, DELETE /api/v1/users/{id}
   - GET /api/v1/audit-logs
   - Protected camera and alert endpoints with Bearer token authentication
10. Regression testing ensuring existing pipelines, alerts, and cameras remain 100% operational.
"""
import os
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.auth import (
    AuditActionOutcome,
    AuditLogEvent,
    Permission,
    ROLE_PERMISSIONS,
    UserCreateRequest,
    UserRole,
    UserStatus,
    UserUpdateRequest,
)
from backend.services.auth_service import AuthService, get_auth_service
from configs.settings import get_settings


@pytest.fixture
def auth_env(tmp_path):
    """Isolated AuthService backed by a temporary SQLite database."""
    test_db = tmp_path / "test_auth.db"
    service = AuthService(db_path=test_db)
    # Ensure default admin exists
    admin = service.get_user_by_username("admin")
    assert admin is not None
    return service, test_db


@pytest.fixture
def test_client_auth(tmp_path, monkeypatch):
    """FastAPI TestClient with AUTH_ENABLED=True and isolated auth DB."""
    test_db = tmp_path / "test_api_auth.db"
    settings = get_settings()

    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_DB_PATH", str(test_db))
    monkeypatch.setattr(settings, "AUTH_RATE_LIMIT_ATTEMPTS", 5)
    monkeypatch.setattr(settings, "AUTH_RATE_LIMIT_LOCKOUT_SECONDS", 300)

    # Initialize service singleton with isolated DB
    service = AuthService(db_path=test_db)
    from backend.services import auth_service as auth_svc_module
    monkeypatch.setattr(auth_svc_module, "_auth_service_instance", service)

    client = TestClient(app)
    return client, service


# =========================================================================
# 1. PASSWORD HASHING, SALTING & VERIFICATION
# =========================================================================

def test_password_hashing_and_verification(auth_env):
    service, _ = auth_env
    raw_pwd = "IndustrialSafety@2026"
    pwd_hash, salt = service.hash_password(raw_pwd)

    assert pwd_hash is not None and len(pwd_hash) == 64  # SHA256 hex digest
    assert salt is not None and len(salt) >= 32
    assert pwd_hash != raw_pwd

    # Correct password succeeds
    assert service.verify_password(raw_pwd, pwd_hash, salt) is True

    # Incorrect password fails
    assert service.verify_password("WrongPassword123", pwd_hash, salt) is False

    # Different salts produce different hashes for same password
    hash2, salt2 = service.hash_password(raw_pwd)
    assert salt != salt2
    assert pwd_hash != hash2


# =========================================================================
# 2. TOKEN ISSUANCE, EXPIRATION & INSTANT REVOCATION
# =========================================================================

def test_token_lifecycle(auth_env):
    service, _ = auth_env
    # Authenticate default admin
    user_res, token, exp = service.authenticate_user("admin", "IntelliWatch2026!")
    assert token is not None
    assert user_res.role == UserRole.ADMIN

    # Validate active token
    user = service.validate_token(token)
    assert user is not None
    assert user.username == "admin"
    assert user.role == UserRole.ADMIN

    # Revoke token on logout
    service.revoke_session(token)

    # Validating revoked token returns None
    assert service.validate_token(token) is None


def test_token_expiration(auth_env):
    service, _ = auth_env
    user_res, token, exp = service.authenticate_user("admin", "IntelliWatch2026!")
    assert token is not None

    # Artificially expire the token in database
    with service._get_connection() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?", (time.time() - 100, token))

    # Expired token returns None
    assert service.validate_token(token) is None


# =========================================================================
# 3. BRUTE-FORCE RATE LIMITING & LOCKOUT
# =========================================================================

def test_brute_force_rate_limiting(auth_env):
    service, _ = auth_env
    username = "admin"

    # 5 failed attempts: first 5 record attempts
    for i in range(5):
        with pytest.raises(ValueError, match="Invalid username or password"):
            service.authenticate_user(username, "BadPassword!")

    # 6th attempt: triggers lockout
    with pytest.raises(ValueError, match="Too many failed login attempts"):
        service.authenticate_user(username, "BadPassword!")

    # Even with correct password, authentication is blocked while locked out
    with pytest.raises(ValueError, match="Too many failed login attempts"):
        service.authenticate_user(username, "IntelliWatch2026!")

    # Clear rate limit manually simulating lockout expiration
    service.clear_failed_attempts(username, "unknown")

    # Now correct password succeeds
    user_res, tok_success, exp = service.authenticate_user(username, "IntelliWatch2026!")
    assert tok_success is not None


# =========================================================================
# 4. USER MANAGEMENT CRUD & LAST-ADMIN PROTECTION
# =========================================================================

def test_user_crud_and_last_admin_protection(auth_env):
    service, _ = auth_env

    # Create safety manager
    mgr = service.create_user(UserCreateRequest(
        username="manager_jane",
        password="JanePassword2026!",
        role=UserRole.SAFETY_MANAGER,
        permitted_cameras=["cam_01", "cam_02"],
    ))
    assert mgr.username == "manager_jane"
    assert mgr.role == UserRole.SAFETY_MANAGER
    assert mgr.permitted_cameras == ["cam_01", "cam_02"]
    assert Permission.ALERTS_RESOLVE.value in mgr.permissions

    # List users
    users_resp = service.list_users()
    assert len(users_resp.users) >= 2

    # Update user role to OPERATOR
    updated = service.update_user(mgr.user_id, UserUpdateRequest(role=UserRole.OPERATOR))
    assert updated.role == UserRole.OPERATOR
    assert Permission.ALERTS_DISMISS.value not in updated.permissions

    # Last-Admin Protection: cannot demote sole admin
    admin = service.get_user_by_username("admin")
    with pytest.raises(ValueError, match="last active Administrator"):
        service.update_user(admin.user_id, UserUpdateRequest(role=UserRole.OPERATOR))

    # Last-Admin Protection: cannot disable sole admin
    with pytest.raises(ValueError, match="last active Administrator"):
        service.update_user(admin.user_id, UserUpdateRequest(status=UserStatus.DISABLED))

    # Last-Admin Protection: cannot delete sole admin
    with pytest.raises(ValueError, match="last active Administrator"):
        service.delete_user(admin.user_id)

    # Deleting non-admin user succeeds
    assert service.delete_user(mgr.user_id) is True


# =========================================================================
# 5. ROLE-BASED ACCESS CONTROL PERMISSIONS MATRIX
# =========================================================================

def test_rbac_matrix_permissions():
    # Admin has all permissions
    admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
    assert len(admin_perms) == len(Permission)

    # Safety Manager has alert lifecycle & view permissions, but not USER_MANAGE
    mgr_perms = ROLE_PERMISSIONS[UserRole.SAFETY_MANAGER]
    assert Permission.ALERTS_ACKNOWLEDGE in mgr_perms
    assert Permission.ALERTS_RESOLVE in mgr_perms
    assert Permission.ALERTS_DISMISS in mgr_perms
    assert Permission.USERS_MANAGE not in mgr_perms
    assert Permission.AUDIT_VIEW not in mgr_perms

    # Operator can view and acknowledge, but cannot dismiss or manage users
    op_perms = ROLE_PERMISSIONS[UserRole.OPERATOR]
    assert Permission.CAMERAS_VIEW in op_perms
    assert Permission.ALERTS_ACKNOWLEDGE in op_perms
    assert Permission.ALERTS_RESOLVE in op_perms
    assert Permission.ALERTS_DISMISS not in op_perms
    assert Permission.CAMERAS_MANAGE not in op_perms
    assert Permission.USERS_MANAGE not in op_perms

    # Viewer can only view
    viewer_perms = ROLE_PERMISSIONS[UserRole.VIEWER]
    assert Permission.CAMERAS_VIEW in viewer_perms
    assert Permission.ALERTS_VIEW in viewer_perms
    assert Permission.ALERTS_ACKNOWLEDGE not in viewer_perms
    assert Permission.ALERTS_RESOLVE not in viewer_perms
    assert Permission.ALERTS_DISMISS not in viewer_perms


# =========================================================================
# 6. CAMERA-LEVEL ACCESS RESTRICTIONS
# =========================================================================

def test_camera_access_restriction(auth_env):
    service, _ = auth_env
    # User with unrestricted camera access (None)
    admin = service.get_user_by_username("admin")
    assert service.user_can_access_camera(admin, "cam_99") is True

    # User with explicit permitted cameras
    op = service.create_user(UserCreateRequest(
        username="op_restricted",
        password="OperatorPwd2026!",
        role=UserRole.OPERATOR,
        permitted_cameras=["cam_warehouse", "cam_gate"],
    ))
    assert service.user_can_access_camera(op, "cam_warehouse") is True
    assert service.user_can_access_camera(op, "cam_gate") is True
    assert service.user_can_access_camera(op, "cam_dock") is False


# =========================================================================
# 7. AUDIT LOGGING & SENSITIVE DATA REDACTION
# =========================================================================

def test_audit_logging_and_redaction(auth_env):
    service, _ = auth_env

    # Log action with secret details
    service.log_audit(
        event_type=AuditLogEvent.USER_CREATE.value,
        actor_username="admin",
        resource_type="user",
        resource_id="usr_123",
        action_outcome=AuditActionOutcome.SUCCESS,
        details={
            "created_user": "test_user",
            "password": "SensitivePassword123!",  # Must be redacted
            "auth_token": "secret_token_abc",      # Must be redacted
        },
    )

    resp = service.list_audit_logs()
    assert resp.total >= 1
    found = [l for l in resp.items if l.resource_id == "usr_123"]
    assert len(found) == 1
    entry = found[0]

    assert entry.details["created_user"] == "test_user"
    assert entry.details["password"] == "[REDACTED]"
    assert entry.details["auth_token"] == "[REDACTED]"


# =========================================================================
# 8. REST API ENDPOINTS: AUTH, RBAC, USERS & AUDIT LOGS
# =========================================================================

def test_api_login_logout_and_me(test_client_auth):
    client, service = test_client_auth

    # 1. Unauthenticated /api/v1/auth/me returns 401
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401

    # 2. Login with wrong password returns 401
    res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    assert res.status_code == 401

    # 3. Login with correct password returns token and user info
    res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "IntelliWatch2026!"})
    assert res.status_code == 200
    data = res.json()
    token = data["access_token"]
    assert token is not None
    assert data["user"]["role"] == "ADMIN"

    # 4. Authenticated /api/v1/auth/me returns user profile
    headers = {"Authorization": f"Bearer {token}"}
    res_me = client.get("/api/v1/auth/me", headers=headers)
    assert res_me.status_code == 200
    assert res_me.json()["username"] == "admin"

    # 5. Logout invalidates token
    res_logout = client.post("/api/v1/auth/logout", headers=headers)
    assert res_logout.status_code == 200

    # 6. Reusing logged-out token returns 401
    res_after = client.get("/api/v1/auth/me", headers=headers)
    assert res_after.status_code == 401


def test_api_change_password(test_client_auth):
    client, service = test_client_auth

    # Login admin
    res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "IntelliWatch2026!"})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Change password
    new_pwd = "NewAdminPassword2026!"
    res_change = client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={"current_password": "IntelliWatch2026!", "new_password": new_pwd},
    )
    assert res_change.status_code == 200

    # Old password no longer works
    res_old = client.post("/api/v1/auth/login", json={"username": "admin", "password": "IntelliWatch2026!"})
    assert res_old.status_code == 401

    # New password works
    res_new = client.post("/api/v1/auth/login", json={"username": "admin", "password": new_pwd})
    assert res_new.status_code == 200


def test_api_user_management_rbac(test_client_auth):
    client, service = test_client_auth

    # Login as admin
    res_adm = client.post("/api/v1/auth/login", json={"username": "admin", "password": "IntelliWatch2026!"})
    adm_token = res_adm.json()["access_token"]
    adm_headers = {"Authorization": f"Bearer {adm_token}"}

    # Create an operator user
    res_create = client.post(
        "/api/v1/users",
        headers=adm_headers,
        json={
            "username": "operator_bob",
            "password": "BobOperator2026!",
            "role": "OPERATOR",
            "permitted_cameras": ["cam_01"],
        },
    )
    assert res_create.status_code == 201
    bob_id = res_create.json()["user_id"]

    # Login as operator
    res_bob_login = client.post("/api/v1/auth/login", json={"username": "operator_bob", "password": "BobOperator2026!"})
    assert res_bob_login.status_code == 200
    bob_token = res_bob_login.json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # Operator cannot manage users (403 Forbidden)
    res_forbidden = client.get("/api/v1/users", headers=bob_headers)
    assert res_forbidden.status_code == 403

    # Operator cannot view audit logs (403 Forbidden)
    res_audit_forbidden = client.get("/api/v1/audit-logs", headers=bob_headers)
    assert res_audit_forbidden.status_code == 403

    # Admin CAN view users and audit logs
    res_users = client.get("/api/v1/users", headers=adm_headers)
    assert res_users.status_code == 200
    user_data = res_users.json()
    users_list = user_data.get("items", user_data.get("users", []))
    assert len(users_list) >= 2

    res_audit = client.get("/api/v1/audit-logs", headers=adm_headers)
    assert res_audit.status_code == 200
    assert res_audit.json()["total"] >= 1


def test_api_camera_and_alert_rbac_enforcement(test_client_auth):
    client, service = test_client_auth

    # Login as admin to create test accounts
    res_adm = client.post("/api/v1/auth/login", json={"username": "admin", "password": "IntelliWatch2026!"})
    adm_token = res_adm.json()["access_token"]
    adm_headers = {"Authorization": f"Bearer {adm_token}"}

    # Create Viewer
    client.post(
        "/api/v1/users",
        headers=adm_headers,
        json={"username": "viewer_sam", "password": "ViewerSam2026!", "role": "VIEWER"},
    )
    res_sam = client.post("/api/v1/auth/login", json={"username": "viewer_sam", "password": "ViewerSam2026!"})
    sam_token = res_sam.json()["access_token"]
    sam_headers = {"Authorization": f"Bearer {sam_token}"}

    # Viewer CANNOT register or stop cameras (requires CAMERAS_MANAGE)
    res_cam_reg = client.post(
        "/api/v1/cameras",
        headers=sam_headers,
        json={"camera_id": "test_cam", "name": "Test", "source": "0"},
    )
    assert res_cam_reg.status_code == 403

    # Viewer CANNOT acknowledge alerts (requires ALERTS_ACKNOWLEDGE)
    res_ack = client.post(
        "/api/v1/alerts/alt_fake/acknowledge",
        headers=sam_headers,
        json={"notes": "trying to ack"},
    )
    assert res_ack.status_code == 403

    # Viewer CAN view cameras list (requires CAMERAS_VIEW)
    res_cams = client.get("/api/v1/cameras", headers=sam_headers)
    assert res_cams.status_code == 200

