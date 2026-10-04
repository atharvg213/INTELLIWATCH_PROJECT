"""
backend/api/auth_routes.py
Step 23 — Authentication, User Management, and Audit Logging REST Endpoints.

Provides:
- Login / Logout / Session endpoints
- Current User Profile (/me)
- Self-service Password Change
- Administrative User CRUD & Password Reset
- Security Audit Log queries with filtering and pagination
"""
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from backend.api.deps import get_current_user, require_permission, require_roles
from backend.schemas.auth import (
    AuditActionOutcome,
    AuditLogEntry,
    AuditLogEvent,
    AuditLogListResponse,
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    Permission,
    ResetPasswordRequest,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
    UserRole,
    UserUpdateRequest,
)
from backend.services.auth_service import get_auth_service

auth_router = APIRouter()


# --------------------------------------------------------------------------
# Authentication Endpoints
# --------------------------------------------------------------------------
@auth_router.post(
    "/api/v1/auth/login",
    response_model=LoginResponse,
    tags=["Authentication & Security"],
    summary="Authenticate user and issue bearer session token",
)
async def login(req: LoginRequest, request: Request) -> LoginResponse:
    """
    Authenticates user with username and password.
    Includes rate-limiting against brute force attacks and constant-time comparison.
    """
    client_ip = request.client.host if request.client else "unknown"
    auth_service = get_auth_service()

    try:
        user, token, expires_in = auth_service.authenticate_user(
            username=req.username,
            password=req.password,
            ip_address=client_ip,
        )
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in,
            user=user,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@auth_router.post(
    "/api/v1/auth/logout",
    tags=["Authentication & Security"],
    summary="Invalidate active session token",
)
async def logout(
    request: Request,
    user: UserResponse = Depends(get_current_user),
):
    """
    Revokes the current session token so it cannot be reused.
    """
    auth_service = get_auth_service()
    auth_hdr = request.headers.get("Authorization")
    if auth_hdr and auth_hdr.startswith("Bearer "):
        token = auth_hdr.split(" ", 1)[1].strip()
        auth_service.revoke_session(token, actor_username=user.username)
    return {"status": "ok", "message": "Successfully logged out."}


@auth_router.get(
    "/api/v1/auth/me",
    response_model=UserResponse,
    tags=["Authentication & Security"],
    summary="Get authenticated user profile and permissions",
)
async def get_my_profile(user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Returns current active user identity, role, permissions, and permitted cameras.
    """
    return user


@auth_router.post(
    "/api/v1/auth/change-password",
    tags=["Authentication & Security"],
    summary="Change own password",
)
async def change_password(
    req: ChangePasswordRequest,
    user: UserResponse = Depends(get_current_user),
):
    """
    Allows authenticated user to update their account password after current password verification.
    Revokes existing sessions.
    """
    auth_service = get_auth_service()
    try:
        auth_service.change_password(
            user_id=user.user_id,
            current_pw=req.current_password,
            new_password=req.new_password,
        )
        return {"status": "ok", "message": "Password changed successfully. Please log in again."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# --------------------------------------------------------------------------
# User Management Endpoints (Administrator only)
# --------------------------------------------------------------------------
@auth_router.get(
    "/api/v1/users",
    response_model=UserListResponse,
    tags=["User Administration"],
    summary="List all system user accounts",
)
async def list_users(
    user: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
) -> UserListResponse:
    """Lists all user accounts (Admin only). Password hashes are never exposed."""
    return get_auth_service().list_users()


@auth_router.post(
    "/api/v1/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["User Administration"],
    summary="Create a new user account",
)
async def create_user(
    req: UserCreateRequest,
    admin: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
) -> UserResponse:
    """Creates a new user account with assigned role and camera permissions."""
    try:
        return get_auth_service().create_user(req=req, actor=admin)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@auth_router.get(
    "/api/v1/users/{user_id}",
    response_model=UserResponse,
    tags=["User Administration"],
    summary="Retrieve user profile by ID",
)
async def get_user_detail(
    user_id: str,
    admin: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
) -> UserResponse:
    """Gets user profile by ID."""
    u = get_auth_service().get_user(user_id)
    if not u:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return u


@auth_router.put(
    "/api/v1/users/{user_id}",
    response_model=UserResponse,
    tags=["User Administration"],
    summary="Update user role, status, or camera permissions",
)
async def update_user(
    user_id: str,
    req: UserUpdateRequest,
    admin: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
) -> UserResponse:
    """Updates user attributes, role, or activation status. Enforces last-administrator protection."""
    try:
        return get_auth_service().update_user(user_id=user_id, req=req, actor=admin)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@auth_router.post(
    "/api/v1/users/{user_id}/reset-password",
    tags=["User Administration"],
    summary="Administrative password reset",
)
async def reset_user_password(
    user_id: str,
    req: ResetPasswordRequest,
    admin: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
):
    """Resets user password administratively and revokes active sessions."""
    try:
        get_auth_service().reset_password(user_id=user_id, new_password=req.new_password, actor=admin)
        return {"status": "ok", "message": f"Password reset for user '{user_id}'."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@auth_router.delete(
    "/api/v1/users/{user_id}",
    tags=["User Administration"],
    summary="Delete a user account",
)
async def delete_user(
    user_id: str,
    admin: UserResponse = Depends(require_permission(Permission.USERS_MANAGE)),
):
    """Permanently removes user account. Enforces last-administrator protection."""
    try:
        get_auth_service().delete_user(user_id=user_id, actor=admin)
        return {"status": "ok", "message": f"User '{user_id}' deleted."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# --------------------------------------------------------------------------
# Audit Logging Endpoints (Administrator only)
# --------------------------------------------------------------------------
@auth_router.get(
    "/api/v1/audit-logs",
    response_model=AuditLogListResponse,
    tags=["Security & Audit Logs"],
    summary="Query and filter security audit log entries",
)
async def list_audit_logs(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    actor: Optional[str] = Query(None, description="Filter by actor username"),
    outcome: Optional[str] = Query(None, description="Filter by outcome (SUCCESS, FAILURE, DENIED)"),
    start_time: Optional[float] = Query(None, description="Start epoch timestamp"),
    end_time: Optional[float] = Query(None, description="End epoch timestamp"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    admin: UserResponse = Depends(require_permission(Permission.AUDIT_VIEW)),
) -> AuditLogListResponse:
    """
    Retrieves filtered audit logs with actor, timestamp, outcome, and redacted details.
    Restricted strictly to administrators.
    """
    return get_auth_service().list_audit_logs(
        event_type=event_type,
        actor=actor,
        outcome=outcome,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )
