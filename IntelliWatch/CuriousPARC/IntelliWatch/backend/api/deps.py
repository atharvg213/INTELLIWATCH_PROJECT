"""
backend/api/deps.py
Step 23 — Authentication and Authorization Dependencies for FastAPI Routes.

Provides:
- Token extraction from Authorization Bearer header or fallback query param
- RBAC permission verification callables
- Camera-level access control enforcement
- Audit logging for unauthorized access attempts
"""
from typing import List, Optional
from fastapi import Depends, Header, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.schemas.auth import (
    AuditActionOutcome,
    AuditLogEvent,
    Permission,
    UserResponse,
    UserRole,
    UserStatus,
)
from backend.services.auth_service import AuthService, get_auth_service
from configs.settings import get_settings

security = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token_query: Optional[str] = Query(None, alias="token", description="Bearer token query param for <img> streams"),
) -> UserResponse:
    """
    Validates authentication token from Bearer header or query parameter.
    If AUTH_ENABLED is False (e.g. testing / dev bypass), returns default administrative profile.
    """
    settings = get_settings()
    if not settings.AUTH_ENABLED:
        # Development / test bypass mode
        return UserResponse(
            user_id="usr_admin_default",
            username="admin",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=0.0,
            permissions=[p.value for p in Permission],
        )

    token = None
    if credentials:
        token = credentials.credentials
    elif token_query:
        token = token_query

    auth_service = get_auth_service()
    if not token:
        # Check Authorization header directly if not caught by bearer
        auth_hdr = request.headers.get("Authorization")
        if auth_hdr and auth_hdr.startswith("Bearer "):
            token = auth_hdr.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_service.validate_token(token)
    if not user:
        client_ip = request.client.host if request.client else "unknown"
        auth_service.log_audit(
            event_type=AuditLogEvent.ACCESS_DENIED.value,
            action_outcome=AuditActionOutcome.DENIED,
            ip_address=client_ip,
            details={"path": request.url.path, "reason": "Invalid, expired, or revoked token"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_permission(permission: Permission):
    """Factory creating an endpoint dependency checking for a specific permission."""

    def _dependency(
        request: Request,
        user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        settings = get_settings()
        if not settings.AUTH_ENABLED:
            return user

        if not AuthService.user_has_permission(user, permission):
            auth_service = get_auth_service()
            client_ip = request.client.host if request.client else "unknown"
            auth_service.log_audit(
                event_type=AuditLogEvent.ACCESS_DENIED.value,
                actor_id=user.user_id,
                actor_username=user.username,
                action_outcome=AuditActionOutcome.DENIED,
                ip_address=client_ip,
                details={
                    "path": request.url.path,
                    "required_permission": permission.value,
                    "user_role": user.role.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Action requires '{permission.value}' permission",
            )
        return user

    return _dependency


def require_roles(allowed_roles: List[UserRole]):
    """Factory checking that the user holds one of the specified roles."""

    def _dependency(
        request: Request,
        user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        settings = get_settings()
        if not settings.AUTH_ENABLED:
            return user

        if user.role not in allowed_roles:
            auth_service = get_auth_service()
            client_ip = request.client.host if request.client else "unknown"
            auth_service.log_audit(
                event_type=AuditLogEvent.ACCESS_DENIED.value,
                actor_id=user.user_id,
                actor_username=user.username,
                action_outcome=AuditActionOutcome.DENIED,
                ip_address=client_ip,
                details={
                    "path": request.url.path,
                    "required_roles": [r.value for r in allowed_roles],
                    "user_role": user.role.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Restricted to roles: {[r.value for r in allowed_roles]}",
            )
        return user

    return _dependency


def verify_camera_access(camera_id: str, user: UserResponse) -> None:
    """Verifies that user is authorized to view or control the specified camera."""
    settings = get_settings()
    if not settings.AUTH_ENABLED:
        return

    if not AuthService.user_can_access_camera(user, camera_id):
        get_auth_service().log_audit(
            event_type=AuditLogEvent.ACCESS_DENIED.value,
            actor_id=user.user_id,
            actor_username=user.username,
            resource_type="camera",
            resource_id=camera_id,
            action_outcome=AuditActionOutcome.DENIED,
            details={"reason": "Camera not in permitted_cameras list"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: You do not have permission to access camera '{camera_id}'",
        )
