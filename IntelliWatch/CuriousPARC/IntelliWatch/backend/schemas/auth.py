"""
backend/schemas/auth.py
Step 23 — Authentication, Role-Based Access Control, Audit Logs & Administration Schemas.

Defines schemas for:
- User Roles and Permission Mappings
- User Lifecycle and Account Management
- Token and Session Contracts
- Audit Logging and Security Monitoring
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator


class UserRole(str, Enum):
    """Hierarchical system roles."""
    ADMIN = "ADMIN"
    SAFETY_MANAGER = "SAFETY_MANAGER"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


class UserStatus(str, Enum):
    """Account activation status."""
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class Permission(str, Enum):
    """Granular system permissions."""
    USERS_MANAGE = "users:manage"
    SYSTEM_CONFIG = "system:config"
    AUDIT_VIEW = "audit:view"
    CAMERAS_MANAGE = "cameras:manage"
    CAMERAS_VIEW = "cameras:view"
    ALERTS_VIEW = "alerts:view"
    ALERTS_ACKNOWLEDGE = "alerts:acknowledge"
    ALERTS_RESOLVE = "alerts:resolve"
    ALERTS_DISMISS = "alerts:dismiss"
    EVIDENCE_VIEW = "evidence:view"
    REPORTS_VIEW = "reports:view"


# Role to Permissions Mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.USERS_MANAGE,
        Permission.SYSTEM_CONFIG,
        Permission.AUDIT_VIEW,
        Permission.CAMERAS_MANAGE,
        Permission.CAMERAS_VIEW,
        Permission.ALERTS_VIEW,
        Permission.ALERTS_ACKNOWLEDGE,
        Permission.ALERTS_RESOLVE,
        Permission.ALERTS_DISMISS,
        Permission.EVIDENCE_VIEW,
        Permission.REPORTS_VIEW,
    },
    UserRole.SAFETY_MANAGER: {
        Permission.CAMERAS_VIEW,
        Permission.ALERTS_VIEW,
        Permission.ALERTS_ACKNOWLEDGE,
        Permission.ALERTS_RESOLVE,
        Permission.ALERTS_DISMISS,
        Permission.EVIDENCE_VIEW,
        Permission.REPORTS_VIEW,
    },
    UserRole.OPERATOR: {
        Permission.CAMERAS_VIEW,
        Permission.ALERTS_VIEW,
        Permission.ALERTS_ACKNOWLEDGE,
        Permission.ALERTS_RESOLVE,
        Permission.EVIDENCE_VIEW,
    },
    UserRole.VIEWER: {
        Permission.CAMERAS_VIEW,
        Permission.ALERTS_VIEW,
        Permission.EVIDENCE_VIEW,
        Permission.REPORTS_VIEW,
    },
}


class UserBase(BaseModel):
    """Base user profile attributes."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: Optional[str] = Field(default=None, max_length=120, description="User email address")
    full_name: Optional[str] = Field(default=None, max_length=120, description="Full descriptive name")
    role: UserRole = Field(default=UserRole.OPERATOR, description="Assigned authorization role")
    permitted_cameras: Optional[List[str]] = Field(
        default=None,
        description="List of permitted camera IDs. None signifies unrestricted access to all cameras.",
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if not v.isalnum() and not all(c.isalnum() or c in "_-." for c in v):
            raise ValueError("Username may only contain alphanumeric characters, hyphens, underscores, and dots.")
        return v


class UserCreateRequest(UserBase):
    """Request model for administrative user creation."""
    password: str = Field(..., min_length=8, max_length=128, description="Initial account password")


class UserUpdateRequest(BaseModel):
    """Request model for updating an existing user account."""
    email: Optional[str] = Field(default=None, max_length=120)
    full_name: Optional[str] = Field(default=None, max_length=120)
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    permitted_cameras: Optional[List[str]] = None


class UserResponse(BaseModel):
    """Public user model returned in API responses (never includes password hashes or salts)."""
    user_id: str = Field(..., description="Unique immutable user identifier")
    username: str = Field(..., description="Login username")
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = Field(..., description="User role")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="Account status")
    permitted_cameras: Optional[List[str]] = Field(
        default=None,
        description="Allowed camera IDs, or None for all cameras",
    )
    created_at: float = Field(..., description="Account creation epoch timestamp")
    last_login: Optional[float] = Field(default=None, description="Most recent login timestamp")
    permissions: List[str] = Field(default_factory=list, description="Computed granular permission keys")


class UserListResponse(BaseModel):
    """List of system users."""
    items: List[UserResponse] = Field(default_factory=list)
    total: int = Field(default=0)

    @property
    def users(self) -> List[UserResponse]:
        """Convenience property for backwards-compatible access."""
        return self.items


class LoginRequest(BaseModel):
    """Login credentials payload."""
    username: str = Field(..., description="Account username")
    password: str = Field(..., description="Account password")


class LoginResponse(BaseModel):
    """Successful authentication response containing token and identity."""
    access_token: str = Field(..., description="Bearer session token")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="Token validity in seconds")
    user: UserResponse = Field(..., description="Authenticated user profile")


class ChangePasswordRequest(BaseModel):
    """Password self-service change payload."""
    current_password: str = Field(..., min_length=1, description="Existing password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


class ResetPasswordRequest(BaseModel):
    """Administrative password reset payload."""
    new_password: str = Field(..., min_length=8, max_length=128, description="Reset password")


class AuditActionOutcome(str, Enum):
    """Outcome status of an audited event."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"


class AuditLogEvent(str, Enum):
    """Controlled taxonomy of audited actions."""
    AUTH_LOGIN_SUCCESS = "AUTH_LOGIN_SUCCESS"
    AUTH_LOGIN_FAILURE = "AUTH_LOGIN_FAILURE"
    AUTH_LOGOUT = "AUTH_LOGOUT"
    AUTH_PASSWORD_CHANGE = "AUTH_PASSWORD_CHANGE"
    AUTH_SESSION_REVOKED = "AUTH_SESSION_REVOKED"
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DISABLE = "USER_DISABLE"
    USER_DELETE = "USER_DELETE"
    USER_PASSWORD_RESET = "USER_PASSWORD_RESET"
    CAMERA_CREATE = "CAMERA_CREATE"
    CAMERA_UPDATE = "CAMERA_UPDATE"
    CAMERA_DELETE = "CAMERA_DELETE"
    ALERT_ACKNOWLEDGE = "ALERT_ACKNOWLEDGE"
    ALERT_RESOLVE = "ALERT_RESOLVE"
    ALERT_DISMISS = "ALERT_DISMISS"
    EVIDENCE_ACCESS = "EVIDENCE_ACCESS"
    CONFIG_UPDATE = "CONFIG_UPDATE"
    ACCESS_DENIED = "ACCESS_DENIED"


class AuditLogEntry(BaseModel):
    """Structured audit trail log record."""
    log_id: str = Field(..., description="Unique audit event ID")
    event_type: str = Field(..., description="Action category")
    timestamp: float = Field(..., description="Event timestamp (epoch seconds)")
    actor_id: Optional[str] = Field(default=None, description="User ID of acting subject")
    actor_username: Optional[str] = Field(default=None, description="Username of acting subject")
    resource_type: Optional[str] = Field(default=None, description="Target entity type (e.g. alert, camera, user)")
    resource_id: Optional[str] = Field(default=None, description="Target entity ID")
    action_outcome: AuditActionOutcome = Field(default=AuditActionOutcome.SUCCESS, description="Outcome")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation identifier")
    details: Dict[str, Any] = Field(default_factory=dict, description="Redacted operational details")


class AuditLogListResponse(BaseModel):
    """Paginated audit trail response."""
    items: List[AuditLogEntry] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=50)
    total_pages: int = Field(default=1)
