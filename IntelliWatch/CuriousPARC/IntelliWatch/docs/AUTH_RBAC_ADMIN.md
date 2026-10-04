# Step 23 — Authentication, Role-Based Access Control, Audit Logs & Administration

## 1. Security Architecture & Threat Model

IntelliWatch implements a multi-tier defense-in-depth security model engineered for industrial control room and edge monitoring deployments. It secures APIs, camera streams, safety alerts, visual evidence snapshots, and configuration management against unauthorized access, privilege escalation, credential brute-forcing, and tampering.

```mermaid
flowchart TD
    subgraph Client ["Clients (Web Dashboard, CCTV Monitor, API Clients)"]
        Browser[Dashboard Browser / Mobile Tablet]
        APIClient[Automated Client / Integrator]
    end

    subgraph Gateway ["FastAPI Gateway & Security Layer"]
        AUTH_MW[Bearer Token Extraction & Validation]
        RATELIMIT[In-Memory IP & Account Rate Limiter]
        RBAC_DEP[Granular Permission & Role Verification]
        CAM_SCOPE[Camera-Level Permitted ID Scoping]
        AUDIT_DEP[Structured Append-Only Audit Logging]
    end

    subgraph Service ["Authentication & Identity Service"]
        HASHER[PBKDF2-HMAC-SHA256 (100k Iterations, 16B Salt)]
        SESSION_MGR[Cryptographic Session Token Store]
        USER_STORE[User Profile & Role Manager]
        LAST_ADMIN[Last Active Admin Deletion Guard]
    end

    subgraph Storage ["Encrypted & Isolated Persistence"]
        AUTH_DB[(SQLite: data/auth.db)]
        AUDIT_TBL[(auth_audit_log Table)]
        ALERT_DB[(data/alerts.db)]
        EVID_DIR[(data/output/alerts/evidence/)]
    end

    Browser & APIClient -->|Bearer Token / Login| RATELIMIT
    RATELIMIT --> AUTH_MW
    AUTH_MW --> SESSION_MGR
    SESSION_MGR <--> AUTH_DB
    AUTH_MW --> RBAC_DEP
    RBAC_DEP --> CAM_SCOPE
    RBAC_DEP & CAM_SCOPE -->|Log Sensitive Action| AUDIT_DEP
    AUDIT_DEP --> AUDIT_TBL
    CAM_SCOPE --> ALERT_DB
    CAM_SCOPE --> EVID_DIR
```

---

## 2. Authentication System

### 2.1 Password Hashing & Key Derivation
- **Algorithm**: `PBKDF2-HMAC-SHA256`
- **Iterations**: 100,000 iterations
- **Salt**: 16 cryptographically secure random bytes generated per user (`secrets.token_hex(16)`)
- **Storage Format**: `pbkdf2:sha256:100000:<salt_hex>:<hash_hex>`
- **Constant-Time Verification**: `hmac.compare_digest` to prevent timing analysis attacks.

### 2.2 Session & Token Lifecycle
- **Token Generation**: 32-byte cryptographically secure hexadecimal tokens (`secrets.token_urlsafe(32)`).
- **Session Expiration**: Configurable duration (default: 480 minutes / 8 hours via `AUTH_TOKEN_EXPIRE_MINUTES`).
- **Server-Side Revocation**: Sessions are tracked in `auth_sessions`. When a user logs out, is disabled, or has credentials reset, all associated active tokens are immediately invalidated server-side.
- **Client Transmission**: Passed via standard HTTP `Authorization: Bearer <token>` header or `X-Auth-Token` header.

### 2.3 Brute-Force Rate Limiting
- **Lockout Policy**: 5 consecutive failed login attempts trigger an automatic 300-second (5-minute) lockout for that username and client IP.
- **Counter Reset**: A successful authentication immediately clears the failed attempt counter.
- **Opaque Errors**: Invalid credentials and non-existent accounts return generic `401 Unauthorized` responses to prevent username enumeration.

---

## 3. Role-Based Access Control (RBAC) Matrix

IntelliWatch implements a hierarchical role model where every user is assigned an explicit role with granular backend permissions:

| Permission | Identifier | `ADMIN` | `SAFETY_MANAGER` | `OPERATOR` | `VIEWER` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Manage Users & Roles** | `users:manage` | **Yes** | No | No | No |
| **Manage Cameras & Feeds** | `cameras:manage` | **Yes** | No | No | No |
| **View Camera Streams** | `cameras:view` | **Yes** | **Yes** | **Yes** | **Yes** |
| **View Safety Alerts** | `alerts:view` | **Yes** | **Yes** | **Yes** | **Yes** |
| **Acknowledge Alerts** | `alerts:acknowledge` | **Yes** | **Yes** | **Yes** | No |
| **Resolve Safety Alerts** | `alerts:resolve` | **Yes** | **Yes** | No | No |
| **Dismiss Safety Alerts** | `alerts:dismiss` | **Yes** | **Yes** | No | No |
| **View Visual Evidence** | `evidence:view` | **Yes** | **Yes** | **Yes** | **Yes** |
| **View Audit Trail** | `audit:view` | **Yes** | No | No | No |
| **System Settings** | `system:settings` | **Yes** | No | No | No |

### 3.1 Camera-Level Scoping (`permitted_cameras`)
In addition to global roles, users can be assigned a granular whitelist of `permitted_cameras`:
- When `permitted_cameras` is empty (`[]` or `None`), the user has access to all configured cameras.
- When `permitted_cameras` is populated (e.g. `["cam_loading_bay_01", "cam_press_line_02"]`), access to camera feeds, alerts, and evidence snapshots is strictly restricted to those camera IDs.
- Attempts to query or act upon alerts from unauthorized cameras return `403 Forbidden` with audit logging of `ACCESS_DENIED`.

---

## 4. User Management & Administration

Administrators manage users through the backend API or the interactive User Management dashboard panel:

### 4.1 Administrative Capabilities
- **User Provisioning**: `POST /api/v1/auth/users` creates users with mandatory password validation (minimum 8 characters), role assignment, and optional camera scoping.
- **Role & Scope Modification**: `PUT /api/v1/auth/users/{user_id}` dynamically reassigns roles, status, and permitted camera lists.
- **Account Disabling**: Toggling `is_active=False` instantly invalidates all active sessions for that account.
- **Secure Password Reset**: Administrators can reset user passwords with instant session invalidation.
- **Last Active Administrator Protection**: The system strictly prevents disabling or deleting the last active `ADMIN` user, preventing accidental administrative lockout.

---

## 5. Audit Logging & Compliance Trail

All security-relevant actions are persistently recorded to `data/auth.db` in the append-only `auth_audit_log` table:

### 5.1 Event Taxonomy
- `AUTH_LOGIN_SUCCESS`, `AUTH_LOGIN_FAILURE`, `AUTH_LOGOUT`, `AUTH_PASSWORD_CHANGE`, `AUTH_SESSION_REVOKED`
- `USER_CREATE`, `USER_UPDATE`, `USER_DISABLE`, `USER_DELETE`, `USER_PASSWORD_RESET`
- `CAMERA_CREATE`, `CAMERA_UPDATE`, `CAMERA_DELETE`
- `ALERT_ACKNOWLEDGE`, `ALERT_RESOLVE`, `ALERT_DISMISS`
- `EVIDENCE_ACCESS`
- `CONFIG_UPDATE`
- `ACCESS_DENIED`

### 5.2 Structured Schema
Each record contains:
- `log_id`: Unique cryptographic identifier (`audit_...`)
- `event_type`: Event category from controlled taxonomy
- `timestamp`: High-precision epoch timestamp
- `actor_id` & `actor_username`: Authenticated user identity
- `resource_type` & `resource_id`: Target entity (camera ID, alert ID, user ID)
- `action_outcome`: `SUCCESS`, `FAILURE`, or `DENIED`
- `ip_address`: Originating client IP
- `correlation_id`: Trace identifier across distributed logs
- `details`: Redacted JSON metadata (passwords, salts, tokens, and raw visual pixels are strictly scrubbed)

---

## 6. Dashboard Integration

The IntelliWatch web dashboard integrates authentication seamlessly:
- **Session Header**: Displays logged-in username, role badge (`ADMIN`, `SAFETY_MANAGER`, `OPERATOR`, `VIEWER`), and direct change-password / logout triggers.
- **Permission-Aware Navigation**:
  - `User Management` tab is only visible to `ADMIN`.
  - `Audit Logs` tab is only visible to `ADMIN`.
  - Action buttons (`Resolve`, `Dismiss`) in the Alerts table automatically adapt based on role capabilities.
- **Login Modal**: Automatically activates when session expires (handling HTTP `401 Unauthorized`) or when visiting unauthenticated.
- **Audit Viewer**: Supports filtering by event type, action outcome, user search, and date ranges with paginated result display.

---

## 7. Environment Variables & Deployment Configuration

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `AUTH_ENABLED` | `true` | Enforces authentication and RBAC across all protected routes. Set to `false` in development/testing. |
| `AUTH_SECRET_KEY` | `""` (Auto-generated) | Cryptographic signing key. In production, provide a 64+ char random string. |
| `AUTH_TOKEN_EXPIRE_MINUTES` | `480` | Session lifetime in minutes (8 hours). |
| `AUTH_DB_PATH` | `data/auth.db` | Filepath to SQLite authentication and audit database. |
| `AUTH_RATE_LIMIT_ATTEMPTS` | `5` | Maximum failed attempts before temporary lockout. |
| `AUTH_RATE_LIMIT_LOCKOUT_SECONDS` | `300` | Duration of brute-force lockout in seconds (5 minutes). |
| `AUTH_DEFAULT_ADMIN_USERNAME` | `admin` | Initial bootstrap administrator account username. |
| `AUTH_DEFAULT_ADMIN_PASSWORD` | `Admin123!` | Initial bootstrap administrator account password. |
| `AUTH_DEFAULT_ADMIN_EMAIL` | `admin@intelliwatch.local` | Initial bootstrap administrator email. |

> [!IMPORTANT]
> **Production Hardening Notice**: Upon initial deployment, immediately log in with the initial administrator credentials and change the password via the User Profile or `POST /api/v1/auth/change-password` endpoint. Configure `AUTH_SECRET_KEY` via an environment variable.
