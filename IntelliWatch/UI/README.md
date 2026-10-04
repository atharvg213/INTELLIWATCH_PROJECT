# IntelliWatch

A self-contained, responsive frontend for an industrial safety operations product. It includes the public site, source onboarding, a detection workspace, live camera view, incident workflow, safety analytics, settings, and light/dark/system themes.

## Run locally

From this folder, run:

```powershell
python server.py
```

Then open <http://127.0.0.1:8000>.

The project uses browser APIs and Python's standard library; it has no package install step.

## Current integration boundary

Uploaded files are previewed in the browser for the current session before analysis is sent to the connected backend. Video analysis, RTSP connection, incident persistence, and analytics depend on backend services; the interface presents unavailable states when those services cannot be reached.
