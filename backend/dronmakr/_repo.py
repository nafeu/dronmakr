"""Repository layout roots (stable regardless of module location)."""

from __future__ import annotations

from pathlib import Path

# backend/dronmakr/_repo.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
ASSETS_ROOT = REPO_ROOT / "assets"
RESOURCES_ROOT = REPO_ROOT / "resources"
FRONTEND_ROOT = REPO_ROOT / "frontend"
LOGS_ROOT = REPO_ROOT / "logs"
