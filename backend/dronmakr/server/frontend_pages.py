"""Shared frontend page registry and Jinja environment for build + dev serving."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dronmakr._repo import ASSETS_ROOT

TEMPLATES_DIR = ASSETS_ROOT / "templates"

# (template_name, dist_filename, active_path for toolbar highlight)
PAGE_SPECS: list[tuple[str, str, str]] = [
    ("startup.html", "startup.html", "/"),
    ("index.html", "index.html", "/"),
    ("auditionr.html", "auditionr.html", "/auditionr"),
    ("beatbuildr.html", "beatbuildr.html", "/beatbuildr"),
    ("folysplitr.html", "folysplitr.html", "/folysplitr"),
    ("collections.html", "collections.html", "/collections"),
    ("settings.html", "settings.html", "/settings"),
    ("onboarding.html", "onboarding.html", "/onboarding"),
    ("about.html", "about.html", "/about"),
]

DIST_FILENAME_TO_SPEC: dict[str, tuple[str, str]] = {
    dist: (template, active_path) for template, dist, active_path in PAGE_SPECS
}

BUILD_URL_FOR_MAP: dict[str, str] = {
    "index": "/",
    "auditionr_page": "/auditionr",
    "beatbuildr_page": "/beatbuildr",
    "folysplitr_page": "/folysplitr",
    "collections_page": "/collections",
    "settings_page": "/settings",
    "onboarding_page": "/onboarding",
    "about_page": "/about",
}


def build_url_for(endpoint: str, **_kwargs: object) -> str:
    """url_for stand-in for static frontend builds (no Flask app)."""
    return BUILD_URL_FOR_MAP.get(endpoint, "/")


def create_jinja_env(*, auto_reload: bool = False) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        auto_reload=auto_reload,
    )


def pagename_for(template_name: str) -> str:
    return Path(template_name).stem
