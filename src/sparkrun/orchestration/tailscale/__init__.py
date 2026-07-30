"""Tailscale orchestration — stdlib REST client + join/status/teardown scripts.

The connectivity substrate that publishes a sparkrun inference endpoint on a
tailnet. This layer is console-free and holds no CLI/api imports; it is driven
by :mod:`sparkrun.api.tailscale`.
"""

from __future__ import annotations

from .api import (
    DEFAULT_API_URL,
    DEFAULT_KEY_EXPIRY_SECONDS,
    DEFAULT_TAG,
    TailscaleApiError,
    TailscaleAuthError,
    TailscaleDevice,
    TailscaleError,
    TailscaleNotConfigured,
    TailscaleSettings,
    TailscaleTagError,
    delete_device,
    fetch_access_token,
    list_devices,
    load_settings,
    mint_auth_key,
    validate_tag,
)
from .local import local_tailscale_dnsname, local_tailscale_ipv4
from .scripts import (
    LOGOUT_FALLBACK_SCRIPT,
    LOGOUT_SCRIPT,
    STATUS_SCRIPT,
    build_join_scripts,
    build_serve_scripts,
    parse_join_result,
)

__all__ = [
    "DEFAULT_API_URL",
    "DEFAULT_KEY_EXPIRY_SECONDS",
    "DEFAULT_TAG",
    "TailscaleApiError",
    "TailscaleAuthError",
    "TailscaleDevice",
    "TailscaleError",
    "TailscaleNotConfigured",
    "TailscaleSettings",
    "TailscaleTagError",
    "delete_device",
    "fetch_access_token",
    "list_devices",
    "load_settings",
    "mint_auth_key",
    "validate_tag",
    "local_tailscale_dnsname",
    "local_tailscale_ipv4",
    "LOGOUT_FALLBACK_SCRIPT",
    "LOGOUT_SCRIPT",
    "STATUS_SCRIPT",
    "build_join_scripts",
    "build_serve_scripts",
    "parse_join_result",
]
