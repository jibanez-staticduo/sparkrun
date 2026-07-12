"""Thunder Compute REST client — stdlib only, no new dependency.

We consume the Bearer token that ``tnr login`` already stored (or
``TNR_API_TOKEN``); we never run the browser OAuth flow ourselves.  Only three
endpoints are needed:

* ``GET  /v1/auth/validate``                      — token check
* ``GET  /v1/instances/list?update_ips=true``     — instances w/ fresh ip+port
* ``POST /v1/instances/{id}/add_key``             — provision an SSH key (PEM)

Token / base-URL resolution mirrors the Go CLI (``cmd/login.go``,
``utils/thunderdir.go``):

* base URL:  ``TNR_API_URL`` → default ``https://api.thundercompute.com:8443``
* token:     ``TNR_API_TOKEN`` → ``$TNR_HOME``/``$HOME``/.thunder/cli_config.json
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.thundercompute.com:8443"
_HTTP_TIMEOUT = 30


def _user_agent() -> str:
    try:
        from sparkrun import __version__

        return "sparkrun/%s" % __version__
    except Exception:  # noqa: BLE001 - version import is best-effort
        return "sparkrun"


_USER_AGENT = _user_agent()


class ThunderError(Exception):
    """Base class for Thunder integration failures."""


class ThunderNotConfigured(ThunderError):
    """No Thunder token available (user has not run ``tnr login``)."""


class ThunderAuthError(ThunderError):
    """Thunder rejected the token (HTTP 401)."""


class ThunderApiError(ThunderError):
    """A Thunder API call failed (non-401 HTTP error or transport error)."""


@dataclass(frozen=True)
class ThunderInstance:
    """One Thunder instance from ``/v1/instances/list``."""

    id: str
    """Positional instance id (e.g. ``"0"``) — used in the ``add_key`` path.
    May be reassigned across sessions, so always resolve it fresh from a list
    call keyed by the stable :attr:`uuid`."""

    uuid: str
    """Stable instance identifier (e.g. ``"ie2pb8eu"``) — used for the key
    filename and as the sparkrun ``provider_ref`` / ssh alias suffix."""

    ip: str | None
    port: int
    status: str
    gpu_type: str
    num_gpus: int
    memory_gb: int
    storage_gb: int
    cpu_cores: str

    @property
    def is_running(self) -> bool:
        return self.status.upper() == "RUNNING"

    @classmethod
    def from_json(cls, raw: dict) -> "ThunderInstance":
        def _int(v, default=0) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        return cls(
            id=str(raw.get("id", "")),
            uuid=str(raw.get("uuid") or raw.get("name") or ""),
            ip=(raw.get("ip") or None),
            port=_int(raw.get("port"), 22),
            status=str(raw.get("status", "")),
            gpu_type=str(raw.get("gpuType", "")),
            num_gpus=_int(raw.get("numGpus"), 1),
            memory_gb=_int(raw.get("memory")),
            storage_gb=_int(raw.get("storage")),
            cpu_cores=str(raw.get("cpuCores", "")),
        )


# ---------------------------------------------------------------------------
# Token / base-URL resolution
# ---------------------------------------------------------------------------


def _thunder_dir() -> Path:
    """Resolve Thunder's state dir: ``$TNR_HOME`` else ``$HOME/.thunder``."""
    explicit = os.environ.get("TNR_HOME")
    if explicit:
        return Path(explicit)
    return Path.home() / ".thunder"


def load_token() -> tuple[str, str]:
    """Return ``(token, base_url)`` for the Thunder API.

    Precedence matches the Go CLI: ``TNR_API_TOKEN`` env wins, else the token in
    ``<thunder-dir>/cli_config.json``.  ``TNR_API_URL`` overrides the base URL.

    Raises:
        ThunderNotConfigured: No token available anywhere.
    """
    base = os.environ.get("TNR_API_URL") or DEFAULT_API_URL

    env_token = os.environ.get("TNR_API_TOKEN")
    if env_token:
        return env_token, base

    config_path = _thunder_dir() / "cli_config.json"
    try:
        data = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise ThunderNotConfigured("No Thunder credentials found. Run `tnr login` (or set TNR_API_TOKEN).") from None
    except (OSError, ValueError) as e:
        raise ThunderNotConfigured("Could not read Thunder config %s: %s" % (config_path, e)) from e

    token = data.get("token")
    if not token:
        raise ThunderNotConfigured("Thunder config %s has no token. Run `tnr login`." % config_path)
    return str(token), (data.get("api_url") or base)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _request(method: str, token: str, base: str, path: str) -> dict | list:
    """Issue a JSON request and return the parsed body.

    Never logs the token.  Maps 401 → :class:`ThunderAuthError`, other HTTP /
    transport failures → :class:`ThunderApiError`.
    """
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, method=method)  # noqa: S310 - fixed https base
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    req.add_header("Thunder-Client", "sparkrun")
    # The API sits behind Cloudflare, which bans the default ``Python-urllib``
    # User-Agent (403, error 1010).  Any other UA is accepted — identify as
    # sparkrun rather than spoofing a browser.
    req.add_header("User-Agent", _USER_AGENT)
    logger.debug("Thunder API %s %s", method, path)
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            body = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ThunderAuthError("Thunder authentication failed (invalid token). Run `tnr login`.") from None
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001 - best-effort error detail
            pass
        raise ThunderApiError("Thunder API %s %s failed: HTTP %d %s" % (method, path, e.code, detail)) from e
    except urllib.error.URLError as e:
        raise ThunderApiError("Thunder API %s %s failed: %s" % (method, path, e.reason)) from e

    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError as e:
        raise ThunderApiError("Thunder API %s %s returned invalid JSON: %s" % (method, path, e)) from e


def validate(token: str, base: str) -> dict:
    """Validate the token (``GET /v1/auth/validate``). Raises on failure."""
    result = _request("GET", token, base, "/v1/auth/validate")
    return result if isinstance(result, dict) else {}


def list_instances(token: str, base: str) -> list[ThunderInstance]:
    """Return all instances with fresh IPs (``/v1/instances/list?update_ips=true``).

    The list endpoint returns a mapping of ``id -> instance``; the ``id`` is
    folded into each item.
    """
    raw = _request("GET", token, base, "/v1/instances/list?update_ips=true")
    instances: list[ThunderInstance] = []
    if isinstance(raw, dict):
        for inst_id, item in raw.items():
            if isinstance(item, dict):
                item = {**item, "id": item.get("id", inst_id)}
                instances.append(ThunderInstance.from_json(item))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                instances.append(ThunderInstance.from_json(item))
    instances.sort(key=lambda i: i.id)
    return instances


def add_key(token: str, base: str, instance_id: str) -> str:
    """Provision an SSH key for *instance_id* and return the private-key PEM.

    ``POST /v1/instances/{id}/add_key`` → ``{"uuid": ..., "key": "<PEM>"}``.
    """
    result = _request("POST", token, base, "/v1/instances/%s/add_key" % instance_id)
    key = result.get("key") if isinstance(result, dict) else None
    if not key:
        raise ThunderApiError("Thunder add_key for instance %s returned no private key" % instance_id)
    return str(key)
