"""Tailscale REST client — stdlib only, no new dependency.

sparkrun authenticates to the Tailscale control plane with an **OAuth client**
(client id + secret) and mints short-lived, tagged auth keys on demand, rather
than storing a long-lived auth key. Only a few endpoints are needed:

* ``POST /api/v2/oauth/token``            — exchange client creds for an access token
* ``POST /api/v2/tailnet/-/keys``         — mint a tagged, pre-authorized auth key
* ``GET  /api/v2/tailnet/-/devices``      — list tailnet devices (status / teardown)
* ``DELETE /api/v2/device/{id}``          — remove a device (``down --remove``)

The tailnet path segment ``-`` refers to the OAuth client's own tailnet.

Credential resolution (highest first):

* client id:     ``TS_API_CLIENT_ID``     → ``tailscale.oauth_client_id`` in config.yaml
* client secret: ``TS_API_CLIENT_SECRET`` → ``tailscale.oauth_client_secret`` in config.yaml
* base URL:      ``TS_API_URL``           → default ``https://api.tailscale.com``
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.tailscale.com"
DEFAULT_TAG = "tag:dgx-spark"
# ACL tags are lowercase DNS-label-ish tokens (`tag:name`). We validate before
# interpolating into a shell assignment on remote hosts (defense-in-depth
# against a `"`-bearing tag breaking out of `--advertise-tags="…"`).
_TAG_RE = re.compile(r"^tag:[a-z0-9][a-z0-9-]*$")


def validate_tag(tag: str) -> str:
    """Return *tag* if it is a well-formed ACL tag, else raise ``TailscaleError``."""
    if not _TAG_RE.match(tag or ""):
        raise TailscaleError("Invalid Tailscale tag %r — expected 'tag:<lowercase-name>'." % tag)
    return tag


def _validate_base_url(base_url: str) -> str:
    """Require an https base URL (the OAuth secret + tokens travel over it).

    A localhost http base is allowed for tests / self-hosted proxies; anything
    else must be https so credentials are never sent in cleartext.
    """
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme == "https":
        return base_url
    if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return base_url
    raise TailscaleError("Tailscale base_url must be https:// (got %r)." % base_url)


# Auth keys are minted just before a join batch and are reusable only for that
# short window; a small TTL bounds argv exposure of the key on joined hosts.
DEFAULT_KEY_EXPIRY_SECONDS = 600
_HTTP_TIMEOUT = 30


def _user_agent() -> str:
    try:
        from sparkrun import __version__

        return "sparkrun/%s" % __version__
    except Exception:  # noqa: BLE001 - version import is best-effort
        return "sparkrun"


_USER_AGENT = _user_agent()


class TailscaleError(Exception):
    """Base class for Tailscale integration failures."""


class TailscaleNotConfigured(TailscaleError):
    """No Tailscale OAuth client credentials available."""


class TailscaleAuthError(TailscaleError):
    """Tailscale rejected the credentials/token (HTTP 401/403)."""


class TailscaleApiError(TailscaleError):
    """A Tailscale API call failed (non-auth HTTP error or transport error)."""


class TailscaleTagError(TailscaleApiError):
    """Key minting failed because of a tag ownership / permission problem.

    Carries actionable guidance: OAuth-minted keys must be tagged, the tag must
    be in the OAuth client's allowed tags, and the tag needs ``tagOwners`` in
    the tailnet ACL policy.
    """


@dataclass(frozen=True)
class TailscaleSettings:
    """Resolved Tailscale integration settings (creds + defaults)."""

    client_id: str
    client_secret: str
    base_url: str = DEFAULT_API_URL
    tag: str = DEFAULT_TAG
    tailnet: str = "-"
    ephemeral: bool = False


@dataclass(frozen=True)
class TailscaleDevice:
    """One device from ``GET /api/v2/tailnet/-/devices``."""

    id: str
    hostname: str
    name: str
    addresses: tuple[str, ...]
    tags: tuple[str, ...]
    online: bool

    @property
    def ipv4(self) -> str | None:
        for addr in self.addresses:
            if ":" not in addr:  # skip IPv6
                return addr
        return None

    @classmethod
    def from_json(cls, raw: dict) -> "TailscaleDevice":
        return cls(
            id=str(raw.get("id", "") or raw.get("nodeId", "")),
            hostname=str(raw.get("hostname", "")),
            name=str(raw.get("name", "")),
            addresses=tuple(str(a) for a in (raw.get("addresses") or [])),
            tags=tuple(str(t) for t in (raw.get("tags") or [])),
            online=bool(raw.get("online", False)),
        )


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def load_settings(config: "SparkrunConfig | None" = None) -> TailscaleSettings:
    """Resolve Tailscale OAuth credentials + defaults from env then config.

    Raises:
        TailscaleNotConfigured: when no client id / secret can be found.
    """
    from scitrera_app_framework import ext_parse_bool

    def _cfg(key: str):
        return config.get(key) if config is not None else None

    client_id = os.environ.get("TS_API_CLIENT_ID") or _cfg("tailscale.oauth_client_id")
    client_secret = os.environ.get("TS_API_CLIENT_SECRET") or _cfg("tailscale.oauth_client_secret")
    if not client_id or not client_secret:
        raise TailscaleNotConfigured(
            "No Tailscale OAuth client configured. Set TS_API_CLIENT_ID / TS_API_CLIENT_SECRET, "
            "or tailscale.oauth_client_id / tailscale.oauth_client_secret in config.yaml. "
            "Create an OAuth client with the 'auth_keys' write scope at "
            "https://login.tailscale.com/admin/settings/oauth"
        )

    base_url = _validate_base_url(os.environ.get("TS_API_URL") or str(_cfg("tailscale.base_url") or DEFAULT_API_URL))
    tag = validate_tag(os.environ.get("TS_TAG") or str(_cfg("tailscale.tag") or DEFAULT_TAG))
    tailnet = str(_cfg("tailscale.tailnet") or "-")
    raw_ephemeral = _cfg("tailscale.ephemeral")
    ephemeral = bool(ext_parse_bool(raw_ephemeral)) if raw_ephemeral is not None else False

    return TailscaleSettings(
        client_id=str(client_id),
        client_secret=str(client_secret),
        base_url=base_url,
        tag=tag,
        tailnet=tailnet,
        ephemeral=ephemeral,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _request(method: str, url: str, *, token: str | None = None, data: bytes | None = None, content_type: str | None = None):
    """Issue an HTTP request and return the parsed JSON body (or ``{}``).

    Never logs the token/secret. Maps 401/403 → :class:`TailscaleAuthError`,
    other HTTP / transport failures → :class:`TailscaleApiError`.
    """
    req = urllib.request.Request(url, method=method, data=data)  # noqa: S310 - base_url is https-validated in load_settings
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if content_type:
        req.add_header("Content-Type", content_type)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _USER_AGENT)
    logger.debug("Tailscale API %s %s", method, urllib.parse.urlsplit(url).path)
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            body = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - best-effort error detail
            pass
        if e.code in (401, 403):
            raise TailscaleAuthError(
                "Tailscale authentication failed (HTTP %d). Check the OAuth client id/secret and its scopes." % e.code
            ) from None
        raise TailscaleApiError("Tailscale API %s failed: HTTP %d %s" % (method, e.code, detail)) from e
    except urllib.error.URLError as e:
        raise TailscaleApiError("Tailscale API %s failed: %s" % (method, e.reason)) from e

    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError as e:
        raise TailscaleApiError("Tailscale API %s returned invalid JSON: %s" % (method, e)) from e


def fetch_access_token(settings: TailscaleSettings) -> str:
    """Exchange the OAuth client credentials for a short-lived access token."""
    url = settings.base_url.rstrip("/") + "/api/v2/oauth/token"
    form = urllib.parse.urlencode({"client_id": settings.client_id, "client_secret": settings.client_secret}).encode("utf-8")
    result = _request("POST", url, data=form, content_type="application/x-www-form-urlencoded")
    token = result.get("access_token") if isinstance(result, dict) else None
    if not token:
        raise TailscaleAuthError("Tailscale OAuth token exchange returned no access_token.")
    return str(token)


def mint_auth_key(
    settings: TailscaleSettings,
    token: str,
    *,
    reusable: bool = True,
    ephemeral: bool | None = None,
    expiry_seconds: int = DEFAULT_KEY_EXPIRY_SECONDS,
    description: str = "sparkrun join",
) -> str:
    """Mint a pre-authorized, tagged auth key and return the ``tskey-…`` secret.

    OAuth-minted keys must carry a tag; :attr:`TailscaleSettings.tag` is used.

    Raises:
        TailscaleTagError: when the mint fails for a tag ownership / permission
            reason (actionable guidance attached).
    """
    if ephemeral is None:
        ephemeral = settings.ephemeral
    url = settings.base_url.rstrip("/") + "/api/v2/tailnet/%s/keys" % urllib.parse.quote(settings.tailnet, safe="")
    payload = {
        "capabilities": {
            "devices": {
                "create": {
                    "reusable": reusable,
                    "ephemeral": ephemeral,
                    "preauthorized": True,
                    "tags": [settings.tag],
                }
            }
        },
        "expirySeconds": expiry_seconds,
        "description": description,
    }
    data = json.dumps(payload).encode("utf-8")
    try:
        result = _request("POST", url, token=token, data=data, content_type="application/json")
    except TailscaleApiError as e:
        msg = str(e).lower()
        if "tag" in msg or "acl" in msg:
            raise TailscaleTagError(
                "Tailscale rejected the tagged auth key (%s). Ensure the tag %r is listed in the OAuth "
                "client's allowed tags AND has a tagOwners entry in your tailnet ACL policy "
                "(https://login.tailscale.com/admin/acls)." % (e, settings.tag)
            ) from e
        raise
    key = result.get("key") if isinstance(result, dict) else None
    if not key:
        raise TailscaleApiError("Tailscale key mint returned no key.")
    return str(key)


def list_devices(settings: TailscaleSettings, token: str) -> list[TailscaleDevice]:
    """Return all devices in the tailnet."""
    url = settings.base_url.rstrip("/") + "/api/v2/tailnet/%s/devices" % urllib.parse.quote(settings.tailnet, safe="")
    result = _request("GET", url, token=token)
    devices: list[TailscaleDevice] = []
    if isinstance(result, dict):
        for item in result.get("devices") or []:
            if isinstance(item, dict):
                devices.append(TailscaleDevice.from_json(item))
    return devices


def delete_device(settings: TailscaleSettings, token: str, device_id: str) -> None:
    """Remove a device from the tailnet by id."""
    url = settings.base_url.rstrip("/") + "/api/v2/device/%s" % urllib.parse.quote(device_id, safe="")
    _request("DELETE", url, token=token)
