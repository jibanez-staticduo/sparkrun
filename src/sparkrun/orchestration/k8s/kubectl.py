"""``kubectl`` binary acquisition and version-cache management.

sparkrun does not assume ``kubectl`` is on ``PATH``.  This module
resolves a usable binary — preferring an explicit config path, then a
version-matched cached download, then any cached download, then a
``PATH`` binary, and finally downloading the ``stable`` release from
``dl.k8s.io``.  Downloads are integrity-checked against the sibling
``.sha256`` published by the Kubernetes release infra.

Cache layout (under ``~/.cache/sparkrun/kubectl/``)::

    kubectl/<version>/<os>-<arch>/kubectl[.exe]

Multiple versions / OS / arch coexist so a control machine that talks to
several clusters can keep a client matching each cluster's server
version (see :func:`ensure_kubectl` ``version=`` and the per-context
pinning wired up by the api layer).

Downloading uses the standard library (``urllib``) — no new dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import stat
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from .errors import KubectlDownloadError, KubectlNotFoundError

logger = logging.getLogger(__name__)

STABLE_URL = "https://dl.k8s.io/release/stable.txt"
RELEASE_BASE = "https://dl.k8s.io/release"

_DOWNLOAD_TIMEOUT = 60
_STABLE_TIMEOUT = 15


@dataclass
class KubectlBinary:
    """A resolved ``kubectl`` executable.

    ``version`` is the known release string (e.g. ``"v1.31.0"``) when the
    binary came from the sparkrun cache or a download; it is ``None`` for
    a ``PATH`` binary whose version has not been probed.  ``source`` is one
    of ``"config"`` / ``"cache"`` / ``"path"`` / ``"download"``.
    """

    path: Path
    version: str | None
    source: str
    os_name: str
    arch: str

    def __fspath__(self) -> str:  # allow use anywhere a path-like is accepted
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_os() -> str:
    """Map :func:`platform.system` onto a kubectl release OS token."""
    system = platform.system().lower()
    if system.startswith("linux"):
        return "linux"
    if system == "darwin":
        return "darwin"
    if system.startswith("win") or system == "windows":
        return "windows"
    # Best-effort passthrough; download will 404 loudly if unsupported.
    return system


def detect_arch() -> str:
    """Map :func:`platform.machine` onto a kubectl release arch token."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("armv7l", "arm"):
        return "arm"
    if machine in ("ppc64le",):
        return "ppc64le"
    if machine in ("s390x",):
        return "s390x"
    return machine


def binary_name(os_name: str) -> str:
    return "kubectl.exe" if os_name == "windows" else "kubectl"


_RELEASE_RE = re.compile(r"v\d+\.\d+\.\d+")


def normalize_release_version(git_version: str | None) -> str | None:
    """Extract a downloadable ``vX.Y.Z`` release tag from a server gitVersion.

    Managed distributions append suffixes (e.g. ``v1.31.2+ck1`` on some
    providers) that are not valid ``dl.k8s.io`` release tags — strip them
    so a version-matched download resolves.  Returns ``None`` when no
    release triple is present.
    """
    match = _RELEASE_RE.search(git_version or "")
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def kubectl_cache_root(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "kubectl"


def cached_binary_path(cache_dir: str | Path, version: str, os_name: str, arch: str) -> Path:
    return kubectl_cache_root(cache_dir) / version / ("%s-%s" % (os_name, arch)) / binary_name(os_name)


def list_cached(cache_dir: str | Path, *, os_name: str | None = None, arch: str | None = None) -> list[KubectlBinary]:
    """Return cached binaries, newest-version-first, optionally filtered by platform."""
    root = kubectl_cache_root(cache_dir)
    if not root.is_dir():
        return []
    found: list[KubectlBinary] = []
    for version_dir in root.iterdir():
        if not version_dir.is_dir():
            continue
        for platform_dir in version_dir.iterdir():
            if not platform_dir.is_dir() or "-" not in platform_dir.name:
                continue
            cached_os, _, cached_arch = platform_dir.name.partition("-")
            if os_name is not None and cached_os != os_name:
                continue
            if arch is not None and cached_arch != arch:
                continue
            binary = platform_dir / binary_name(cached_os)
            if binary.is_file():
                found.append(
                    KubectlBinary(
                        path=binary,
                        version=version_dir.name,
                        source="cache",
                        os_name=cached_os,
                        arch=cached_arch,
                    )
                )
    found.sort(key=lambda b: _version_key(b.version or ""), reverse=True)
    return found


def _version_key(version: str) -> tuple:
    """Sort key for ``vX.Y.Z`` strings (non-numeric segments sort low)."""
    cleaned = version.lstrip("v").split("-", 1)[0]
    parts: list[int] = []
    for segment in cleaned.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def fetch_stable_version(*, timeout: int = _STABLE_TIMEOUT) -> str:
    """Return the current stable kubectl release (contents of ``stable.txt``)."""
    try:
        with urllib.request.urlopen(STABLE_URL, timeout=timeout) as resp:  # noqa: S310 — fixed https host
            version = resp.read().decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 — surfaced as a typed error
        raise KubectlDownloadError("Failed to fetch kubectl stable version: %s" % exc) from exc
    if not version.startswith("v"):
        raise KubectlDownloadError("Unexpected stable.txt contents: %r" % version)
    return version


def _release_url(version: str, os_name: str, arch: str) -> str:
    return "%s/%s/bin/%s/%s/%s" % (RELEASE_BASE, version, os_name, arch, binary_name(os_name))


def _fetch_bytes(url: str, *, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        return resp.read()


def download_kubectl(
    cache_dir: str | Path,
    version: str,
    os_name: str,
    arch: str,
    *,
    verify: bool = True,
    timeout: int = _DOWNLOAD_TIMEOUT,
) -> Path:
    """Download and cache ``kubectl`` *version* for *os_name*/*arch*.

    The binary is verified against the published ``.sha256`` unless
    *verify* is False, then written atomically and made executable.
    Returns the cached binary path.
    """
    dest = cached_binary_path(cache_dir, version, os_name, arch)
    url = _release_url(version, os_name, arch)
    logger.info("Downloading kubectl %s (%s/%s) from %s", version, os_name, arch, url)
    try:
        payload = _fetch_bytes(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise KubectlDownloadError("Failed to download kubectl from %s: %s" % (url, exc)) from exc

    if verify:
        try:
            expected = _fetch_bytes(url + ".sha256", timeout=timeout).decode("utf-8").strip().split()[0]
        except Exception as exc:  # noqa: BLE001
            raise KubectlDownloadError("Failed to fetch kubectl checksum for %s: %s" % (url, exc)) from exc
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise KubectlDownloadError("kubectl checksum mismatch for %s: expected %s, got %s" % (url, expected, actual))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, dest)
    logger.info("Cached kubectl %s at %s", version, dest)
    return dest


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def ensure_kubectl(
    cache_dir: str | Path,
    *,
    version: str | None = None,
    explicit_path: str | Path | None = None,
    allow_path: bool = True,
    allow_download: bool = True,
    verify: bool = True,
    os_name: str | None = None,
    arch: str | None = None,
) -> KubectlBinary:
    """Resolve a usable ``kubectl`` binary.

    Resolution order:

    1. *explicit_path* (config ``k8s.kubectl.path``) if given and present.
    2. A cached binary matching *version* (when *version* is pinned).
    3. Any cached binary for this platform (newest first) — only when no
       *version* is requested.
    4. A ``kubectl`` on ``PATH`` (when *allow_path*) — only when no
       *version* is requested (a PATH binary's version is unknown).
    5. Download (*version* or the current ``stable``) when *allow_download*.

    Raises :class:`KubectlNotFoundError` when nothing resolves.
    """
    os_name = os_name or detect_os()
    arch = arch or detect_arch()

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return KubectlBinary(path=path, version=version, source="config", os_name=os_name, arch=arch)
        raise KubectlNotFoundError("Configured kubectl path does not exist: %s" % path)

    if version:
        cached = cached_binary_path(cache_dir, version, os_name, arch)
        if cached.is_file():
            return KubectlBinary(path=cached, version=version, source="cache", os_name=os_name, arch=arch)
    else:
        for candidate in list_cached(cache_dir, os_name=os_name, arch=arch):
            return candidate
        if allow_path:
            path_binary = which("kubectl")
            if path_binary:
                return KubectlBinary(path=Path(path_binary), version=None, source="path", os_name=os_name, arch=arch)

    if allow_download:
        resolved_version = version or fetch_stable_version()
        cached = cached_binary_path(cache_dir, resolved_version, os_name, arch)
        if not cached.is_file():
            cached = download_kubectl(cache_dir, resolved_version, os_name, arch, verify=verify)
        return KubectlBinary(path=cached, version=resolved_version, source="download", os_name=os_name, arch=arch)

    raise KubectlNotFoundError("No kubectl binary available (version=%r, downloads disabled)." % version)


__all__ = [
    "KubectlBinary",
    "STABLE_URL",
    "RELEASE_BASE",
    "detect_os",
    "detect_arch",
    "binary_name",
    "normalize_release_version",
    "kubectl_cache_root",
    "cached_binary_path",
    "list_cached",
    "fetch_stable_version",
    "download_kubectl",
    "ensure_kubectl",
]
