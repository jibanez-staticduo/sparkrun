"""Filesystem helpers that behave the same on POSIX and Windows."""

from __future__ import annotations

import os

__all__ = ("open_private_write",)


def open_private_write(path: str | os.PathLike) -> int:
    """Open *path* for writing, owner-only, without following a symlink.

    ``O_NOFOLLOW`` refuses to write *through* a symlink, so another local user
    can't pre-create one at a predictable path and capture what we write. It is
    POSIX-only: on Windows ``os.O_NOFOLLOW`` does not exist, and naming it
    directly raises ``AttributeError`` — which callers that wrap the write in a
    broad ``except`` will swallow, leaving the file silently unwritten. (That is
    exactly how a Windows control machine ended up with no job metadata for jobs
    it had launched, so ``logs`` and ``stop`` could not find their hosts.)

    ``O_BINARY`` matters on Windows for the same reason it does elsewhere in
    sparkrun: text mode would rewrite ``\\n`` on write.

    Returns:
        A file descriptor opened ``O_WRONLY|O_CREAT|O_TRUNC`` with mode 0600.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    # Absent on Windows; the symlink hardening simply doesn't apply there.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o600)
