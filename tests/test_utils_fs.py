"""Tests for ``sparkrun.utils.fs`` — the private-write helper.

Every caller of this helper writes a file that may carry an api_key, so the
POSIX hardening must survive; and every caller may run on a Windows control
node, where the hardening flag simply doesn't exist.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from sparkrun.utils.fs import open_private_write


def _write(path: Path, text: str) -> None:
    with os.fdopen(open_private_write(path), "w") as f:
        f.write(text)


def test_creates_owner_only_file(tmp_path: Path):
    target = tmp_path / "secret.yaml"
    _write(target, "api_key: hunter2\n")
    assert target.read_text() == "api_key: hunter2\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_truncates_an_existing_file(tmp_path: Path):
    target = tmp_path / "secret.yaml"
    target.write_text("stale and much longer than the replacement")
    _write(target, "new\n")
    assert target.read_text() == "new\n"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX-only hardening")
def test_refuses_to_write_through_a_symlink(tmp_path: Path):
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    link = tmp_path / "link.yaml"
    link.symlink_to(victim)

    with pytest.raises(OSError):
        _write(link, "secret\n")
    assert victim.read_text() == "untouched"


def test_still_writes_when_o_nofollow_is_absent(tmp_path: Path, monkeypatch):
    """Simulates Windows: the flag is gone, but the write must still happen."""
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    target = tmp_path / "secret.yaml"
    _write(target, "written anyway\n")
    assert target.read_text() == "written anyway\n"
