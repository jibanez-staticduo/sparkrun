"""Hard stop on outbound telemetry from the test suite.

``isolate_stateful`` sets ``SPARKRUN_NO_TELEMETRY=1``, but that is *policy*: it
is one environment variable, any test may drop it with ``monkeypatch.delenv``
(several do, deliberately), and a branch or sibling repo whose conftest predates
it has no protection at all.  This is the *mechanism* — the HTTP call itself is
replaced, so nothing the suite does can reach the collector.

Real leaks this backstops (observed in collected data, 2026-07-31 / 08-05 /
08-07): a ``MagicMock`` config makes ``telemetry_enabled`` fail open, so mock
objects were posted to the production endpoint as a benchmark's category,
framework and profile.
"""

from __future__ import annotations

#: Patch target — the single ``urlopen`` every telemetry send funnels through.
TELEMETRY_URLOPEN = "sparkrun.telemetry.client.urlopen"


def install_telemetry_blocker(monkeypatch) -> list[str]:
    """Replace the telemetry HTTP call and return the list of attempted URLs.

    Recording is the load-bearing half.  Raising alone would not fail anything:
    every ``sparkrun.telemetry.emit_*`` helper wraps its send in
    ``except Exception`` and logs at DEBUG, which is precisely the silence that
    let mock objects reach the real collector unnoticed.  Callers assert the
    returned list is empty at teardown.
    """
    attempts: list[str] = []

    def _blocked(request, timeout=None):
        attempts.append(str(getattr(request, "full_url", request)))
        raise AssertionError("telemetry POST blocked: the test suite must never send telemetry")

    monkeypatch.setattr(TELEMETRY_URLOPEN, _blocked)
    return attempts


def describe_escapes(attempts: list[str]) -> str:
    """Render the teardown failure message for a non-empty attempt list."""
    return (
        "telemetry escaped the test suite: %d POST(s) to %s. Telemetry must be disabled in tests -- if this test needs the enabled path, patch %s itself."
        % (
            len(attempts),
            ", ".join(sorted(set(attempts))),
            TELEMETRY_URLOPEN,
        )
    )
