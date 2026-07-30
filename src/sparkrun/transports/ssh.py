"""Default SSH transport — a pure no-op.

Clusters whose hosts are already reachable via plain ``ssh`` (every cluster
today) use this transport.  :meth:`prepare` does nothing, so wiring the
transport seam into the run path is byte-identical to the pre-transport
behavior.
"""

from __future__ import annotations

from sparkrun.transports.base import Transport


class SshTransport(Transport):
    """No-op transport for plain-SSH clusters (the default)."""

    transport_name = "ssh"
