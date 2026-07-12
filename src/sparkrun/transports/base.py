"""Transport seam: how sparkrun reaches / prepares a cluster's hosts.

A :class:`Transport` owns the *connectivity* concern — everything that must
happen before sparkrun's SSH machinery (``orchestration.ssh``) can talk to a
cluster's hosts.  For ordinary clusters this is nothing: the hosts are already
reachable via plain ``ssh``, so :class:`~sparkrun.transports.ssh.SshTransport`
is a pure no-op and behavior is byte-identical to before transports existed.

Provider-backed clusters (e.g. Thunder Compute) override :meth:`prepare` to
materialize connection details — refresh ephemeral IP/port, provision SSH keys,
and write a managed ``ssh_config`` alias — so that once ``prepare`` returns,
every host in the cluster is a plain SSH host and the rest of sparkrun runs
unchanged.

Transport (how you *reach* the host) is orthogonal to Executor
(``orchestration.executors`` — how you *run the workload* on the host).  A
Thunder-transport cluster still uses the default docker executor.

Layering: this package depends on ``core`` + ``orchestration``; ``orchestration``
never imports ``transports`` (it stays the generic SSH/Docker leaf).
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition

logger = logging.getLogger(__name__)


class TransportError(Exception):
    """Raised when a transport cannot prepare a cluster for connectivity."""


class Transport(ABC):
    """Base class for cluster transports.

    The default implementation is a no-op — subclasses override :meth:`prepare`
    only when the cluster's hosts need out-of-band setup before SSH.
    """

    name: str = "ssh"
    """Selector matching :attr:`ClusterDefinition.transport`."""

    def prepare(self, cluster: "ClusterDefinition", *, dry_run: bool = False) -> None:
        """Ensure every host in *cluster* is reachable via plain ``ssh``.

        Called at run/connect init before any SSH fan-out.  The default is a
        no-op.  Implementations must be idempotent and safe to call repeatedly.
        When *dry_run* is True they must not mutate local state or make
        privileged/expensive remote calls beyond read-only lookups.
        """
        return None
