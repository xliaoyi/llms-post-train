"""Harbor's singularity environment for TACC, without modifying Harbor (-e tacc_env:TaccSingularityEnvironment).

- The container runs under --fakeroot, so commands already execute as root; Harbor wraps root commands in
  `su root`, which fails where setgroups is denied (TACC). Skip the wrapper for root.
- exec() without timeout_sec gets a 600 s HTTP timeout, which kills agent runs (installed agents call it without
  one) after 10 minutes. Default to just over the 2-hour agent timeout so Harbor's own agent timeout applies.
"""
from harbor.environments.singularity.singularity import SingularityEnvironment

DEFAULT_TIMEOUT_SEC = 7300


class TaccSingularityEnvironment(SingularityEnvironment):
    def _resolve_user(self, user):
        resolved = super()._resolve_user(user)
        return None if resolved in ("root", 0) else resolved

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec or DEFAULT_TIMEOUT_SEC,
                                  user=user)
