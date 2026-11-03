"""Harbor's Qwen Code agent with a pinned, offline install (-a qwen_agent:QwenCodePinned).

Harbor's install() downloads nvm, Node and `@qwen-code/qwen-code@latest` in every trial. Instead, use the Node 22 +
Qwen Code 0.26.5 tree in agents/, bind-mounted read-only at /opt/agents by run_trial.sh, and fail if the version
differs. run() is Harbor's unchanged; it sources ~/.nvm/nvm.sh, which here only puts /opt/agents on PATH.
"""
from harbor.agents.installed.qwen_code import QwenCode
from harbor.environments.base import BaseEnvironment

VERSION = "0.26.5"
SHIM = "export PATH=/opt/agents/node/bin:/opt/agents/npm/bin:$PATH"


class QwenCodePinned(QwenCode):
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(
            environment,
            command=(f"set -euo pipefail; mkdir -p ~/.nvm && echo '{SHIM}' > ~/.nvm/nvm.sh && . ~/.nvm/nvm.sh && "
                     f"v=$(qwen --version) && echo \"qwen $v\" && test \"$v\" = {VERSION}"),
        )
