"""Minimal Harbor agent for Altimate Code (-a altimate_agent:AltimateCode), modeled on Harbor's opencode.py.

  harbor run -p <task> -e tacc_env:TaccSingularityEnvironment -a altimate_agent:AltimateCode -m vllm/student \
      --ae ALTIMATE_BASE_URL=http://localhost:8102/v1 [--ae ALTIMATE_API_KEY=...] [--ae ALTIMATE_MAX_TURNS=100] \
      [--ae ALTIMATE_BINARY=/opt/altimate-dist/altimate]

install() copies the pinned binary at $ALTIMATE_BINARY (mounted read-only) into the container without network
access, else runs Altimate's official installer for VERSION. run() executes `altimate --yolo run "<instruction>"` in the task workspace with the
model configured from environment variables (inline OPENCODE_CONFIG_CONTENT: an OpenAI-compatible provider at
$ALTIMATE_BASE_URL). Altimate's stdout/stderr go to altimate.txt in the agent log dir; Harbor's verifier
computes the reward. No trajectory conversion.
"""
import json
import shlex

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

VERSION = "0.12.8"
PATH = 'export PATH="$HOME/.altimate/bin:$PATH"'
# GIGACAGE_ENABLED=0: Bun's JSC otherwise reserves ~70 GB of address space per process, which exhausts the
# commit limit on hosts with vm.overcommit_memory=2. The rest keep the run local (no hosted provider, telemetry,
# auto-update or model catalog fetch).
RUN_ENV = {"GIGACAGE_ENABLED": "0", "ALTIMATE_BASE_AUTO_REGISTER": "0", "ALTIMATE_TELEMETRY_DISABLED": "1",
           "ALTIMATE_CLI_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
           "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_SHARE": "1"}


class AltimateCode(BaseInstalledAgent):
    _OUTPUT_FILENAME = "altimate.txt"

    @staticmethod
    def name() -> str:
        return "altimate-code"

    def get_version_command(self) -> str | None:
        return f"{PATH}; GIGACAGE_ENABLED=0 altimate --version"

    async def install(self, environment: BaseEnvironment) -> None:
        binary = self.extra_env.get("ALTIMATE_BINARY")
        if binary:  # pinned binary mounted from the host: install offline, fail on a version mismatch
            await self.exec_as_agent(
                environment,
                command=(f"set -euo pipefail; mkdir -p $HOME/.altimate/bin && "
                         f"cp {shlex.quote(binary)} $HOME/.altimate/bin/altimate && {PATH} && "
                         f"v=$(GIGACAGE_ENABLED=0 altimate --version 2>/dev/null | tail -1) && echo \"altimate $v\" && "
                         f"test \"$v\" = {VERSION}"),
            )
            return
        await self.ensure_system_dependencies(environment, ("curl", "bash"))
        await self.exec_as_agent(
            environment,
            command=(f"set -euo pipefail; curl -fsSL https://www.altimate.sh/install | bash -s -- --version {VERSION} "
                     f"&& {PATH} && GIGACAGE_ENABLED=0 altimate --version"),
        )

    def _provider_config(self) -> dict:
        provider, _, model = (self.model_name or "").partition("/")
        if not provider or not model:
            raise ValueError("Model name must be provider/model, e.g. vllm/student")
        base_url = self.extra_env.get("ALTIMATE_BASE_URL")
        if not base_url:
            raise ValueError("Set the model endpoint with --ae ALTIMATE_BASE_URL=<OpenAI-compatible /v1 URL>")
        limit = {"context": int(self.extra_env.get("ALTIMATE_CONTEXT", 262144)), "output": 32768}
        return {"provider": {provider: {
                    "npm": "@ai-sdk/openai-compatible", "name": provider,
                    "options": {"baseURL": base_url, "apiKey": self.extra_env.get("ALTIMATE_API_KEY", "unused")},
                    "models": {model: {"name": model, "limit": limit}}}},
                "model": self.model_name, "disabled_providers": ["altimate-base"], "autoupdate": False,
                "share": "disabled", "permission": {"webfetch": "deny", "websearch": "deny"}}

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        env = {**RUN_ENV, "OPENCODE_CONFIG_CONTENT": json.dumps(self._provider_config())}
        max_turns = self.extra_env.get("ALTIMATE_MAX_TURNS")
        turns = f"--max-turns {int(max_turns)} " if max_turns else ""
        log = shlex.quote((self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix())
        await self.exec_as_agent(
            environment,
            command=(f"{PATH}; altimate --yolo run {turns}--format json -m {shlex.quote(self.model_name)} "
                     f"{shlex.quote(instruction)} 2>&1 </dev/null | stdbuf -oL tee {log}"),
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass
