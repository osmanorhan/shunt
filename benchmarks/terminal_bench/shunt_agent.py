import json
import os
import shlex
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class ShuntAgent(BaseAgent):
    """Harbor custom agent that runs shunt through the official harness."""

    _DEFAULT_ENDPOINT = "http://host.docker.internal:8080"
    _DEFAULT_TIMEOUT_SECS = 300
    _DEFAULT_RUN_TIMEOUT_SECS = 1800
    _DEFAULT_SETUP_TIMEOUT_SECS = 600
    _PATH_EXPORT = 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"'

    def __init__(
        self,
        *args: Any,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_secs: int | None = None,
        run_timeout_secs: int | None = None,
        setup_timeout_secs: int | None = None,
        install_command: str | None = None,
        workdir: str | None = None,
        version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint = (
            endpoint or os.environ.get("SHUNT_ENDPOINT") or self._DEFAULT_ENDPOINT
        )
        self.shunt_model = (
            model or os.environ.get("SHUNT_MODEL") or self._model_from_harbor()
        )
        self.timeout_secs = int(
            timeout_secs or os.environ.get("SHUNT_TIMEOUT_SECS") or self._DEFAULT_TIMEOUT_SECS
        )
        self.run_timeout_secs = int(
            run_timeout_secs
            or os.environ.get("SHUNT_RUN_TIMEOUT_SECS")
            or self._DEFAULT_RUN_TIMEOUT_SECS
        )
        self.setup_timeout_secs = int(
            setup_timeout_secs
            or os.environ.get("SHUNT_SETUP_TIMEOUT_SECS")
            or self._DEFAULT_SETUP_TIMEOUT_SECS
        )
        self.install_command = install_command or os.environ.get("SHUNT_INSTALL_COMMAND")
        self.workdir = workdir or os.environ.get("SHUNT_WORKDIR")
        self._version = version or os.environ.get("SHUNT_AGENT_VERSION")

    @staticmethod
    def name() -> str:
        return "shunt"

    def version(self) -> str | None:
        return self._version

    async def setup(self, environment: BaseEnvironment) -> None:
        if await self._shunt_available(environment):
            return

        if self.install_command:
            result = await environment.exec(
                command=self._shell(self.install_command),
                cwd=self.workdir,
                timeout_sec=self.setup_timeout_secs,
            )
            self._write_log("setup-stdout.txt", result.stdout)
            self._write_log("setup-stderr.txt", result.stderr)
            if result.return_code != 0:
                raise RuntimeError(f"SHUNT_INSTALL_COMMAND failed with exit {result.return_code}")
            await self._publish_shunt_binary(environment)
            if not await self._shunt_available(environment):
                raise RuntimeError("SHUNT_INSTALL_COMMAND completed but shunt is not on PATH")
            return

        await self._install_curl_if_needed(environment)
        install = (
            "curl --proto '=https' --tlsv1.2 -LsSf "
            "https://github.com/osmanorhan/shunt/releases/latest/download/shunt-cli-installer.sh | sh"
        )
        result = await environment.exec(
            command=self._shell(install),
            cwd=self.workdir,
            timeout_sec=self.setup_timeout_secs,
        )
        self._write_log("setup-stdout.txt", result.stdout)
        self._write_log("setup-stderr.txt", result.stderr)
        if result.return_code != 0:
            raise RuntimeError(f"shunt installer failed with exit {result.return_code}")
        await self._publish_shunt_binary(environment)
        if not await self._shunt_available(environment):
            raise RuntimeError("shunt installer completed but shunt is not on PATH")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.shunt_model:
            raise ValueError("set SHUNT_MODEL or pass Harbor -m/--model")

        config = self._config_toml()
        command = (
            "mkdir -p .shunt && "
            f"cat > .shunt/config.toml <<'SHUNT_CONFIG'\n{config}\nSHUNT_CONFIG\n"
            f"shunt agent --once {shlex.quote(instruction)}"
        )
        result = await environment.exec(
            command=self._shell(command),
            cwd=self.workdir,
            timeout_sec=self.run_timeout_secs,
        )
        self._write_log("shunt-stdout.txt", result.stdout)
        self._write_log("shunt-stderr.txt", result.stderr)
        await self._capture_shunt_files(environment)

        context.metadata = {
            "endpoint": self.endpoint,
            "model": self.shunt_model,
            "return_code": result.return_code,
        }

        if result.return_code != 0:
            raise RuntimeError(f"shunt failed with exit {result.return_code}")

    def _model_from_harbor(self) -> str | None:
        if not self.model_name:
            return None
        return self.model_name.split("/", maxsplit=1)[-1]

    async def _shunt_available(self, environment: BaseEnvironment) -> bool:
        result = await environment.exec(
            command=self._shell("command -v shunt"),
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code == 0:
            self._write_log("shunt-path.txt", result.stdout)
            return True
        self._write_log("shunt-path-stderr.txt", result.stderr)
        return False

    async def _install_curl_if_needed(self, environment: BaseEnvironment) -> None:
        command = (
            "command -v curl >/dev/null 2>&1 || "
            "if command -v apt-get >/dev/null 2>&1; then "
            "DEBIAN_FRONTEND=noninteractive apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates xz-utils; "
            "elif command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache curl ca-certificates xz; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum install -y curl ca-certificates xz; "
            "else echo 'curl is required to install shunt' >&2; exit 1; fi"
        )
        result = await environment.exec(
            command=self._shell(command),
            cwd=self.workdir,
            timeout_sec=self.setup_timeout_secs,
            user=0,
        )
        self._write_log("setup-curl-stdout.txt", result.stdout)
        self._write_log("setup-curl-stderr.txt", result.stderr)
        if result.return_code != 0:
            raise RuntimeError(f"curl setup failed with exit {result.return_code}")

    async def _publish_shunt_binary(self, environment: BaseEnvironment) -> None:
        command = (
            "bin=$(command -v shunt 2>/dev/null || true); "
            "if [ -n \"$bin\" ] && [ -x \"$bin\" ]; then "
            "install -m 0755 \"$bin\" /usr/local/bin/shunt; exit 0; fi; "
            "for candidate in /root/.cargo/bin/shunt /home/*/.cargo/bin/shunt; do "
            "if [ -x \"$candidate\" ]; then "
            "install -m 0755 \"$candidate\" /usr/local/bin/shunt; exit 0; fi; "
            "done; "
            "echo 'installed shunt binary not found' >&2; exit 1"
        )
        result = await environment.exec(
            command=self._shell(command),
            cwd=self.workdir,
            timeout_sec=30,
            user=0,
        )
        self._write_log("setup-path-stdout.txt", result.stdout)
        self._write_log("setup-path-stderr.txt", result.stderr)
        if result.return_code != 0:
            raise RuntimeError(f"shunt PATH setup failed with exit {result.return_code}")

    async def _capture_shunt_files(self, environment: BaseEnvironment) -> None:
        for source, log_name in (
            (".shunt/debug.log", "shunt-debug.log"),
            (".shunt/config.toml", "shunt-config.toml"),
        ):
            result = await environment.exec(
                command=self._shell(f"test -f {shlex.quote(source)} && cat {shlex.quote(source)}"),
                cwd=self.workdir,
                timeout_sec=30,
            )
            if result.return_code == 0:
                self._write_log(log_name, result.stdout)

    def _config_toml(self) -> str:
        return "\n".join(
            [
                f"endpoint = {json.dumps(self.endpoint)}",
                f"model = {json.dumps(self.shunt_model)}",
                f"timeout_secs = {self.timeout_secs}",
                "",
                "[agent]",
                "max_turns = 48",
            ]
        )

    def _shell(self, command: str) -> str:
        return f"set -e; {self._PATH_EXPORT}; {command}"

    def _write_log(self, name: str, text: str | None) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / name).write_text(text or "")
