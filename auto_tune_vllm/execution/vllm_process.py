import atexit
import logging
import os
import socket
import subprocess
import time

import requests

from auto_tune_vllm.core.trial import TrialConfig


class vLLMProcess:
    def __init__(self, logger: logging.Logger, trial_config: TrialConfig):
        self.process: subprocess.Popen[str] | None = None
        self.logger: logging.Logger = logger
        self._trial_config: TrialConfig = trial_config
        self._started: bool = False
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def started(self) -> bool:
        return self._started

    def build_cmd(self) -> list[str]:
        self._port = self._get_available_port()
        cmd = [
            "vllm",
            "serve",
            "--model",
            self._trial_config.benchmark_config.model,
            "--port",
            str(self._port),
            "--host",
            "0.0.0.0",
            "--no-enable-prefix-caching",
        ]
        cmd.extend(self._trial_config.vllm_args)
        self.logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        return cmd

    def get_os_env_vars(self):
        """
        Grab OS environments and update them with anything from the trial config
        """
        env = os.environ.copy()
        trial_env_vars = self._trial_config.environment_vars
        if trial_env_vars:
            env.update(trial_env_vars)
            self.logger.info(f"Environment variables: {trial_env_vars}")

        return env

    def start(self):
        """Start the vLLM serve command in a subprocess."""
        if self.process:
            return {
                "port": self._port,
                "url": f"http://localhost:{self._port}/v1",
                "pid": self.process.pid if self.process else None,
            }

        cmd = self.build_cmd()
        env = self.get_os_env_vars()
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            universal_newlines=True,
            start_new_session=True,
        )
        atexit.register(self.cleanup)

        return {
            "port": self._port,
            "url": self.url_for("v1"),
            "pid": self.process.pid,
        }

    def cleanup(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    @property
    def url_root(self) -> str:
        return f"http://localhost:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def wait_for_server_ready(self, timeout: int = 300):
        """Wait for vLLM server to be ready."""

        start_time = time.time()
        health_url = self.url_for("health")

        self.logger.info(
            f"Waiting for vLLM server to be ready at {health_url} (timeout: {timeout}s)"
        )

        while time.time() - start_time < timeout:
            # Check if vLLM process has died during startup
            if self.process and self.process.poll() is not None:
                status = self.process.returncode
                self.logger.error(
                    f"vLLM process died during startup with exit code {status}"
                )
                raise RuntimeError(
                    f"vLLM process died during startup with exit code {status}"
                )

            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"vLLM server ready at {self.url_root}")
                    return
            except requests.exceptions.RequestException as e:
                self.logger.debug(f"Health check failed: {e}")
            time.sleep(2)
        self.logger.error(f"vLLM server failed to start within {timeout} seconds")

        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds")

    def _get_available_port(self) -> int:
        if self._port:
            return self._port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        return port

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        self.cleanup()

        return False
