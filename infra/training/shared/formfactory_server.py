"""FormFactory Flask server lifecycle manager for training.

Extracted from infra/eval/pipelines/eval_benchmark.py to be shared
between the eval pipeline and online GRPO training.

Manages starting/stopping the FormFactory Flask application as a
subprocess on localhost.
"""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class FormFactoryServer:
    """Manages the FormFactory Flask server lifecycle."""

    def __init__(self, formfactory_dir: Path, port: int = 5050):
        self.formfactory_dir = formfactory_dir
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(self) -> bool:
        """Start the FormFactory Flask server.

        Returns True if server started successfully.
        """
        app_py = self.formfactory_dir / "app.py"
        if not app_py.exists():
            logger.error(
                f"FormFactory app.py not found at {app_py}. "
                "Run: uv run infra/eval/scripts/download_datasets.py --datasets formfactory"
            )
            return False

        # Check if something is already running on the port
        if self._is_running():
            logger.info(f"FormFactory server already running on port {self.port}")
            return True

        logger.info(f"Starting FormFactory server on port {self.port}...")
        self.process = subprocess.Popen(
            [
                sys.executable, "-c",
                f"from app import app; app.run(host='127.0.0.1', port={self.port}, debug=False)",
            ],
            cwd=str(self.formfactory_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        for _attempt in range(30):
            time.sleep(1)
            if self._is_running():
                logger.info(f"FormFactory server ready on port {self.port}")
                return True
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                logger.error(f"FormFactory server exited unexpectedly: {stderr[:500]}")
                return False

        logger.error("FormFactory server did not start within 30 seconds")
        self.stop()
        return False

    def stop(self):
        """Stop the FormFactory Flask server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping FormFactory server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            logger.info("FormFactory server stopped")
            self.process = None

    def restart(self) -> bool:
        """Stop and restart the FormFactory server.

        Returns True if the server restarted successfully.
        """
        logger.info("Restarting FormFactory server...")
        self.stop()
        return self.start()

    def is_healthy(self) -> bool:
        """Check if the server is alive and responding."""
        return self._is_running()

    def _is_running(self) -> bool:
        """Check if the server is responding."""
        try:
            urlopen(f"http://127.0.0.1:{self.port}/", timeout=2)
            return True
        except (URLError, OSError):
            return False
