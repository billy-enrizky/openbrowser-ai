"""Installer help explains the public frontend commands before installation."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = "https://docs.openbrowser.me/customize/integrations/native-cli"


@pytest.mark.skipif(os.name == "nt", reason="The POSIX installer targets macOS and Linux")
def test_posix_help_describes_frontends_without_installer_dependencies(tmp_path):
    shell = shutil.which("sh")
    assert shell is not None
    result = subprocess.run(
        [shell, str(ROOT / "install.sh"), "--help"],
        env={"PATH": str(tmp_path), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "openbrowser-ai-py" in result.stdout
    assert "0.1.61" in result.stdout
    assert GUIDE in result.stdout


def test_powershell_help_declares_the_python_frontend_commands():
    text = (ROOT / "install.ps1").read_text()
    match = re.search(r"\{ \$_ -in '--help', '-h' \} \{(.*?)exit 0", text, re.DOTALL)
    assert match is not None
    assert "openbrowser-ai-py" in match[1]
    assert "Python" in match[1]
    assert GUIDE in match[1]


def test_both_installer_summaries_link_to_frontend_recovery():
    for name in ["install.sh", "install.ps1"]:
        text = (ROOT / name).read_text()
        assert text.count(GUIDE) >= 2
        assert text.count("openbrowser-ai-py") >= 2
