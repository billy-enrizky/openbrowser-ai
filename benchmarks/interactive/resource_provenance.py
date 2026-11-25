"""Bind an installed console launcher to the application source being compared.

The probe reads import specifications and distribution metadata. It never imports
OpenBrowser or invokes the console entry point. Public results contain digests and
closed reason codes, not local installation or checkout paths.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

_PROBE = """
import sys
sys.path[0] = sys.argv[1]
import importlib.metadata
import importlib.util
import json
try:
    if any(name == "openbrowser" or name.startswith("openbrowser.") for name in sys.modules):
        result = {"error": "application_already_imported"}
    else:
        spec = importlib.util.find_spec("openbrowser")
        dist = importlib.metadata.distribution("openbrowser-ai")
        result = {
            "origin": spec.origin if spec is not None else None,
            "locations": list(spec.submodule_search_locations or []) if spec is not None else [],
            "dist_name": dist.metadata.get("Name"),
            "dist_version": dist.version,
            "dist_location": str(dist.locate_file("")),
        }
except Exception:
    result = {"error": "probe_failed"}
print(json.dumps(result))
"""

_SIMPLE_ENTRY = """
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    sys.exit(cli())
"""
_UV_ENTRY = """
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(cli())
"""
_PIP_ENTRY = """
import re
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\\.pyw|\\.exe)?$", "", sys.argv[0])
    sys.exit(cli())
"""
_ENTRY_TREES = frozenset(ast.dump(ast.parse(body)) for body in (_SIMPLE_ENTRY, _UV_ENTRY, _PIP_ENTRY))


def application_digest(package_dir: Path) -> str:
    """Hash package files, including runtime data, independently of their root.

    Python bytecode caches are ignored. The caller supplies the application
    package, not the repository, so harnesses and benchmark results are outside
    this digest. Symlinked content is unsupported rather than silently omitted.
    """
    package_dir = package_dir.resolve(strict=True)
    if not package_dir.is_dir() or not (package_dir / "__init__.py").is_file():
        raise ValueError("application_package_required")
    digest = hashlib.sha256(b"openbrowser-application-v1\0")
    for path in sorted(package_dir.rglob("*"), key=lambda item: item.relative_to(package_dir).as_posix()):
        relative = path.relative_to(package_dir)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("symlinked_package_content")
        if not path.is_file():
            continue
        name = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _launcher_interpreter(content: bytes) -> str | None:
    try:
        source = content.decode("utf-8")
        first_line = source.splitlines()[0]
        if not first_line.startswith("#!") or ast.dump(ast.parse(source)) not in _ENTRY_TREES:
            return None
        interpreter = first_line[2:].strip()
        path = Path(interpreter)
        if (
            not path.is_absolute()
            or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", path.name) is None
            or any(character.isspace() for character in interpreter)
            or not path.is_file()
            or not os.access(path, os.X_OK)
        ):
            return None
        # Do not resolve a venv's interpreter symlink: its invocation path selects the venv.
        return interpreter
    except (IndexError, UnicodeError, SyntaxError, OSError, ValueError):
        return None


def bind_openbrowser(binary: str, expected_package: Path, *, env: dict[str, str], cwd: Path) -> dict[str, object]:
    """Fail closed unless the launcher resolves to the exact or identical app.

    ``env`` is the adapter's environment. PYTHONPATH is removed and the probe's
    initial import directory is the resolved launcher directory, as for executing
    that script, not the benchmark's current directory. This binds app content,
    not dependencies, interpreter contents, or a future race-free execution.
    """
    result: dict[str, object] = {
        "verified": False,
        "reason": "binding_failed",
        "entry_sha256": None,
        "expected_app_digest": None,
        "actual_app_digest": None,
        "dist_version": None,
        "origin_matches_expected": False,
    }
    try:
        expected_package = expected_package.resolve(strict=True)
        result["expected_app_digest"] = application_digest(expected_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "expected_package_unreadable"
        return result

    try:
        if os.path.dirname(binary):
            entry = Path(binary)
            if not entry.is_absolute():
                entry = cwd / entry
        else:
            search_path = os.pathsep.join(
                str(Path(part) if Path(part).is_absolute() else cwd / part)
                for part in env.get("PATH", os.defpath).split(os.pathsep)
            )
            found = shutil.which(binary, path=search_path)
            if found is None:
                result["reason"] = "entry_not_found"
                return result
            entry = Path(found)
            if not entry.is_absolute():
                entry = cwd / entry
        entry = entry.resolve(strict=True)
        content = entry.read_bytes()
        result["entry_sha256"] = hashlib.sha256(content).hexdigest()
        interpreter = _launcher_interpreter(content)
        if interpreter is None or not os.access(entry, os.X_OK):
            result["reason"] = "unsupported_launcher"
            return result
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "entry_unreadable"
        return result

    probe_env = {key: value for key, value in env.items() if key != "PYTHONPATH"}
    try:
        probe = subprocess.run(
            [interpreter, "-c", _PROBE, str(entry.parent)],
            env=probe_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0:
            result["reason"] = "probe_failed"
            return result
        payload = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        result["reason"] = "probe_failed"
        return result
    if not isinstance(payload, dict) or "error" in payload:
        result["reason"] = "probe_invalid"
        return result

    version = payload.get("dist_version")
    if (
        payload.get("dist_name") != "openbrowser-ai"
        or not isinstance(version, str)
        or re.fullmatch(r"[0-9][A-Za-z0-9.+!_-]{0,127}", version) is None
        or not isinstance(payload.get("dist_location"), str)
    ):
        result["reason"] = "distribution_metadata_invalid"
        return result
    result["dist_version"] = version
    origin, locations = payload.get("origin"), payload.get("locations")
    if (
        not isinstance(origin, str)
        or not isinstance(locations, list)
        or len(locations) != 1
        or not isinstance(locations[0], str)
    ):
        result["reason"] = "ambiguous_package"
        return result
    try:
        origin_path = Path(origin).resolve(strict=True)
        actual_package = Path(locations[0]).resolve(strict=True)
        if origin_path.name != "__init__.py" or origin_path.parent != actual_package:
            result["reason"] = "inconsistent_package_spec"
            return result
        result["origin_matches_expected"] = actual_package == expected_package
        result["actual_app_digest"] = application_digest(actual_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "actual_package_unreadable"
        return result
    if result["actual_app_digest"] != result["expected_app_digest"]:
        result["reason"] = "application_digest_mismatch"
        return result
    result["verified"] = True
    result["reason"] = "exact_package" if result["origin_matches_expected"] else "byte_identical_package"
    return result
