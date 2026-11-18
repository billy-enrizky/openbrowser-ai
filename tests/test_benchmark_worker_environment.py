import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/navigation_compare.py'
spec = importlib.util.spec_from_file_location('navigation_compare_environment_tests', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def worker_launch(worker_root, source_root, arm='openbrowser', mode=None):
    worker_root = Path(worker_root)
    runtime_root = worker_root.parent
    if worker_root.name != arm:
        worker_root = runtime_root / arm
    return benchmark.make_worker_launch(
        arm=arm,
        mode=mode or benchmark.WorkerMode.NAVIGATION,
        source_root=Path(source_root),
        candidate_python=Path(sys.executable),
        browser_use_python=Path(sys.executable),
        runtime_root=runtime_root,
        worker_root=worker_root,
        working_directory=worker_root,
        log_directory=runtime_root.parent / 'logs',
        log_prefix='worker',
        owner_token='private-owner',
    )


def test_worker_environment_drops_parent_sentinels(tmp_path, monkeypatch):
    worker_root = tmp_path / 'worker'
    worker_root.mkdir()
    source_root = tmp_path / 'candidate'
    source_root.mkdir()
    parent_values = {
        'AWS_ACCESS_KEY_ID': 'parent-access-key',
        'AWS_SECRET_ACCESS_KEY': 'parent-secret-key',
        'AWS_WEB_IDENTITY_TOKEN_FILE': '/parent/token',
        'AWS_CONTAINER_CREDENTIALS_FULL_URI': 'http://parent/credentials',
        'AWS_PROFILE': 'parent-profile',
        'AWS_EC2_METADATA_DISABLED': 'false',
        'AWS_CONFIG_FILE': '/parent/config',
        'AWS_SHARED_CREDENTIALS_FILE': '/parent/credentials',
        'NAV_BENCH_SOURCE': '/parent/source',
        'NAVIGATION_BENCHMARK_OWNER': 'parent-owner',
        'UV_PROJECT_ENVIRONMENT': '/parent/venv',
        'BH_RUNTIME_DIR': '/parent/runtime',
        'OPENBROWSER_CHROME_PATH': '/parent/chrome',
        'AGENT_BROWSER_SOCKET_DIR': '/parent/socket',
        'PYTHONPATH': '/parent/pythonpath',
        'PATH': '/parent/path',
    }
    for name, value in parent_values.items():
        monkeypatch.setenv(name, value)

    env = benchmark.worker_environment(worker_launch(worker_root, source_root))

    credential_keys = {
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_WEB_IDENTITY_TOKEN_FILE',
        'AWS_CONTAINER_CREDENTIALS_FULL_URI', 'AWS_PROFILE', 'NAV_BENCH_SOURCE',
        'UV_PROJECT_ENVIRONMENT',
    }
    assert credential_keys.isdisjoint(env)
    assert env['AWS_EC2_METADATA_DISABLED'] == 'true'
    assert env['AWS_CONFIG_FILE'] == '/dev/null'
    assert env['AWS_SHARED_CREDENTIALS_FILE'] == '/dev/null'
    assert env['NAVIGATION_BENCHMARK_OWNER'] != parent_values['NAVIGATION_BENCHMARK_OWNER']
    assert env['BH_RUNTIME_DIR'] != parent_values['BH_RUNTIME_DIR']
    assert env['AGENT_BROWSER_SOCKET_DIR'] != parent_values['AGENT_BROWSER_SOCKET_DIR']
    assert 'OPENBROWSER_CHROME_PATH' not in env
    assert env['PYTHON_DOTENV_DISABLED'] == '1'
    assert env['PATH'] != parent_values['PATH']
    assert env['PYTHONPATH'] == str(source_root / 'src')
    assert env['NAVIGATION_BENCHMARK_OWNER'] == 'private-owner'
    assert env['AWS_EC2_METADATA_DISABLED'] == 'true'
    assert env['AWS_CONFIG_FILE'] == '/dev/null'
    assert env['AWS_SHARED_CREDENTIALS_FILE'] == '/dev/null'


def test_openbrowser_import_ignores_source_env_file(tmp_path, monkeypatch):
    source_root = tmp_path / 'candidate'
    package = source_root / 'src' / 'openbrowser'
    package.parent.mkdir(parents=True)
    shutil.copytree(SCRIPT.parents[1] / 'src' / 'openbrowser', package)
    sentinel = str(tmp_path / 'source-env-secret')
    (source_root / '.env').write_text(
        f'OPENBROWSER_DEBUG_LOG_FILE={sentinel}\nAWS_ACCESS_KEY_ID=fake-source-key\n'
    )
    worker_root = tmp_path / 'worker'
    worker_root.mkdir()
    worker_cwd = tmp_path / 'worker-cwd'
    worker_cwd.mkdir()
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '0')
    monkeypatch.delenv('OPENBROWSER_DEBUG_LOG_FILE', raising=False)

    env = benchmark.worker_environment(worker_launch(worker_root, source_root))
    for name in ('HOME', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'BH_HOME', 'BH_RUNTIME_DIR',
                 'BH_TMP_DIR', 'AGENT_BROWSER_SOCKET_DIR'):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    dotenv_script = (
        'import os; import openbrowser.logging_config; '
        'assert os.getenv("OPENBROWSER_DEBUG_LOG_FILE") is None'
    )
    dotenv_result = subprocess.run(
        [sys.executable, '-c', dotenv_script], cwd=source_root, env=env,
        capture_output=True, check=False, text=True,
    )
    assert dotenv_result.returncode == 0, dotenv_result.stderr

    settings_script = (
        'from openbrowser.config import CONFIG; '
        f'assert str(CONFIG.OPENBROWSER_DEBUG_LOG_FILE) != {sentinel!r}'
    )
    settings_result = subprocess.run(
        [sys.executable, '-c', settings_script], cwd=worker_cwd, env=env,
        capture_output=True, check=False, text=True,
    )

    assert settings_result.returncode == 0, settings_result.stderr
