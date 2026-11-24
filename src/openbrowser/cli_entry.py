"""Console entry point for openbrowser.

The CLI keeps its fast paths before heavy imports so `--mcp`, `-c`, and daemon
commands do not load the agent stack. This wrapper executes `openbrowser.cli`
as `__main__`, preserving that ordering without making a normal library import
of `openbrowser.cli` execute a command.
"""

import runpy
import sys

from openbrowser.cli_modes import CliMode, select_cli_mode


def _execute_code_fast_path(argv: list[str]) -> bool:
	"""Run `-c` against a warm daemon without importing the Click tree."""
	if select_cli_mode(argv[1:]) is not CliMode.EXECUTE_CODE:
		return False
	try:
		code_index = argv.index('-c')
	except ValueError:
		return False
	code = argv[code_index + 1] if code_index + 1 < len(argv) else None
	if code is None:
		return False
	if code == '-':
		code = sys.stdin.read().strip()

	from openbrowser.daemon.thin_client import execute_code_sync

	response = execute_code_sync(code)
	if response is None:
		# No warm daemon. The normal CLI path owns cold startup.
		return False
	if response.success:
		if response.output:
			print(response.output)
		return True
	print(response.output or response.error, file=sys.stderr)
	if response.error_code:
		print(f'Error code: {response.error_code}', file=sys.stderr)
	if response.hint:
		print(f'Hint: {response.hint}', file=sys.stderr)
	raise SystemExit(1)


def cli() -> None:
	if _execute_code_fast_path(sys.argv):
		return
	runpy.run_module('openbrowser.cli', run_name='__main__', alter_sys=False)


if __name__ == '__main__':
	cli()
