"""Console entry point for openbrowser.

The CLI keeps its fast paths before heavy imports so `--mcp`, `-c`, and daemon
commands do not load the agent stack. This wrapper executes `openbrowser.cli`
as `__main__`, preserving that ordering without making a normal library import
of `openbrowser.cli` execute a command.
"""

import runpy


def cli() -> None:
	runpy.run_module('openbrowser.cli', run_name='__main__', alter_sys=False)


if __name__ == '__main__':
	cli()
