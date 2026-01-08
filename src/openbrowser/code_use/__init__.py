"""Code-use mode - Jupyter notebook-like code execution for browser automation."""

from openbrowser.code_use.namespace import create_namespace
from openbrowser.code_use.notebook_export import export_to_ipynb, session_to_python_script
from openbrowser.code_use.service import CodeAgent
from openbrowser.code_use.views import CodeCell, ExecutionStatus, NotebookSession

__all__ = [
	'CodeAgent',
	'create_namespace',
	'export_to_ipynb',
	'session_to_python_script',
	'CodeCell',
	'ExecutionStatus',
	'NotebookSession',
]
