"""Code-use mode - Jupyter notebook-like code execution for browser automation."""

__all__ = [
	'CodeAgent',
	'create_namespace',
	'export_to_ipynb',
	'session_to_python_script',
	'CodeCell',
	'ExecutionStatus',
	'NotebookSession',
]

# Lazy imports to avoid pulling in LLM dependencies when only
# create_namespace is needed (e.g. MCP server mode).
_LAZY_IMPORTS = {
	'CodeAgent': ('openbrowser.code_use.service', 'CodeAgent'),
	'create_namespace': ('openbrowser.code_use.namespace', 'create_namespace'),
	'export_to_ipynb': ('openbrowser.code_use.notebook_export', 'export_to_ipynb'),
	'session_to_python_script': ('openbrowser.code_use.notebook_export', 'session_to_python_script'),
	'CodeCell': ('openbrowser.code_use.views', 'CodeCell'),
	'ExecutionStatus': ('openbrowser.code_use.views', 'ExecutionStatus'),
	'NotebookSession': ('openbrowser.code_use.views', 'NotebookSession'),
}


def __getattr__(name: str):
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		import importlib
		module = importlib.import_module(module_path)
		return getattr(module, attr_name)
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
