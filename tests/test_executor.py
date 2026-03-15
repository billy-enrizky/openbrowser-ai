# tests/test_executor.py
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_browser_session():
    session = MagicMock()
    session.start = AsyncMock()
    session.kill = AsyncMock()
    session.reset = AsyncMock()
    session.event_bus = MagicMock()
    session.event_bus.dispatch = AsyncMock()
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = '/tmp/test-downloads'
    session.browser_profile.user_data_dir = '/tmp/test-profile'
    return session


@pytest.fixture
def mock_namespace(mock_browser_session):
    """A minimal namespace dict that mimics create_namespace() output."""
    return {
        'browser': mock_browser_session,
        'file_system': None,
        'json': __import__('json'),
        'asyncio': __import__('asyncio'),
    }


class TestCodeExecutor:
    """Tests for CodeExecutor code wrapping and execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('print("hello world")')
        assert result.success is True
        assert 'hello world' in result.output

    @pytest.mark.asyncio
    async def test_execute_variable_assignment(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('x = 42')
        assert result.success is True
        assert mock_namespace.get('x') == 42

    @pytest.mark.asyncio
    async def test_execute_variable_persistence(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        await executor.execute('my_var = "persisted"')
        result = await executor.execute('print(my_var)')
        assert result.success is True
        assert 'persisted' in result.output

    @pytest.mark.asyncio
    async def test_execute_error_returns_traceback(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('raise ValueError("test error")')
        assert result.success is False
        assert 'ValueError' in result.output
        assert 'test error' in result.output

    @pytest.mark.asyncio
    async def test_execute_no_output(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('pass')
        assert result.success is True
        assert 'executed successfully' in result.output

    @pytest.mark.asyncio
    async def test_execute_stdout_before_error(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('print("before")\nraise RuntimeError("boom")')
        assert result.success is False
        assert 'before' in result.output
        assert 'RuntimeError' in result.output

    @pytest.mark.asyncio
    async def test_execute_return_value_display(self, mock_namespace):
        """Test that __ob_result__ triggers the return-value display path."""
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor()
        executor._namespace = mock_namespace

        result = await executor.execute('__ob_result__ = {"answer": 42}')
        assert result.success is True
        assert '42' in result.output
        assert 'answer' in result.output

    @pytest.mark.asyncio
    async def test_output_truncation(self, mock_namespace):
        from openbrowser.code_use.executor import CodeExecutor

        executor = CodeExecutor(max_output_chars=100)
        executor._namespace = mock_namespace

        result = await executor.execute('print("x" * 500)')
        assert result.success is True
        assert len(result.output) <= 200  # some overhead for truncation message
        assert 'truncated' in result.output
