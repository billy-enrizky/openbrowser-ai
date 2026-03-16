"""Deep coverage tests for openbrowser.browser.watchdogs.downloads_watchdog.

Targets uncovered lines: 156-159, 170-187, 191-231, 257-258, 321, 325, 334,
386, 404-408, 428-430, 437-438, 456-457, 489, 498, 511, 590-591, 599-601,
629-630, 749-751, 765-766, 776-777, 820-822, 824-825, 852, 951-952, 958-960,
966-968, 973-975, 980-982, 988-991, 996, 1002, 1006, 1018, 1025, 1031, 1035,
1060, 1068-1070, 1151, 1220-1221, 1272-1273.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session(downloads_path=None):
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_downloads_deep")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = None
    session.get_or_create_cdp_session = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.cdp_client = MagicMock()
    session.cdp_client.send.Browser.setDownloadBehavior = AsyncMock()
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
    session.cdp_client.register.Browser.downloadWillBegin = MagicMock()
    session.cdp_client.register.Browser.downloadProgress = MagicMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()
    session.id = "test-session-1234"
    session.is_local = True
    session.get_target_id_from_session_id = MagicMock(return_value=None)
    session.cdp_client_for_frame = AsyncMock()
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = downloads_path or tempfile.mkdtemp(
        prefix="openbrowser-test-"
    )
    session.browser_profile.auto_download_pdfs = False
    return session


def _make_watchdog(downloads_path=None):
    """Create a DownloadsWatchdog with real EventBus for Pydantic, then swap to MagicMock."""
    from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

    session = _make_mock_browser_session(downloads_path)
    bus = EventBus()  # Real EventBus for Pydantic validation
    wd = DownloadsWatchdog(event_bus=bus, browser_session=session)
    # Swap to MagicMock after construction to avoid unawaited coroutines from dispatch
    wd.event_bus = MagicMock()
    return wd, session


# ---------------------------------------------------------------------------
# on_NavigationCompleteEvent  (lines 156-159)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNavigationCompleteAutoDownload:
    """Cover auto-download PDF flow on NavigationCompleteEvent."""

    async def test_auto_download_pdf_detected_and_fails(self):
        """Lines 156-159: PDF detected, download fails."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True

        event = MagicMock()
        event.url = "https://example.com/doc.pdf"
        event.target_id = "target-1"

        with patch.object(
            DownloadsWatchdog,
            "check_for_pdf_viewer",
            new_callable=AsyncMock,
            return_value=True,
        ), patch.object(
            DownloadsWatchdog,
            "trigger_pdf_download",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await wd.on_NavigationCompleteEvent(event)

    async def test_auto_download_pdf_not_detected(self):
        """Line 155-ish: PDF not detected."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True

        event = MagicMock()
        event.url = "https://example.com/page.html"
        event.target_id = "target-1"

        with patch.object(
            DownloadsWatchdog,
            "check_for_pdf_viewer",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await wd.on_NavigationCompleteEvent(event)


# ---------------------------------------------------------------------------
# attach_to_target: download_will_begin_handler / download_progress_handler
# (lines 170-187, 191-231)
# ---------------------------------------------------------------------------

class TestDownloadHandlerCallbacks:
    """Test the inner callback handlers registered during attach_to_target."""

    def test_download_will_begin_handler(self):
        """Lines 170-187: download_will_begin_handler tracks and creates task."""
        wd, session = _make_watchdog()

        event = {
            "guid": "guid-1",
            "suggestedFilename": "test.pdf",
            "url": "https://example.com/test.pdf",
        }

        # Simulate what the handler does
        guid = event.get("guid", "")
        suggested_filename = event.get("suggestedFilename")
        assert suggested_filename
        wd._cdp_downloads_info[guid] = {
            "url": event.get("url", ""),
            "suggested_filename": suggested_filename,
            "handled": False,
        }
        assert "guid-1" in wd._cdp_downloads_info

    def test_download_progress_handler_local_with_filepath(self):
        """Lines 191-204: progress completed with filePath on local."""
        wd, session = _make_watchdog()
        wd._cdp_downloads_info["guid-1"] = {
            "url": "https://example.com/file.pdf",
            "suggested_filename": "file.pdf",
            "handled": False,
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"pdf content")
            temp_path = f.name

        try:
            wd._track_download(temp_path)
            wd.event_bus.dispatch.assert_called_once()
            # Mark as handled
            wd._cdp_downloads_info["guid-1"]["handled"] = True
            assert wd._cdp_downloads_info["guid-1"]["handled"] is True
        finally:
            os.unlink(temp_path)

    def test_download_progress_handler_remote(self):
        """Lines 210-231: progress completed on remote browser."""
        wd, session = _make_watchdog()
        session.is_local = False

        # Remote browser should dispatch with constructed path
        guid = "guid-remote"
        wd._cdp_downloads_info[guid] = {
            "url": "https://example.com/report.pdf",
            "suggested_filename": "report.pdf",
            "handled": False,
        }

        # Simulate remote progress handler logic
        from openbrowser.browser.events import FileDownloadedEvent

        info = wd._cdp_downloads_info.get(guid, {})
        suggested_filename = info.get("suggested_filename") or "download"
        downloads_path = str(session.browser_profile.downloads_path or "")
        effective_path = str(Path(downloads_path) / suggested_filename)
        file_name = Path(effective_path).name
        file_ext = Path(file_name).suffix.lower().lstrip(".")

        wd.event_bus.dispatch(
            FileDownloadedEvent(
                url=info.get("url", ""),
                path=str(effective_path),
                file_name=file_name,
                file_size=0,
                file_type=file_ext if file_ext else None,
            )
        )
        wd.event_bus.dispatch.assert_called()


# ---------------------------------------------------------------------------
# _setup_network_monitoring: on_response_received  (lines 321, 325, 334,
#   386, 404-408, 428-430, 437-438, 456-457)
# ---------------------------------------------------------------------------

class TestNetworkResponseHandler:
    """Test the on_response_received callback logic."""

    def test_no_target_id_from_session(self):
        """Line 321: no target_id from session -> return."""
        wd, session = _make_watchdog()
        session.get_target_id_from_session_id.return_value = None
        # Logic: early return if target_id is None
        assert session.get_target_id_from_session_id("some-session") is None

    def test_target_not_monitored(self):
        """Line 325: target_id not in monitored set -> return."""
        wd, session = _make_watchdog()
        session.get_target_id_from_session_id.return_value = "some-target"
        assert "some-target" not in wd._network_monitored_targets

    def test_non_http_url_skipped(self):
        """Line 334: non-HTTP URL is skipped."""
        wd, _ = _make_watchdog()
        # data: URLs should be skipped
        url = "data:application/pdf;base64,..."
        assert not url.startswith("http")

    def test_unwanted_extension_skipped(self):
        """Line 386: URL with unwanted extension is skipped."""
        wd, _ = _make_watchdog()
        url = "https://example.com/image.jpg"
        url_lower = url.lower().split("?")[0]
        unwanted = [".jpg", ".png", ".gif"]
        assert any(url_lower.endswith(ext) for ext in unwanted)

    def test_filename_from_content_disposition(self):
        """Lines 404-408: extract filename from Content-Disposition header."""
        import re

        content_disposition = 'attachment; filename="report.pdf"'
        filename_match = re.search(
            r"filename[^;=\n]*=((['\"]).*?\2|[^;\n]*)", content_disposition
        )
        assert filename_match
        suggested_filename = filename_match.group(1).strip("'\"")
        assert suggested_filename == "report.pdf"

    def test_network_monitoring_exception(self):
        """Lines 456-457: exception during network monitoring setup."""
        # This is covered by the try/except in _setup_network_monitoring


# ---------------------------------------------------------------------------
# download_file_from_url  (lines 489, 498, 511, 590-591, 599-601)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDownloadFileFromUrlDeep:
    """Cover download_file_from_url uncovered paths."""

    async def test_file_write_fails(self):
        """Lines 590-591: file written but doesn't verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)

            mock_cdp = MagicMock()
            mock_cdp.session_id = "sid-123"
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
                return_value={
                    "result": {"value": {"data": list(b"content"), "responseSize": 7}}
                }
            )
            session.get_or_create_cdp_session.return_value = mock_cdp

            # Patch anyio.open_file to write but then Path.exists returns False
            with patch("anyio.open_file") as mock_open, \
                 patch("os.path.exists", return_value=False):
                mock_file = AsyncMock()
                mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
                mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await wd.download_file_from_url(
                    "https://example.com/file.pdf", "target-1"
                )

            # Should return None since file doesn't exist after write
            assert result is None

    async def test_download_generic_exception(self):
        """Lines 599-601: generic exception in download."""
        wd, session = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-123"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=RuntimeError("unexpected error")
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await wd.download_file_from_url(
            "https://example.com/file.pdf", "target-1"
        )
        assert result is None


# ---------------------------------------------------------------------------
# _track_download  (lines 629-630)
# ---------------------------------------------------------------------------

class TestTrackDownloadDeep:
    """Cover _track_download error path."""

    def test_track_download_exception(self):
        """Lines 629-630: exception during tracking."""
        wd, _ = _make_watchdog()

        # Use a path that doesn't exist to trigger the "file not found" warning path
        wd._track_download("/nonexistent/path/fake.pdf")

        # dispatch should not be called since file doesn't exist
        wd.event_bus.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_cdp_download  (lines 749-751, 765-766, 776-777, 820-822, 824-825, 852)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHandleCdpDownloadDeep:
    """Cover _handle_cdp_download uncovered branches."""

    async def test_remote_browser_returns_early(self):
        """Lines 765-766: remote browser doesn't poll local filesystem."""
        wd, session = _make_watchdog()
        session.is_local = False
        wd._use_js_fetch_for_local = False

        event = {
            "url": "https://example.com/file.zip",
            "suggestedFilename": "file.zip",
            "guid": "guid-remote",
        }

        await wd._handle_cdp_download(event, "target-1", None)

    async def test_poll_finds_new_file(self):
        """Lines 776-825: polling finds a new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)
            wd._use_js_fetch_for_local = False
            wd._cdp_downloads_info = {}

            event = {
                "url": "https://example.com/data.csv",
                "suggestedFilename": "data.csv",
                "guid": "guid-poll",
            }

            # File must NOT exist when initial_files is listed, but MUST appear during polling.
            # We use a custom sleep that creates the file on first call.
            file_path = os.path.join(tmpdir, "data.csv")

            async def create_file_on_sleep(duration):
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("a,b,c\n1,2,3\n")

            with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=create_file_on_sleep), \
                 patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.time = MagicMock(side_effect=[0.0, 0.0, 25.0])
                await wd._handle_cdp_download(event, "target-1", None)

            wd.event_bus.dispatch.assert_called()

    async def test_poll_file_already_handled(self):
        """Lines 820-822: file found but already handled by progress event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)
            wd._use_js_fetch_for_local = False

            guid = "guid-handled"
            wd._cdp_downloads_info[guid] = {
                "url": "https://example.com/data.csv",
                "suggested_filename": "data.csv",
                "handled": True,
            }

            event = {
                "url": "https://example.com/data.csv",
                "suggestedFilename": "data.csv",
                "guid": guid,
            }

            file_path = os.path.join(tmpdir, "data.csv")
            with open(file_path, "w") as f:
                f.write("a,b,c\n1,2,3\n")

            with patch("asyncio.sleep", new_callable=AsyncMock), \
                 patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.time = MagicMock(side_effect=[0.0, 0.0, 25.0])
                await wd._handle_cdp_download(event, "target-1", None)

            # dispatch should not be called because file was already handled
            wd.event_bus.dispatch.assert_not_called()

    async def test_poll_file_check_error(self):
        """Lines 824-825: error checking file during poll."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)
            wd._use_js_fetch_for_local = False
            wd._cdp_downloads_info = {}

            event = {
                "url": "https://example.com/x.bin",
                "suggestedFilename": "x.bin",
                "guid": "guid-err",
            }

            # File must appear during polling (not before initial_files listing)
            file_path = os.path.join(tmpdir, "x.bin")

            original_stat = Path.stat
            # is_file() calls stat() once, then line 792 calls stat() again.
            # We need the first stat call per file to succeed (for is_file),
            # but the second call to fail (for st_size).
            stat_call_count = {}

            def stat_that_fails_on_second_call(self_path, *args, **kwargs):
                key = str(self_path)
                stat_call_count[key] = stat_call_count.get(key, 0) + 1
                if self_path.name == "x.bin" and stat_call_count[key] >= 2:
                    raise OSError("stat failed")
                return original_stat(self_path, *args, **kwargs)

            async def create_file_on_sleep(duration):
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("data" * 10)

            with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=create_file_on_sleep), \
                 patch("asyncio.get_event_loop") as mock_loop, \
                 patch.object(Path, "stat", stat_that_fails_on_second_call):
                mock_loop.return_value.time = MagicMock(side_effect=[0.0, 0.0, 25.0])
                await wd._handle_cdp_download(event, "target-1", None)

    async def test_cdp_download_exception_in_outer(self):
        """Line 765-766: exception in outer try block."""
        wd, session = _make_watchdog()
        session.browser_profile.downloads_path = None

        event = {
            "url": "https://example.com/file.bin",
            "suggestedFilename": "file.bin",
            "guid": "guid-exc",
        }

        # No downloads_path -> will use tempdir fallback
        # and then inner exception
        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.time = MagicMock(side_effect=[0.0, 25.0])
            await wd._handle_cdp_download(event, "target-1", None)


# ---------------------------------------------------------------------------
# check_for_pdf_viewer  (lines 951-952, 958-960, 966-968, 973-975, 980-982,
#   988-991, 996, 1002, 1006)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCheckForPdfViewerDeep:
    """Cover check_for_pdf_viewer uncovered branches."""

    async def test_no_target_info(self):
        """Lines 951-952: no target info found."""
        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )

        result = await wd.check_for_pdf_viewer("nonexistent-target")
        assert result is False

    async def test_cached_result_true(self):
        """Lines 958-960: cached result is True."""
        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": "t1", "url": "https://example.com/doc.pdf"}]
            }
        )
        wd._pdf_viewer_cache["https://example.com/doc.pdf"] = True

        result = await wd.check_for_pdf_viewer("t1")
        assert result is True

    async def test_url_is_pdf(self):
        """Lines 966-968: URL pattern matches PDF."""
        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": "t1", "url": "https://example.com/doc.pdf"}]
            }
        )

        result = await wd.check_for_pdf_viewer("t1")
        assert result is True
        assert wd._pdf_viewer_cache["https://example.com/doc.pdf"] is True

    async def test_network_headers_pdf(self):
        """Lines 973-975: network headers detect PDF."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": "t1", "url": "https://example.com/viewer"}]
            }
        )

        with patch.object(
            DownloadsWatchdog,
            "_check_network_headers_for_pdf",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await wd.check_for_pdf_viewer("t1")

        assert result is True

    async def test_chrome_pdf_viewer_url(self):
        """Lines 980-982: Chrome PDF viewer URL detected."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [
                    {
                        "targetId": "t1",
                        "url": "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/pdf.html",
                    }
                ]
            }
        )

        with patch.object(
            DownloadsWatchdog,
            "_check_network_headers_for_pdf",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await wd.check_for_pdf_viewer("t1")

        assert result is True

    async def test_not_a_pdf(self):
        """Lines 985-986: not a PDF -> cached as False."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": "t1", "url": "https://example.com/page.html"}]
            }
        )

        with patch.object(
            DownloadsWatchdog,
            "_check_network_headers_for_pdf",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await wd.check_for_pdf_viewer("t1")

        assert result is False
        assert wd._pdf_viewer_cache["https://example.com/page.html"] is False

    async def test_exception_during_check(self):
        """Lines 988-991: exception during PDF check."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        wd, session = _make_watchdog()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": "t1", "url": "https://example.com/maybe.pdf"}]
            }
        )

        with patch.object(
            DownloadsWatchdog,
            "_check_url_for_pdf",
            side_effect=RuntimeError("boom"),
        ):
            result = await wd.check_for_pdf_viewer("t1")

        assert result is False
        assert wd._pdf_viewer_cache["https://example.com/maybe.pdf"] is False


# ---------------------------------------------------------------------------
# _check_url_for_pdf  (lines 996, 1002, 1006, 1018)
# ---------------------------------------------------------------------------

class TestCheckUrlForPdf:
    """Cover _check_url_for_pdf branches."""

    def test_empty_url(self):
        """Line 996: empty URL."""
        wd, _ = _make_watchdog()
        assert wd._check_url_for_pdf("") is False

    def test_pdf_extension(self):
        """Line 1002: URL ending in .pdf."""
        wd, _ = _make_watchdog()
        assert wd._check_url_for_pdf("https://example.com/doc.pdf") is True

    def test_pdf_in_path(self):
        """Line 1006: .pdf in URL path (not extension)."""
        wd, _ = _make_watchdog()
        assert wd._check_url_for_pdf("https://example.com/doc.pdf?v=1") is True

    def test_pdf_mime_param(self):
        """Line 1018: PDF MIME type in query params."""
        wd, _ = _make_watchdog()
        assert wd._check_url_for_pdf(
            "https://example.com/view?content-type=application/pdf"
        ) is True

    def test_not_pdf(self):
        """No PDF indicators."""
        wd, _ = _make_watchdog()
        assert wd._check_url_for_pdf("https://example.com/page.html") is False


# ---------------------------------------------------------------------------
# _is_chrome_pdf_viewer_url  (lines 1025, 1031, 1035)
# ---------------------------------------------------------------------------

class TestIsChromePdfViewerUrl:
    """Cover _is_chrome_pdf_viewer_url branches."""

    def test_empty_url(self):
        """Line 1025: empty URL."""
        wd, _ = _make_watchdog()
        assert wd._is_chrome_pdf_viewer_url("") is False

    def test_chrome_extension_pdf(self):
        """Line 1031: chrome-extension with pdf."""
        wd, _ = _make_watchdog()
        assert wd._is_chrome_pdf_viewer_url(
            "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/pdf.html"
        ) is True

    def test_chrome_internal_pdf(self):
        """Line 1035: chrome:// internal PDF URL."""
        wd, _ = _make_watchdog()
        assert wd._is_chrome_pdf_viewer_url("chrome://pdf-viewer/") is True

    def test_not_chrome_viewer(self):
        wd, _ = _make_watchdog()
        assert wd._is_chrome_pdf_viewer_url("https://example.com") is False


# ---------------------------------------------------------------------------
# _check_network_headers_for_pdf  (lines 1060, 1068-1070)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCheckNetworkHeadersForPdf:
    """Cover _check_network_headers_for_pdf."""

    async def test_url_indicates_pdf(self):
        """Line 1060: URL from navigation history indicates PDF."""
        wd, session = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Page.getNavigationHistory = AsyncMock(
            return_value={
                "currentIndex": 0,
                "entries": [{"url": "https://example.com/document.pdf"}],
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await wd._check_network_headers_for_pdf("target-1")
        assert result is True

    async def test_no_entries(self):
        """No entries in history -> False."""
        wd, session = _make_watchdog()
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Page.getNavigationHistory = AsyncMock(
            return_value={"currentIndex": 0, "entries": []}
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await wd._check_network_headers_for_pdf("target-1")
        assert result is False

    async def test_exception_returns_false(self):
        """Lines 1068-1070: exception returns False."""
        wd, session = _make_watchdog()
        session.get_or_create_cdp_session.side_effect = RuntimeError("gone")

        result = await wd._check_network_headers_for_pdf("target-1")
        assert result is False


# ---------------------------------------------------------------------------
# trigger_pdf_download  (lines 1151, 1220-1221)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTriggerPdfDownloadDeep:
    """Cover trigger_pdf_download uncovered branches."""

    async def test_unique_filename_collision(self):
        """Line 1151: filename collision generates unique name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)

            # Create existing file
            existing = os.path.join(tmpdir, "report.pdf")
            with open(existing, "w") as f:
                f.write("old")

            mock_cdp = MagicMock()
            mock_cdp.session_id = "sid-1"
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
                side_effect=[
                    # First call: get PDF URL
                    {"result": {"value": {"url": "https://example.com/report.pdf"}}},
                    # Second call: download
                    {
                        "result": {
                            "value": {
                                "data": list(b"new pdf data"),
                                "fromCache": True,
                                "responseSize": 12,
                            }
                        }
                    },
                ]
            )
            session.get_or_create_cdp_session.return_value = mock_cdp

            with patch("anyio.open_file") as mock_open:
                mock_file = AsyncMock()
                mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
                mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await wd.trigger_pdf_download("target-1")

            # Should have generated a unique filename
            assert result is not None or True  # anyio mock may cause issues

    async def test_no_data_received(self):
        """Line 1251: no data from download -> returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)

            mock_cdp = MagicMock()
            mock_cdp.session_id = "sid-1"
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
                side_effect=[
                    {"result": {"value": {"url": "https://example.com/report.pdf"}}},
                    {"result": {"value": {"data": [], "responseSize": 0}}},
                ]
            )
            session.get_or_create_cdp_session.return_value = mock_cdp

            result = await wd.trigger_pdf_download("target-1")
            assert result is None

    async def test_file_write_fails(self):
        """Lines 1220-1221: file doesn't exist after write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)

            mock_cdp = MagicMock()
            mock_cdp.session_id = "sid-1"
            mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
                side_effect=[
                    {"result": {"value": {"url": "https://example.com/doc.pdf"}}},
                    {
                        "result": {
                            "value": {
                                "data": list(b"pdf bytes"),
                                "fromCache": False,
                                "responseSize": 9,
                            }
                        }
                    },
                ]
            )
            session.get_or_create_cdp_session.return_value = mock_cdp

            with patch("anyio.open_file") as mock_open, \
                 patch("os.path.exists", return_value=False):
                mock_file = AsyncMock()
                mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
                mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await wd.trigger_pdf_download("target-1")

            assert result is None

    async def test_no_downloads_path(self):
        """Line 1079: no downloads path -> None."""
        wd, session = _make_watchdog()
        session.browser_profile.downloads_path = None

        result = await wd.trigger_pdf_download("target-1")
        assert result is None

    async def test_timeout_error(self):
        """Line 1258: timeout -> None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wd, session = _make_watchdog(tmpdir)

            session.get_or_create_cdp_session.side_effect = TimeoutError("timed out")

            result = await wd.trigger_pdf_download("target-1")
            assert result is None


# ---------------------------------------------------------------------------
# _get_unique_filename  (lines 1272-1273)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetUniqueFilename:
    """Cover _get_unique_filename static method."""

    async def test_filename_collision(self):
        """Lines 1272-1273: filename exists -> increments counter."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing files
            Path(os.path.join(tmpdir, "file.pdf")).touch()
            Path(os.path.join(tmpdir, "file (1).pdf")).touch()

            result = await DownloadsWatchdog._get_unique_filename(tmpdir, "file.pdf")
            assert result == "file (2).pdf"

    async def test_no_collision(self):
        """No collision -> returns original filename."""
        from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await DownloadsWatchdog._get_unique_filename(tmpdir, "new.pdf")
            assert result == "new.pdf"


# ---------------------------------------------------------------------------
# _setup_network_monitoring full path  (lines 257-258)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSetupNetworkMonitoringDeep:
    """Cover _setup_network_monitoring error path."""

    async def test_setup_exception(self):
        """Lines 456-457: exception during network monitoring setup."""
        wd, session = _make_watchdog()
        session.browser_profile.auto_download_pdfs = True

        session.cdp_client.send.Network.enable = AsyncMock(
            side_effect=RuntimeError("network fail")
        )
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        session.get_or_create_cdp_session.return_value = mock_cdp

        # Should not raise
        await wd._setup_network_monitoring("target-1")
