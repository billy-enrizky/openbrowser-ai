"""Behavior tests for explicit PDF detection and validated auto-download."""

# Test assertions intentionally inspect files synchronously after awaiting the operation.
# ruff: noqa: ASYNC240

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.download_state import (
    DownloadOutcome,
    DownloadPhase,
    DownloadTimeoutError,
    PdfDetectionPhase,
    PdfDetectionSnapshot,
    downloads_summary,
    pdf_validation_error,
    redacted_outcome,
    redacted_pdf_detection_snapshot,
    redacted_text,
)
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

pytest.importorskip("pypdf")

MINIMAL_PDF = (
    b"%PDF-1.0\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n212\n%%EOF\n"
)


_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []


@pytest.fixture(autouse=True)
def _cleanup_pdf_test_dirs():
    yield
    while _TEMP_DIRS:
        _TEMP_DIRS.pop().cleanup()


def _make_watchdog(downloads_path: str | None = None):
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test-pdf-detection")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = None
    session.get_or_create_cdp_session = AsyncMock()
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    session.cdp_client = MagicMock()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
    session.id = "test-session-1234"
    session.is_local = True
    session.browser_profile = MagicMock()
    if downloads_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="pdf-test-", ignore_cleanup_errors=True)
        _TEMP_DIRS.append(temp_dir)
        downloads_path = temp_dir.name
    session.browser_profile.downloads_path = downloads_path
    session.browser_profile.auto_download_pdfs = True
    bus = EventBus()
    watchdog = DownloadsWatchdog(event_bus=bus, browser_session=session)
    watchdog.event_bus = MagicMock()
    return watchdog, session


def _targets(*items):
    return {"targetInfos": [{"targetId": target_id, "url": url} for target_id, url in items]}


@pytest.mark.asyncio
async def test_missing_target_is_check_failed_and_retryable():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

    result = await watchdog.check_for_pdf_viewer("missing", url="https://example.com/viewer")

    snapshot = watchdog.pdf_detection_snapshot()
    assert result is False
    assert snapshot.phase is PdfDetectionPhase.CHECK_FAILED
    assert snapshot.url == "https://example.com/viewer"
    assert snapshot.error and "No target info" in snapshot.error


@pytest.mark.asyncio
async def test_cancellation_records_a_cancelled_attempt_and_reraises():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await watchdog.check_for_pdf_viewer("t1", url="https://example.com/viewer")

    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.CANCELLED
    assert snapshot.url == "https://example.com/viewer"


@pytest.mark.asyncio
async def test_transient_history_failure_is_not_cached_and_retries():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", "https://example.com/viewer")))

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=[(None, None), (True, "https://example.com/viewer")],
    ) as history:
        first = await watchdog.check_for_pdf_viewer("t1")
        first_snapshot = watchdog.pdf_detection_snapshot()
        second = await watchdog.check_for_pdf_viewer("t1")
        second_snapshot = watchdog.pdf_detection_snapshot()

    assert first is False
    assert first_snapshot.phase is PdfDetectionPhase.CHECK_FAILED
    assert second is True
    assert second_snapshot.phase is PdfDetectionPhase.PDF
    assert history.await_count == 2


@pytest.mark.asyncio
async def test_definitive_miss_rechecks_on_the_next_attempt():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", "https://example.com/viewer")))

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(False, "https://example.com/viewer"),
    ) as history:
        assert await watchdog.check_for_pdf_viewer("t1") is False
        assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.NOT_PDF
        assert await watchdog.check_for_pdf_viewer("t1") is False

    assert history.await_count == 2


@pytest.mark.asyncio
async def test_positive_url_wins_before_a_transient_history_failure():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(
        return_value=_targets(("t1", "https://example.com/report.pdf"))
    )

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=AssertionError("history must not run"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is True

    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.PDF
    assert snapshot.resource_url == "https://example.com/report.pdf"


@pytest.mark.asyncio
async def test_chrome_viewer_wins_before_a_transient_history_failure():
    watchdog, session = _make_watchdog()
    viewer = "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/pdf.html"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", viewer)))

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(None, None),
    ) as history:
        assert await watchdog.check_for_pdf_viewer("t1") is True

    assert history.await_count == 0
    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.PDF
    assert snapshot.resource_url is None


@pytest.mark.asyncio
async def test_document_mime_handles_an_extensionless_pdf():
    watchdog, session = _make_watchdog()
    url = "https://example.com/export?format=pdf"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    watchdog._note_document_mime("t1", url, "application/pdf")

    assert await watchdog.check_for_pdf_viewer("t1") is True
    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.PDF
    assert snapshot.resource_url == url


@pytest.mark.asyncio
async def test_network_observer_records_main_document_mime():
    watchdog, session = _make_watchdog()
    url = "https://example.com/export?format=pdf"
    cdp_session = MagicMock()
    cdp_session.session_id = "sid-document"
    session.get_or_create_cdp_session.return_value = cdp_session
    session.get_target_id_from_session_id.return_value = "t1"
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()

    with patch.object(DownloadsWatchdog, "download_file_from_url", new_callable=AsyncMock):
        await watchdog._setup_network_monitoring("t1")
        callback = session.cdp_client.register.Network.responseReceived.call_args.args[0]
        callback(
            {
                "type": "Other",
                "response": {
                    "url": url,
                    "mimeType": "application/pdf",
                    "headers": {},
                },
            },
            "sid-document",
        )
        await asyncio.sleep(0)

    assert watchdog._pdf_document_mime[("t1", url)] == "application/pdf"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    assert await watchdog.check_for_pdf_viewer("t1") is True
    assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.PDF


@pytest.mark.asyncio
async def test_network_observer_html_reload_replaces_stale_pdf_mime():
    watchdog, session = _make_watchdog()
    url = "https://example.com/export?format=pdf"
    cdp_session = MagicMock()
    cdp_session.session_id = "sid-document"
    session.get_or_create_cdp_session.return_value = cdp_session
    session.get_target_id_from_session_id.return_value = "t1"
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()

    with patch.object(DownloadsWatchdog, "download_file_from_url", new_callable=AsyncMock):
        await watchdog._setup_network_monitoring("t1")
        callback = session.cdp_client.register.Network.responseReceived.call_args.args[0]
        callback(
            {
                "type": "Other",
                "response": {
                    "url": url,
                    "mimeType": "application/pdf",
                    "headers": {},
                },
            },
            "sid-document",
        )
        callback(
            {
                "type": "Document",
                "response": {
                    "url": url,
                    "mimeType": "text/html",
                    "headers": {},
                },
            },
            "sid-document",
        )
        await asyncio.sleep(0)

    assert watchdog._pdf_document_mime[("t1", url)] == "text/html"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(False, "https://example.com/viewer"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is False
    assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.NOT_PDF


@pytest.mark.asyncio
async def test_failed_network_observer_attempt_allows_a_retry():
    watchdog, session = _make_watchdog()
    url = "https://example.com/report.pdf"
    cdp_session = MagicMock()
    cdp_session.session_id = "sid-document"
    session.get_or_create_cdp_session.return_value = cdp_session
    session.get_target_id_from_session_id.return_value = "t1"
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()

    with patch.object(
        DownloadsWatchdog,
        "download_file_from_url",
        new_callable=AsyncMock,
        side_effect=[None, "/tmp/report.pdf"],
    ) as download:
        await watchdog._setup_network_monitoring("t1")
        callback = session.cdp_client.register.Network.responseReceived.call_args.args[0]
        event = {
            "type": "Other",
            "response": {
                "url": url,
                "mimeType": "application/pdf",
                "headers": {},
            },
        }
        callback(event, "sid-document")
        await asyncio.gather(*list(watchdog._cdp_event_tasks))
        assert url not in watchdog._detected_downloads

        callback(event, "sid-document")
        await asyncio.gather(*list(watchdog._cdp_event_tasks))

    assert download.await_count == 2


@pytest.mark.asyncio
async def test_network_observer_logs_redact_signed_query_values(caplog):
    watchdog, session = _make_watchdog()
    url = "https://example.com/report.pdf?token=SECRET"
    cdp_session = MagicMock()
    cdp_session.session_id = "sid-document"
    session.get_or_create_cdp_session.return_value = cdp_session
    session.get_target_id_from_session_id.return_value = "t1"
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()
    caplog.set_level(logging.INFO)

    with patch.object(DownloadsWatchdog, "download_file_from_url", new_callable=AsyncMock):
        await watchdog._setup_network_monitoring("t1")
        callback = session.cdp_client.register.Network.responseReceived.call_args.args[0]
        callback(
            {
                "type": "Other",
                "response": {
                    "url": url,
                    "mimeType": "application/pdf",
                    "headers": {},
                },
            },
            "sid-document",
        )
        await asyncio.gather(*list(watchdog._cdp_event_tasks), return_exceptions=True)

    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_network_observer_reads_mixed_case_content_disposition():
    watchdog, session = _make_watchdog()
    url = "https://example.com/download?id=1"
    cdp_session = MagicMock()
    cdp_session.session_id = "sid-document"
    session.get_or_create_cdp_session.return_value = cdp_session
    session.get_target_id_from_session_id.return_value = "t1"
    session.cdp_client.send.Network.enable = AsyncMock()
    session.cdp_client.register.Network.responseReceived = MagicMock()

    with patch.object(
        DownloadsWatchdog,
        "download_file_from_url",
        new_callable=AsyncMock,
        return_value="/tmp/report.pdf",
    ) as download:
        await watchdog._setup_network_monitoring("t1")
        callback = session.cdp_client.register.Network.responseReceived.call_args.args[0]
        callback(
            {
                "type": "Other",
                "response": {
                    "url": url,
                    "mimeType": "application/octet-stream",
                    "headers": {"Content-Disposition": 'attachment; filename="report.pdf"'},
                },
            },
            "sid-document",
        )
        await asyncio.gather(*list(watchdog._cdp_event_tasks), return_exceptions=True)

    assert download.await_args is not None
    assert download.await_args.kwargs["suggested_filename"] == "report.pdf"


@pytest.mark.asyncio
async def test_fetch_failure_logs_redact_signed_query_values(caplog, tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": {}}})
    session.get_or_create_cdp_session.return_value = mock_cdp
    caplog.set_level(logging.DEBUG)

    result = await watchdog.download_file_from_url(
        "https://example.com/report.pdf?token=SECRET",
        "t1",
    )

    assert result is None
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_same_url_html_reload_removes_stale_pdf_mime():
    watchdog, session = _make_watchdog()
    url = "https://example.com/export?format=pdf"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    watchdog._note_document_mime("t1", url, "application/pdf")
    watchdog._note_document_mime("t1", url, "text/html")

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(False, "https://example.com/viewer"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is False

    assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.NOT_PDF


@pytest.mark.asyncio
async def test_pdf_suffix_with_observed_html_is_not_pdf():
    watchdog, session = _make_watchdog()
    url = "https://example.com/report.pdf"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    watchdog._note_document_mime("t1", url, "text/html")

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=AssertionError("history must not run after a definitive MIME miss"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is False

    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.NOT_PDF
    assert snapshot.detail and "non-PDF document mime" in snapshot.detail


@pytest.mark.asyncio
async def test_pdf_suffix_with_observed_image_is_not_pdf():
    watchdog, session = _make_watchdog()
    url = "https://example.com/report.pdf"
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url)))
    watchdog._note_document_mime("t1", url, "image/png")

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=AssertionError("history must not run after a definitive MIME miss"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is False

    assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.NOT_PDF


@pytest.mark.asyncio
async def test_non_pdf_reload_invalidates_a_cached_pdf_path(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    cached = tmp_path / "cached.pdf"
    cached.write_bytes(MINIMAL_PDF)
    watchdog._session_pdf_urls[url] = str(cached)
    watchdog._note_document_mime("t1", url, "text/html")
    assert url not in watchdog._session_pdf_urls

    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(b"<html>not a PDF</html>"),
                    "fromCache": False,
                    "responseSize": 22,
                    "contentType": "text/html",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    result = await watchdog.trigger_pdf_download("t1", resource_url=url)

    assert result is None
    assert cached.read_bytes() == MINIMAL_PDF
    outcome = watchdog._download_registry.latest()
    assert outcome is not None and outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_inflight_fetch_does_not_repopulate_a_reloaded_cache(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"

    async def evaluate(*_args, **_kwargs):
        watchdog._note_document_mime("t1", url, "text/html")
        return {
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }

    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=evaluate)
    session.get_or_create_cdp_session.return_value = mock_cdp

    result = await watchdog.trigger_pdf_download("t1", resource_url=url)

    assert result is not None
    assert url not in watchdog._session_pdf_urls


@pytest.mark.asyncio
async def test_queued_trigger_does_not_repopulate_a_reloaded_cache(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    await watchdog._js_fetch_lock.acquire()
    try:
        task = asyncio.create_task(watchdog.trigger_pdf_download("t1", resource_url=url))
        await asyncio.sleep(0)
        watchdog._note_document_mime("t1", url, "text/html")
    finally:
        watchdog._js_fetch_lock.release()

    result = await task
    assert result is not None
    assert url not in watchdog._session_pdf_urls


@pytest.mark.asyncio
async def test_queued_network_fetch_does_not_repopulate_a_reloaded_cache(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    await watchdog._js_fetch_lock.acquire()
    try:
        task = asyncio.create_task(watchdog.download_file_from_url(url, "t1", content_type="application/pdf"))
        await asyncio.sleep(0)
        watchdog._note_document_mime("t1", url, "text/html")
    finally:
        watchdog._js_fetch_lock.release()

    result = await task
    assert result is not None
    assert url not in watchdog._session_pdf_urls


@pytest.mark.asyncio
async def test_two_targets_keep_independent_document_mime():
    watchdog, session = _make_watchdog()
    url = "https://example.com/shared"
    watchdog._note_document_mime("t1", url, "application/pdf")
    watchdog._note_document_mime("t2", url, "text/html")

    session.cdp_client.send.Target.getTargets = AsyncMock(return_value=_targets(("t1", url), ("t2", url)))
    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(False, "https://example.com/viewer"),
    ):
        assert await watchdog.check_for_pdf_viewer("t1") is True
        assert await watchdog.check_for_pdf_viewer("t2") is False


@pytest.mark.asyncio
async def test_newer_detection_owns_the_published_snapshot():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(
        side_effect=[
            _targets(("t1", "https://example.com/one")),
            _targets(("t2", "https://example.com/two")),
        ]
    )
    one_started = asyncio.Event()
    two_started = asyncio.Event()
    release_one = asyncio.Event()
    release_two = asyncio.Event()

    async def history(target_id):
        if target_id == "t1":
            one_started.set()
            await release_one.wait()
            return True, "https://example.com/one"
        two_started.set()
        await release_two.wait()
        return False, "https://example.com/two"

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=history,
    ):
        first = asyncio.create_task(watchdog._detect_pdf("t1", url="https://example.com/one"))
        await one_started.wait()
        second = asyncio.create_task(watchdog._detect_pdf("t2", url="https://example.com/two"))
        await two_started.wait()
        release_two.set()
        second_result = await second
        release_one.set()
        first_result = await first

    assert first_result.phase is PdfDetectionPhase.PDF
    assert second_result.phase is PdfDetectionPhase.NOT_PDF
    snapshot = watchdog.pdf_detection_snapshot()
    assert snapshot.phase is PdfDetectionPhase.NOT_PDF
    assert snapshot.url == "https://example.com/two"


@pytest.mark.asyncio
async def test_cancelling_an_older_detection_does_not_replace_a_newer_attempt():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(
        side_effect=[
            _targets(("t1", "https://example.com/one")),
            _targets(("t2", "https://example.com/two")),
        ]
    )
    one_started = asyncio.Event()
    two_started = asyncio.Event()
    release_two = asyncio.Event()

    async def history(target_id):
        if target_id == "t1":
            one_started.set()
            await asyncio.Event().wait()
        two_started.set()
        await release_two.wait()
        return False, "https://example.com/two"

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        side_effect=history,
    ):
        first = asyncio.create_task(watchdog._detect_pdf("t1", url="https://example.com/one"))
        await one_started.wait()
        second = asyncio.create_task(watchdog._detect_pdf("t2", url="https://example.com/two"))
        await two_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        snapshot = watchdog.pdf_detection_snapshot()
        assert snapshot.phase is PdfDetectionPhase.CHECKING
        assert snapshot.url == "https://example.com/two"
        release_two.set()
        await second

    assert watchdog.pdf_detection_snapshot().phase is PdfDetectionPhase.NOT_PDF


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/report.pdf",
        "https://example.com/report.PDF?download=1",
        "https://example.com/export?content-type=application/pdf",
        "https://example.com/export?mimetype=application%2Fpdf",
        "https://example.com/export?type=application/pdf",
        "data:application/pdf;base64,AAAA",
    ],
)
def test_strict_pdf_url_positives(url):
    watchdog, _ = _make_watchdog()
    assert watchdog._check_url_for_pdf(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/pdf-downloads",
        "https://example.com/pdf.js",
        "https://example.com/report.pdf.txt",
        "https://example.com/view?file=report.pdf",
        "https://example.com/view#report.pdf",
        "chrome://settings/?search=pdf",
        "chrome-extension://other/pdf.html",
        "file:///tmp/report.pdf",
        "https:///report.pdf",
        "http:report.pdf",
        "https://exa mple.com/report.pdf",
        "http:// /report.pdf",
        "https://example.com:bad/report.pdf",
        "https://exa%ZZmple.com/report.pdf",
        "http://bad%2fhost/report.pdf",
        "data:application/pdfx;base64,AAAA",
        "data:text/html;base64,AAAA",
        "not a url",
    ],
)
def test_strict_pdf_url_negatives(url):
    watchdog, _ = _make_watchdog()
    assert watchdog._check_url_for_pdf(url) is False


def test_chrome_viewer_identity_is_exact():
    watchdog, _ = _make_watchdog()
    assert watchdog._is_chrome_pdf_viewer_url("chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/index.html") is True
    assert watchdog._is_chrome_pdf_viewer_url("chrome://pdf-viewer/index.html") is True
    assert watchdog._is_chrome_pdf_viewer_url("chrome://settings/?search=pdf") is False
    assert watchdog._is_chrome_pdf_viewer_url("chrome-extension://other/pdf.html") is False


def test_pdf_validation_rejects_spoofed_html():
    html = b"<html>%PDF-1.0<script>startxref %%EOF</script></html>"
    assert pdf_validation_error(html) is not None


def test_pdf_validation_accepts_minimal_pdf():
    assert pdf_validation_error(MINIMAL_PDF) is None


def test_pdf_validation_missing_pypdf_fails_with_install_hint():
    with patch("openbrowser.browser.download_state._load_pdf_reader", return_value=None):
        error = pdf_validation_error(MINIMAL_PDF)
    assert error is not None
    assert "openbrowser-ai[pdf]" in error


def test_downloads_summary_defaults_to_idle_pdf_detection():
    summary = downloads_summary(None)
    assert summary["pdf_detection"]["phase"] == "idle"


def test_downloads_summary_redacts_data_url_payload():
    class Owner:
        def download_outcomes(self):
            return []

        def pdf_detection_snapshot(self):
            return PdfDetectionSnapshot(
                phase=PdfDetectionPhase.PDF,
                url="data:application/pdf;base64,SECRET",
            )

    summary = downloads_summary(Owner())
    assert summary["pdf_detection"]["url"] == "data:application/pdf;base64,<redacted>"


def test_public_download_outcomes_redact_data_url_payloads():
    watchdog, _ = _make_watchdog()
    watchdog._download_registry.start(
        "data-download",
        url="data:application/pdf;base64,SECRET",
        source="pdf",
    )

    outcome = watchdog.download_outcomes()[0]
    latest = watchdog.latest_download_outcome()

    assert outcome["url"] == "data:application/pdf;base64,<redacted>"
    assert latest is not None and latest["url"] == "data:application/pdf;base64,<redacted>"
    assert "SECRET" not in str(outcome)
    assert "SECRET" not in str(latest)


def test_download_timeout_outcome_redacts_data_url_payloads():
    outcome = DownloadOutcome(
        download_id="data-timeout",
        phase=DownloadPhase.STARTED,
        url="data:application/pdf;base64,SECRET",
    )

    error = DownloadTimeoutError("timed out", outcome)

    assert error.outcome is not None
    assert error.outcome["url"] == "data:application/pdf;base64,<redacted>"


def test_redacted_text_removes_a_data_url_payload():
    text = redacted_text("validation failed for data:application/pdf;base64,SECRET")

    assert isinstance(text, str)
    assert "SECRET" not in text
    assert "<redacted>" in text

    uppercase = redacted_text("validation failed for DATA:application/pdf;base64,SECRET")
    assert isinstance(uppercase, str)
    assert "SECRET" not in uppercase


def test_safe_log_url_removes_query_userinfo_and_data_payloads():
    watchdog, _ = _make_watchdog()

    assert watchdog._safe_log_url("https://user:secret@example.com/report.pdf?token=secret") == (
        "https://example.com/report.pdf?redacted"
    )
    assert watchdog._safe_log_url("data:application/pdf;base64,SECRET") == ("data:application/pdf;base64,<redacted>")


@pytest.mark.asyncio
async def test_trigger_rejects_unresolved_viewer_before_fetch():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        viewer = "chrome-extension://mhjfbmdgcfjbbpaeojofohoefgiehjai/index.html"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": {"url": None}}})
        session.get_or_create_cdp_session.return_value = mock_cdp
        watchdog._pdf_detection_resource_url = None
        watchdog._pdf_detection_url = viewer

        result = await watchdog.trigger_pdf_download("t1")

        assert result is None
        outcome = watchdog._download_registry.get(next(iter(watchdog._download_registry._outcomes)))
        assert outcome.phase is DownloadPhase.FAILED
        assert outcome.stage == "resolve_url"


@pytest.mark.asyncio
async def test_viewer_resolver_skips_an_unusable_embed_candidate():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {
                    "result": {
                        "value": {
                            "embed": "blob:not-fetchable",
                            "file": resource,
                            "src": None,
                        }
                    }
                },
                {
                    "result": {
                        "value": {
                            "data": list(MINIMAL_PDF),
                            "fromCache": False,
                            "responseSize": len(MINIMAL_PDF),
                            "contentType": "application/pdf",
                        }
                    }
                },
            ]
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.trigger_pdf_download("t1")

        assert result is not None
        assert Path(result).read_bytes() == MINIMAL_PDF


@pytest.mark.asyncio
async def test_navigation_complete_passes_detected_resource_to_trigger():
    watchdog, _session = _make_watchdog()
    watchdog.browser_session.browser_profile.auto_download_pdfs = True
    resource = "https://example.com/report.pdf"
    event = MagicMock()
    event.url = resource
    event.target_id = "t1"

    with (
        patch.object(
            DownloadsWatchdog,
            "_detect_pdf",
            new_callable=AsyncMock,
            return_value=PdfDetectionSnapshot(
                phase=PdfDetectionPhase.PDF,
                url=resource,
                resource_url=resource,
            ),
        ),
        patch.object(
            DownloadsWatchdog,
            "trigger_pdf_download",
            new_callable=AsyncMock,
            return_value="/tmp/report.pdf",
        ) as trigger,
    ):
        await watchdog.on_NavigationCompleteEvent(event)

    trigger.assert_awaited_once_with("t1", resource_url=resource)


@pytest.mark.asyncio
async def test_navigation_complete_does_not_trigger_without_pdf():
    watchdog, _session = _make_watchdog()
    watchdog.browser_session.browser_profile.auto_download_pdfs = True
    event = MagicMock()
    event.url = "https://example.com/page.html"
    event.target_id = "t1"

    with (
        patch.object(
            DownloadsWatchdog,
            "_detect_pdf",
            new_callable=AsyncMock,
            return_value=PdfDetectionSnapshot(
                phase=PdfDetectionPhase.NOT_PDF,
                url=event.url,
            ),
        ),
        patch.object(
            DownloadsWatchdog,
            "trigger_pdf_download",
            new_callable=AsyncMock,
        ) as trigger,
    ):
        await watchdog.on_NavigationCompleteEvent(event)

    trigger.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_rejects_html_with_a_pdf_query_and_writes_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/search?type=application/pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": {"url": resource}}},
                {
                    "result": {
                        "value": {
                            "data": list(b"<html>not a pdf</html>"),
                            "fromCache": False,
                            "responseSize": 22,
                            "contentType": "application/pdf",
                        }
                    }
                },
            ]
        )
        session.get_or_create_cdp_session.return_value = mock_cdp
        watchdog._pdf_detection_resource_url = None
        watchdog._pdf_detection_url = resource

        result = await watchdog.trigger_pdf_download("t1")

        assert result is None
        assert list(Path(tmpdir).glob("*.pdf")) == []
        outcome = watchdog._download_registry.get(next(iter(watchdog._download_registry._outcomes)))
        assert outcome.phase is DownloadPhase.FAILED
        assert outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_trigger_saves_a_validated_pdf_and_completes_once():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": {"url": resource}}},
                {
                    "result": {
                        "value": {
                            "data": list(MINIMAL_PDF),
                            "fromCache": False,
                            "responseSize": len(MINIMAL_PDF),
                            "contentType": "application/pdf",
                        }
                    }
                },
            ]
        )
        session.get_or_create_cdp_session.return_value = mock_cdp
        watchdog._pdf_detection_resource_url = None
        watchdog._pdf_detection_url = resource

        result = await watchdog.trigger_pdf_download("t1")

        assert result is not None
        path = Path(result)
        assert path.exists()
        assert path.read_bytes() == MINIMAL_PDF
        outcome = watchdog._download_registry.get(next(iter(watchdog._download_registry._outcomes)))
        assert outcome.phase is DownloadPhase.COMPLETED
        assert outcome.path == result
        watchdog.event_bus.dispatch.assert_called_once()


@pytest.mark.asyncio
async def test_network_fetch_rejects_pdf_labelled_html_and_writes_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(b"<html>not a PDF</html>"),
                        "responseSize": 22,
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.download_file_from_url(
            resource,
            "t1",
            content_type="application/pdf",
        )

        assert result is None
        assert list(Path(tmpdir).iterdir()) == []
        outcome = watchdog._download_registry.latest()
        assert outcome is not None
        assert outcome.phase is DownloadPhase.FAILED
        assert outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_network_fetch_validates_a_pdf_named_by_content_disposition(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(b"<html>not a PDF</html>"),
                    "responseSize": 22,
                    "contentType": "application/octet-stream",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    result = await watchdog.download_file_from_url(
        "https://example.com/download?id=1",
        "t1",
        content_type="application/octet-stream",
        suggested_filename="report.pdf",
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []
    outcome = watchdog._download_registry.latest()
    assert outcome is not None and outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_network_fetch_confines_a_server_suggested_filename(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    result = await watchdog.download_file_from_url(
        "https://example.com/download?id=1",
        "t1",
        content_type="application/pdf",
        suggested_filename="../../escape.pdf",
    )

    assert result is not None
    assert Path(result).parent == tmp_path
    assert Path(result).name == "escape.pdf"


@pytest.mark.asyncio
async def test_network_fetch_confines_a_windows_drive_like_url_filename(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(b"payload"),
                    "responseSize": 7,
                    "contentType": "application/octet-stream",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    result = await watchdog.download_file_from_url(
        "https://example.com/D:payload.exe",
        "t1",
    )

    assert result is not None
    assert Path(result).parent == tmp_path
    assert Path(result).name == "D_payload.exe"


@pytest.mark.asyncio
async def test_trigger_drops_invalid_session_cache_and_refetches(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    resource = "https://example.com/report.pdf"
    cached = tmp_path / "cached.pdf"
    cached.write_bytes(b"<html>not a PDF</html>")
    watchdog._session_pdf_urls[resource] = str(cached)
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    path = await watchdog.trigger_pdf_download("t1", resource_url=resource)

    assert path is not None
    assert Path(path) != cached
    assert Path(path).read_bytes() == MINIMAL_PDF
    assert cached.read_bytes() == b"<html>not a PDF</html>"
    assert watchdog._session_pdf_urls[resource] == path


@pytest.mark.asyncio
async def test_cache_validation_failure_keeps_a_newer_cache_entry(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    old_path = tmp_path / "old.pdf"
    new_path = tmp_path / "new.pdf"
    old_path.write_bytes(MINIMAL_PDF)
    new_path.write_bytes(MINIMAL_PDF)
    watchdog._session_pdf_urls[url] = str(old_path)

    async def replace_during_validation(_data):
        watchdog._session_pdf_urls[url] = str(new_path)
        return "invalid old cache"

    with patch.object(
        DownloadsWatchdog,
        "_pdf_validation_error_async",
        new_callable=AsyncMock,
        side_effect=replace_during_validation,
    ):
        path, error = await watchdog._validated_pdf_cache_entry(url)

    assert path is None
    assert error == "invalid old cache"
    assert watchdog._session_pdf_urls[url] == str(new_path)


@pytest.mark.asyncio
async def test_viewer_trigger_ignores_stale_global_resource():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        own_resource = "https://example.com/own.pdf"
        watchdog._pdf_detection_resource_url = "https://example.com/other.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": {"url": own_resource}}},
                {
                    "result": {
                        "value": {
                            "data": list(MINIMAL_PDF),
                            "fromCache": False,
                            "responseSize": len(MINIMAL_PDF),
                            "contentType": "application/pdf",
                        }
                    }
                },
            ]
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.trigger_pdf_download("t1")
        outcome = watchdog._download_registry.latest()

        assert result is not None
        assert outcome is not None and outcome.url == own_resource


@pytest.mark.asyncio
async def test_trigger_hash_mismatch_leaves_no_partial_or_final_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        with patch.object(DownloadsWatchdog, "_sha256_file", return_value="wrong"):
            result = await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert result is None
        assert list(Path(tmpdir).iterdir()) == []
        outcome = watchdog._download_registry.latest()
        assert outcome is not None
        assert outcome.phase is DownloadPhase.FAILED
        assert outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_atomic_writer_never_overwrites_an_independent_file(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    destination = tmp_path / "report.pdf"
    destination.write_bytes(b"independent native download")

    size, stage, error = await watchdog._write_bytes_atomically(str(destination), MINIMAL_PDF)

    assert size is None
    assert stage == "destination_exists"
    assert error is not None
    assert destination.read_bytes() == b"independent native download"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_native_completion_rejects_html_named_as_pdf(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"<html>not a PDF</html>")
    watchdog._download_registry.start(
        "native-bad",
        url="https://example.com/bad.pdf",
        file_name=path.name,
        source="cdp",
    )

    await watchdog._complete_native_download(
        download_id="native-bad",
        file_path=str(path),
        url="https://example.com/bad.pdf",
        total_bytes=len(b"<html>not a PDF</html>"),
        source="cdp",
    )

    assert not path.exists()
    outcome = watchdog._download_registry.latest()
    assert outcome is not None
    assert outcome.phase is DownloadPhase.FAILED
    assert outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_native_completion_uses_observed_mime_for_an_extensionless_file(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/export?format=pdf"
    path = tmp_path / "export.bin"
    path.write_bytes(b"<html>not a PDF</html>")
    watchdog._note_document_mime("t1", url, "application/pdf")
    watchdog._download_registry.start(
        "native-extensionless",
        url=url,
        file_name=path.name,
        source="cdp",
    )

    await watchdog._complete_native_download(
        download_id="native-extensionless",
        file_path=str(path),
        url=url,
        total_bytes=22,
        source="cdp",
        target_id="t1",
    )

    assert not path.exists()
    outcome = watchdog._download_registry.latest()
    assert outcome is not None
    assert outcome.phase is DownloadPhase.FAILED
    assert outcome.stage == "content_validation"


@pytest.mark.asyncio
async def test_native_completion_without_a_target_does_not_use_other_tab_mime(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/export?format=pdf"
    path = tmp_path / "export.bin"
    path.write_bytes(b"<html>not a PDF</html>")
    watchdog._note_document_mime("other-tab", url, "application/pdf")
    watchdog._download_registry.start(
        "native-unscoped",
        url=url,
        file_name=path.name,
        source="cdp",
    )

    await watchdog._complete_native_download(
        download_id="native-unscoped",
        file_path=str(path),
        url=url,
        total_bytes=22,
        source="cdp",
    )

    assert path.exists()
    outcome = watchdog._download_registry.latest()
    assert outcome is not None
    assert outcome.phase is DownloadPhase.COMPLETED


@pytest.mark.asyncio
async def test_native_completion_for_another_tab_keeps_its_non_pdf_file(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/export?format=pdf"
    path = tmp_path / "export.bin"
    path.write_bytes(b"<html>not a PDF</html>")
    watchdog._note_document_mime("tab-one", url, "application/pdf")
    watchdog._note_document_mime("tab-two", url, "text/html")
    watchdog._download_registry.start(
        "native-other-tab",
        url=url,
        file_name=path.name,
        source="cdp",
    )

    await watchdog._complete_native_download(
        download_id="native-other-tab",
        file_path=str(path),
        url=url,
        total_bytes=22,
        source="cdp",
        target_id="tab-two",
    )

    assert path.exists()
    outcome = watchdog._download_registry.latest()
    assert outcome is not None and outcome.phase is DownloadPhase.COMPLETED


@pytest.mark.asyncio
async def test_download_target_is_resolved_from_the_event_session():
    watchdog, session = _make_watchdog()
    session.get_target_id_from_session_id.return_value = "tab-two"

    assert await watchdog._resolve_download_target_id("sid-two", "frame-two") == "tab-two"


@pytest.mark.asyncio
async def test_download_target_is_resolved_from_the_frame_tree():
    watchdog, session = _make_watchdog()
    session.get_target_id_from_session_id.return_value = None
    session.get_all_frames = AsyncMock(return_value=({"frame-two": {}}, {}))
    session.find_frame_target = AsyncMock(return_value={"frameTargetId": "tab-two"})

    assert await watchdog._resolve_download_target_id(None, "frame-two") == "tab-two"


@pytest.mark.asyncio
async def test_native_completion_validates_and_caches_a_pdf(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    path = tmp_path / "report.pdf"
    path.write_bytes(MINIMAL_PDF)
    watchdog._download_registry.start(
        "native-good",
        url=url,
        file_name=path.name,
        source="cdp",
    )

    await watchdog._complete_native_download(
        download_id="native-good",
        file_path=str(path),
        url=url,
        total_bytes=len(MINIMAL_PDF),
        source="cdp",
    )

    outcome = watchdog._download_registry.latest()
    assert outcome is not None
    assert outcome.phase is DownloadPhase.COMPLETED
    assert outcome.source == "cdp"
    assert watchdog._session_pdf_urls[url] == str(path)

    with patch.object(DownloadsWatchdog, "_track_download") as duplicate_track:
        await watchdog._complete_native_download(
            download_id="native-good",
            file_path=str(path),
            url=url,
            total_bytes=len(MINIMAL_PDF),
            source="cdp",
        )
    duplicate_track.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_native_completion_cannot_publish_or_cache(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    path = tmp_path / "report.pdf"
    path.write_bytes(MINIMAL_PDF)
    watchdog._download_registry.reject(
        "rejected-native",
        stage="capacity",
        error="capacity full",
        url=url,
        source="cdp",
    )

    with patch.object(DownloadsWatchdog, "_track_download") as track:
        await watchdog._complete_native_download(
            download_id="rejected-native",
            file_path=str(path),
            url=url,
            total_bytes=len(MINIMAL_PDF),
            source="cdp",
        )

    track.assert_not_called()
    assert url not in watchdog._session_pdf_urls
    outcome = watchdog._download_registry.latest()
    assert outcome is not None and outcome.phase is DownloadPhase.FAILED


@pytest.mark.asyncio
async def test_native_completion_error_log_redacts_exception_url(caplog, tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    path = tmp_path / "report.pdf"
    path.write_bytes(MINIMAL_PDF)
    watchdog._download_registry.start(
        "native-log-error",
        url=url,
        file_name=path.name,
        source="cdp",
    )
    caplog.set_level(logging.WARNING)

    with patch.object(
        DownloadsWatchdog,
        "_track_download",
        side_effect=RuntimeError("https://example.com/export?token=LEAK_ME"),
    ):
        await watchdog._complete_native_download(
            download_id="native-log-error",
            file_path=str(path),
            url=url,
            total_bytes=len(MINIMAL_PDF),
            source="cdp",
        )

    assert "LEAK_ME" not in caplog.text


@pytest.mark.asyncio
async def test_trigger_cancellation_during_write_cleans_the_temporary_file():
    class PartialWriter:
        def __init__(self, path):
            self.path = Path(path)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def write(self, data):
            self.path.write_bytes(data[:10])
            raise asyncio.CancelledError()

    async def open_partial(path, _mode):
        return PartialWriter(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        with (
            patch(
                "openbrowser.browser.watchdogs.downloads_watchdog.anyio.open_file",
                side_effect=open_partial,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert list(Path(tmpdir).iterdir()) == []
        outcome = watchdog._download_registry.latest()
        assert outcome is not None and outcome.phase is DownloadPhase.CANCELLED


@pytest.mark.asyncio
async def test_trigger_cancelled_while_queued_records_a_cancelled_outcome():
    watchdog, _ = _make_watchdog()
    await watchdog._js_fetch_lock.acquire()
    try:
        task = asyncio.create_task(watchdog.trigger_pdf_download("t1", resource_url="https://example.com/report.pdf"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        watchdog._js_fetch_lock.release()

    outcome = watchdog._download_registry.latest()
    assert outcome is not None
    assert outcome.phase is DownloadPhase.CANCELLED
    assert outcome.url == "https://example.com/report.pdf"
    assert outcome.source == "pdf"


@pytest.mark.asyncio
async def test_trigger_missing_pypdf_fails_closed_without_a_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp
        with patch("openbrowser.browser.download_state._load_pdf_reader", return_value=None):
            result = await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert result is None
        assert list(Path(tmpdir).glob("*.pdf")) == []
        outcome = watchdog._download_registry.get(next(iter(watchdog._download_registry._outcomes)))
        assert outcome.phase is DownloadPhase.FAILED
        assert outcome.stage == "content_validation"
        assert outcome.error and "openbrowser-ai[pdf]" in outcome.error


@pytest.mark.asyncio
async def test_trigger_uses_a_safe_filename_for_data_urls():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "data:application/pdf;base64,AAAA"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp
        result = await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert result is not None
        assert Path(result).name == "document.pdf"
        assert Path(result).read_bytes() == MINIMAL_PDF


@pytest.mark.asyncio
async def test_trigger_preserves_an_uppercase_pdf_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.PDF"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        result = await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert result is not None
        assert Path(result).name == "report.PDF"


@pytest.mark.asyncio
async def test_pdf_validation_timeout_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            }
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        def slow_validation(_data):
            time.sleep(0.05)
            return None

        with (
            patch(
                "openbrowser.browser.watchdogs.downloads_watchdog.PDF_VALIDATION_TIMEOUT_SECONDS",
                0.01,
            ),
            patch(
                "openbrowser.browser.watchdogs.downloads_watchdog.pdf_validation_error",
                side_effect=slow_validation,
            ),
        ):
            result = await watchdog.trigger_pdf_download("t1", resource_url=resource)

        assert result is None
        assert list(Path(tmpdir).iterdir()) == []
        outcome = watchdog._download_registry.latest()
        assert outcome is not None
        assert outcome.stage == "content_validation"
        assert outcome.error and "timed out" in outcome.error


@pytest.mark.asyncio
async def test_wait_for_download_accepts_the_public_download_ref(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    path = tmp_path / "report.pdf"
    path.write_bytes(MINIMAL_PDF)
    watchdog._download_registry.complete(
        "internal-download-id",
        path=str(path),
        file_size=len(MINIMAL_PDF),
        file_name=path.name,
        source="pdf",
    )
    download_ref = watchdog.download_outcomes()[0]["download_ref"]
    assert isinstance(download_ref, str)

    outcome = await watchdog.wait_for_download(timeout=1.0, download_ref=download_ref)

    assert outcome["download_ref"] == download_ref
    assert outcome["path"] == str(path)


@pytest.mark.asyncio
async def test_session_cache_reports_file_size_and_missing_path_refetches(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    resource = "https://example.com/report.pdf"
    existing = tmp_path / "cached.pdf"
    existing.write_bytes(MINIMAL_PDF)
    watchdog._session_pdf_urls[resource] = str(existing)
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    fetch_response = {
        "result": {
            "value": {
                "data": list(MINIMAL_PDF),
                "fromCache": False,
                "responseSize": len(MINIMAL_PDF),
                "contentType": "application/pdf",
            }
        }
    }
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        side_effect=[
            {"result": {"value": {"url": resource}}},
            {"result": {"value": {"url": resource}}},
            fetch_response,
        ]
    )
    session.get_or_create_cdp_session.return_value = mock_cdp
    watchdog._pdf_detection_resource_url = None
    watchdog._pdf_detection_url = resource

    path = await watchdog.trigger_pdf_download("t1")
    outcome = watchdog._download_registry.get(next(iter(watchdog._download_registry._outcomes)))

    assert path == str(existing)
    assert outcome is not None and outcome.phase is DownloadPhase.COMPLETED
    assert outcome.file_size == len(MINIMAL_PDF)
    assert outcome.source == "session_cache"

    watchdog._session_pdf_urls[resource] = str(tmp_path / "missing.pdf")
    path = await watchdog.trigger_pdf_download("t1")
    assert path is not None and Path(path).exists()
    assert watchdog._session_pdf_urls[resource] == path


@pytest.mark.asyncio
async def test_network_fetch_and_trigger_share_one_locked_file(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    resource = "https://example.com/report.pdf"
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    triggered, fetched = await asyncio.gather(
        watchdog.trigger_pdf_download("t1", resource_url=resource),
        watchdog.download_file_from_url(resource, "t1", content_type="application/pdf"),
    )

    assert triggered is not None
    assert fetched == triggered
    assert mock_cdp.cdp_client.send.Runtime.evaluate.await_count == 1
    assert list(tmp_path.glob("*.pdf")) == [Path(triggered)]


def test_data_url_redaction_is_bounded_without_a_comma():
    redacted = redacted_outcome({"url": "data:application/pdf;base64" + "A" * 2000})
    assert redacted["url"] == "data:<redacted>"

    redacted = redacted_outcome({"url": "data:application/pdf;base64,SECRET"})
    assert redacted["url"] == "data:application/pdf;base64,<redacted>"

    redacted = redacted_outcome({"url": "data:application/vnd.unknown;base64,SECRET"})
    assert redacted["url"] == "data:application/octet-stream;base64,<redacted>"


def test_downloads_summary_redacts_dictionary_snapshots():
    class Owner:
        def download_outcomes(self):
            return []

        def pdf_detection_snapshot(self):
            return {
                "phase": "pdf",
                "url": "https://user:pass@example.com/doc.pdf?token=secret",
                "resource_url": "https://example.com/doc.pdf?sig=secret",
                "error": "https://user:pass@example.com/doc.pdf?token=secret",
            }

    summary = downloads_summary(Owner())

    assert summary["pdf_detection"]["url"] == "https://example.com/doc.pdf?redacted"
    assert summary["pdf_detection"]["resource_url"] == "https://example.com/doc.pdf?redacted"
    assert "secret" not in str(summary)


def test_downloads_summary_reports_pdf_reader_failure_in_the_same_response():
    class Owner:
        def download_outcomes(self):
            return []

        def pdf_detection_snapshot(self):
            raise RuntimeError("boom https://user:pass@example.com/doc.pdf?token=secret")

    summary = downloads_summary(Owner())

    assert summary["pdf_detection"]["phase"] == "idle"
    assert "pdf_detection_error" in summary
    assert "secret" not in str(summary)


def test_pdf_validation_rejects_an_oversized_payload(monkeypatch):
    from openbrowser.browser import download_state as download_state_module

    monkeypatch.setattr(download_state_module, "PDF_VALIDATION_MAX_BYTES", 1024)
    error = download_state_module.pdf_validation_error(b"%PDF-" + b"x" * 2048)

    assert error is not None
    assert "validation limit" in error


@pytest.mark.asyncio
async def test_pdf_validation_uses_the_dedicated_executor():
    watchdog, _ = _make_watchdog()

    with patch(
        "openbrowser.browser.watchdogs.downloads_watchdog.asyncio.to_thread",
        side_effect=AssertionError("default executor must not run PDF parsing"),
    ):
        error = await watchdog._pdf_validation_error_async(MINIMAL_PDF)

    from openbrowser.browser.watchdogs.downloads_watchdog import _PDF_VALIDATION_EXECUTOR

    assert error is None
    assert _PDF_VALIDATION_EXECUTOR._max_workers == 2


def test_only_main_document_responses_become_document_mime_evidence():
    assert DownloadsWatchdog._should_record_document_mime("document", "text/html") is True
    assert DownloadsWatchdog._should_record_document_mime("other", "application/pdf") is True
    assert DownloadsWatchdog._should_record_document_mime("fetch", "application/pdf") is False
    assert DownloadsWatchdog._should_record_document_mime("xhr", "application/pdf") is False


def test_stale_epoch_token_fails_after_the_bounded_map_evicts_the_url():
    watchdog, _ = _make_watchdog()
    url = "https://example.com/report.pdf"
    token = watchdog._begin_pdf_epoch(url)

    assert watchdog._pdf_epoch_is_current(url, token)

    for index in range(205):
        watchdog._note_document_mime("t1", f"https://example.com/{index}.html", "text/html")

    assert not watchdog._pdf_epoch_is_current(url, token)


@pytest.mark.asyncio
async def test_history_match_uses_the_committed_url_not_the_stale_target_url():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(
        return_value=_targets(("t1", "https://example.com/stale-viewer"))
    )

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(True, "https://example.com/late.pdf"),
    ):
        snapshot = await watchdog._detect_pdf("t1", url="https://example.com/stale-viewer")

    assert snapshot.phase is PdfDetectionPhase.PDF
    assert snapshot.url == "https://example.com/late.pdf"
    assert snapshot.resource_url == "https://example.com/late.pdf"


@pytest.mark.asyncio
async def test_trigger_falls_back_to_the_page_url_when_no_viewer_candidate_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        watchdog, session = _make_watchdog(tmpdir)
        resource = "https://example.com/report.pdf"
        mock_cdp = MagicMock()
        mock_cdp.session_id = "sid-1"
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": {"href": resource, "embed": None, "file": None, "src": None}}},
                {
                    "result": {
                        "value": {
                            "data": list(MINIMAL_PDF),
                            "fromCache": False,
                            "responseSize": len(MINIMAL_PDF),
                            "contentType": "application/pdf",
                        }
                    }
                },
            ]
        )
        session.get_or_create_cdp_session.return_value = mock_cdp

        path = await watchdog.trigger_pdf_download("t1")

        assert path is not None
        assert Path(path).read_bytes() == MINIMAL_PDF
        assert watchdog._session_pdf_urls[resource] == path


@pytest.mark.asyncio
async def test_cached_path_disappearing_before_size_read_refetches(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    resource = "https://example.com/report.pdf"
    existing = tmp_path / "cached.pdf"
    existing.write_bytes(MINIMAL_PDF)
    watchdog._session_pdf_urls[resource] = str(existing)
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        side_effect=[
            {"result": {"value": {"url": resource}}},
            {
                "result": {
                    "value": {
                        "data": list(MINIMAL_PDF),
                        "fromCache": False,
                        "responseSize": len(MINIMAL_PDF),
                        "contentType": "application/pdf",
                    }
                }
            },
        ]
    )
    session.get_or_create_cdp_session.return_value = mock_cdp
    watchdog._pdf_detection_resource_url = None
    watchdog._pdf_detection_url = resource

    real_getsize = os.path.getsize
    size_calls = 0

    def getsize(path):
        nonlocal size_calls
        size_calls += 1
        if size_calls == 1:
            raise FileNotFoundError("gone")
        return real_getsize(path)

    with patch(
        "openbrowser.browser.watchdogs.downloads_watchdog.os.path.getsize",
        side_effect=getsize,
    ):
        path = await watchdog.trigger_pdf_download("t1")

    assert path is not None
    assert path != str(existing)
    assert Path(path).read_bytes() == MINIMAL_PDF
    assert watchdog._session_pdf_urls[resource] == path


@pytest.mark.asyncio
async def test_download_target_resolution_includes_background_frames():
    watchdog, session = _make_watchdog()
    session.get_all_frames = AsyncMock(return_value=({"frame-bg": {"frameTargetId": "bg-target"}}, {}))
    session.find_frame_target = AsyncMock(return_value={"frameTargetId": "bg-target"})

    target_id = await watchdog._resolve_download_target_id(None, "frame-bg")

    assert target_id == "bg-target"
    session.get_all_frames.assert_awaited_once_with(include_background_targets=True)


@pytest.mark.asyncio
async def test_poll_timeout_defers_to_a_pending_native_completion(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    download_id = "native-pending"
    watchdog._download_registry.start(download_id, url="https://example.com/report.pdf", source="cdp")
    watchdog._native_completion_pending.add(download_id)

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_clock = MagicMock()
        mock_clock.time.side_effect = [0, 0, 25]
        mock_loop.return_value = mock_clock
        with patch("asyncio.sleep", new=AsyncMock()):
            await watchdog._poll_for_download_file(
                download_id=download_id,
                download_url="https://example.com/report.pdf",
                downloads_dir=tmp_path,
                suggested_filename="report.pdf",
                guid="g-pending",
            )

    outcome = watchdog._download_registry.get(download_id)
    assert outcome is not None
    assert outcome.phase is DownloadPhase.STARTED


@pytest.mark.asyncio
async def test_poll_does_not_publish_after_cancellation_during_validation(tmp_path):
    watchdog, _ = _make_watchdog(str(tmp_path))
    download_id = "cancel-during-validation"
    watchdog._download_registry.start(download_id, url="https://example.com/report.pdf", source="cdp")

    async def create_file(_duration):
        (tmp_path / "report.pdf").write_bytes(MINIMAL_PDF)

    async def validate_then_cancel(*_args, **_kwargs):
        watchdog._download_registry.cancel(download_id, detail="cancelled during validation")
        return True

    with patch("asyncio.get_event_loop") as mock_loop:
        mock_clock = MagicMock()
        mock_clock.time.side_effect = [0, 0, 5, 25]
        mock_loop.return_value = mock_clock
        with (
            patch("asyncio.sleep", side_effect=create_file),
            patch.object(watchdog, "_validate_pdf_file_path", side_effect=validate_then_cancel),
        ):
            await watchdog._poll_for_download_file(
                download_id=download_id,
                download_url="https://example.com/report.pdf",
                downloads_dir=tmp_path,
                suggested_filename="report.pdf",
                guid="g-cancel",
            )

    outcome = watchdog._download_registry.get(download_id)
    assert outcome is not None
    assert outcome.phase is DownloadPhase.CANCELLED
    watchdog.event_bus.dispatch.assert_not_called()


def test_non_pdf_response_clears_download_suppression():
    watchdog, _ = _make_watchdog()
    url = "https://example.com/report.pdf"
    watchdog._session_pdf_urls[url] = "/tmp/report.pdf"
    watchdog._detected_downloads.add(url)

    watchdog._note_document_mime("t1", url, "text/html")

    assert url not in watchdog._session_pdf_urls
    assert url not in watchdog._detected_downloads


def test_pdf_snapshot_redacts_error_and_detail():
    watchdog, _ = _make_watchdog()
    snapshot = watchdog._pdf_snapshot(
        PdfDetectionPhase.CHECK_FAILED,
        url="https://example.com/viewer",
        started_at=time.time(),
        error="https://user:pass@example.com/report.pdf?token=secret",
        detail="failed at https://user:pass@example.com/report.pdf?token=secret",
    )

    assert snapshot.error is not None and "secret" not in snapshot.error
    assert snapshot.detail is not None and "secret" not in snapshot.detail


@pytest.mark.asyncio
async def test_pdf_validation_exception_is_redacted(monkeypatch):
    watchdog, _ = _make_watchdog()

    def fail_validation(_data):
        raise RuntimeError("https://user:pass@example.com/report.pdf?token=secret")

    monkeypatch.setattr(
        "openbrowser.browser.watchdogs.downloads_watchdog.pdf_validation_error",
        fail_validation,
    )

    error = await watchdog._pdf_validation_error_async(MINIMAL_PDF)

    assert error is not None
    assert "secret" not in error


@pytest.mark.asyncio
async def test_playwright_invalidated_url_is_not_cached(tmp_path):
    watchdog, _session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    download = MagicMock()
    download.url = url
    download.suggested_filename = "report.pdf"
    download.failure = AsyncMock(return_value=None)

    async def fake_save_as(path):
        Path(path).write_bytes(MINIMAL_PDF)
        watchdog._note_document_mime("t1", url, "text/html")

    download.save_as = fake_save_as

    await watchdog._handle_download(download)

    outcome = watchdog.latest_download_outcome()
    assert outcome is not None
    assert outcome["phase"] == "completed"
    assert Path(str(outcome["path"])).read_bytes() == MINIMAL_PDF
    assert url not in watchdog._session_pdf_urls


@pytest.mark.asyncio
async def test_network_callback_does_not_let_a_fetch_pdf_replace_document_mime():
    watchdog, session = _make_watchdog()
    session.browser_profile.auto_download_pdfs = True
    session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock(session_id="sid-net"))
    session.cdp_client.send.Network.enable = AsyncMock()
    await watchdog._setup_network_monitoring("t1")
    handler = session.cdp_client.register.Network.responseReceived.call_args[0][0]
    session.get_target_id_from_session_id.return_value = "t1"
    url = "https://example.com/page"

    handler(
        {"type": "Document", "response": {"url": url, "mimeType": "text/html", "headers": {}}},
        "sid-document",
    )
    assert watchdog._pdf_document_mime[("t1", url)] == "text/html"

    with patch.object(DownloadsWatchdog, "download_file_from_url", new_callable=AsyncMock):
        handler(
            {"type": "Fetch", "response": {"url": url, "mimeType": "application/pdf", "headers": {}}},
            "sid-fetch",
        )
        await asyncio.gather(*watchdog._cdp_event_tasks, return_exceptions=True)

    assert watchdog._pdf_document_mime[("t1", url)] == "text/html"


@pytest.mark.asyncio
async def test_cdp_fallback_does_not_cache_after_invalidation(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    url = "https://example.com/report.pdf"
    watchdog._use_js_fetch_for_local = True
    session.is_local = True
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"

    async def evaluate(*_args, **_kwargs):
        watchdog._note_document_mime("t1", url, "text/html")
        return {
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "size": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }

    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=evaluate)
    session.cdp_client_for_frame = AsyncMock(return_value=mock_cdp)

    download_id = "cdp-fallback"
    watchdog._download_registry.start(download_id, url=url, source="cdp")
    await watchdog._handle_cdp_download(
        {
            "url": url,
            "suggestedFilename": "report.pdf",
            "guid": "g-cdp-fallback",
            "frameId": "frame-1",
        },
        "t1",
        "sid-1",
        download_id=download_id,
    )

    outcome = watchdog._download_registry.get(download_id)
    assert outcome is not None
    assert outcome.phase is DownloadPhase.COMPLETED
    assert outcome.path is not None
    assert Path(outcome.path).read_bytes() == MINIMAL_PDF
    assert url not in watchdog._session_pdf_urls


def test_redacted_pdf_detection_snapshot_hides_url_credentials_and_tokens():
    raw = PdfDetectionSnapshot(
        phase=PdfDetectionPhase.PDF,
        url="https://user:pass@example.com/doc.pdf?token=secret",
        resource_url="https://example.com/doc.pdf?sig=secret",
        error="https://user:pass@example.com/doc.pdf?token=secret",
        detail="failed at https://user:pass@example.com/doc.pdf?token=secret",
        started_at=1.0,
        finished_at=2.0,
        duration_ms=1000.0,
    )

    redacted = redacted_pdf_detection_snapshot(raw)

    assert redacted.url == "https://example.com/doc.pdf?redacted"
    assert redacted.resource_url == "https://example.com/doc.pdf?redacted"
    assert redacted.error is not None and "secret" not in redacted.error
    assert redacted.detail is not None and "secret" not in redacted.detail
    assert redacted.started_at == 1.0
    assert redacted.finished_at == 2.0


def test_non_pdf_response_clears_suppression_without_a_cache_entry():
    watchdog, _ = _make_watchdog()
    url = "https://example.com/report.pdf"
    watchdog._detected_downloads.add(url)

    watchdog._note_document_mime("t1", url, "text/html")

    assert url not in watchdog._detected_downloads


@pytest.mark.asyncio
async def test_cache_invalidated_during_size_read_refetches(tmp_path):
    watchdog, session = _make_watchdog(str(tmp_path))
    resource = "https://example.com/report.pdf"
    existing = tmp_path / "cached.pdf"
    existing.write_bytes(MINIMAL_PDF)
    watchdog._session_pdf_urls[resource] = str(existing)
    mock_cdp = MagicMock()
    mock_cdp.session_id = "sid-1"
    mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={
            "result": {
                "value": {
                    "data": list(MINIMAL_PDF),
                    "fromCache": False,
                    "responseSize": len(MINIMAL_PDF),
                    "contentType": "application/pdf",
                }
            }
        }
    )
    session.get_or_create_cdp_session.return_value = mock_cdp

    real_getsize = os.path.getsize
    invalidated = False

    def getsize(path):
        nonlocal invalidated
        if not invalidated:
            invalidated = True
            watchdog._note_document_mime("t1", resource, "text/html")
        return real_getsize(path)

    with patch(
        "openbrowser.browser.watchdogs.downloads_watchdog.os.path.getsize",
        side_effect=getsize,
    ):
        result = await watchdog.download_file_from_url(resource, "t1", content_type="application/pdf")

    assert result is not None
    assert result != str(existing)
    assert Path(result).read_bytes() == MINIMAL_PDF
    assert watchdog._session_pdf_urls.get(resource) != str(existing)


@pytest.mark.asyncio
async def test_history_match_respects_recorded_non_pdf_mime():
    watchdog, session = _make_watchdog()
    session.cdp_client.send.Target.getTargets = AsyncMock(
        return_value=_targets(("t1", "https://example.com/stale-viewer"))
    )
    watchdog._note_document_mime("t1", "https://example.com/late.pdf", "text/html")

    with patch.object(
        DownloadsWatchdog,
        "_check_network_headers_for_pdf",
        new_callable=AsyncMock,
        return_value=(True, "https://example.com/late.pdf"),
    ):
        snapshot = await watchdog._detect_pdf("t1", url="https://example.com/stale-viewer")

    assert snapshot.phase is PdfDetectionPhase.NOT_PDF
