"""Provenance checks for the benchmark comparison pages.

These tests keep the pages from presenting historical runs as current
version comparisons or citing raw data that is not in the repository.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _page(name):
    return (ROOT / 'docs' / name).read_text()


def test_mcp_comparison_does_not_claim_unrecorded_latest_versions():
    page = _page('comparison.mdx')

    assert 'Playwright MCP (latest)' not in page
    assert 'Chrome DevTools MCP (latest)' not in page
    assert 'resolved version not recorded' in page


def test_mcp_comparison_does_not_cite_missing_raw_data():
    page = _page('comparison.mdx')

    for name in ('playwright_results.json', 'cdp_results.json', 'openbrowser_results.json'):
        assert name not in page
    assert 'not retained in this repository' in page


def test_mcp_comparison_scopes_current_and_universal_feature_claims():
    page = _page('comparison.mdx')

    assert 'Features no competitor offers' not in page
    assert '3 current' not in page
    assert '`get_status` and `reset_browser` were added after v0.1.26' in page
    assert '1 at benchmark (`execute_code`)' in page
    assert '(CodeAgent, 3 tools)' not in page
    assert 'The benchmark itself used `execute_code` only.' in page
    assert 'Feature lists below are from the versions tested on 2026-02-21' in page


def test_comparison_pages_state_historical_scope_and_no_universal_winner():
    for name in ('comparison.mdx', 'cli-comparison.mdx'):
        page = _page(name)
        assert 'Historical benchmark, not a current-version ranking' in page
        assert 'No universal winner' in page


def test_cli_comparison_states_unrecorded_tool_versions_and_seed():
    page = _page('cli-comparison.mdx')

    assert 'does not record the CLI tool versions' in page
    assert 'random seed' in page


def test_native_navigation_lane_is_not_cited_on_either_comparison_page():
    for name in ('comparison.mdx', 'cli-comparison.mdx'):
        page = _page(name)
        assert 'navigation_compare' not in page
        assert 'navigation_results' not in page
