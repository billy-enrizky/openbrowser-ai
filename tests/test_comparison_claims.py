"""Current-evidence checks for the benchmark comparison pages.

These tests keep the pages scoped to the current native lanes, prevent
unrecorded version claims, and verify that local links resolve.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _page(name):
    return (ROOT / 'docs' / name).read_text()


def _sections(text):
    return [title for title in re.findall(r'^## (.+)$', text, re.M) if title != 'Contents']


def _local_links(text):
    pattern = re.compile(r'(?:\]\(([^)]+)\)|<img[^>]+src="([^"]+)")')
    for match in pattern.finditer(text):
        target = next((group for group in match.groups() if group), '')
        target = target.split('#')[0]
        if target.startswith('https://github.com/billy-enrizky/openbrowser-ai/blob/main/'):
            yield ROOT / target.split('/blob/main/', 1)[1]
            continue
        if not target or target.startswith(('http://', 'https://', 'mailto:')):
            continue
        if target.startswith('/images/'):
            yield ROOT / 'docs' / 'images' / target.removeprefix('/images/')
        elif target.startswith('benchmarks/'):
            yield ROOT / target
        elif target.startswith('/'):
            route = target.strip('/')
            candidates = [ROOT / 'docs' / f'{route}.mdx', ROOT / 'docs' / f'{route}.md']
            yield next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        else:
            yield ROOT / 'docs' / target


def test_comparison_page_has_current_sections_only():
    assert _sections(_page('comparison.mdx')) == [
        '1. Benchmark index',
        '2. How to read this page',
        '3. Current interactive lanes, 2026-11-25',
        '4. OpenBrowser no-browser memory, 2026-11-25',
        '5. Other native local lanes, 2026-11-24',
        '6. How to choose a surface',
        '7. What these lanes show',
        '8. What this page does not measure',
    ]


def test_benchmarks_article_is_a_current_guide():
    article = _page('examples/apps/benchmarks.mdx')
    assert 'title: "Benchmark Guide"' in article
    assert 'How to run Agent and CodeAgent benchmarks' in article
    assert _sections(article) == [
        '1. Overview',
        '2. Quick Comparison',
        '3. Running Benchmarks',
        '4. Key Findings',
        '5. Test Tasks',
        '6. Source Code',
    ]


def test_github_main_links_are_checked():
    text = '[source](https://github.com/billy-enrizky/openbrowser-ai/blob/main/benchmarks/interactive/run_benchmark.py)'
    assert list(_local_links(text)) == [ROOT / 'benchmarks' / 'interactive' / 'run_benchmark.py']


def test_comparison_page_links_resolve():
    for path in _local_links(_page('comparison.mdx')):
        assert path.exists(), path


def test_article_links_resolve():
    for path in _local_links(_page('examples/apps/benchmarks.mdx')):
        assert path.exists(), path


def test_benchmark_readme_links_resolve():
    for name in ('interactive', 'scraping'):
        text = (ROOT / 'benchmarks' / name / 'README.md').read_text()
        for path in _local_links(text):
            assert path.exists(), path


def test_benchmark_readmes_have_current_sections_only():
    interactive = (ROOT / 'benchmarks' / 'interactive' / 'README.md').read_text()
    scraping = (ROOT / 'benchmarks' / 'scraping' / 'README.md').read_text()
    assert _sections(interactive) == [
        '1. How to read these numbers',
        '2. Current results, 2026-11-25, rotated N=5',
        '3. Interactive resource lane, 2026-11-25',
        '4. Arms',
        '5. Tasks',
        '6. Fair-run procedure',
        '7. Metrics',
        '8. OpenBrowser lean profile',
        '9. Limitations',
        '10. Setup',
        '11. Run',
        '12. Files',
        '13. Safety and cleanup',
    ]
    assert _sections(scraping) == [
        '1. How to read these numbers',
        '2. Current results, 2026-11-25, and reference lanes',
        '3. Arms',
        '4. Tasks',
        '5. Run',
        '6. Batch harness',
        '7. Firecrawl lane reference procedure',
        '8. Limits',
    ]


def test_cli_comparison_route_redirects_to_comparison():
    assert not (ROOT / 'docs' / 'cli-comparison.mdx').exists()
    config = json.loads((ROOT / 'docs' / 'docs.json').read_text())
    assert {'source': '/cli-comparison', 'destination': '/comparison'} in config.get('redirects', [])
    pages = []
    for group in config.get('navigation', {}).get('groups', []):
        pages.extend(page for page in group.get('pages', []) if isinstance(page, str))
    assert 'cli-comparison' not in pages


def test_comparison_page_does_not_claim_a_universal_winner():
    page = _page('comparison.mdx')
    assert 'No universal winner is claimed' in page
    assert 'navigation_compare' not in page
    assert 'navigation_results' not in page


def test_current_docs_do_not_cite_removed_benchmark_scripts():
    readme = (ROOT / 'README.md').read_text()
    plugin = (ROOT / 'plugin' / 'README.md').read_text()
    for text in (readme, plugin):
        assert 'openbrowser_benchmark.py' not in text
        assert 'playwright_benchmark.py' not in text
        assert 'cdp_benchmark.py' not in text
    assert '## 13. Benchmark: Native Local Lanes' in readme
    assert '## 4. Benchmark: Native Local Lanes' in plugin
