"""
Static asset integrity.

These exist because of a real failure. Three.js was vendored by downloading
`three.module.js` and `OrbitControls.js`, and both were verified to serve 200 --
but `three.module.js` (r167 and later) itself does `import './three.core.js'`,
which was never downloaded. The browser fetched it, got a 404, the module graph
failed to link, and Chrome reported the failure against the ENTRY point
(`viewer.js`) rather than the file that was actually missing.

The lesson generalises: vendoring a library means vendoring its whole import
graph, and checking the files you chose to download proves nothing about the
files they depend on. So these tests walk the graph instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"

# Matching import statements with a regex needs care: a naive pattern also
# matches ordinary strings. three.module.js contains
#     console.warn( 'THREE.WebGLRenderer: Texture has been resized from (' + ... )
# which a loose /(?:from|import)\s*['"]/ happily reports as importing
# " + dimensions.width + ". So these patterns are anchored to the start of a
# line and forbid a quote before the specifier, which excludes string contents.
#
# The forms that must be caught:
#     import * as THREE from 'three';          line begins with `import`
#     } from 'three';                          continuation of a multi-line import
#     import './side-effect.js';               bare side-effect import
#     import('./viewer.js')                    dynamic import
_FROM_CLAUSE = r"""(?m)^(?:\s*(?:import|export)\b[^'"\n]*|\s*\}[^'"\n]*)from\s*['"]([^'"]+)['"]"""
_SIDE_EFFECT = r"""(?m)^\s*import\s*['"]([^'"]+)['"]"""
_DYNAMIC = r"""import\s*\(\s*['"]([^'"]+)['"]\s*\)"""

_ALL_SPECIFIERS = [re.compile(p) for p in (_FROM_CLAUSE, _SIDE_EFFECT, _DYNAMIC)]


def specifiers(source: Path) -> list[str]:
    """Every module specifier imported by a file."""
    text = source.read_text()
    found: list[str] = []
    for pattern in _ALL_SPECIFIERS:
        found.extend(pattern.findall(text))
    return found


def is_relative(specifier: str) -> bool:
    return specifier.startswith(".")


def javascript_files() -> list[Path]:
    return sorted(STATIC.rglob("*.js"))


def import_map() -> dict[str, str]:
    html = (STATIC / "index.html").read_text()
    match = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert match, "index.html must contain an import map"
    return json.loads(match.group(1))["imports"]


def test_there_are_javascript_files_to_check() -> None:
    """Guard against the suite silently passing because it found nothing."""
    assert javascript_files(), f"no .js files under {STATIC}"


@pytest.mark.parametrize("source", javascript_files(), ids=lambda p: p.name)
def test_every_relative_import_resolves(source: Path) -> None:
    """
    Walk the module graph. This is the test that would have caught the missing
    three.core.js before it reached the browser.
    """
    for specifier in (s for s in specifiers(source) if is_relative(s)):
        target = (source.parent / specifier).resolve()
        assert target.is_file(), (
            f"{source.name} imports {specifier!r}, which does not exist at {target}. "
            "If this is a vendored library, it ships more files than were downloaded."
        )


@pytest.mark.parametrize("source", javascript_files(), ids=lambda p: p.name)
def test_every_bare_specifier_is_in_the_import_map(source: Path) -> None:
    """
    A browser cannot resolve a bare specifier without an import map entry. There
    is no build step here to paper over a missing one.
    """
    mapping = import_map()
    for specifier in (s for s in specifiers(source) if not is_relative(s)):
        assert specifier in mapping, (
            f"{source.name} imports bare specifier {specifier!r}, which the import "
            f"map does not define. Known: {sorted(mapping)}"
        )


def test_import_map_targets_exist() -> None:
    for specifier, target in import_map().items():
        resolved = (STATIC / target.lstrip("./")).resolve()
        assert resolved.is_file(), (
            f"import map points {specifier!r} at {target}, which does not exist"
        )


def test_no_external_urls_in_the_page() -> None:
    """
    Everything must be served from this origin.

    The point of vendoring was that the page has no external dependency at
    review time; a CDN link creeping back in would silently undo that, and
    would only fail on a network that blocks it.
    """
    html = (STATIC / "index.html").read_text()
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert not external, f"index.html references external URLs: {external}"
