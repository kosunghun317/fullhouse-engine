"""Checks that Markdown Mermaid blocks use renderer-safe syntax."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mermaid_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    pos = 0
    while True:
        start = text.find("```mermaid", pos)
        if start < 0:
            return
        first_newline = text.find("\n", start)
        end = text.find("```", first_newline + 1)
        assert end >= 0, f"Unclosed Mermaid block in {path}"
        yield text[first_newline + 1:end].strip()
        pos = end + 3


def test_mermaid_blocks_use_conservative_graph_syntax():
    docs_root = Path(__file__).resolve().parents[1] / "docs"
    blocks = []
    for path in sorted(docs_root.rglob("*.md")):
        blocks.extend((path, block) for block in _mermaid_blocks(path))

    assert blocks, "Expected at least one Mermaid diagram in docs/"
    for path, block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        assert lines[0] == "graph TD", f"{path} must start Mermaid blocks with graph TD"
        assert "sequenceDiagram" not in block, f"{path} uses less portable sequenceDiagram syntax"
        assert "graph LR" not in block and "graph TB" not in block, f"{path} uses nonstandard graph direction"
        assert "\\n" not in block, f"{path} uses escaped newlines inside labels"
