"""Unit tests for baking tuned heuristic env values into submissions."""

from __future__ import annotations

import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.heuristic_env_overrides import bake_heuristic_source, heuristic_env_scope, load_env_overrides
from tools.package_heuristic import package


def test_load_env_overrides_accepts_summary_json_and_shell_exports(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"best_env": {"HEURISTIC_CALL_MARGIN_BASE": "0.091", "IGNORED": "x"}}),
        encoding="utf-8",
    )
    shell = tmp_path / "best_env.sh"
    shell.write_text(
        "export HEURISTIC_DRY_BLUFF_PROB=0.37\nexport OTHER=value\n",
        encoding="utf-8",
    )

    assert load_env_overrides(summary) == {"HEURISTIC_CALL_MARGIN_BASE": "0.091"}
    assert load_env_overrides(shell) == {"HEURISTIC_DRY_BLUFF_PROB": "0.37"}


def test_bake_heuristic_source_replaces_float_and_int_defaults():
    source = '\n'.join([
        'CALL_MARGIN_BASE = _float_env("HEURISTIC_CALL_MARGIN_BASE", 0.075)',
        'FLOP_SAMPLES = _int_env("HEURISTIC_FLOP_SAMPLES", 520)',
    ])

    baked, report = bake_heuristic_source(
        source,
        {
            "HEURISTIC_CALL_MARGIN_BASE": "0.091",
            "HEURISTIC_FLOP_SAMPLES": "640.2",
        },
    )

    assert '_float_env("HEURISTIC_CALL_MARGIN_BASE", 0.091)' in baked
    assert '_int_env("HEURISTIC_FLOP_SAMPLES", 640)' in baked
    assert report["baked_count"] == 2


def test_heuristic_env_scope_clears_stale_heuristic_values(monkeypatch):
    monkeypatch.setenv("HEURISTIC_CALL_MARGIN_BASE", "0.111")
    monkeypatch.setenv("HEURISTIC_DRY_BLUFF_PROB", "0.222")

    with heuristic_env_scope({"HEURISTIC_CALL_MARGIN_BASE": "0.091"}):
        assert os.environ["HEURISTIC_CALL_MARGIN_BASE"] == "0.091"
        assert "HEURISTIC_DRY_BLUFF_PROB" not in os.environ

    assert os.environ["HEURISTIC_CALL_MARGIN_BASE"] == "0.111"
    assert os.environ["HEURISTIC_DRY_BLUFF_PROB"] == "0.222"


def test_package_bakes_env_file_into_zip_bot_py(tmp_path):
    env_file = tmp_path / "best_env.json"
    env_file.write_text(json.dumps({"HEURISTIC_CALL_MARGIN_BASE": "0.091"}), encoding="utf-8")
    output = tmp_path / "heuristic_bot.zip"

    package(output, env_file=env_file)

    with zipfile.ZipFile(output) as zf:
        bot_py = zf.read("bot.py").decode("utf-8")
    assert '_float_env("HEURISTIC_CALL_MARGIN_BASE", 0.091)' in bot_py
