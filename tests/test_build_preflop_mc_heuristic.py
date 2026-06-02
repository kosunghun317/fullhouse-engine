"""Tests for building preflop-MC heuristic candidates."""

from __future__ import annotations

import zipfile
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_preflop_mc_heuristic import build_preflop_mc_candidate, preflop_mc_override_source, zip_candidate


def test_preflop_mc_override_replaces_preflop_score_name():
    source = preflop_mc_override_source(samples=1024, budget_s=0.2, min_samples=128, mc_probability=0.5)

    assert 'PREFLOP_MC_SAMPLES = _int_env("HEURISTIC_PREFLOP_MC_SAMPLES", 1024)' in source
    assert 'PREFLOP_MC_PROBABILITY = _float_env("HEURISTIC_PREFLOP_MC_PROBABILITY", 0.500000)' in source
    assert "_TABLE_PREFLOP_SCORE = _preflop_score" in source
    assert "random.random() >= mc_probability" in source
    assert "def _preflop_score(cards):" in source
    assert "eval7.evaluate(hero + board)" in source


def test_build_preflop_mc_candidate_from_zip(tmp_path):
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as zf:
        zf.writestr("bot.py", "def _preflop_score(cards):\n    return 42\n")
        zf.writestr("data/tables.npz", b"not-real-npz")

    output = tmp_path / "candidate"
    report = build_preflop_mc_candidate(
        input_path=source_zip,
        output_dir=output,
        samples=512,
        budget_s=0.1,
        min_samples=64,
        mc_probability=0.5,
    )

    assert report["output"] == str(output.resolve())
    assert (output / "bot.py").is_file()
    assert (output / "data" / "tables.npz").is_file()
    text = (output / "bot.py").read_text(encoding="utf-8")
    assert "PREFLOP_MC_SAMPLES" in text
    assert 'HEURISTIC_PREFLOP_MC_SAMPLES", 512' in text
    assert report["mc_probability"] == 0.5
    assert 'HEURISTIC_PREFLOP_MC_PROBABILITY", 0.500000' in text


def test_build_preflop_mc_candidate_refuses_existing_output(tmp_path):
    source = tmp_path / "bot.py"
    source.write_text("def _preflop_score(cards):\n    return 42\n", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        build_preflop_mc_candidate(source, output, samples=128, budget_s=0.1)


def test_zip_candidate_contains_bot_and_data(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "data").mkdir(parents=True)
    (candidate / "bot.py").write_text("def decide(state):\n    return {'action': 'check'}\n", encoding="utf-8")
    (candidate / "data" / "tables.npz").write_bytes(b"fake")
    (candidate / "__pycache__").mkdir()
    (candidate / "__pycache__" / "bot.cpython-310.pyc").write_bytes(b"ignored")

    output = tmp_path / "candidate.zip"
    zip_candidate(candidate, output)

    with zipfile.ZipFile(output) as zf:
        assert sorted(zf.namelist()) == ["bot.py", "data/tables.npz"]
