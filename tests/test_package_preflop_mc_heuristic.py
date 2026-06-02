"""Tests for packaging preflop-MC heuristic submissions."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.package_preflop_mc_heuristic import create_submission_zip, resolve_candidate


VALID_BOT = """
def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"}
"""


def test_create_submission_zip_contains_only_submission_files(tmp_path):
    candidate = tmp_path / "run" / "bot"
    (candidate / "data" / "__pycache__").mkdir(parents=True)
    (candidate / "bot.py").write_text(VALID_BOT, encoding="utf-8")
    (candidate / "data" / "tables.npz").write_bytes(b"fake")
    (candidate / "data" / ".DS_Store").write_bytes(b"ignored")
    (candidate / "data" / "__pycache__" / "x.pyc").write_bytes(b"ignored")

    report = create_submission_zip(candidate, tmp_path / "submission.zip", validate_zip=False)

    assert report["members"] == ["bot.py", "data/tables.npz"]
    with zipfile.ZipFile(report["output"]) as zf:
        assert sorted(zf.namelist()) == ["bot.py", "data/tables.npz"]


def test_create_submission_zip_uses_unique_output_without_overwriting(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "bot.py").write_text(VALID_BOT, encoding="utf-8")
    requested = tmp_path / "submission.zip"
    requested.write_bytes(b"existing")

    report = create_submission_zip(candidate, requested, validate_zip=False)

    assert Path(report["output"]) != requested
    assert requested.read_bytes() == b"existing"
    assert Path(report["output"]).is_file()


def test_create_submission_zip_validates_submission(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "bot.py").write_text(VALID_BOT, encoding="utf-8")

    report = create_submission_zip(candidate, tmp_path / "submission.zip")

    assert report["validator_passed"] is True
    assert report["errors"] == []


def test_resolve_candidate_accepts_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    bot_dir = run_dir / "bot"
    bot_dir.mkdir(parents=True)
    (bot_dir / "bot.py").write_text(VALID_BOT, encoding="utf-8")

    assert resolve_candidate(run_dir) == bot_dir.resolve()


def test_create_submission_zip_rejects_python_files_in_data(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "data").mkdir(parents=True)
    (candidate / "bot.py").write_text(VALID_BOT, encoding="utf-8")
    (candidate / "data" / "helper.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data/ may not contain Python files"):
        create_submission_zip(candidate, tmp_path / "submission.zip", validate_zip=False)
