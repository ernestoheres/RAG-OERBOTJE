"""Cache-only tests: no notebook import, model loading, or external API calls."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from notebook_cache import ExperimentCache


def test_lazy_initialization_and_persistence_in_another_process(tmp_path):
    root = tmp_path / "cache"
    cache = ExperimentCache(root)
    assert not root.exists()
    cache.put("translations", {"text": "Vraag", "model": "nl-en"}, "Question")
    result = subprocess.run(
        [sys.executable, "-c", (
            "from notebook_cache import ExperimentCache; import sys; "
            "assert ExperimentCache(sys.argv[1]).get('translations', "
            "{'model': 'nl-en', 'text': 'Vraag'}) == 'Question'"
        ), str(root)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changed", [
    {"text": "Andere vraag"}, {"model": "other-model"}, {"scope": "few-shot-run"},
    {"revision": "other-revision"}, {"version": 2},
])
def test_keys_invalidate_changed_content_model_scope_and_versions(tmp_path, changed):
    cache = ExperimentCache(tmp_path / "cache")
    identity = {"text": "Vraag", "model": "nl-en", "scope": "zero-shot-run", "revision": "abc", "version": 1}
    cache.put("translations", identity, "Question")
    assert cache.get("translations", dict(reversed(list(identity.items())))) == "Question"
    assert cache.get("translations", {**identity, **changed}) is None
    assert cache.get("answers", identity) is None


def test_failed_computation_is_retried_but_success_is_reused(tmp_path):
    cache = ExperimentCache(tmp_path / "cache")
    calls = []

    def compute():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return ""

    with pytest.raises(RuntimeError, match="temporary failure"):
        cache.get_or_compute("translations", "key", compute)
    assert cache.get("translations", "key") is None
    assert cache.get_or_compute("translations", "key", compute) == ""
    assert cache.get_or_compute("translations", "key", compute) == ""
    assert len(calls) == 2


def test_checkpoint_survives_interrupted_export_and_retains_timings(tmp_path, monkeypatch):
    cache = ExperimentCache(tmp_path / "cache")
    export = tmp_path / "answers.json"
    old = {"fingerprint": "run", "samples": []}
    new = {"fingerprint": "run", "samples": [{"id": "1", "seconds": 12.345}]}
    cache.save_checkpoint("answers", "run", old, export)

    def interrupted_replace(*args):
        raise OSError("interrupted export")

    with monkeypatch.context() as patch:
        patch.setattr("notebook_cache.os.replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted export"):
            cache.save_checkpoint("answers", "run", new, export)
    # The readable mirror stays valid, and the newer DB checkpoint is authoritative.
    assert json.loads(export.read_text()) == old
    reopened = ExperimentCache(tmp_path / "cache")
    recovered = reopened.get("answers", "run")
    assert recovered == new
    assert not list(tmp_path.glob(".*.tmp"))
    reopened.save_checkpoint("answers", "run", recovered, export)
    assert json.loads(export.read_text()) == new
    export.unlink()
    assert reopened.get("answers", "run") == new


def test_partial_scores_are_separate_per_run_and_judge(tmp_path):
    cache = ExperimentCache(tmp_path / "cache")
    export = tmp_path / "scores.json"
    identity = {"run": "a", "evaluator": "judge-a"}
    payload = {"samples": [{"id": "1", "faithfulness": 0.75, "factual_correctness": None}]}
    cache.save_checkpoint("scores", identity, payload, export)
    assert cache.get("scores", identity) == payload
    assert cache.get("scores", {"run": "a", "evaluator": "judge-b"}) is None
    assert cache.get("scores", {"run": "b", "evaluator": "judge-a"}) is None
    export.write_text('{"samples":')
    # Corrupting an export cannot destroy a previously committed checkpoint.
    assert cache.get("scores", identity) == payload


def test_nonfinite_scores_do_not_overwrite_valid_checkpoint(tmp_path):
    cache = ExperimentCache(tmp_path / "cache")
    cache.put("scores", "run", {"faithfulness": 0.8})
    with pytest.raises(ValueError):
        cache.put("scores", "run", {"faithfulness": float("nan")})
    assert cache.get("scores", "run") == {"faithfulness": 0.8}
