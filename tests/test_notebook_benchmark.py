"""Exercise pure table cells and scoring with fake metrics, without running models."""

import ast
import asyncio
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import pytest

from notebook_cache import ExperimentCache


NOTEBOOK = Path(__file__).resolve().parents[1] / "RAG-OERBOTJE.py"
CELLS = ast.parse(NOTEBOOK.read_text()).body
METRICS = ["faithfulness", "context_precision_with_reference", "factual_correctness"]


def cell_outputs(name, **inputs):
    """Compile only the requested cell; never import or execute the notebook app."""
    cell = next(cell for cell in CELLS if isinstance(cell, ast.FunctionDef) and any(
        (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name)
        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        for node in ast.walk(cell)
    ))
    cell = copy.deepcopy(cell)
    standalone = any(isinstance(item, ast.Attribute) and item.attr == "function" for item in cell.decorator_list)
    cell.decorator_list = []
    scope = dict(inputs)
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    if standalone:
        return {cell.name: scope[cell.name]}
    result = scope[cell.name](**{arg.arg: inputs[arg.arg] for arg in cell.args.args})
    names = [node.id for node in cell.body[-1].value.elts]
    return dict(zip(names, result))


def frames(answers=None, scores=None):
    mo = SimpleNamespace(ui=SimpleNamespace(dropdown=lambda **kwargs: SimpleNamespace(value=kwargs["value"])))
    variants = cell_outputs("ablation_variants", mo=mo)["ablation_variants"]
    return cell_outputs(
        "variant_table", ablation_variants=variants, benchmark_metric_names=METRICS,
        saved_answers=answers or {}, saved_scores=scores or {}, pd=pd,
    )


def test_empty_results_still_show_all_eight_variants():
    data = frames()
    overview = cell_outputs("overview", benchmark_questions=[None] * 100, **data)["overview"]
    timing = cell_outputs("timing_table", **data)["timing_table"]
    delta = cell_outputs("paired_few_shot_delta", **data)["paired_few_shot_delta"]
    effects = cell_outputs("few_shot_effect", paired_few_shot_delta=delta, **data)["few_shot_effect"]
    assert len(overview) == len(timing) == 8
    assert len(effects) == 4
    assert overview["answers"].eq(0).all()
    assert overview["valid F/P/F1"].eq("0/0/0").all()
    assert not overview["complete"].any()


def test_overview_counts_valid_scores_and_marks_partial_runs():
    answers = {"nl_zero": [{"id": "1", "seconds": 1}, {"id": "2", "seconds": 3}]}
    scores = {"nl_zero": [
        {"id": "1", **dict.fromkeys(METRICS, 0.8)},
        {"id": "2", "faithfulness": 0.6, "context_precision_with_reference": None, "factual_correctness": None},
    ]}
    data = frames(answers, scores)
    row = cell_outputs("overview", benchmark_questions=[None, None], **data)["overview"].set_index("variant").loc["nl_zero"]
    assert row["faithfulness"] == 0.7
    assert row["factual_f1"] == 0.8
    assert row["valid F/P/F1"] == "2/1/1"
    assert row["seconds_mean"] == 2
    assert row["seconds_p95"] == 2.9
    assert not row["complete"]


def test_few_shot_comparison_pairs_question_ids_before_averaging():
    data = frames(
        answers={
            "nl_zero": [{"id": "1", "seconds": 100}, {"id": "2", "seconds": 4}],
            "nl_few": [{"id": "2", "seconds": 7}, {"id": "3", "seconds": 200}],
        },
        scores={
            "nl_zero": [{"id": "1", "factual_correctness": 0.1}, {"id": "2", "factual_correctness": 0.6}],
            "nl_few": [{"id": "2", "factual_correctness": 0.9}, {"id": "3", "factual_correctness": 0.2}],
        },
    )
    delta = cell_outputs("paired_few_shot_delta", **data)["paired_few_shot_delta"]
    effects = cell_outputs("few_shot_effect", paired_few_shot_delta=delta, **data)["few_shot_effect"].set_index("route")
    assert effects.loc["nl", "ΔF1 few − zero"] == 0.3
    assert effects.loc["nl", "Δtijd few − zero (s)"] == 3
    assert effects.loc["nl", "Gepaarde F1-vragen"] == 1
    assert effects.loc["nl", "Gepaarde timingvragen"] == 1


def test_scoring_error_surfaces_and_retry_resumes_after_last_metric(tmp_path):
    cache = ExperimentCache(tmp_path / "cache")
    (tmp_path / "Data").mkdir()
    mo = SimpleNamespace(status=SimpleNamespace(progress_bar=lambda items, **kwargs: items))
    evaluate = cell_outputs(
        "evaluate_benchmark_run", benchmark_dir=tmp_path, benchmark_metric_names=METRICS,
        experiment_cache=cache, mo=mo, perf_counter=perf_counter, math=math,
        ragas_metric_inputs=cell_outputs("ragas_metric_inputs")["ragas_metric_inputs"],
    )["evaluate_benchmark_run"]

    class FakeMetric:
        def __init__(self, fail=False):
            self.fail, self.calls = fail, 0

        async def ascore(self, **kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("judge unavailable")
            return SimpleNamespace(value=0.8)

    metrics = {name: FakeMetric(fail=name == METRICS[1]) for name in METRICS}
    run = {"fingerprint": "run", "stem": "ablation_test", "variant": "nl_zero", "samples": [{
        "id": "1", "vraag": "Question", "response": "Answer", "antwoord": "Reference",
        "retrieved_contexts": ["Context"],
    }]}
    evaluator = ({"model": "fake-judge"}, "judge")
    with pytest.raises(RuntimeError, match="judge unavailable"):
        asyncio.run(evaluate(run, metrics, evaluator))
    saved = cache.get("scores", {"run": "run", "evaluator": "judge"})["samples"][0]
    assert saved[METRICS[0]] == 0.8
    assert saved[METRICS[1]] is None

    metrics[METRICS[1]].fail = False
    result = asyncio.run(evaluate(run, metrics, evaluator))
    assert all(result["samples"][0][name] == 0.8 for name in METRICS)
    assert [metrics[name].calls for name in METRICS] == [1, 2, 1]
    assert result["cache_status"] == {"cached_metrics": 1, "new_metrics": 2}

    result = asyncio.run(evaluate(run, metrics, evaluator))
    assert result["cache_status"] == {"cached_metrics": 3, "new_metrics": 0}
    assert [metrics[name].calls for name in METRICS] == [1, 2, 1]


def test_openrouter_judge_identity_includes_endpoint_and_model():
    inputs = {
        "cache_versions": {"ragas": "test-version"}, "hashlib": hashlib, "json": json,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_model": "openai/gpt-4.1-mini",
    }
    result = cell_outputs("evaluator_fingerprint", **inputs)
    settings, fingerprint = result["display_evaluator"]
    assert settings["provider"] == "openrouter"
    assert settings["base_url"] == inputs["openrouter_base_url"]
    assert not any("key" in key for key in settings)
    for changed in ({"openrouter_model": "openai/gpt-4.1"}, {"openrouter_base_url": "https://other.example/v1"}):
        assert cell_outputs("evaluator_fingerprint", **{**inputs, **changed})["evaluator_fingerprint"] != fingerprint


def test_interactive_stages_cache_only_complete_answers_and_invalidate_changed_question(tmp_path):
    calls = []

    def generate(messages, max_tokens):
        question = messages[0]["content"]
        calls.append(question)
        return {
            "response": question, "seconds": 1.23, "input_tokens": 5, "output_tokens": 2,
            "output_limit_reached": False, "prompt": f"Prompt: {question}",
        }

    inputs = {
        "ablation_choice": SimpleNamespace(value="nl_zero"),
        "experiment_cache": ExperimentCache(tmp_path / "cache"),
        "mo": SimpleNamespace(stop=lambda *args: None, md=lambda text: text),
        "rag_question": SimpleNamespace(value="Question"),
        "run_preview": SimpleNamespace(value="Ja"), "generate_rag_pass": generate,
        "demo_messages": [{"role": "user", "content": "Question"}],
        "generation_settings": {"answer_max_new_tokens": 256}, "translation_settings": {},
        "cache_versions": {"schema": 1}, "llm_id": "test-model", "llm_runtime": {},
    }

    first = cell_outputs("rag_result_cached", **inputs)
    assert not first["rag_result_cached"]
    # An interrupted first/second pass must not leave a misleading completed cache entry.
    assert inputs["experiment_cache"].get("preview_answers", first["preview_identity"]) is None
    saved = cell_outputs(
        "rag_result", **first, experiment_cache=inputs["experiment_cache"],
        demo_final_response=first["first_response"], demo_contexts=["Source"],
        demo_model_question="Question", demo_messages=inputs["demo_messages"],
        demo_second_messages=[], demo_translation_seconds=0.1,
        demo_translation_cache={"hits": 0, "misses": 0}, second_pass=None,
    )["rag_result"]
    second = cell_outputs("rag_result_cached", **inputs)
    assert second["rag_result_cached"]
    assert second["preview_cached_result"] == saved
    assert second["first_prompt"] == saved["model_prompt"]
    assert calls == ["Question"]
    inputs["rag_question"].value = "Another question"
    inputs["demo_messages"] = [{"role": "user", "content": "Another question"}]
    assert not cell_outputs("rag_result_cached", **inputs)["rag_result_cached"]
    assert calls == ["Question", "Another question"]


def test_cached_translation_reports_hit_without_loading_opus(tmp_path):
    cache = ExperimentCache(tmp_path / "cache")
    identity = {"text": "Vraag", "settings": {"model": "opus"}, "versions": {"schema": 1}, "scope": "interactive"}
    cache.put("translations", identity, "Question")
    translate = cell_outputs(
        "translate_nl_en", cache_versions=identity["versions"], experiment_cache=cache,
        translation_settings=identity["settings"], load_opus_translator=None,
        split_translation_text=None, translate_chunks=None, rejoin_translation=None,
    )["translate_nl_en"]
    stats = {"hits": 0, "misses": 0}
    assert translate("Vraag", cache_stats=stats) == "Question"
    assert stats == {"hits": 1, "misses": 0}
    assert translate("\n ", cache_stats=stats) == "\n "
    assert stats == {"hits": 1, "misses": 0}


def test_translation_splitting_keeps_all_source_characters():
    split = cell_outputs(
        "split_translation_text", re=re, translation_settings={"max_source_tokens": 20},
    )["split_translation_text"]
    source = "OER  regel  met\nuitzonderingen " * 8
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    chunks = split(source, tokenizer)
    assert "".join(chunks) == source
    assert all(len(chunk) <= 20 for chunk in chunks)
