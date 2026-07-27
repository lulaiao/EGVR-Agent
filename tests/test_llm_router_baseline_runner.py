from __future__ import annotations

import json

import pytest

from egvr import llm_router_baseline_runner as runner_mod
from egvr.llm_router_baseline_runner import run_llm_router_baseline


def test_llm_router_baseline_heuristic_reports_valid_plans(tmp_path):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": (
                    "Generate 5 molecules for protein_path=/tmp/receptor.pdb "
                    "pocket_center=[1,2,3] for synthesizability and toxicity."
                ),
                "metadata": {"dataset": "CrossDocked2020"},
                "expected_tools": ["rxnflow", "scscore", "toxicity"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        output_path=tmp_path / "llm_router_summary.json",
        project_root=tmp_path,
    )

    row = payload["row"]
    assert row["task_count"] == 1
    assert row["valid_json_count"] == 1
    assert row["valid_schema_count"] == 1
    assert row["hallucinated_tool_count"] == 0
    assert row["tool_recall"] == 1.0
    assert payload["task_results"][0]["raw_response"]


def test_llm_router_baseline_detects_hallucinated_tools(tmp_path):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt pocket_center=[1,2,3] box_size=[20,20,20].",
                "metadata": {"dataset": "PDBbind+"},
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        router_mode="heuristic_invalid_tool",
        output_path=tmp_path / "llm_router_summary.json",
        project_root=tmp_path,
    )

    assert payload["row"]["hallucinated_tool_count"] == 1
    assert payload["task_results"][0]["hallucinated_tools"] == ["imaginary_docking_tool"]


def test_llm_router_baseline_allows_generated_smiles_as_downstream_input(tmp_path):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": (
                    "Generate 5 molecules for protein_path=/tmp/receptor.pdb "
                    "pocket_center=[1,2,3] and evaluate synthesizability."
                ),
                "metadata": {"dataset": "CrossDocked2020"},
                "expected_tools": ["rxnflow", "scscore"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = {
        "task_id": "task_1",
        "planner_type": "llm_as_router",
        "selected_tools": ["rxnflow", "scscore"],
        "tool_sequence": [
            {
                "tool_name": "rxnflow",
                "reason": "Generate molecules from the supplied protein pocket.",
                "action": "default",
                "expected_outputs": ["generated_smiles"],
                "required_inputs": ["protein_path"],
                "optional_inputs": ["pocket_center", "num_samples"],
                "parameters": {},
            },
            {
                "tool_name": "scscore",
                "reason": "Evaluate synthesizability of generated molecules.",
                "action": "default",
                "expected_outputs": ["scscore"],
                "required_inputs": ["input_smiles (from rxnflow.generated_smiles)", "candidate_0_smiles"],
                "optional_inputs": [],
                "parameters": {},
            },
        ],
        "expected_outputs": ["generated_smiles", "scscore"],
        "notes": [],
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps({"task_id": "task_1", "raw_response": json.dumps(response)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        responses_jsonl=responses,
        output_path=tmp_path / "llm_router_summary.json",
        project_root=tmp_path,
    )

    assert payload["task_results"][0]["missing_required_inputs"] == []
    assert payload["row"]["missing_required_input_count"] == 0


def test_llm_router_api_mode_requires_explicit_api_config(tmp_path, monkeypatch):
    for key in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt pocket_center=[1,2,3] box_size=[20,20,20].",
                "metadata": {"dataset": "PDBbind+"},
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="router_mode='api' requires explicit"):
        run_llm_router_baseline(
            benchmark_paths=[benchmark],
            router_mode="api",
            output_path=tmp_path / "llm_router_summary.json",
            project_root=tmp_path,
        )


def test_llm_router_api_mode_saves_raw_response_and_fixed_temperature(tmp_path, monkeypatch):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt pocket_center=[1,2,3] box_size=[20,20,20].",
                "metadata": {"dataset": "PDBbind+"},
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_workflow = json.dumps(
        {
            "task_id": "task_1",
            "planner_type": "llm_as_router",
            "selected_tools": ["vina"],
            "tool_sequence": [
                {
                    "tool_name": "vina",
                    "reason": "Dock the supplied ligand against the supplied receptor.",
                    "action": "default",
                    "expected_outputs": ["docking_score", "docked_pose_path"],
                    "required_inputs": ["protein_path", "ligand_path", "pocket_center", "box_size"],
                    "optional_inputs": ["exhaustiveness"],
                    "parameters": {},
                }
            ],
            "expected_outputs": ["docking_score", "docked_pose_path"],
            "notes": [],
        },
        sort_keys=True,
    )
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "chatcmpl-test",
                "choices": [{"message": {"content": raw_workflow}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(runner_mod.requests, "post", fake_post)

    output_path = tmp_path / "llm_router_summary.json"
    response_log_path = tmp_path / "llm_router_summary.responses.jsonl"
    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        router_mode="api",
        output_path=output_path,
        project_root=tmp_path,
        llm_model="test-router-model",
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_api_key="secret-key",
        llm_extra_body_json='{"enable_thinking": false}',
        temperature=0.0,
        max_tokens=512,
        request_timeout_sec=3.0,
    )

    assert calls[0]["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-key"
    assert calls[0]["json"]["model"] == "test-router-model"
    assert calls[0]["json"]["temperature"] == 0.0
    assert calls[0]["json"]["max_tokens"] == 512
    assert calls[0]["json"]["enable_thinking"] is False
    assert calls[0]["timeout"] == 3.0
    assert payload["row"]["valid_json_count"] == 1
    assert payload["row"]["valid_schema_count"] == 1
    assert payload["task_results"][0]["response_source"] == "api"
    assert payload["task_results"][0]["raw_response"] == raw_workflow
    assert payload["task_results"][0]["api_metadata"]["api_model"] == "test-router-model"
    assert payload["task_results"][0]["api_metadata"]["api_protocol"] == (
        "openai_compatible_chat_completions"
    )
    assert payload["task_results"][0]["api_metadata"]["request_payload"]["enable_thinking"] is False
    assert payload["task_results"][0]["api_metadata"]["response_payload"]["id"] == "chatcmpl-test"
    assert payload["task_results"][0]["prompt_hash"]
    assert payload["task_results"][0]["request_hash"]
    assert payload["task_results"][0]["plan_hash"]
    assert payload["task_results"][0]["prompt_messages"]
    assert "secret-key" not in output_path.read_text(encoding="utf-8")
    assert response_log_path.exists()
    response_log_text = response_log_path.read_text(encoding="utf-8")
    response_log_row = json.loads(response_log_text.strip())
    assert response_log_row["raw_response"] == raw_workflow
    assert response_log_row["prompt_hash"]
    assert response_log_row["request_hash"]
    assert response_log_row["plan_hash"]
    assert response_log_row["api_metadata"]["usage"]["completion_tokens"] == 20
    assert "secret-key" not in response_log_text


def test_llm_router_api_mode_loads_explicit_dotenv(tmp_path, monkeypatch):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": (
                    "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt "
                    "pocket_center=[1,2,3] box_size=[20,20,20]."
                ),
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dotenv = tmp_path / ".env.test"
    dotenv.write_text(
        "TEST_ROUTER_KEY=dotenv-secret\nTEST_ROUTER_MODEL=unused\n",
        encoding="utf-8",
    )
    raw_workflow = json.dumps(
        {
            "task_id": "task_1",
            "planner_type": "llm_as_router",
            "selected_tools": ["vina"],
            "tool_sequence": [],
            "expected_outputs": [],
            "notes": [],
        }
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "dotenv-test",
                "choices": [{"message": {"content": raw_workflow}, "finish_reason": "stop"}],
                "usage": {},
            }

    def fake_post(url, *, headers, json, timeout):
        assert headers["Authorization"] == "Bearer dotenv-secret"
        return FakeResponse()

    monkeypatch.setattr(runner_mod.requests, "post", fake_post)
    output = tmp_path / "out.json"
    run_llm_router_baseline(
        benchmark_paths=[benchmark],
        router_mode="api",
        output_path=output,
        project_root=tmp_path,
        dotenv_path=dotenv,
        llm_model="dotenv-model",
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_api_key_env="TEST_ROUTER_KEY",
    )

    assert "dotenv-secret" not in output.read_text(encoding="utf-8")


def test_llm_router_api_mode_resumes_from_response_log_without_new_api_call(tmp_path, monkeypatch):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt pocket_center=[1,2,3] box_size=[20,20,20].",
                "metadata": {"dataset": "PDBbind+"},
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_workflow = json.dumps(
        {
            "task_id": "task_1",
            "planner_type": "llm_as_router",
            "selected_tools": ["vina"],
            "tool_sequence": [
                {
                    "tool_name": "vina",
                    "reason": "Dock the supplied ligand against the supplied receptor.",
                    "action": "default",
                    "expected_outputs": ["docking_score"],
                    "required_inputs": ["protein_path", "ligand_path", "pocket_center", "box_size"],
                    "optional_inputs": [],
                    "parameters": {},
                }
            ],
            "expected_outputs": ["docking_score"],
            "notes": [],
        },
        sort_keys=True,
    )
    response_log = tmp_path / "cached.responses.jsonl"
    response_log.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_response": raw_workflow,
                "recorded_at": "2026-07-25T00:00:00Z",
                "prompt_hash": "cached-prompt-hash",
                "request_hash": "cached-request-hash",
                "api_metadata": {
                    "api_model": "test-router-model",
                    "usage": {"total_tokens": 12},
                    "request_payload": {
                        "model": "test-router-model",
                        "messages": [{"role": "user", "content": "cached"}],
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_post(*args, **kwargs):
        raise AssertionError("API should not be called for cached response rows")

    monkeypatch.setattr(runner_mod.requests, "post", fail_post)

    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        response_log_jsonl=response_log,
        router_mode="api",
        output_path=tmp_path / "llm_router_summary.json",
        project_root=tmp_path,
        llm_model="test-router-model",
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_api_key="secret-key",
    )

    assert payload["row"]["valid_schema_count"] == 1
    assert payload["task_results"][0]["response_source"] == "responses_jsonl"
    assert payload["task_results"][0]["api_metadata"]["usage"]["total_tokens"] == 12
    assert payload["task_results"][0]["prompt_hash"] == "cached-prompt-hash"
    assert payload["task_results"][0]["request_hash"] == "cached-request-hash"
    assert payload["task_results"][0]["api_response_recorded_at"] == "2026-07-25T00:00:00Z"


def test_llm_router_accepts_single_markdown_json_fence_without_repairing_content(tmp_path):
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_user_query": (
                    "Dock ligand_path=/tmp/lig.sdf protein_path=/tmp/rec.pdbqt "
                    "pocket_center=[1,2,3] box_size=[20,20,20]."
                ),
                "expected_tools": ["vina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workflow = {
        "task_id": "task_1",
        "planner_type": "llm_as_router",
        "selected_tools": ["vina"],
        "tool_sequence": [],
        "expected_outputs": [],
        "notes": [],
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "raw_response": "```json\n" + json.dumps(workflow) + "\n```",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_llm_router_baseline(
        benchmark_paths=[benchmark],
        responses_jsonl=responses,
        router_mode="api_replay",
        output_path=tmp_path / "out.json",
        project_root=tmp_path,
    )

    assert payload["row"]["valid_json_count"] == 1
    assert payload["row"]["valid_schema_count"] == 1
    assert payload["task_results"][0]["response_normalization"] == "markdown_json_fence_removed"
