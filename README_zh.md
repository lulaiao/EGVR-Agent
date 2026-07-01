# FullCopilot

FullCopilot 是一个面向可信工具调用的 biomedical agent 框架。它将用户请求解析为结构化任务，按任务选择必要工具，执行工具调用，并用 verifier 检查真实输出是否足以支持任务完成。

## 核心思想

FullCopilot 关注的不是“给大模型更多工具”，而是让 agent 在执行后留下可验证证据：选了什么工具、为什么调用、工具返回了什么、哪些证据通过验证、失败时如何修复或明确标记未完成。

## 主要功能

- 将自然语言请求解析为 `ParsedTask`。
- 根据任务目标生成 `PlannedWorkflow`。
- 通过结构化 executor 调用工具或 offline wrapper。
- 将工具输出归一化为候选结果和证据记录。
- 使用 verifier 检查 SMILES、分数、证据字段、provenance 等是否完整。
- 在证据缺失或工具失败时执行 conservative repair / fallback。
- 保存 JSONL execution trace，便于复现、审计和后续学习型 planner 训练。
- 提供 benchmark runner、baseline runner 和 release audit 工具。

## 快速开始

```bash
conda create -n fullcopilot python=3.11
conda activate fullcopilot
pip install -e ".[dev]"
python -m pytest tests/test_domain_router.py tests/test_clinical_trial_verifier.py tests/test_drug_target_verifier.py
```

运行一个 offline benchmark 示例：

```bash
python -m CAi.toolkit.agent_planner.biomedical_benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl \
  --output /tmp/fullcopilot_offline_summary.json
```

生成一个紧凑的 biomedical generalization 表：

```bash
python -m CAi.toolkit.agent_planner.biomedical_generalization_table \
  --benchmark CAi/toolkit/agent_planner/benchmarks/clinical_trial_outcome_prediction_v2_offline.jsonl \
  --benchmark CAi/toolkit/agent_planner/benchmarks/drug_target_evidence_v2_offline.jsonl \
  --output /tmp/biomedical_generalization_table.json
```

运行一个 mock benchmark 示例：

```bash
python -m CAi.toolkit.agent_planner.benchmark_runner \
  --benchmark CAi/toolkit/agent_planner/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --output /tmp/fullcopilot_mock_summary.json
```

## 发布边界

GitHub 首版只包含核心代码、轻量示例 benchmark、测试和文档；不包含 `.env`、API key、真实 logs、trace、workspace、模型权重、大型数据集或第三方工具源码下载物。

发布前请执行：

```bash
python scripts/audit_release_tree.py --root .
```
