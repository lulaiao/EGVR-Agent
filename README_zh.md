# EGVR-Agent

EGVR-Agent 是论文 **Evidence Before Success: Execution-Grounded
Verification and Targeted Repair for Biomedical Tool-Using Agents** 的代码
artifact。

项目实现的是外部工具之上的可信执行层：只有真实执行输出满足任务对应的
verifier checks，系统才声明任务成功。证据缺失或格式错误时，verifier 可以
授权有预算限制的 targeted retry 或已声明 fallback；不可恢复的情况会明确
保留为 incomplete/failed。

## 核心模块

- 结构化 `ParsedTask`、`PlannedWorkflow`、`ToolCallRecord`、
  `CandidateRecord`、`EvidenceRecord` 和 `VerifierResult`。
- 基于 typed tool registry 的 task-conditioned planning。
- 按声明参数执行工具，不依赖 LLM 临场编写执行代码。
- 将异构 backend 输出归一化为稳定证据对象。
- execution-grounded verification 与 conservative success gate。
- verifier-guided retry/fallback 和显式 repair budget。
- JSONL trace、重复记录检测和一致性审计。
- controlled reliability、tool-menu、LLM-router 和 biomedical evidence
  benchmark runners。

本仓库定位为论文研究代码，而不是通用聊天产品，因此不包含 Web UI、对话
管理器、REPL 外壳、模型权重、私有 traces 或第三方科学工具源码。

## 目录

```text
egvr/                  # 规划、执行、验证、修复和评测核心
egvr/adapters/         # 可选外部 backend HTTP contract
egvr/benchmarks/       # 轻量公开 benchmark
tests/                 # 无网络回归测试
scripts/               # 离线 artifact 与发布审计
docs/                  # 架构、复现和 claim 映射
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr_artifact
```

运行 mock benchmark：

```bash
python -m egvr.benchmark_runner \
  --benchmark egvr/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --planner-baseline egvr_agent \
  --output /tmp/egvr_mock_summary.json
```

## 外部工具接口

真实工具通过 `egvr.adapters.tool_server` 接入：

```bash
export EGVR_TOOL_SERVER_URL=http://127.0.0.1:8001
```

外部服务只需实现 `/run/{tool}/{action}`、`/job/{job_id}` 和 `/health`
接口。科学工具、模型、环境和数据集由使用者独立安装，不与可信执行框架绑定。

## Claim 边界

公开 offline tasks 验证机制行为和 evidence-interface transfer，不声称分子
生成、临床预测、DTI、ADMET 或药物发现 SOTA。论文真实工具结果、模型供应商
响应、私有 backend I/O 和受许可证约束的数据均不进入公开仓。

旧结果中的 `full_copilot` 标识仍可读取；新实验统一使用 `egvr_agent`。

发布前运行：

```bash
python scripts/audit_release_tree.py --root .
```

项目采用 Apache-2.0，历史兼容边界和来源说明见 [NOTICE](NOTICE)。
