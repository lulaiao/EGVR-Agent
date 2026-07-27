# EGVR-Agent

[![CI](https://github.com/lulaiao/EGVR-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lulaiao/EGVR-Agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

EGVR-Agent 是论文 **Evidence Before Success: Execution-Grounded
Verification and Targeted Repair for Biomedical Tool-Using Agents** 的研究
代码。

它在外部工具之上提供 evidence-first 可信执行层：只有真实执行输出满足任务
对应的 verifier checks，系统才声明任务成功。证据缺失或格式错误时，verifier
可以授权有预算限制的 targeted retry 或已声明 fallback；不可恢复的情况会
明确保留为 incomplete/failed。

## 为什么使用 EGVR-Agent？

工具运行完成不等于任务完成。EGVR-Agent 将流程拆分为：

```text
任务 -> 规划 -> 执行 -> 归一化 -> 验证 -> 修复或不完整 -> Trace
```

- **Task-conditioned planning**：只向任务暴露相关工具。
- **Deterministic execution**：执行声明好的调用，不依赖 LLM 临场写代码。
- **Execution-grounded verification**：检查真实存在的输出、分数和 artifacts。
- **Targeted repair**：由 verifier failure reason 和 repair budget 约束。
- **Traceable decisions**：将执行与判断写成结构化 JSONL trace。

本仓库定位为研究 artifact，而不是聊天产品，因此不包含 Web UI、对话管理器、
模型权重、私有 traces、受许可证约束的数据集或第三方科学工具源码。

## 环境要求

- Python 3.11 或更高版本
- 公开离线流程推荐 Linux 或 macOS
- 30 秒 demo 不需要 API key、GPU、模型权重或工具服务
- 可选真实执行需要用户独立部署 backend

## 安装

克隆仓库，并使用以方法名命名的 Conda 环境：

```bash
git clone https://github.com/lulaiao/EGVR-Agent.git
cd EGVR-Agent

conda create -n egvr-agent python=3.11 -y
conda activate egvr-agent
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

不使用 Conda 时，创建包含方法名的 venv：

```bash
python3.11 -m venv .egvr-agent-venv
source .egvr-agent-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 30 秒 Demo

以下命令在无网络条件下跑通 parser、planner、executor、verifier 和 trace：

```bash
python -m examples.minimal_mock \
  --output-dir /tmp/egvr-agent-demo
```

预期摘要：

```json
{
  "candidate_count": 2,
  "failure_reason": null,
  "selected_tools": [
    "reinvent4_denovo",
    "scscore",
    "toxicity"
  ],
  "task_id": "public_minimal_demo",
  "task_success": true,
  "tool_call_count": 4,
  "trace_path": "/tmp/egvr-agent-demo/YYYYMMDD_traces.jsonl"
}
```

该示例证明系统会先归一化并验证工具输出，再记录任务成功。

## 接入自己的 Python 工具

运行自定义工具示例：

```bash
python -m examples.custom_tool_adapter
```

完整代码位于
[`examples/custom_tool_adapter.py`](examples/custom_tool_adapter.py)。核心
调用方式如下：

```python
from egvr import WorkflowExecutor, parse_task, plan_workflow, verify_workflow

task = parse_task("Generate de novo molecules and evaluate synthesizability.")
workflow = plan_workflow(task)
executor = WorkflowExecutor(
    tool_functions={
        "reinvent4_denovo": my_generator,
        "scscore": my_synthesis_evaluator,
    }
)
calls, candidates = executor.execute(task, workflow)
result = verify_workflow(task, workflow, calls, candidates)
print(result.success, result.failure_reason)
```

## 命令行

安装后可以直接运行公开 mock benchmark：

```bash
egvr-benchmark \
  --benchmark egvr/benchmarks/molecular_agent_tasks.example.jsonl \
  --execution-mode mock \
  --planner-baseline egvr_agent \
  --output /tmp/egvr-mock-summary.json
```

运行完整的无网络 paper artifact：

```bash
python scripts/run_offline_artifact.py \
  --output-dir /tmp/egvr-artifact
```

## 外部工具 HTTP Contract

配置服务地址：

```bash
export EGVR_TOOL_SERVER_URL=http://127.0.0.1:8001
```

服务需要实现：

- `GET /health`
- `POST /run/{tool}/{action}`，返回 `{"job_id": "job-123"}`
- `GET /job/{job_id}`，返回 `running`、`finished` 或 `failed`

完成状态示例：

```json
{
  "status": "finished",
  "data": {
    "success": true,
    "summary": {},
    "results": {}
  }
}
```

详细 client 实现见
[`egvr/adapters/tool_server.py`](egvr/adapters/tool_server.py)。缺失、超时、
格式错误或失败输出不会被转换成科学任务成功。

## 目录结构

```text
egvr/                  # 规划、执行、验证、修复和评测
egvr/adapters/         # 外部 backend contract
egvr/benchmarks/       # 轻量公开 benchmark
examples/              # 无网络使用示例
tests/                 # 无网络回归测试
scripts/               # 离线 artifact 与发布审计
docs/                  # 架构和复现文档
```

## 复现与测试

```bash
python -m pytest
python scripts/audit_release_tree.py --root .
```

- [Artifact 说明](ARTIFACT.md)
- [系统架构](docs/architecture.md)
- [复现指南](docs/reproducibility.md)
- [论文 claim 与代码映射](docs/paper_artifact_mapping.md)
- [发布检查清单](docs/release_checklist.md)

## Claim 边界

公开 offline tasks 验证机制行为和 evidence-interface transfer，不复现私有真实
工具实验，也不声称分子生成、临床预测、DTI、ADMET 或药物发现 SOTA。模型
响应、私有 backend I/O、权重和受许可证约束的数据均不进入公开仓。

## 常见问题

- **找不到 `egvr`**：激活 `egvr-agent`，在仓库根目录重新执行
  `python -m pip install -e ".[dev]"`。
- **Python 版本错误**：确认 `python --version` 不低于 3.11。
- **工具服务不可达**：使用 `--execution-mode mock`，或检查
  `EGVR_TOOL_SERVER_URL` 和 `/health`。
- **工具返回成功但 verifier 失败**：检查 JSONL trace 中的
  `failure_reason`、required checks、输出和 artifact 引用；这是保守门控的
  预期行为。
- **读取旧实验文件**：参考
  [`docs/naming_and_migration.md`](docs/naming_and_migration.md)。

## 引用

软件引用信息见 [`CITATION.cff`](CITATION.cff)。论文正式公开后会补充最终
BibTeX。

## 许可证与来源

EGVR-Agent 使用 Apache-2.0。历史兼容边界和第三方来源见 [`NOTICE`](NOTICE)。
