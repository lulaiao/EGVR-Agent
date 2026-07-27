from __future__ import annotations

from egvr.executor import WorkflowExecutor
from egvr.tool_registry import build_default_tool_registry


def test_registry_wrappers_resolve_inside_standalone_package():
    registry = build_default_tool_registry()
    executor = WorkflowExecutor(registry=registry)

    for tool_name in (
        "rxnflow",
        "reinvent4_denovo",
        "reinvent4_mol2mol",
        "vina",
        "scscore",
        "toxicity",
    ):
        metadata = registry.require(tool_name)
        assert metadata.wrapper_function.startswith("egvr.")
        assert callable(executor._resolve_function(tool_name))
