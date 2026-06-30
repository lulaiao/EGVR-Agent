from __future__ import annotations

import pytest

from CAi.toolkit.agent_planner.task_schema import ToolMetadata
from CAi.toolkit.agent_planner.tool_registry import ChemistryToolRegistry, build_default_tool_registry


CORE_TOOLS = {
    "rxnflow",
    "reinvent4_denovo",
    "reinvent4_mol2mol",
    "reinvent4_libinvent",
    "scaffold",
    "libinvent",
    "vina",
    "scscore",
    "toxicity",
    "pmic",
    "sa_score",
    "posebusters",
    "rdkit_property_verifier",
}


def test_default_registry_contains_core_tools():
    registry = build_default_tool_registry()

    assert set(registry.names()) == CORE_TOOLS
    assert len(registry) == len(CORE_TOOLS)


def test_default_registry_keeps_backend_actions_structured():
    registry = build_default_tool_registry()

    assert registry.require("reinvent4_denovo").backend_tool_name == "reinvent4"
    assert registry.require("reinvent4_denovo").backend_action == "de_novo"
    assert registry.require("reinvent4_mol2mol").backend_action == "mol2mol"
    assert registry.require("rxnflow").wrapper_function.endswith("generate_molecules_for_pocket")


def test_registry_filters_by_task_type_and_tag():
    registry = build_default_tool_registry()

    docking_tools = {tool.tool_name for tool in registry.tools_for_task("docking_evaluation")}
    generation_tools = {tool.tool_name for tool in registry.tools_with_tag("generation")}

    property_tools = {tool.tool_name for tool in registry.tools_with_tag("properties")}

    assert docking_tools == {"vina", "posebusters"}
    assert {"rxnflow", "reinvent4_denovo", "scaffold"}.issubset(generation_tools)
    assert property_tools == {"rdkit_property_verifier"}


def test_registry_exposes_downstream_tools():
    registry = build_default_tool_registry()

    downstream = [tool.tool_name for tool in registry.downstream_of("rxnflow")]

    assert downstream == ["vina", "scscore", "toxicity"]


def test_registry_register_and_require_behavior():
    registry = ChemistryToolRegistry()
    tool = ToolMetadata(
        tool_name="demo",
        description="Demo",
        supported_task_types=["de_novo_generation"],
        required_inputs=[],
        optional_inputs=[],
        outputs=["generated_smiles"],
        typical_failures=[],
        estimated_cost="low",
        downstream_tools=[],
        chemistry_role="demo",
        backend_tool_name="demo",
    )

    registry.register(tool)

    assert "demo" in registry
    assert registry.get("demo") is tool
    with pytest.raises(KeyError, match="missing"):
        registry.require("missing")
