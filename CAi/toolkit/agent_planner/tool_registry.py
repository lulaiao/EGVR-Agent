"""Structured chemistry tool registry for planning and benchmarking."""

from __future__ import annotations

from collections.abc import Iterable

from .task_schema import ToolMetadata


class ChemistryToolRegistry:
    """Small in-memory registry for chemistry-aware planner metadata."""

    def __init__(self, tools: Iterable[ToolMetadata] = ()) -> None:
        self._tools: dict[str, ToolMetadata] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolMetadata) -> None:
        self._tools[tool.tool_name] = tool

    def get(self, tool_name: str) -> ToolMetadata | None:
        return self._tools.get(tool_name)

    def require(self, tool_name: str) -> ToolMetadata:
        tool = self.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown chemistry tool: {tool_name}")
        return tool

    def all(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def tools_for_task(self, task_type: str) -> list[ToolMetadata]:
        return [tool for tool in self._tools.values() if tool.supports_task_type(task_type)]

    def tools_with_tag(self, tag: str) -> list[ToolMetadata]:
        return [tool for tool in self._tools.values() if tag in tool.tags]

    def downstream_of(self, tool_name: str) -> list[ToolMetadata]:
        tool = self.require(tool_name)
        return [self.require(name) for name in tool.downstream_tools if name in self._tools]

    def to_dict(self) -> dict[str, dict]:
        return {name: tool.to_dict() for name, tool in self._tools.items()}

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


DEFAULT_TOOL_METADATA: tuple[ToolMetadata, ...] = (
    ToolMetadata(
        tool_name="rxnflow",
        description="Pocket-conditioned, synthesis-aware molecular generation from a protein pocket.",
        supported_task_types=[
            "pocket_conditioned_generation",
            "multi_objective_screening",
        ],
        required_inputs=["protein_path"],
        optional_inputs=["pocket_center", "ref_ligand_path", "num_samples"],
        outputs=["generated_smiles", "proxy_scores", "result_csv"],
        typical_failures=[
            "missing protein structure",
            "missing pocket center or reference ligand",
            "model environment unavailable",
            "GPU unavailable",
        ],
        estimated_cost="high",
        downstream_tools=["vina", "scscore", "toxicity"],
        chemistry_role="pocket_conditioned_generator",
        backend_tool_name="rxnflow",
        wrapper_function="CAi.toolkit.functions.generation.generate_molecules_for_pocket",
        tags=["generation", "pocket", "synthesis_aware", "gpu"],
    ),
    ToolMetadata(
        tool_name="reinvent4_denovo",
        description="REINVENT4 de novo generation for broad chemical-space exploration.",
        supported_task_types=[
            "de_novo_generation",
            "pocket_conditioned_generation",
            "multi_objective_screening",
            "failure_recovery",
        ],
        required_inputs=[],
        optional_inputs=["num_variants"],
        outputs=["generated_smiles"],
        typical_failures=["model environment unavailable", "GPU unavailable", "empty generation output"],
        estimated_cost="medium",
        downstream_tools=["vina", "scscore", "toxicity"],
        chemistry_role="fallback_denovo_generator",
        backend_tool_name="reinvent4",
        backend_action="de_novo",
        wrapper_function="CAi.toolkit.functions.generation.generate_molecules_reinvent4_denovo",
        tags=["generation", "denovo", "fallback", "gpu"],
    ),
    ToolMetadata(
        tool_name="reinvent4_mol2mol",
        description="REINVENT4 Mol2Mol analog generation from complete input molecules.",
        supported_task_types=[
            "hit_to_lead_optimization",
            "multi_objective_screening",
            "failure_recovery",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=["num_variants", "strategy", "temperature"],
        outputs=["generated_smiles"],
        typical_failures=["invalid input SMILES", "wildcard scaffold supplied", "model environment unavailable"],
        estimated_cost="medium",
        downstream_tools=["vina", "scscore", "toxicity", "pmic"],
        chemistry_role="analog_generator",
        backend_tool_name="reinvent4",
        backend_action="mol2mol",
        wrapper_function="CAi.toolkit.functions.generation.generate_molecules_reinvent4_mol2mol",
        tags=["generation", "optimization", "mol2mol", "gpu"],
    ),
    ToolMetadata(
        tool_name="reinvent4_libinvent",
        description="REINVENT4 LibInvent scaffold decoration at wildcard attachment points.",
        supported_task_types=[
            "scaffold_conditioned_generation",
            "multi_objective_screening",
            "failure_recovery",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=["num_variants"],
        outputs=["generated_smiles"],
        typical_failures=["missing attachment point", "unsupported stereochemistry", "model environment unavailable"],
        estimated_cost="medium",
        downstream_tools=["vina", "scscore", "toxicity", "pmic"],
        chemistry_role="scaffold_decorator",
        backend_tool_name="reinvent4",
        backend_action="libinvent",
        wrapper_function="CAi.toolkit.functions.generation.generate_molecules_reinvent4_libinvent",
        tags=["generation", "scaffold", "decoration", "gpu"],
    ),
    ToolMetadata(
        tool_name="scaffold",
        description="RNN-based scaffold analog generation from a scaffold SMILES with an attachment point.",
        supported_task_types=["scaffold_conditioned_generation", "multi_objective_screening"],
        required_inputs=["input_smiles"],
        optional_inputs=["num_analogs"],
        outputs=["generated_smiles"],
        typical_failures=["missing attachment point", "unsupported stereochemistry", "empty generation output"],
        estimated_cost="medium",
        downstream_tools=["scscore", "toxicity", "vina"],
        chemistry_role="scaffold_analog_generator",
        backend_tool_name="scaffold",
        wrapper_function="CAi.toolkit.functions.generation.generate_scaffold_analogs",
        tags=["generation", "scaffold", "gpu"],
    ),
    ToolMetadata(
        tool_name="libinvent",
        description="Lib-INVENT reaction-based decoration of scaffold attachment points.",
        supported_task_types=["scaffold_conditioned_generation", "multi_objective_screening"],
        required_inputs=["input_smiles"],
        optional_inputs=["num_decorations"],
        outputs=["generated_smiles", "decoration_rows"],
        typical_failures=["missing attachment point", "unsupported stereochemistry", "empty generation output"],
        estimated_cost="medium",
        downstream_tools=["scscore", "toxicity", "vina"],
        chemistry_role="reaction_based_scaffold_decorator",
        backend_tool_name="libinvent",
        wrapper_function="CAi.toolkit.functions.generation.generate_libinvent_decorations",
        tags=["generation", "scaffold", "decoration"],
    ),
    ToolMetadata(
        tool_name="vina",
        description="AutoDock Vina docking evaluation for receptor and ligand structures.",
        supported_task_types=[
            "docking_evaluation",
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
        ],
        required_inputs=["protein_path", "ligand_path", "pocket_center", "box_size"],
        optional_inputs=["exhaustiveness"],
        outputs=["docking_score", "docked_pose_path", "minimized_pose_path"],
        typical_failures=["missing receptor file", "missing ligand file", "invalid docking box", "conversion failure"],
        estimated_cost="high",
        downstream_tools=["scscore", "toxicity"],
        chemistry_role="binding_affinity_evaluator",
        backend_tool_name="vina",
        wrapper_function="CAi.toolkit.functions.evaluation.perform_molecular_docking_vina",
        tags=["evaluation", "docking", "structure_based"],
    ),
    ToolMetadata(
        tool_name="scscore",
        description="Synthetic complexity and synthesizability scoring for candidate SMILES.",
        supported_task_types=[
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
            "scaffold_conditioned_generation",
            "de_novo_generation",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=["model_type"],
        outputs=["scscore", "canonical_smiles"],
        typical_failures=["invalid SMILES", "model files unavailable"],
        estimated_cost="low",
        downstream_tools=["toxicity", "pmic"],
        chemistry_role="synthesizability_evaluator",
        backend_tool_name="scscore",
        wrapper_function="CAi.toolkit.functions.evaluation.calculate_scscore",
        tags=["evaluation", "synthesizability", "batchable"],
    ),
    ToolMetadata(
        tool_name="toxicity",
        description="Hepatotoxicity prediction with optional structural explanation artifacts.",
        supported_task_types=[
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
            "scaffold_conditioned_generation",
            "de_novo_generation",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=[],
        outputs=["toxicity_score", "toxicity_verdict", "explanation_artifact"],
        typical_failures=["invalid SMILES", "model environment unavailable", "image artifact write failure"],
        estimated_cost="low",
        downstream_tools=[],
        chemistry_role="toxicity_evaluator",
        backend_tool_name="toxicity",
        wrapper_function="CAi.toolkit.functions.evaluation.predict_molecule_toxicity",
        tags=["evaluation", "toxicity"],
    ),
    ToolMetadata(
        tool_name="pmic",
        description="Antibacterial potency prediction from complete molecule SMILES.",
        supported_task_types=["hit_to_lead_optimization", "multi_objective_screening"],
        required_inputs=["input_smiles"],
        optional_inputs=[],
        outputs=["pmic_score", "estimated_mic_um"],
        typical_failures=["invalid SMILES", "model environment unavailable"],
        estimated_cost="low",
        downstream_tools=[],
        chemistry_role="bioactivity_evaluator",
        backend_tool_name="pmic",
        wrapper_function="CAi.toolkit.functions.evaluation.predict_antibacterial_pmic",
        tags=["evaluation", "bioactivity", "antibacterial"],
    ),
    ToolMetadata(
        tool_name="sa_score",
        description="RDKit synthetic accessibility scoring used as a lightweight synthesizability verifier.",
        supported_task_types=[
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
            "scaffold_conditioned_generation",
            "de_novo_generation",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=[],
        outputs=["sa_score", "canonical_smiles"],
        typical_failures=["RDKit unavailable", "invalid SMILES", "SA_Score contrib files unavailable"],
        estimated_cost="low",
        downstream_tools=[],
        chemistry_role="synthesizability_verifier",
        backend_tool_name="sa_score",
        wrapper_function="CAi.toolkit.agent_planner.verifier_evidence_runner.calculate_sa_scores",
        tags=["evaluation", "synthesizability", "verifier", "batchable"],
    ),
    ToolMetadata(
        tool_name="posebusters",
        description="Pose-level sanity checking for docking outputs when PoseBusters and compatible pose files are available.",
        supported_task_types=[
            "docking_evaluation",
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
        ],
        required_inputs=["protein_path", "docked_pose_path"],
        optional_inputs=["ligand_path", "reference_ligand_path"],
        outputs=["posebusters_pass", "posebusters_checks", "posebusters_status"],
        typical_failures=["PoseBusters unavailable", "pose format unsupported", "missing receptor or ligand pose"],
        estimated_cost="medium",
        downstream_tools=[],
        chemistry_role="pose_sanity_verifier",
        backend_tool_name="posebusters",
        wrapper_function="CAi.toolkit.agent_planner.verifier_evidence_runner.collect_posebusters_evidence",
        tags=["evaluation", "docking", "pose_sanity", "verifier"],
    ),
    ToolMetadata(
        tool_name="rdkit_property_verifier",
        description="Lightweight RDKit property checks for QED, LogP, molecular weight, Lipinski, PAINS, and Brenk alerts.",
        supported_task_types=[
            "pocket_conditioned_generation",
            "hit_to_lead_optimization",
            "multi_objective_screening",
            "scaffold_conditioned_generation",
            "de_novo_generation",
        ],
        required_inputs=["input_smiles"],
        optional_inputs=[],
        outputs=[
            "qed",
            "molwt",
            "logp",
            "tpsa",
            "lipinski_violations",
            "pains_flags",
            "brenk_flags",
        ],
        typical_failures=["RDKit unavailable", "invalid SMILES", "filter catalog unavailable"],
        estimated_cost="low",
        downstream_tools=[],
        chemistry_role="property_verifier",
        backend_tool_name="rdkit_property_verifier",
        wrapper_function="CAi.toolkit.agent_planner.rdkit_property_runner.calculate_rdkit_properties",
        tags=["evaluation", "properties", "verifier", "batchable"],
    ),
)


def build_default_tool_registry() -> ChemistryToolRegistry:
    """Return a fresh registry populated with the first-stage core tools."""

    return ChemistryToolRegistry(DEFAULT_TOOL_METADATA)
