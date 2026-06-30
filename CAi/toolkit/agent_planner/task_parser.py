"""Rule-based task parser for chemistry-aware planning."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from .task_schema import ParsedTask, TaskConstraints


SUPPORTED_TASK_TYPES = {
    "pocket_conditioned_generation",
    "hit_to_lead_optimization",
    "docking_evaluation",
    "multi_objective_screening",
    "scaffold_conditioned_generation",
    "de_novo_generation",
    "failure_recovery",
    "unknown",
}

_PATH_RE = re.compile(
    r"(?<![\w/.-])(?:~|\.{1,2}|/|[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.~/-]+"
    r"\.(?:pdbqt|pdb|sdf|mol2|mol|smi|csv|json|txt)\b",
    re.IGNORECASE,
)
_FIELD_VALUE_TEMPLATE = r"\b(?:{fields})\b\s*[:=]\s*(?P<value>\[[^\]]+\]|\([^)]+\)|[^\s,;]+)"
_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

_OBJECTIVE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("binding", ("binding", "affinity", "dock", "docking", "vina")),
    ("synthesizability", ("synthesizability", "synthesizable", "synthetic", "synthesis", "scscore")),
    ("toxicity", ("toxicity", "toxic", "hepatotoxicity", "safe", "safety")),
    ("bioactivity", ("pmic", "mic", "antibacterial", "activity", "potency")),
    ("diversity", ("diverse", "diversity")),
    ("novelty", ("novel", "novelty", "de novo", "denovo")),
)

_FAILURE_KEYWORDS = ("fail", "failed", "failure", "fallback", "recover", "repair", "retry")
_GENERATION_KEYWORDS = ("generate", "generation", "design", "create", "sample", "propose")
_OPTIMIZATION_KEYWORDS = ("optimize", "optimise", "hit-to-lead", "hit to lead", "lead optimization", "analog")
_SCAFFOLD_KEYWORDS = ("scaffold", "decorate", "decoration", "r-group", "r group", "libinvent")
_MULTI_OBJECTIVE_KEYWORDS = ("multi-objective", "multiobjective", "pareto", "rank", "ranking", "screen", "screening")
_DENOVO_KEYWORDS = ("de novo", "denovo", "from scratch", "novel molecules")
_DOCKING_KEYWORDS = ("dock", "docking", "vina")


def parse_task(
    raw_user_query: str,
    *,
    task_id: str | None = None,
    context_files: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedTask:
    """Parse a chemistry task with conservative deterministic rules."""

    return RuleBasedTaskParser().parse(
        raw_user_query,
        task_id=task_id,
        context_files=context_files,
        metadata=metadata,
    )


class RuleBasedTaskParser:
    """Keyword and explicit-field parser for first-stage benchmarking."""

    parser_type = "rule_based"

    def parse(
        self,
        raw_user_query: str,
        *,
        task_id: str | None = None,
        context_files: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedTask:
        query = raw_user_query or ""
        parser_metadata = dict(metadata or {})
        parser_metadata.setdefault("parser_type", self.parser_type)

        protein_path = _extract_field_value(
            query,
            ("protein_path", "protein", "receptor_path", "receptor", "receptor_pdbqt_path"),
        )
        ligand_path = _extract_field_value(query, ("ligand_path", "ligand", "ligand_pdbqt_path"))
        ref_ligand_path = _extract_field_value(query, ("ref_ligand_path", "reference_ligand", "ref_ligand"))
        pocket_center = _extract_xyz(query, ("pocket_center", "center_xyz", "center"))
        box_size = _extract_xyz(query, ("box_size", "box"))
        run_seed = _extract_run_seed(query, parser_metadata)
        input_smiles = _extract_smiles(query)
        extracted_paths = _unique(_extract_paths(query))
        keyword_lower = _remove_paths_for_keyword_matching(query).lower()
        objectives = _infer_objectives(keyword_lower)
        constraints = _infer_constraints(query, keyword_lower, objectives)

        combined_context_files = _unique([*(context_files or ()), *extracted_paths])
        if protein_path and protein_path not in combined_context_files:
            combined_context_files.append(protein_path)
        if ligand_path and ligand_path not in combined_context_files:
            combined_context_files.append(ligand_path)
        if ref_ligand_path and ref_ligand_path not in combined_context_files:
            combined_context_files.append(ref_ligand_path)

        task_type = _infer_task_type(
            lower=keyword_lower,
            objectives=objectives,
            protein_path=protein_path,
            pocket_center=pocket_center,
            ligand_path=ligand_path,
            input_smiles=input_smiles,
        )

        parser_metadata["extracted_paths"] = extracted_paths
        parser_metadata["explicit_fields"] = sorted(
            field
            for field, value in {
                "protein_path": protein_path,
                "ligand_path": ligand_path,
                "ref_ligand_path": ref_ligand_path,
                "pocket_center": pocket_center,
                "box_size": box_size,
                "run_seed": run_seed,
                "input_smiles": input_smiles,
            }.items()
            if value is not None and value != [] and value != ""
        )
        if run_seed is not None:
            parser_metadata["run_seed"] = run_seed

        return ParsedTask(
            task_id=task_id or _stable_task_id(query),
            raw_user_query=query,
            task_type=task_type,
            objectives=objectives,
            constraints=constraints,
            protein_path=protein_path,
            pocket_center=pocket_center,
            ligand_path=ligand_path,
            ref_ligand_path=ref_ligand_path,
            box_size=box_size,
            input_smiles=input_smiles,
            context_files=combined_context_files,
            metadata=parser_metadata,
        )


def _stable_task_id(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    return f"task_{digest}"


def _extract_field_value(query: str, fields: tuple[str, ...]) -> str | None:
    field_pattern = "|".join(re.escape(field) for field in fields)
    match = re.search(_FIELD_VALUE_TEMPLATE.format(fields=field_pattern), query, flags=re.IGNORECASE)
    if not match:
        return None
    return _strip_value(match.group("value"))


def _extract_xyz(query: str, fields: tuple[str, ...]) -> list[float] | None:
    field_pattern = "|".join(re.escape(field) for field in fields)
    pattern = rf"\b(?:{field_pattern})\b\s*[:=]?\s*[\[(]?\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*,\s*({_FLOAT_RE})\s*[\])]?"
    match = re.search(pattern, query, flags=re.IGNORECASE)
    if not match:
        return None
    return [float(match.group(idx)) for idx in range(1, 4)]


def _extract_paths(query: str) -> list[str]:
    return [_strip_value(match.group(0)) for match in _PATH_RE.finditer(query)]


def _remove_paths_for_keyword_matching(query: str) -> str:
    return _PATH_RE.sub(" ", query)


def _extract_smiles(query: str) -> list[str]:
    smiles_values: list[str] = []
    list_pattern = re.compile(
        r"\b(?:smiles_list|input_smiles|smiles|hit_smiles|scaffold_smiles)\b\s*[:=]\s*\[(?P<value>[^\]]+)\]",
        re.IGNORECASE,
    )
    for match in list_pattern.finditer(query):
        smiles_values.extend(_split_smiles_list(match.group("value")))

    single_pattern = re.compile(
        r"\b(?:smiles|input_smiles|hit_smiles|scaffold_smiles|molecule_smiles)\b\s*[:=]\s*(?P<value>[^\s,;]+)",
        re.IGNORECASE,
    )
    for match in single_pattern.finditer(query):
        value = _strip_value(match.group("value"))
        if not value.startswith("["):
            smiles_values.append(value)

    return _unique(value for value in smiles_values if value)


def _split_smiles_list(value: str) -> list[str]:
    return [_strip_value(item) for item in re.split(r"[,;\s]+", value) if _strip_value(item)]


def _infer_objectives(lower: str) -> list[str]:
    objectives: list[str] = []
    for objective, keywords in _OBJECTIVE_KEYWORDS:
        if any(keyword in lower for keyword in keywords):
            objectives.append(objective)
    return objectives


def _infer_constraints(query: str, lower: str, objectives: list[str]) -> TaskConstraints:
    constraints = TaskConstraints()
    constraints.require_docking = "binding" in objectives
    constraints.require_synthesizability = "synthesizability" in objectives
    constraints.require_toxicity = "toxicity" in objectives
    constraints.require_ranking = any(keyword in lower for keyword in ("rank", "ranking", "pareto", "top"))

    max_candidates = _extract_int_constraint(query, ("num_samples", "num_variants", "num_analogs", "num_candidates"))
    if max_candidates is None:
        max_candidates = _extract_generate_count(query)
    constraints.max_candidates = max_candidates
    constraints.min_candidates = _extract_min_candidates(query)
    constraints.max_scscore = _extract_float_threshold(query, ("scscore",), ("<=", "<", "below", "under", "max"))
    constraints.max_toxicity_score = _extract_float_threshold(
        query,
        ("toxicity", "toxicity_score", "toxicity_probability"),
        ("<=", "<", "below", "under", "max"),
    )
    constraints.min_pmic = _extract_float_threshold(query, ("pmic", "pMIC"), (">=", ">", "above", "min"))
    return constraints


def _extract_int_constraint(query: str, fields: tuple[str, ...]) -> int | None:
    field_pattern = "|".join(re.escape(field) for field in fields)
    match = re.search(rf"\b(?:{field_pattern})\b\s*[:=]\s*(\d+)", query, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_generate_count(query: str) -> int | None:
    match = re.search(r"\b(?:generate|sample|propose|top)\s+(\d+)\b", query, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_min_candidates(query: str) -> int | None:
    match = re.search(r"\b(?:at least|min(?:imum)?)\s+(\d+)\b", query, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_run_seed(query: str, metadata: dict[str, Any]) -> int | None:
    for key in ("rxnflow_seed", "random_seed", "run_seed", "seed"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return _extract_int_constraint(query, ("rxnflow_seed", "random_seed", "run_seed", "seed"))


def _extract_float_threshold(query: str, fields: tuple[str, ...], operators: tuple[str, ...]) -> float | None:
    field_pattern = "|".join(re.escape(field) for field in fields)
    op_pattern = "|".join(re.escape(op) for op in operators)
    match = re.search(rf"\b(?:{field_pattern})\b\s*(?:{op_pattern})\s*({_FLOAT_RE})", query, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _infer_task_type(
    *,
    lower: str,
    objectives: list[str],
    protein_path: str | None,
    pocket_center: list[float] | None,
    ligand_path: str | None,
    input_smiles: list[str],
) -> str:
    has_generation = any(keyword in lower for keyword in _GENERATION_KEYWORDS)
    has_optimization = any(keyword in lower for keyword in _OPTIMIZATION_KEYWORDS)
    has_multi = any(keyword in lower for keyword in _MULTI_OBJECTIVE_KEYWORDS)
    has_scaffold = any(keyword in lower for keyword in _SCAFFOLD_KEYWORDS) or any("*" in smiles for smiles in input_smiles)
    has_denovo = any(keyword in lower for keyword in _DENOVO_KEYWORDS)
    has_docking = any(keyword in lower for keyword in _DOCKING_KEYWORDS)
    has_failure = any(keyword in lower for keyword in _FAILURE_KEYWORDS)

    if has_failure:
        return "failure_recovery"
    if has_multi or (len(objectives) >= 2 and ("screen" in lower or "rank" in lower)):
        return "multi_objective_screening"
    if has_docking and ligand_path and not has_optimization:
        return "docking_evaluation"
    if has_scaffold:
        return "scaffold_conditioned_generation"
    if has_optimization and input_smiles:
        return "hit_to_lead_optimization"
    if protein_path and pocket_center and not input_smiles:
        return "pocket_conditioned_generation"
    if has_denovo or (has_generation and not input_smiles and not protein_path):
        return "de_novo_generation"
    if has_docking and not has_generation and not has_optimization:
        return "docking_evaluation"
    return "unknown"


def _strip_value(value: str) -> str:
    return value.strip().strip("\"'`[](){}<>.,;")


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values
