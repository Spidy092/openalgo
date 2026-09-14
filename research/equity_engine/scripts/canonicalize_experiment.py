from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "research/equity_engine/src/equity_engine"
CORE_PATH = PKG / "_experiment_v3_core.py"
WRAPPER_PATH = PKG / "experiment.py"


def decorated_start(node: ast.ClassDef) -> int:
    starts = [node.lineno]
    starts.extend(item.lineno for item in node.decorator_list)
    return min(starts)


def source_for(text: str, node: ast.AST) -> str:
    lines = text.splitlines(keepends=True)
    assert hasattr(node, "lineno") and hasattr(node, "end_lineno")
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"class {name} not found")


def find_method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise RuntimeError(f"method {node.name}.{name} not found")


def main() -> None:
    core = CORE_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
    core_tree = ast.parse(core)
    wrapper_tree = ast.parse(wrapper)

    # Take the hardened public corporate-action evidence boundary from the compatibility
    # layer and place it directly into the V3 implementation.
    wrapper_mismatch = find_class(wrapper_tree, "CorporateActionMismatchError")
    wrapper_ca = find_class(wrapper_tree, "CorporateActionEvidenceIdentity")
    hardened_ca = (
        source_for(wrapper, wrapper_mismatch)
        + "\n\n_UNSET_COVERED_INSTRUMENTS = object()\n\n\n"
        + source_for(wrapper, wrapper_ca)
        + "\n\n"
    )

    core_ca = find_class(core_tree, "CorporateActionEvidenceIdentity")
    core_cost = find_class(core_tree, "CostModelIdentity")
    lines = core.splitlines(keepends=True)
    start = decorated_start(core_ca) - 1
    end = decorated_start(core_cost) - 1
    core = "".join(lines[:start]) + hardened_ca + "".join(lines[end:])

    core = core.replace(
        "from collections.abc import Mapping\n",
        "from collections.abc import Iterable, Mapping\n",
        1,
    )
    corporate_import = "from .corporate_actions import CorporateActionEvaluationMode\n"
    if corporate_import not in core:
        marker = "from .cost_ledger import (\n"
        if marker not in core:
            raise RuntimeError("cost_ledger import marker not found")
        core = core.replace(marker, corporate_import + marker, 1)

    # Preserve the mature V3 algorithms as private implementation helpers, then make the
    # hardened public methods the only evaluate/validate entry points on the single class.
    core = core.replace(
        "    def evaluate_promotion_gate(\n",
        "    def _evaluate_v3_promotion_gate(\n",
        1,
    )
    core = core.replace(
        "    def validate_integrity(\n",
        "    def _validate_v3_integrity(\n",
        1,
    )

    wrapper_exp = find_class(wrapper_tree, "ExperimentArtifact")
    method_names = (
        "_validate_corporate_action_structure",
        "validate_structure",
        "evaluate_promotion_gate",
        "validate_integrity",
    )
    methods: list[str] = []
    for name in method_names:
        method = find_method(wrapper_exp, name)
        method_source = source_for(wrapper, method)
        method_source = method_source.replace(
            "super().evaluate_promotion_gate(",
            "self._evaluate_v3_promotion_gate(",
        ).replace(
            "super().validate_integrity(",
            "self._validate_v3_integrity(",
        )
        methods.append(method_source.rstrip() + "\n")

    # Reparse after the replacements so insertion occurs at the true end of the V3 class.
    modified_tree = ast.parse(core)
    exp = find_class(modified_tree, "ExperimentArtifact")
    core_lines = core.splitlines(keepends=True)
    insertion = "\n" + "\n".join(methods) + "\n"
    core = "".join(core_lines[: exp.end_lineno]) + insertion + "".join(core_lines[exp.end_lineno :])

    # Hard invariants for the canonical result.
    final_tree = ast.parse(core)
    classes = [n for n in final_tree.body if isinstance(n, ast.ClassDef)]
    if sum(n.name == "ExperimentArtifact" for n in classes) != 1:
        raise RuntimeError("canonical experiment.py must contain exactly one ExperimentArtifact")
    if sum(n.name == "CorporateActionEvidenceIdentity" for n in classes) != 1:
        raise RuntimeError("canonical experiment.py must contain exactly one CorporateActionEvidenceIdentity")
    exp = find_class(final_tree, "ExperimentArtifact")
    public_methods = [n.name for n in exp.body if isinstance(n, ast.FunctionDef)]
    if public_methods.count("evaluate_promotion_gate") != 1:
        raise RuntimeError("exactly one public evaluate_promotion_gate is required")
    if public_methods.count("validate_integrity") != 1:
        raise RuntimeError("exactly one public validate_integrity is required")
    if "_evaluate_v3_promotion_gate" not in public_methods or "_validate_v3_integrity" not in public_methods:
        raise RuntimeError("V3 implementation helpers were not preserved")

    WRAPPER_PATH.write_text(core, encoding="utf-8")
    CORE_PATH.unlink()

    references: list[str] = []
    for path in PKG.rglob("*.py"):
        if "_experiment_v3_core" in path.read_text(encoding="utf-8"):
            references.append(str(path.relative_to(ROOT)))
    if references:
        raise RuntimeError(f"stale _experiment_v3_core references remain: {references}")


if __name__ == "__main__":
    main()
