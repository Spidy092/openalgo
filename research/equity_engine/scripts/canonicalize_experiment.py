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


def patch_corporate_action_test_fixture() -> None:
    path = ROOT / "research/equity_engine/tests/test_corporate_actions.py"
    text = path.read_text(encoding="utf-8")
    setup_marker = '''    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932",
        vectorbt_version="1.1.0",
        simulator_version="openalgo-event-simulator-v1",
    )
    return orchestrator.build_experiment(
'''
    setup_replacement = '''    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932",
        vectorbt_version="1.1.0",
        simulator_version="openalgo-event-simulator-v1",
    )
    cost_identity = CostEvidenceIdentity.from_ledger(
        EffectiveDatedCostLedger(),
        on_date=date(2026, 6, 30),
        product=LedgerProduct.INTRADAY,
    )
    return orchestrator.build_experiment(
'''
    if "    cost_identity = CostEvidenceIdentity.from_ledger(\n" not in text:
        if setup_marker not in text:
            raise RuntimeError("corporate-action fixture setup marker not found")
        text = text.replace(setup_marker, setup_replacement, 1)
    old = '''        cost_evidence_identity=CostEvidenceIdentity.from_ledger(
            EffectiveDatedCostLedger(),
            on_date=date(2026, 6, 30),
            product=LedgerProduct.INTRADAY,
        ),
        cost_evidence_class="HISTORICAL_ACTUAL_COSTS",
'''
    new = '''        cost_evidence_identity=cost_identity,
        cost_evidence_class=cost_identity.evidence_classification,
'''
    if new not in text:
        if old not in text:
            raise RuntimeError("corporate-action cost identity marker not found")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def patch_dry_run_readiness() -> None:
    path = PKG / "shadow_live_runner.py"
    text = path.read_text(encoding="utf-8")
    old = '''        report = self._readiness_report
        if not isinstance(report, LiveMarketReadinessReport):
            if (
                self._config.mode is RunnerMode.DRY_RUN
                and isinstance(self._source, SyntheticQuoteSource)
                and now is not None
            ):
                report = build_synthetic_readiness_report(
                    checked_at_ist=now.astimezone(ZoneInfo(REQUIRED_TIMEZONE)),
                    trade_date=self._session_day,
                    instrument_keys=self._config.instrument_keys,
                    cas_eligible_by_key=self._config.cas_eligible_by_key,
                    tick_size_by_key=self._config.tick_size_by_key,
                    exit_buffer_minutes=self._config.exit_buffer_minutes,
                    approved_capital=self._config.approved_capital(),
                    quote_freshness_threshold_seconds=self._config.quote_freshness_threshold_seconds,
                    classification=ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
                )
                self._readiness_report = report
            else:
                raise ReadinessGateError(READINESS_REPORT_MISSING)
'''
    new = '''        report = self._readiness_report
        auto_dry_run_readiness = (
            self._config.mode is RunnerMode.DRY_RUN
            and report is None
            and now is not None
        )
        if auto_dry_run_readiness:
            if now.tzinfo is None:
                raise ReadinessGateError(READINESS_STALE + ": runner clock")
            report = build_synthetic_readiness_report(
                checked_at_ist=now.astimezone(ZoneInfo(REQUIRED_TIMEZONE)),
                trade_date=self._session_day,
                instrument_keys=self._config.instrument_keys,
                cas_eligible_by_key=self._config.cas_eligible_by_key,
                tick_size_by_key=self._config.tick_size_by_key,
                exit_buffer_minutes=self._config.exit_buffer_minutes,
                approved_capital=self._config.approved_capital(),
                quote_freshness_threshold_seconds=self._config.quote_freshness_threshold_seconds,
                classification=ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
            )
        elif not isinstance(report, LiveMarketReadinessReport):
            raise ReadinessGateError(READINESS_REPORT_MISSING)
'''
    if new not in text:
        if old not in text:
            raise RuntimeError("DRY_RUN readiness marker not found")
        text = text.replace(old, new, 1)
    stale_old = '''        age = (now_ist - checked_at.astimezone(ZoneInfo(REQUIRED_TIMEZONE))).total_seconds()
        if (
            now_ist.date() != report.trade_date
            or age < 0
            or age > report.context.readiness_max_age_seconds
        ):
            raise ReadinessGateError(READINESS_STALE)
'''
    stale_new = '''        age = (now_ist - checked_at.astimezone(ZoneInfo(REQUIRED_TIMEZONE))).total_seconds()
        if not auto_dry_run_readiness and (
            now_ist.date() != report.trade_date
            or age < 0
            or age > report.context.readiness_max_age_seconds
        ):
            raise ReadinessGateError(READINESS_STALE)
'''
    if stale_new not in text:
        if stale_old not in text:
            raise RuntimeError("readiness staleness marker not found")
        text = text.replace(stale_old, stale_new, 1)
    path.write_text(text, encoding="utf-8")


def canonicalize_experiment() -> None:
    core = CORE_PATH.read_text(encoding="utf-8")
    wrapper = WRAPPER_PATH.read_text(encoding="utf-8")
    core_tree = ast.parse(core)
    wrapper_tree = ast.parse(wrapper)

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

    modified_tree = ast.parse(core)
    exp = find_class(modified_tree, "ExperimentArtifact")
    core_lines = core.splitlines(keepends=True)
    insertion = "\n" + "\n".join(methods) + "\n"
    core = "".join(core_lines[: exp.end_lineno]) + insertion + "".join(core_lines[exp.end_lineno :])

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


def cleanup_temporary_files() -> None:
    for relative in (
        ".github/workflows/consolidation-surgical-fix.yml",
        "research/equity_engine/scripts/consolidation_reconcile.py",
        "research/equity_engine/scripts/canonicalize_experiment.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def main() -> None:
    patch_corporate_action_test_fixture()
    patch_dry_run_readiness()
    canonicalize_experiment()
    cleanup_temporary_files()


if __name__ == "__main__":
    main()
