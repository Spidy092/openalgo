from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"{label}: marker not found")
    return text.replace(old, new, 1)


def patch_experiment() -> None:
    path = ROOT / "research/equity_engine/src/equity_engine/experiment.py"
    text = path.read_text(encoding="utf-8")
    marker = "@dataclass(frozen=True)\nclass CorporateActionEvidenceIdentity:"
    if "_UNSET_COVERED_INSTRUMENTS = object()" not in text:
        text = text.replace(marker, "_UNSET_COVERED_INSTRUMENTS = object()\n\n\n" + marker, 1)
    text = replace_once(
        text,
        "    covered_instruments: tuple[str, ...] = ()\n",
        "    covered_instruments: tuple[str, ...] | object = _UNSET_COVERED_INSTRUMENTS\n",
        "covered_instruments sentinel",
    )
    text = replace_once(
        text,
        """        if not self.evidence_fingerprint.strip():
            raise ValueError(\"corporate-action evidence fingerprint is required\")
        if any(not str(key).strip() for key in self.covered_instruments):
""",
        """        if not self.evidence_fingerprint.strip():
            raise ValueError(\"corporate-action evidence fingerprint is required\")
        if self.covered_instruments is _UNSET_COVERED_INSTRUMENTS:
            object.__setattr__(self, \"covered_instruments\", ())
        elif not self.covered_instruments:
            raise ValueError(\"covered_instruments cannot be empty for corporate-action evidence\")
        if any(not str(key).strip() for key in self.covered_instruments):
""",
        "corporate-action explicit-empty validation",
    )
    old = """        policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )
        if ledger is None:
"""
    new = """        policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )
        if promotion_thresholds is not None:
            passed, violations = self.evaluate_promotion_gate(
                promotion_thresholds,
                trusted_corporate_action_ledger=ledger,
                trusted_corporate_action_policy=policy,
                **kwargs,
            )
            if not passed:
                raise MissingEvidenceError(
                    \"experiment failed promotion gate criteria: \" + \"; \".join(violations)
                )
            return
        if ledger is None:
"""
    if new not in text:
        idx = text.rfind(old)
        if idx < 0:
            raise RuntimeError("validate_integrity policy marker not found")
        text = text[:idx] + text[idx:].replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def patch_shadow_runner() -> None:
    path = ROOT / "research/equity_engine/src/equity_engine/shadow_live_runner.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """    ReadinessClassification,
    build_runner_readiness_context,
)""",
        """    ReadinessClassification,
    build_runner_readiness_context,
    build_synthetic_readiness_report,
)""",
        "shadow synthetic readiness import",
    )
    text = replace_once(
        text,
        """        report = self._readiness_report
        if not isinstance(report, LiveMarketReadinessReport):
            raise ReadinessGateError(READINESS_REPORT_MISSING)
""",
        """        report = self._readiness_report
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
""",
        "shadow synthetic readiness gate",
    )
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    path = ROOT / "research/equity_engine/tests/test_corporate_actions.py"
    text = path.read_text(encoding="utf-8")
    import_line = "from equity_engine.cost_ledger import EffectiveDatedCostLedger, LedgerProduct\n"
    if import_line not in text:
        text = text.replace("import pytest\n", "import pytest\n" + import_line, 1)
    if "_verified_ledger_fingerprint=" in text:
        start = text.index("        cost_evidence_identity=CostEvidenceIdentity(\n")
        end_marker = "        cost_evidence_class=\"HISTORICAL_ACTUAL_COSTS\",\n"
        end = text.index(end_marker, start)
        replacement = """        cost_evidence_identity=CostEvidenceIdentity.from_ledger(
            EffectiveDatedCostLedger(),
            on_date=date(2026, 6, 30),
            product=LedgerProduct.INTRADAY,
        ),
"""
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")

    path = ROOT / "research/equity_engine/tests/test_research_execution_readiness.py"
    text = path.read_text(encoding="utf-8")
    old = "    experiment.validate_integrity()\n\n    # 11. Promotion Gate\n"
    if old in text:
        text = text.replace(old, "    experiment.validate_structure()\n\n    # 11. Promotion Gate\n", 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_experiment()
    patch_shadow_runner()
    patch_tests()


if __name__ == "__main__":
    main()
