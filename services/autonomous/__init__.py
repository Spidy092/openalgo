"""Safe autonomous execution integration boundaries.

The package coordinates already-approved research candidates with deterministic
platform risk and Analyzer-only execution. It does not contain a live broker
order path.
"""

from services.autonomous.portfolio_analyzer_bridge import (
    AnalyzerPortfolioSnapshotAdapter,
    AnalyzerRiskContext,
    AnalyzerSnapshotUnavailable,
    PortfolioAnalyzerBridge,
    PortfolioAnalyzerResult,
    TradeCandidatePortfolioIntentAdapter,
)

__all__ = [
    "AnalyzerPortfolioSnapshotAdapter",
    "AnalyzerRiskContext",
    "AnalyzerSnapshotUnavailable",
    "PortfolioAnalyzerBridge",
    "PortfolioAnalyzerResult",
    "TradeCandidatePortfolioIntentAdapter",
]
