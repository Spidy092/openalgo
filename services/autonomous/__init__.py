"""Safe autonomous execution integration boundaries.

The package coordinates already-approved research candidates with deterministic
platform risk and Analyzer-only execution. It does not contain a live broker
order path.
"""

from services.autonomous.execution_state import (
    CandidateIdentityConflict,
    DuplicateExecution,
    ExecutionEvent,
    ExecutionNeedsReconciliation,
    ExecutionNotFound,
    ExecutionRecord,
    ExecutionState,
    ExecutionStateError,
    ExecutionStateMachine,
    InvalidExecutionTransition,
    SqliteExecutionStateStore,
    TERMINAL_EXECUTION_STATES,
    deterministic_execution_key,
)
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
    "CandidateIdentityConflict",
    "DuplicateExecution",
    "ExecutionEvent",
    "ExecutionNeedsReconciliation",
    "ExecutionNotFound",
    "ExecutionRecord",
    "ExecutionState",
    "ExecutionStateError",
    "ExecutionStateMachine",
    "InvalidExecutionTransition",
    "PortfolioAnalyzerBridge",
    "PortfolioAnalyzerResult",
    "SqliteExecutionStateStore",
    "TERMINAL_EXECUTION_STATES",
    "TradeCandidatePortfolioIntentAdapter",
    "deterministic_execution_key",
]
