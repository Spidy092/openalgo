# Autonomous Integration Status

This branch consolidates research, evidence, readiness, shadow/analyzer execution, and autonomous orchestration work.

## Hard safety boundary

Autonomous v1 is limited to NSE/BSE cash equity in SHADOW/ANALYZER modes. It cannot place live broker orders. The analyzer executor calls OpenAlgo's sandbox implementation directly, so a mode-toggle race cannot fall through to live execution.

## Consolidation rules

- Exact-SHA-identical branch work is treated as already absorbed.
- Diverged branches are not blindly merged.
- Files unchanged since their merge base may be transplanted as exact blobs.
- Files with independent later edits are merged surgically and must retain both lines of development.
- Unknown or unverifiable evidence fails closed.

## Current work

- Autonomous trade candidate contract: added.
- Deterministic risk gate: added.
- Analyzer-only executor: added.
- Analyzer executor tests: added.
- Branch-family audit: in progress.
- PIT corporate-action evidence: identified as genuinely missing and being integrated.
- Autonomous session journal/watchdog: pending.
- CI verification and integration PR: pending.
