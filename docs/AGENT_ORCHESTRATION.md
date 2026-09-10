# OpenAlgo Agent Orchestration

This document defines ownership and integration authority for the research and
engineering work around OpenAlgo. It does not grant any agent authority to
place live orders or approve capital deployment.

## Ownership

| Owner | Responsibility |
| --- | --- |
| **Product Owner** | Final business, capital, and live-authorization decisions. |
| **ChatGPT** | System architecture, cross-agent orchestration, dependency and gate decisions, exact-SHA verification, and integration approval. |
| **Codex** | Core engineering, shared interfaces, integration, CI, and release assembly. Codex is the only agent that assembles approved commits. |
| **Kiro** | Historical and point-in-time datasets, acquisition, provenance, and data quality. |
| **Antigravity** | Experiments, strategy evaluation, walk-forward/out-of-sample validation, and research-lifecycle/promotion evidence. |
| **OpenCode** | Transaction costs, fees, fills, slippage, market microstructure, and execution realism. |

Ownership means an agent may develop within its scope and report an exact
commit. It does not authorize that agent to merge, deploy, trade, or change
another agent's branch.

## Engineering rules

1. Every agent works in a separate isolated worktree and branch.
2. Agents never edit another agent's branch or worktree.
3. Each subsystem has one canonical implementation. Do not create duplicate
   implementations to bypass ownership or integration conflicts.
4. Commits are integrated only by Codex after ChatGPT's orchestration approval
   and exact-SHA verification. Product Owner approval remains required for
   business, capital, or live authorization.
5. Use `uv run` for repository Python commands and CI-equivalent tooling.
6. Never output or persist credentials, access tokens, API secrets, or personal
   broker identity data.
7. Research code must not place, modify, or cancel live orders. Read-only broker
   checks must remain explicitly separate from order paths.
8. Do not mutate, delete, migrate, or silently rewrite existing research data.
   Raw and derived artifacts must remain traceable and hash-bound.
9. A current snapshot is not historical eligibility. Historical decisions must
   use dated, point-in-time evidence.
10. Unknown evidence is not zero, false, or complete evidence; unresolved
    ambiguity fails closed.
11. Prevent lookahead and survivorship leakage in data acquisition, universe
    formation, signal generation, validation, and promotion.
12. PR #1 remains draft and unmerged until the Product Owner explicitly approves
    the business, capital, and live-authorization boundary.

## Integration handoff

Each handoff must state the branch, parent/base SHA, exact commit SHA, changed
files, tests and CI results, unresolved assumptions, and any external-data
access performed. ChatGPT resolves dependency order and gates; Codex performs
the approved assembly and release checks; the Product Owner makes the final
authorization decision.
