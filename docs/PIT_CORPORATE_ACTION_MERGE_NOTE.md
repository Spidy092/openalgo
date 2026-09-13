# PIT Corporate-Action Merge Provenance

Source branch: `agent/pit-corporate-action-evidence`

The following files were confirmed unchanged on `codex/autonomous-integration-v1` since merge base `6b0e3c9ef3eb6120e2f92ddfccbf0b3c0ae2322a`, so their latest source-branch blobs may be transplanted without overwriting independent later work:

- `research/equity_engine/src/equity_engine/corporate_actions.py`
- `research/equity_engine/src/equity_engine/cross_sectional_walk_forward.py`
- `research/equity_engine/tests/test_corporate_actions.py`

`experiment.py` is not safe for wholesale replacement because the integration branch contains newer independent cost/evidence work. Its corporate-action changes must be merged selectively.
