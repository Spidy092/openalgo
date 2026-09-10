"""Generate the compact deterministic historical-acquisition V2 example."""

from __future__ import annotations

from pathlib import Path

from equity_engine.historical_acquisition_plan import build_example_historical_acquisition_plan

OUTPUT_PATH = Path(__file__).parents[1] / "examples" / "historical_acquisition_plan_example.json"


def main() -> int:
    plan = build_example_historical_acquisition_plan()
    payload = plan.to_json(indent=None)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    if plan.deterministic_fingerprint() not in payload:
        raise RuntimeError("generated fixture is missing its deterministic fingerprint")
    print(
        f"wrote {OUTPUT_PATH} ({len(payload.encode('utf-8'))} bytes, "
        f"{len(payload.splitlines())} lines)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
