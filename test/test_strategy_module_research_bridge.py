"""Research candidates import into inert Strategy Module configurations."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
import pytz
from flask import Flask

sys.path.insert(0, str(Path(__file__).parents[1]))

from blueprints import strategy_module  # noqa: E402
from database import strategy_module_db as store  # noqa: E402
from database.engine_factory import create_db_engine  # noqa: E402
from limiter import limiter  # noqa: E402
from services.strategy_module.research_bridge import (  # noqa: E402
    ResearchCandidateError,
    candidate_to_strategy_config,
)


def candidate(**overrides):
    body = {
        "schema_version": "equity-trade-candidate/v1",
        "candidate_id": "itc-orb-20260915",
        "symbol": "ITC",
        "exchange": "NSE",
        "strategy_id": "orb-15m",
        "strategy_version": "1.0.0",
        "side": "BUY",
        "quantity": 1,
        "product": "MIS",
        "price_type": "MARKET",
        "entry_price": "260.00",
        "stop_price": "258.00",
        "target_price": "264.00",
        "expected_edge_bps": "15",
        "confidence": "0.70",
        "valid_until": "2099-09-15T10:00:00+05:30",
        "dataset_fingerprint": "a" * 64,
        "research_fingerprint": "b" * 64,
    }
    body.update(overrides)
    return body


def test_bridge_maps_candidate_to_stopped_sandbox_configuration():
    config, provenance = candidate_to_strategy_config(candidate())

    assert config["strategy_kind"] == "signal"
    assert config["direction"] == "long_only"
    assert config["legs"] == [
        {
            "id": 1,
            "symbol": "ITC",
            "exchange": "NSE",
            "side": "long",
            "segment": "cash",
            "qty_mode": "units",
            "qty": 1,
            "risk_unit": "points",
            "sl_pts": 2.0,
            "target_pts": 4.0,
        }
    ]
    assert config["scheduler"]["enabled"] is False
    assert config["scheduler"]["default_mode"] == "sandbox"
    assert provenance["dataset_fingerprint"] == "a" * 64


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"price_type": "LIMIT"}, "LIMIT"),
        ({"valid_until": "2020-01-01T00:00:00+00:00"}, "stale"),
        ({"dataset_fingerprint": "not-a-hash"}, "SHA-256"),
        ({"side": "SELL", "product": "CNC"}, "MIS"),
        ({"extra": "drift"}, "unknown"),
    ],
)
def test_bridge_fails_closed(override, message):
    with pytest.raises(ResearchCandidateError, match=message):
        candidate_to_strategy_config(candidate(**override))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'bridge.db').as_posix()}")
    store.db_session.remove()
    store.db_session.configure(bind=engine)
    store.engine = engine
    store.Base.metadata.create_all(bind=engine)

    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="bridge-test")
    app.register_blueprint(strategy_module.strategy_module_bp)
    test_client = app.test_client()
    with test_client.session_transaction() as flask_session:
        flask_session["logged_in"] = True
        flask_session["user"] = "researcher"
        flask_session["login_time"] = datetime.now(
            pytz.timezone("Asia/Kolkata")
        ).isoformat()
    yield test_client
    store.db_session.remove()
    engine.dispose()


def test_import_endpoint_creates_inert_strategy_and_preserves_provenance(client):
    response = client.post(
        "/strategy/api/strategies/import-research-candidate",
        json=candidate(),
    )

    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    assert body["data"]["status"] == "stopped"
    assert body["data"]["live_enabled"] is False
    assert body["data"]["scheduler"]["enabled"] is False
    assert body["data"]["scheduler"]["default_mode"] == "sandbox"
    assert body["provenance"]["research_fingerprint"] == "b" * 64
    assert body["webhook_token"].startswith(store.WEBHOOK_TOKEN_PREFIX)

    events = store.list_events(body["data"]["id"])
    assert "dataset_sha256=" + "a" * 64 in events[0]["message"]


def test_import_endpoint_never_starts_or_enables_live(client):
    response = client.post(
        "/strategy/api/strategies/import-research-candidate",
        json=candidate(candidate_id="safety-proof"),
    )
    strategy_id = response.get_json()["data"]["id"]
    row = store.get_strategy(strategy_id, "researcher")

    assert row.status == "stopped"
    assert row.live_enabled is False
    assert store.list_runs(strategy_id, "researcher") == []
