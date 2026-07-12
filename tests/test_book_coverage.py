import pandas as pd

from option_repair_risk.pipeline import AuditConfig, build_synthetic_call_panel, compute_book_coverage


def test_book_coverage_filters_unusable_quotes_and_checks_nextday():
    price = pd.DataFrame(
        [
            {
                "currency": "BTC",
                "date": 1000,
                "expiration_date": 1000 + 10 * 86400,
                "DTE": 10.0,
                "claim_type": "call",
                "k": 0.0,
                "strike": 100.0,
                "instrument_name": "BTC-BAD-C",
                "usable_price_box": False,
            },
            {
                "currency": "BTC",
                "date": 1000,
                "expiration_date": 1000 + 10 * 86400,
                "DTE": 10.0,
                "claim_type": "call",
                "k": 0.015,
                "strike": 101.5,
                "instrument_name": "BTC-GOOD-C",
                "usable_price_box": True,
            },
            {
                "currency": "BTC",
                "date": 1000,
                "expiration_date": 1000 + 10 * 86400,
                "DTE": 10.0,
                "claim_type": "put",
                "k": 0.0,
                "strike": 100.0,
                "instrument_name": "BTC-GOOD-P",
                "usable_price_box": True,
            },
        ]
    )
    next_keys = {
        ("BTC", 1000 + 86400, "BTC-GOOD-C"),
    }

    out = compute_book_coverage(price, AuditConfig(book_tolerance=0.02), next_keys)
    atm = out[out["book"] == "ATM_straddle"].iloc[0]

    assert atm["constructible"]
    assert "BTC-BAD-C" not in atm["selected_instruments"]
    assert "BTC-GOOD-C" in atm["selected_instruments"]
    assert "BTC-GOOD-P" in atm["selected_instruments"]
    assert not atm["nextday_available"]
    assert not atm["nextday_ready"]


def test_synthetic_call_panel_uses_common_forward_for_otm_selection():
    price = pd.DataFrame(
        [
            {
                "currency": "BTC",
                "date": 1000,
                "expiration_date": 1000 + 10 * 86400,
                "strike": 100.0,
                "forward_price": 99.9,
                "underlyer_spot": 100.0,
                "common_forward_price": 100.0,
                "tau": 10 / 365,
                "DTE": 10.0,
                "dte_bucket": "8-14",
                "k": 0.0,
                "claim_type": "call",
                "bid_price": 0.01,
                "ask_price": 0.02,
                "instrument_name": "BTC-C",
                "usable_price_box": True,
                "relspread": 0.1,
            },
            {
                "currency": "BTC",
                "date": 1000,
                "expiration_date": 1000 + 10 * 86400,
                "strike": 100.0,
                "forward_price": 100.1,
                "underlyer_spot": 100.0,
                "common_forward_price": 100.0,
                "tau": 10 / 365,
                "DTE": 10.0,
                "dte_bucket": "8-14",
                "k": 0.0,
                "claim_type": "put",
                "bid_price": 0.01,
                "ask_price": 0.02,
                "instrument_name": "BTC-P",
                "usable_price_box": True,
                "relspread": 0.1,
            },
        ]
    )

    panel = build_synthetic_call_panel(price)

    assert len(panel) == 1
    assert panel.iloc[0]["source"] == "call"
    assert not panel.duplicated(["currency", "date", "expiration_date", "strike"]).any()
