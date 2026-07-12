import pandas as pd
import pytest

from option_repair_risk.core import bs_inverse_coin_price, bs_inverse_usd_delta, bs_inverse_usd_gamma
from option_repair_risk.pipeline import greek_convention_stats


def test_greek_convention_stats_identifies_usd_delta_gamma_convention():
    spot = 100.0
    forward = 100.5
    tau = 14 / 365
    sigma = 0.6
    rows = []
    for claim_type, strike in [
        ("call", 90.0),
        ("call", 100.0),
        ("call", 110.0),
        ("put", 90.0),
        ("put", 100.0),
        ("put", 110.0),
    ]:
        rows.append(
            {
                "currency": "BTC",
                "claim_type": claim_type,
                "underlyer_spot": spot,
                "forward_price": forward,
                "strike": strike,
                "tau": tau,
                "DTE": tau * 365,
                "mark_iv": sigma * 100,
                "mark_price": bs_inverse_coin_price(
                    claim_type=claim_type,
                    spot=spot,
                    forward=forward,
                    strike=strike,
                    tau=tau,
                    sigma=sigma,
                ),
                "delta": bs_inverse_usd_delta(
                    claim_type=claim_type,
                    spot=spot,
                    forward=forward,
                    strike=strike,
                    tau=tau,
                    sigma=sigma,
                ),
                "gamma": bs_inverse_usd_gamma(
                    spot=spot,
                    forward=forward,
                    strike=strike,
                    tau=tau,
                    sigma=sigma,
                ),
                "vega": 1.0,
            }
        )

    stats = greek_convention_stats(pd.DataFrame(rows), iv_scale=100.0)

    assert stats["usd_delta_corr_with_deribit_delta"] == pytest.approx(1.0)
    assert stats["usd_delta_median_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert stats["usd_gamma_corr_with_deribit_gamma"] == pytest.approx(1.0)
    assert stats["usd_gamma_median_abs_diff"] == pytest.approx(0.0, abs=1e-12)
