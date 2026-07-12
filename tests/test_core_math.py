import math
from datetime import datetime, timezone

import pytest

from option_repair_risk.core import (
    bs_inverse_coin_price,
    bs_inverse_usd_delta,
    bs_inverse_usd_gamma,
    infer_iv_scale,
    parse_deribit_instrument,
    tau_years,
)


def test_parse_deribit_instrument_uses_0800_utc_expiry():
    parsed = parse_deribit_instrument("BTC-14OCT22-10000-C")

    assert parsed.currency == "BTC"
    assert parsed.strike == 10000.0
    assert parsed.claim_type == "call"
    assert parsed.expiration_ts == int(
        datetime(2022, 10, 14, 8, 0, tzinfo=timezone.utc).timestamp()
    )


def test_parse_deribit_instrument_accepts_mixed_case_months():
    parsed = parse_deribit_instrument("ETH-26Jun26-2400-C")

    assert parsed.currency == "ETH"
    assert parsed.strike == 2400.0
    assert parsed.claim_type == "call"
    assert parsed.expiration_ts == int(
        datetime(2026, 6, 26, 8, 0, tzinfo=timezone.utc).timestamp()
    )


def test_tau_years_uses_seconds_and_calendar_years():
    start = int(datetime(2022, 10, 8, 0, 0, tzinfo=timezone.utc).timestamp())
    expiry = int(datetime(2022, 10, 14, 8, 0, tzinfo=timezone.utc).timestamp())

    assert tau_years(start, expiry) == pytest.approx((6 + 8 / 24) / 365)


def test_infer_iv_scale_distinguishes_percent_from_decimal_units():
    assert infer_iv_scale(65.0) == 100.0
    assert infer_iv_scale(0.65) == 1.0


def test_inverse_coin_price_reconstructs_sample_deep_itm_call():
    start = int(datetime(2022, 10, 8, 0, 0, tzinfo=timezone.utc).timestamp())
    expiry = int(datetime(2022, 10, 14, 8, 0, tzinfo=timezone.utc).timestamp())
    tau = tau_years(start, expiry)

    price = bs_inverse_coin_price(
        claim_type="call",
        spot=19533.07,
        forward=19531.35,
        strike=10000.0,
        tau=tau,
        sigma=143.51 / 100.0,
    )

    assert math.isfinite(price)
    assert price == pytest.approx(0.4880, abs=2e-3)


def test_inverse_usd_greeks_match_spot_forward_finite_difference():
    spot = 100.0
    forward = 101.0
    strike = 100.0
    tau = 30 / 365
    sigma = 0.55
    carry = forward / spot
    h = 1e-3

    up = bs_inverse_coin_price(
        claim_type="call",
        spot=spot + h,
        forward=(spot + h) * carry,
        strike=strike,
        tau=tau,
        sigma=sigma,
    ) * (spot + h)
    mid = bs_inverse_coin_price(
        claim_type="call",
        spot=spot,
        forward=forward,
        strike=strike,
        tau=tau,
        sigma=sigma,
    ) * spot
    down = bs_inverse_coin_price(
        claim_type="call",
        spot=spot - h,
        forward=(spot - h) * carry,
        strike=strike,
        tau=tau,
        sigma=sigma,
    ) * (spot - h)

    fd_delta = (up - down) / (2 * h)
    fd_gamma = (up - 2 * mid + down) / (h * h)

    assert bs_inverse_usd_delta(
        claim_type="call",
        spot=spot,
        forward=forward,
        strike=strike,
        tau=tau,
        sigma=sigma,
    ) == pytest.approx(fd_delta, rel=1e-7, abs=1e-7)
    assert bs_inverse_usd_gamma(
        spot=spot,
        forward=forward,
        strike=strike,
        tau=tau,
        sigma=sigma,
    ) == pytest.approx(fd_gamma, rel=1e-5, abs=1e-5)
