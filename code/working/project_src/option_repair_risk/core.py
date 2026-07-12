from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

INSTRUMENT_RE = re.compile(
    r"^(?P<currency>[A-Z]+)-(?P<day>\d{1,2})(?P<month>[A-Za-z]{3})(?P<year>\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
)


@dataclass(frozen=True)
class ParsedInstrument:
    currency: str
    expiration_ts: int
    strike: float
    claim_type: str


def parse_deribit_instrument(instrument_name: str) -> ParsedInstrument:
    match = INSTRUMENT_RE.match(instrument_name)
    if not match:
        raise ValueError(f"Unrecognized Deribit instrument: {instrument_name!r}")
    parts = match.groupdict()
    month = MONTHS[parts["month"].upper()]
    year = 2000 + int(parts["year"])
    expiry = datetime(
        year,
        month,
        int(parts["day"]),
        8,
        0,
        0,
        tzinfo=timezone.utc,
    )
    claim_type = "call" if parts["cp"] == "C" else "put"
    return ParsedInstrument(
        currency=parts["currency"],
        expiration_ts=int(expiry.timestamp()),
        strike=float(parts["strike"]),
        claim_type=claim_type,
    )


def tau_years(snapshot_ts: int | float, expiration_ts: int | float) -> float:
    tau = (float(expiration_ts) - float(snapshot_ts)) / (365.0 * 24.0 * 3600.0)
    return max(tau, 0.0)


def infer_iv_scale(raw_median_iv: float) -> float:
    if not math.isfinite(raw_median_iv) or raw_median_iv <= 0:
        raise ValueError(f"Cannot infer IV scale from median {raw_median_iv!r}")
    return 100.0 if raw_median_iv > 3.0 else 1.0


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def inverse_black_d1(*, forward: float, strike: float, tau: float, sigma: float) -> float:
    if forward <= 0 or strike <= 0 or tau <= 0 or sigma <= 0 or not math.isfinite(sigma):
        return float("nan")
    vol_sqrt_t = sigma * math.sqrt(tau)
    return (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol_sqrt_t


def bs_inverse_coin_price(
    *,
    claim_type: str,
    spot: float,
    forward: float,
    strike: float,
    tau: float,
    sigma: float,
) -> float:
    """Deribit inverse-option Black price in base currency.

    The USD Black price uses the continuously compounded rate implied by
    `forward / spot`. Dividing by spot gives the base-currency premium:
    call = N(d1) - K/F N(d2), put = K/F N(-d2) - N(-d1).
    """
    if spot <= 0 or forward <= 0 or strike <= 0:
        return float("nan")
    if tau <= 0 or sigma <= 0 or not math.isfinite(sigma):
        intrinsic = max(1.0 - strike / forward, 0.0)
        if claim_type == "call":
            return intrinsic
        if claim_type == "put":
            return max(strike / forward - 1.0, 0.0)
        return float("nan")

    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    cp = claim_type.lower()
    if cp in {"call", "c"}:
        return normal_cdf(d1) - (strike / forward) * normal_cdf(d2)
    if cp in {"put", "p"}:
        return (strike / forward) * normal_cdf(-d2) - normal_cdf(-d1)
    return float("nan")


def bs_inverse_usd_price(
    *,
    claim_type: str,
    spot: float,
    forward: float,
    strike: float,
    tau: float,
    sigma: float,
) -> float:
    coin = bs_inverse_coin_price(
        claim_type=claim_type,
        spot=spot,
        forward=forward,
        strike=strike,
        tau=tau,
        sigma=sigma,
    )
    return coin * spot if math.isfinite(coin) else float("nan")


def bs_inverse_usd_delta(
    *,
    claim_type: str,
    spot: float,
    forward: float,
    strike: float,
    tau: float,
    sigma: float,
) -> float:
    """USD-premium spot delta when forward/spot carry is held fixed."""
    if spot <= 0:
        return float("nan")
    d1 = inverse_black_d1(forward=forward, strike=strike, tau=tau, sigma=sigma)
    if not math.isfinite(d1):
        return float("nan")
    cp = claim_type.lower()
    if cp in {"call", "c"}:
        return normal_cdf(d1)
    if cp in {"put", "p"}:
        return normal_cdf(d1) - 1.0
    return float("nan")


def bs_inverse_usd_gamma(
    *,
    spot: float,
    forward: float,
    strike: float,
    tau: float,
    sigma: float,
) -> float:
    """USD-premium spot gamma when forward/spot carry is held fixed."""
    if spot <= 0:
        return float("nan")
    d1 = inverse_black_d1(forward=forward, strike=strike, tau=tau, sigma=sigma)
    if not math.isfinite(d1):
        return float("nan")
    return normal_pdf(d1) / (spot * sigma * math.sqrt(tau))


def self_check_core() -> None:
    parsed = parse_deribit_instrument("BTC-14OCT22-10000-C")
    assert parsed.expiration_ts == int(
        datetime(2022, 10, 14, 8, tzinfo=timezone.utc).timestamp()
    )
    tau = tau_years(
        int(datetime(2022, 10, 8, 0, tzinfo=timezone.utc).timestamp()),
        parsed.expiration_ts,
    )
    assert abs(tau - (6 + 8 / 24) / 365) < 1e-12
    assert infer_iv_scale(65.0) == 100.0
    price = bs_inverse_coin_price(
        claim_type="call",
        spot=19533.07,
        forward=19531.35,
        strike=10000.0,
        tau=tau,
        sigma=143.51 / 100.0,
    )
    assert abs(price - 0.4880) < 0.002
    delta = bs_inverse_usd_delta(
        claim_type="call",
        spot=19533.07,
        forward=19531.35,
        strike=10000.0,
        tau=tau,
        sigma=143.51 / 100.0,
    )
    gamma = bs_inverse_usd_gamma(
        spot=19533.07,
        forward=19531.35,
        strike=10000.0,
        tau=tau,
        sigma=143.51 / 100.0,
    )
    assert 0.0 < delta < 1.0
    assert gamma > 0.0
