from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import ndtr

from .audit_helpers import (
    EPS,
    classify_dte_bucket,
    convexity_violations,
    elastic_monotone_convex_slack,
    monotonicity_violations,
)
from .core import infer_iv_scale, parse_deribit_instrument


NUMERIC_COLUMNS = [
    "strike",
    "open_interest",
    "volume",
    "underlyer_spot",
    "forward_price",
    "mark_price",
    "mark_iv",
    "bid_iv",
    "best_bid_price",
    "best_bid_amount",
    "best_ask_price",
    "best_ask_amount",
    "ask_iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]

REQUIRED_COLUMNS = [
    "currency",
    "strike",
    "claim_type",
    "expiration_date",
    "date",
    "granularity",
    "instrument_name",
    "underlyer_spot",
    "forward_price",
    "mark_price",
    "mark_iv",
    "bid_iv",
    "best_bid_price",
    "best_ask_price",
    "ask_iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]

DTE_BUCKETS = ["0-1", "2-7", "8-14", "15-30", "31-45", "46+"]
ARTIFACT_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class AuditConfig:
    data_root: Path = Path(
        os.environ.get("REPAIR_RISK_RAW_DATA_ROOT", ARTIFACT_ROOT / "data" / "raw")
    )
    output_root: Path = Path(
        os.environ.get("REPAIR_RISK_PANEL_BUILD_ROOT", ARTIFACT_ROOT / "results" / "panel_build")
    )
    main_snapshot_hour: int = 9
    main_dte_min: float = 2.0
    main_dte_max: float = 45.0
    pricing_sample_per_currency: int = 500_000
    greek_sample_per_currency: int = 200_000
    book_tolerance: float = 0.02
    feasibility_max_lp_per_currency: int = 10_000

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def parquet_dir(self) -> Path:
        return self.output_root / "parquet"


@dataclass
class AuditState:
    config: AuditConfig
    iv_scale: float | None = None
    price_convention: str | None = None
    audit_summary: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.audit_summary is None:
            self.audit_summary = []

    def record(self, audit: str, metric: str, value, passed: bool | None = None) -> None:
        assert self.audit_summary is not None
        self.audit_summary.append(
            {
                "audit": audit,
                "metric": metric,
                "value": value,
                "passed": "" if passed is None else bool(passed),
            }
        )


def ensure_dirs(config: AuditConfig) -> None:
    for path in [
        config.output_root,
        config.tables_dir,
        config.figures_dir,
        config.logs_dir,
        config.parquet_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def option_files(config: AuditConfig, currency: str) -> list[Path]:
    return sorted((config.data_root / currency).glob("*.parquet"))


def scan_options(config: AuditConfig, currency: str) -> pl.LazyFrame:
    files = option_files(config, currency)
    if not files:
        raise FileNotFoundError(f"No parquet files for {currency} in {config.data_root}")
    lf = pl.scan_parquet([str(p) for p in files])
    casts = [
        pl.col(c).cast(pl.Float64).alias(c)
        for c in NUMERIC_COLUMNS
        if c in lf.collect_schema().names()
    ]
    return lf.with_columns(casts)


def scan_all(config: AuditConfig) -> pl.LazyFrame:
    return pl.concat([scan_options(config, "BTC"), scan_options(config, "ETH")])


def add_time_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    tau = (pl.col("expiration_date") - pl.col("date")) / (365.0 * 24.0 * 3600.0)
    return lf.with_columns(
        [
            pl.from_epoch(pl.col("date"), time_unit="s").dt.replace_time_zone("UTC").alias("datetime_utc"),
            pl.from_epoch(pl.col("expiration_date"), time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("expiration_utc"),
            (pl.col("date") % 86_400 // 3_600).cast(pl.Int64).alias("hour_utc"),
            (pl.col("date") // 86_400).cast(pl.Int64).alias("date_day"),
            tau.alias("tau"),
            (tau * 365.0).alias("DTE"),
        ]
    )


def main_sample_filter(config: AuditConfig):
    return (
        (pl.col("hour_utc") == config.main_snapshot_hour)
        & pl.col("DTE").is_between(config.main_dte_min, config.main_dte_max)
    )


def valid_core_filter():
    return (
        (pl.col("underlyer_spot") > 0)
        & (pl.col("forward_price") > 0)
        & (pl.col("strike") > 0)
        & (pl.col("tau") > 0)
        & (pl.col("mark_price") >= 0)
        & (pl.col("mark_iv") > 0)
    )


def write_csv(df: pl.DataFrame | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df.write_csv(path)
    else:
        df.to_csv(path, index=False)


def self_check_table(path: Path, min_rows: int = 1) -> None:
    if not path.exists():
        raise AssertionError(f"Missing expected output: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if len(df) < min_rows:
            raise AssertionError(f"Expected at least {min_rows} rows in {path}")


def vector_inverse_coin_price(
    claim_type: np.ndarray,
    spot: np.ndarray,
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    spot = spot.astype(float)
    forward = forward.astype(float)
    strike = strike.astype(float)
    tau = tau.astype(float)
    sigma = sigma.astype(float)
    out = np.full(len(strike), np.nan, dtype=float)
    valid = (spot > 0) & (forward > 0) & (strike > 0) & (tau > 0) & (sigma > 0)
    if not np.any(valid):
        return out
    vol_sqrt_t = sigma[valid] * np.sqrt(tau[valid])
    d1 = (np.log(forward[valid] / strike[valid]) + 0.5 * sigma[valid] ** 2 * tau[valid]) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    k_over_f = strike[valid] / forward[valid]
    cp = pd.Series(claim_type[valid]).astype(str).str.lower().to_numpy()
    val = np.full(cp.shape[0], np.nan, dtype=float)
    call = (cp == "call") | (cp == "c")
    put = (cp == "put") | (cp == "p")
    val[call] = ndtr(d1[call]) - k_over_f[call] * ndtr(d2[call])
    val[put] = k_over_f[put] * ndtr(-d2[put]) - ndtr(-d1[put])
    out[valid] = val
    return out


def vector_inverse_usd_greeks(
    claim_type: np.ndarray,
    spot: np.ndarray,
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spot = spot.astype(float)
    forward = forward.astype(float)
    strike = strike.astype(float)
    tau = tau.astype(float)
    sigma = sigma.astype(float)
    delta = np.full(len(strike), np.nan, dtype=float)
    gamma = np.full(len(strike), np.nan, dtype=float)
    valid = (spot > 0) & (forward > 0) & (strike > 0) & (tau > 0) & (sigma > 0)
    if not np.any(valid):
        return delta, gamma
    vol_sqrt_t = sigma[valid] * np.sqrt(tau[valid])
    d1 = (np.log(forward[valid] / strike[valid]) + 0.5 * sigma[valid] ** 2 * tau[valid]) / vol_sqrt_t
    norm_pdf = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    cp = pd.Series(claim_type[valid]).astype(str).str.lower().to_numpy()
    val_delta = np.full(cp.shape[0], np.nan, dtype=float)
    call = (cp == "call") | (cp == "c")
    put = (cp == "put") | (cp == "p")
    val_delta[call] = ndtr(d1[call])
    val_delta[put] = ndtr(d1[put]) - 1.0
    delta[valid] = val_delta
    gamma[valid] = norm_pdf / (spot[valid] * sigma[valid] * np.sqrt(tau[valid]))
    return delta, gamma


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return float("nan")
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok])))


def finite_median_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    return float(np.median(np.abs(a[ok] - b[ok])))


def finite_p90_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    return float(np.quantile(np.abs(a[ok] - b[ok]), 0.90))


def finite_ratio_median(actual: np.ndarray, model: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    model = np.asarray(model, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(model) & (np.abs(model) > 1e-18)
    if not np.any(ok):
        return float("nan")
    return float(np.median(actual[ok] / model[ok]))


def greek_convention_stats(df: pd.DataFrame, iv_scale: float) -> dict:
    if len(df) == 0:
        return {"sample_rows": 0}
    sigma = df["mark_iv"].to_numpy(dtype=float) / iv_scale
    spot = df["underlyer_spot"].to_numpy(dtype=float)
    forward = df["forward_price"].to_numpy(dtype=float)
    strike = df["strike"].to_numpy(dtype=float)
    tau = df["tau"].to_numpy(dtype=float)
    claim = df["claim_type"].to_numpy()
    deribit_delta = df["delta"].to_numpy(dtype=float)
    deribit_gamma = df["gamma"].to_numpy(dtype=float)
    usd_delta, usd_gamma = vector_inverse_usd_greeks(claim, spot, forward, strike, tau, sigma)

    h = np.maximum(spot * 1e-4, 1e-6)
    up_spot = spot + h
    down_spot = np.maximum(spot - h, 1e-12)
    up_forward = forward * up_spot / spot
    down_forward = forward * down_spot / spot
    up_coin = vector_inverse_coin_price(claim, up_spot, up_forward, strike, tau, sigma)
    down_coin = vector_inverse_coin_price(claim, down_spot, down_forward, strike, tau, sigma)
    coin_fd_per_dollar = (up_coin - down_coin) / (up_spot - down_spot)
    fd_usd_delta = (up_coin * up_spot - down_coin * down_spot) / (up_spot - down_spot)

    call = df["claim_type"].astype(str).str.lower().isin(["call", "c"])
    put = df["claim_type"].astype(str).str.lower().isin(["put", "p"])
    return {
        "sample_rows": int(len(df)),
        "call_delta_positive_rate": float((df.loc[call, "delta"] > 0).mean()) if call.any() else float("nan"),
        "put_delta_negative_rate": float((df.loc[put, "delta"] < 0).mean()) if put.any() else float("nan"),
        "gamma_positive_rate": float((df["gamma"] > 0).mean()),
        "vega_positive_rate": float((df["vega"] > 0).mean()) if "vega" in df else float("nan"),
        "usd_delta_corr_with_deribit_delta": finite_corr(usd_delta, deribit_delta),
        "usd_delta_median_abs_diff": finite_median_abs_diff(usd_delta, deribit_delta),
        "usd_delta_p90_abs_diff": finite_p90_abs_diff(usd_delta, deribit_delta),
        "fd_usd_delta_corr_with_deribit_delta": finite_corr(fd_usd_delta, deribit_delta),
        "fd_usd_delta_median_abs_diff": finite_median_abs_diff(fd_usd_delta, deribit_delta),
        "coin_fd_per_dollar_corr_with_deribit_delta": finite_corr(coin_fd_per_dollar, deribit_delta),
        "coin_fd_per_dollar_median_abs_diff": finite_median_abs_diff(coin_fd_per_dollar, deribit_delta),
        "usd_gamma_corr_with_deribit_gamma": finite_corr(usd_gamma, deribit_gamma),
        "usd_gamma_median_abs_diff": finite_median_abs_diff(usd_gamma, deribit_gamma),
        "usd_gamma_ratio_median": finite_ratio_median(deribit_gamma, usd_gamma),
        "note": "Main check is USD-premium spot delta/gamma with forward/spot carry fixed; coin-FD-per-dollar is a convention contrast.",
    }


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), eps)


def audit0_integrity(state: AuditState) -> None:
    config = state.config
    rows = []
    required = set(REQUIRED_COLUMNS)
    for currency in ["BTC", "ETH"]:
        lf = add_time_columns(scan_options(config, currency))
        schema_cols = set(lf.collect_schema().names())
        missing_required = sorted(required - schema_cols)
        basic = lf.select(
            [
                pl.len().alias("rows"),
                pl.col("instrument_name").n_unique().alias("unique_instruments"),
                pl.col("date").n_unique().alias("unique_timestamps"),
                pl.col("date").min().alias("min_ts"),
                pl.col("date").max().alias("max_ts"),
                (pl.col("date") % 3600).max().alias("max_ts_mod_3600"),
            ]
        ).collect().row(0, named=True)
        dups1 = (
            lf.group_by(["currency", "date", "instrument_name"])
            .len()
            .filter(pl.col("len") > 1)
            .select((pl.col("len") - 1).sum().fill_null(0).alias("duplicate_rows"))
            .collect()
            .item()
        )
        dups2 = (
            lf.group_by(["currency", "date", "expiration_date", "strike", "claim_type"])
            .len()
            .filter(pl.col("len") > 1)
            .select((pl.col("len") - 1).sum().fill_null(0).alias("duplicate_rows"))
            .collect()
            .item()
        )
        expected_hours = int((basic["max_ts"] - basic["min_ts"]) / 3600) + 1
        missing_ts_rate = 1.0 - basic["unique_timestamps"] / expected_hours
        rows.append(
            {
                "currency": currency,
                "rows": basic["rows"],
                "unique_instruments": basic["unique_instruments"],
                "unique_timestamps": basic["unique_timestamps"],
                "min_ts": basic["min_ts"],
                "max_ts": basic["max_ts"],
                "min_datetime_utc": pd.to_datetime(basic["min_ts"], unit="s", utc=True),
                "max_datetime_utc": pd.to_datetime(basic["max_ts"], unit="s", utc=True),
                "duplicate_rate_currency_date_instrument": dups1 / basic["rows"],
                "duplicate_rate_expiry_strike_claim": dups2 / basic["rows"],
                "missing_timestamp_rate": missing_ts_rate,
                "max_ts_mod_3600": basic["max_ts_mod_3600"],
                "missing_required_columns": ";".join(missing_required),
            }
        )
        state.record("Audit 0", f"{currency} hourly timestamps", basic["max_ts_mod_3600"], basic["max_ts_mod_3600"] == 0)
        state.record("Audit 0", f"{currency} duplicate key rate", dups1 / basic["rows"], dups1 == 0)
    out = pd.DataFrame(rows)
    path = config.tables_dir / "audit0_integrity.csv"
    write_csv(out, path)
    self_check_audit0(path)


def self_check_audit0(path: Path) -> None:
    self_check_table(path, min_rows=2)
    df = pd.read_csv(path)
    if not (df["rows"] > 1_000_000).all():
        raise AssertionError("Audit 0 row counts are unexpectedly small")
    if not (df["max_ts_mod_3600"] == 0).all():
        raise AssertionError("Audit 0 found non-hourly timestamps")


def audit1_time_axis(state: AuditState) -> None:
    config = state.config
    lf = add_time_columns(scan_all(config))
    valid_bidask = (pl.col("best_ask_price") >= pl.col("best_bid_price")) & (pl.col("best_ask_price") > 0)
    relspread = (pl.col("best_ask_price") - pl.col("best_bid_price")) / (pl.col("mark_price") + 1e-12)
    hourly = (
        lf.with_columns(relspread.alias("relspread_raw"))
        .group_by(["currency", "hour_utc"])
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("date_day").n_unique().alias("calendar_days"),
                pl.struct(["date", "expiration_date"]).n_unique().alias("chain_slices"),
                (pl.col("bid_iv") <= 0).mean().alias("missing_or_zero_bid_iv_rate"),
                (pl.col("ask_iv") <= 0).mean().alias("missing_or_zero_ask_iv_rate"),
                valid_bidask.mean().alias("valid_explicit_bidask_rate"),
                pl.col("relspread_raw").filter(valid_bidask).median().alias("median_raw_relspread"),
            ]
        )
        .sort(["currency", "hour_utc"])
        .collect()
    )
    path = config.tables_dir / "audit1_hour_quality.csv"
    write_csv(hourly, path)

    heatmap = (
        lf.group_by(["currency", "date_day", "hour_utc"])
        .agg(pl.len().alias("rows"))
        .sort(["currency", "date_day", "hour_utc"])
        .collect()
    )
    write_csv(heatmap, config.tables_dir / "audit1_date_hour_row_counts.csv")
    plot_hour_quality(hourly, config.figures_dir / "audit1_hour_quality.png")
    self_check_audit1(path, config.main_snapshot_hour)
    selected = hourly.filter(pl.col("hour_utc") == config.main_snapshot_hour)
    for row in selected.iter_rows(named=True):
        state.record("Audit 1", f"{row['currency']} selected hour days", row["calendar_days"], row["calendar_days"] > 1000)


def plot_hour_quality(hourly: pl.DataFrame, path: Path) -> None:
    pdf = hourly.to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for currency, g in pdf.groupby("currency"):
        axes[0].plot(g["hour_utc"], g["calendar_days"], marker="o", label=currency)
        axes[1].plot(g["hour_utc"], g["median_raw_relspread"], marker="o", label=currency)
    axes[0].set_title("Calendar-day coverage by UTC hour")
    axes[0].set_xlabel("UTC hour")
    axes[0].set_ylabel("days")
    axes[1].set_title("Median explicit relative spread")
    axes[1].set_xlabel("UTC hour")
    axes[1].set_ylabel("(ask-bid)/mark")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def self_check_audit1(path: Path, main_hour: int) -> None:
    self_check_table(path, min_rows=48)
    df = pd.read_csv(path)
    if main_hour not in set(df["hour_utc"]):
        raise AssertionError(f"Main snapshot hour {main_hour} is absent")


def audit2_instrument_parsing(state: AuditState) -> None:
    config = state.config
    rows = []
    examples = []
    for currency in ["BTC", "ETH"]:
        lf = scan_options(config, currency)
        unique = (
            lf.group_by("instrument_name")
            .agg(
                [
                    pl.col("currency").first(),
                    pl.col("expiration_date").first(),
                    pl.col("strike").first(),
                    pl.col("claim_type").first(),
                    pl.len().alias("rows"),
                ]
            )
            .collect()
            .to_pandas()
        )
        parsed_rows = []
        parse_errors = 0
        for row in unique.itertuples(index=False):
            try:
                parsed = parse_deribit_instrument(row.instrument_name)
                parsed_rows.append((parsed.currency, parsed.expiration_ts, parsed.strike, parsed.claim_type))
            except ValueError:
                parse_errors += 1
                parsed_rows.append((None, None, np.nan, None))
        parsed_df = pd.DataFrame(parsed_rows, columns=["parsed_currency", "parsed_expiration", "parsed_strike", "parsed_claim"])
        joined = pd.concat([unique.reset_index(drop=True), parsed_df], axis=1)
        joined["currency_mismatch"] = joined["currency"] != joined["parsed_currency"]
        joined["expiration_mismatch"] = joined["expiration_date"].astype("int64") != joined["parsed_expiration"].astype("Int64")
        joined["strike_mismatch"] = np.abs(joined["strike"].astype(float) - joined["parsed_strike"].astype(float)) > 1e-9
        joined["claim_mismatch"] = joined["claim_type"].str.lower() != joined["parsed_claim"]
        rows.append(
            {
                "currency": currency,
                "unique_instruments": len(joined),
                "parse_error_rate": parse_errors / max(len(joined), 1),
                "currency_mismatch_rate": joined["currency_mismatch"].mean(),
                "expiration_mismatch_rate": joined["expiration_mismatch"].mean(),
                "strike_mismatch_rate": joined["strike_mismatch"].mean(),
                "claim_mismatch_rate": joined["claim_mismatch"].mean(),
            }
        )
        examples.append(joined.head(20))
        state.record("Audit 2", f"{currency} parse error rate", parse_errors / max(len(joined), 1), parse_errors == 0)
    out = pd.DataFrame(rows)
    path = config.tables_dir / "audit2_instrument_parsing.csv"
    write_csv(out, path)
    pd.concat(examples, ignore_index=True).to_csv(config.tables_dir / "audit2_instrument_examples.csv", index=False)
    self_check_audit2(path)


def self_check_audit2(path: Path) -> None:
    self_check_table(path, min_rows=2)
    df = pd.read_csv(path)
    bad_cols = [c for c in df.columns if c.endswith("_rate")]
    if not (df[bad_cols] < 1e-8).all().all():
        raise AssertionError("Audit 2 found instrument parsing or field consistency failures")


def audit3_iv_units(state: AuditState) -> None:
    config = state.config
    lf = scan_all(config)
    q = (
        lf.with_columns((pl.col("date") // (365 * 24 * 3600) + 1970).alias("rough_year"))
        .group_by("currency")
        .agg(
            [
                pl.col("mark_iv").quantile(0.01).alias("mark_iv_p01"),
                pl.col("mark_iv").median().alias("mark_iv_median"),
                pl.col("mark_iv").quantile(0.99).alias("mark_iv_p99"),
                (pl.col("mark_iv") <= 0).mean().alias("mark_iv_nonpositive_rate"),
                (pl.col("bid_iv") <= 0).mean().alias("bid_iv_nonpositive_rate"),
                (pl.col("ask_iv") <= 0).mean().alias("ask_iv_nonpositive_rate"),
            ]
        )
        .sort("currency")
        .collect()
    )
    pdf = q.to_pandas()
    pdf["inferred_iv_scale"] = pdf["mark_iv_median"].map(infer_iv_scale)
    scales = set(pdf["inferred_iv_scale"])
    if len(scales) != 1:
        raise AssertionError(f"Inconsistent IV scales across currencies: {scales}")
    state.iv_scale = float(next(iter(scales)))
    path = config.tables_dir / "audit3_iv_units.csv"
    write_csv(pdf, path)
    state.record("Audit 3", "inferred_iv_scale", state.iv_scale, state.iv_scale == 100.0)
    self_check_audit3(path)


def self_check_audit3(path: Path) -> None:
    self_check_table(path, min_rows=2)
    df = pd.read_csv(path)
    if df["inferred_iv_scale"].nunique() != 1:
        raise AssertionError("Audit 3 did not infer a unique IV unit")


def collect_pricing_sample(config: AuditConfig, currency: str, n: int) -> pd.DataFrame:
    lf = (
        add_time_columns(scan_options(config, currency))
        .filter(main_sample_filter(config) & valid_core_filter())
        .select(
            [
                "currency",
                "date",
                "expiration_date",
                "instrument_name",
                "claim_type",
                "strike",
                "underlyer_spot",
                "forward_price",
                "mark_price",
                "mark_iv",
                "bid_iv",
                "ask_iv",
                "best_bid_price",
                "best_ask_price",
                "delta",
                "gamma",
                "vega",
                "theta",
                "tau",
                "DTE",
            ]
        )
        .head(n)
    )
    return lf.collect().to_pandas()


def audit4_price_convention(state: AuditState) -> None:
    if state.iv_scale is None:
        raise AssertionError("Audit 4 requires Audit 3 IV scale")
    config = state.config
    rows = []
    floor_rows = []
    for currency in ["BTC", "ETH"]:
        df = collect_pricing_sample(config, currency, config.pricing_sample_per_currency)
        sigma = df["mark_iv"].to_numpy(dtype=float) / state.iv_scale
        coin = vector_inverse_coin_price(
            df["claim_type"].to_numpy(),
            df["underlyer_spot"].to_numpy(dtype=float),
            df["forward_price"].to_numpy(dtype=float),
            df["strike"].to_numpy(dtype=float),
            df["tau"].to_numpy(dtype=float),
            sigma,
        )
        usd = coin * df["underlyer_spot"].to_numpy(dtype=float)
        mark = df["mark_price"].to_numpy(dtype=float)
        for convention, model in [("base_currency", coin), ("usd", usd)]:
            err = rel_error(model, mark)
            abs_err = np.abs(model - mark)
            rows.append(
                {
                    "currency": currency,
                    "convention": convention,
                    "sample_rows": len(df),
                    "median_relative_error": float(np.nanmedian(err)),
                    "p95_relative_error": float(np.nanpercentile(err, 95)),
                    "p99_relative_error": float(np.nanpercentile(err, 99)),
                }
            )
            for floor in [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
                mask = mark >= floor
                if not np.any(mask):
                    continue
                floor_rows.append(
                    {
                        "currency": currency,
                        "convention": convention,
                        "mark_price_floor": floor,
                        "sample_rows": int(mask.sum()),
                        "median_relative_error": float(np.nanmedian(err[mask])),
                        "p95_relative_error": float(np.nanpercentile(err[mask], 95)),
                        "p99_relative_error": float(np.nanpercentile(err[mask], 99)),
                        "p95_absolute_error": float(np.nanpercentile(abs_err[mask], 95)),
                    }
                )
    out = pd.DataFrame(rows)
    path = config.tables_dir / "audit4_price_convention.csv"
    write_csv(out, path)
    floor_path = config.tables_dir / "audit4_price_convention_by_floor.csv"
    write_csv(pd.DataFrame(floor_rows), floor_path)
    base_med = out.loc[out["convention"] == "base_currency", "median_relative_error"].median()
    usd_med = out.loc[out["convention"] == "usd", "median_relative_error"].median()
    floors = pd.DataFrame(floor_rows)
    base_floor_p95 = floors[
        (floors["convention"] == "base_currency") & (floors["mark_price_floor"] == 0.0005)
    ]["p95_relative_error"].median()
    state.price_convention = "base_currency" if base_med < usd_med else "usd"
    state.record("Audit 4", "price_convention", state.price_convention, state.price_convention == "base_currency")
    state.record("Audit 4", "base_median_error", base_med, base_med < 0.005)
    state.record("Audit 4", "base_p95_error_mark_floor_0.0005", base_floor_p95, base_floor_p95 < 0.05)
    self_check_audit4(path)


def self_check_audit4(path: Path) -> None:
    self_check_table(path, min_rows=4)
    df = pd.read_csv(path)
    base = df[df["convention"] == "base_currency"]["median_relative_error"].median()
    usd = df[df["convention"] == "usd"]["median_relative_error"].median()
    if not (base < usd and base < 0.005):
        raise AssertionError(f"Audit 4 failed price convention check: base={base}, usd={usd}")
    floor_path = path.parent / "audit4_price_convention_by_floor.csv"
    floors = pd.read_csv(floor_path)
    base_floor_p95 = floors[
        (floors["convention"] == "base_currency") & (floors["mark_price_floor"] == 0.0005)
    ]["p95_relative_error"].median()
    if not base_floor_p95 < 0.05:
        raise AssertionError(f"Audit 4 liquid-price p95 reconstruction error too high: {base_floor_p95}")


def audit5_price_box(state: AuditState) -> None:
    if state.iv_scale is None or state.price_convention is None:
        raise AssertionError("Audit 5 requires Audits 3-4")
    config = state.config
    frames = []
    summary = []
    for currency in ["BTC", "ETH"]:
        df = collect_pricing_sample(config, currency, 10_000_000)
        for iv_col, out_col in [("bid_iv", "bid_price_iv"), ("ask_iv", "ask_price_iv"), ("mark_iv", "mark_price_rebuilt")]:
            sigma = df[iv_col].to_numpy(dtype=float) / state.iv_scale
            coin = vector_inverse_coin_price(
                df["claim_type"].to_numpy(),
                df["underlyer_spot"].to_numpy(dtype=float),
                df["forward_price"].to_numpy(dtype=float),
                df["strike"].to_numpy(dtype=float),
                df["tau"].to_numpy(dtype=float),
                sigma,
            )
            df[out_col] = coin if state.price_convention == "base_currency" else coin * df["underlyer_spot"].to_numpy(dtype=float)
        df["bid_price"] = df["best_bid_price"].astype(float)
        df["ask_price"] = df["best_ask_price"].astype(float)
        df["relspread"] = (df["ask_price"] - df["bid_price"]) / (df["mark_price"].abs() + 1e-12)
        slice_keys = ["currency", "date", "expiration_date"]
        df["common_forward_price"] = df.groupby(slice_keys)["forward_price"].transform("median")
        df["common_underlyer_spot"] = df.groupby(slice_keys)["underlyer_spot"].transform("median")
        df["k"] = np.log(df["strike"].astype(float) / df["common_forward_price"].astype(float))
        df["dte_bucket"] = df["DTE"].map(classify_dte_bucket)
        df["bid_iv_ask_iv_available"] = (df["bid_iv"] > 0) & (df["ask_iv"] > 0)
        df["iv_box_ordered"] = df["bid_price_iv"] <= df["ask_price_iv"]
        df["explicit_box_ordered"] = df["bid_price"] <= df["ask_price"]
        df["usable_price_box"] = df["explicit_box_ordered"] & (df["ask_price"] > 0)
        df["mark_inside_iv_box"] = (df["mark_price"] >= df["bid_price_iv"] - 1e-12) & (
            df["mark_price"] <= df["ask_price_iv"] + 1e-12
        )
        df["mark_inside_explicit_box"] = (df["mark_price"] >= df["bid_price"] - 1e-12) & (
            df["mark_price"] <= df["ask_price"] + 1e-12
        )
        summary.append(
            {
                "currency": currency,
                "rows": len(df),
                "both_bid_ask_iv_available_rate": df["bid_iv_ask_iv_available"].mean(),
                "bid_iv_gt_ask_iv_rate": (df["bid_iv"] > df["ask_iv"]).mean(),
                "iv_bid_price_gt_ask_price_rate": (~df["iv_box_ordered"]).mean(),
                "explicit_bid_price_gt_ask_price_rate": (~df["explicit_box_ordered"]).mean(),
                "usable_explicit_price_box_rate": df["usable_price_box"].mean(),
                "mark_inside_iv_box_rate": df["mark_inside_iv_box"].mean(),
                "mark_inside_explicit_box_rate": df["mark_inside_explicit_box"].mean(),
                "median_relative_spread": df["relspread"].replace([np.inf, -np.inf], np.nan).median(),
                "p90_relative_spread": df["relspread"].replace([np.inf, -np.inf], np.nan).quantile(0.90),
                "p99_relative_spread": df["relspread"].replace([np.inf, -np.inf], np.nan).quantile(0.99),
            }
        )
        keep_cols = [
            "currency",
            "date",
            "expiration_date",
            "instrument_name",
            "claim_type",
            "strike",
            "forward_price",
            "common_forward_price",
            "underlyer_spot",
            "common_underlyer_spot",
            "tau",
            "DTE",
            "dte_bucket",
            "k",
            "bid_price",
            "ask_price",
            "mark_price",
            "bid_price_iv",
            "ask_price_iv",
            "mark_price_rebuilt",
            "bid_iv",
            "ask_iv",
            "mark_iv",
            "relspread",
            "usable_price_box",
            "delta",
            "gamma",
            "vega",
            "theta",
        ]
        frames.append(df[keep_cols])
    price_box = pd.concat(frames, ignore_index=True)
    price_path = config.output_root / "price_box.parquet"
    price_box.to_parquet(price_path, index=False)
    summary_df = pd.DataFrame(summary)
    write_csv(summary_df, config.tables_dir / "audit5_price_box.csv")
    state.record("Audit 5", "price_box_rows", len(price_box), len(price_box) > 100_000)
    self_check_audit5(price_path, config.tables_dir / "audit5_price_box.csv")


def self_check_audit5(price_path: Path, summary_path: Path) -> None:
    if not price_path.exists():
        raise AssertionError("Audit 5 did not write price_box.parquet")
    df = pd.read_csv(summary_path)
    if not (df["usable_explicit_price_box_rate"] > 0.80).all():
        raise AssertionError("Audit 5 explicit bid/ask boxes are mostly unusable")


def load_price_box(config: AuditConfig) -> pd.DataFrame:
    return pd.read_parquet(config.output_root / "price_box.parquet")


def audit6_put_call_parity(state: AuditState) -> None:
    config = state.config
    df = load_price_box(config)
    df = df[df["usable_price_box"]].copy()
    calls = df[df["claim_type"].str.lower() == "call"].copy()
    puts = df[df["claim_type"].str.lower() == "put"].copy()
    keys = ["currency", "date", "expiration_date", "strike"]
    pairs = calls.merge(puts, on=keys, suffixes=("_call", "_put"), how="inner")
    parity = 1.0 - pairs["strike"] / pairs["common_forward_price_call"]
    residual = pairs["mark_price_call"] - pairs["mark_price_put"] - parity
    spread = (pairs["ask_price_call"] - pairs["bid_price_call"]) + (
        pairs["ask_price_put"] - pairs["bid_price_put"]
    )
    pairs["pcp_abs_residual"] = np.abs(residual)
    pairs["pcp_scaled_residual"] = pairs["pcp_abs_residual"] / (spread.abs() + 1e-12)
    out_rows = []
    for currency, g in pairs.groupby("currency"):
        out_rows.append(
            {
                "currency": currency,
                "matched_pairs": len(g),
                "matched_pair_coverage_vs_calls": len(g) / max((calls["currency"] == currency).sum(), 1),
                "median_abs_residual": g["pcp_abs_residual"].median(),
                "p90_abs_residual": g["pcp_abs_residual"].quantile(0.90),
                "p99_abs_residual": g["pcp_abs_residual"].quantile(0.99),
                "median_scaled_residual": g["pcp_scaled_residual"].replace([np.inf, -np.inf], np.nan).median(),
                "p90_scaled_residual": g["pcp_scaled_residual"].replace([np.inf, -np.inf], np.nan).quantile(0.90),
                "p99_scaled_residual": g["pcp_scaled_residual"].replace([np.inf, -np.inf], np.nan).quantile(0.99),
            }
        )
    out = pd.DataFrame(out_rows)
    write_csv(out, config.tables_dir / "audit6_put_call_parity.csv")
    pairs[
        keys
        + [
            "instrument_name_call",
            "instrument_name_put",
            "pcp_abs_residual",
            "pcp_scaled_residual",
        ]
    ].to_parquet(config.parquet_dir / "audit6_pcp_pairs.parquet", index=False)
    state.record("Audit 6", "pcp_pairs", len(pairs), len(pairs) > 10_000)
    self_check_audit6(config.tables_dir / "audit6_put_call_parity.csv")


def self_check_audit6(path: Path) -> None:
    self_check_table(path, min_rows=2)
    df = pd.read_csv(path)
    if not (df["matched_pair_coverage_vs_calls"] > 0.5).all():
        raise AssertionError("Audit 6 has poor call-put match coverage")


def build_synthetic_call_panel(price: pd.DataFrame) -> pd.DataFrame:
    df = price.copy()
    df = df[df["usable_price_box"]].copy()
    forward_col = "common_forward_price" if "common_forward_price" in df.columns else "forward_price"
    spot_col = "common_underlyer_spot" if "common_underlyer_spot" in df.columns else "underlyer_spot"
    calls = df[(df["claim_type"].str.lower() == "call") & (df["strike"] >= df[forward_col])].copy()
    calls["C_bid"] = calls["bid_price"]
    calls["C_ask"] = calls["ask_price"]
    calls["source"] = "call"
    puts = df[(df["claim_type"].str.lower() == "put") & (df["strike"] < df[forward_col])].copy()
    parity = 1.0 - puts["strike"] / puts[forward_col]
    puts["C_bid"] = puts["bid_price"] + parity
    puts["C_ask"] = puts["ask_price"] + parity
    puts["source"] = "synthetic_from_put"
    panel = pd.concat([calls, puts], ignore_index=True)
    panel["forward_price"] = panel[forward_col]
    panel["underlyer_spot"] = panel[spot_col]
    panel["k"] = np.log(panel["strike"].astype(float) / panel["forward_price"].astype(float))
    panel["C_mid"] = 0.5 * (panel["C_bid"] + panel["C_ask"])
    panel["intrinsic"] = np.maximum(1.0 - panel["strike"] / panel["forward_price"], 0.0)
    panel["synthetic_valid"] = (
        (panel["C_bid"] <= panel["C_ask"] + 1e-12)
        & (panel["C_mid"] >= panel["intrinsic"] - 1e-8)
        & (panel["C_mid"] <= 1.0 + 1e-8)
    )
    panel.sort_values(["currency", "date", "expiration_date", "strike", "source"], inplace=True)
    if panel.duplicated(["currency", "date", "expiration_date", "strike"]).any():
        agg_spec = {
            "forward_price": "median",
            "underlyer_spot": "median",
            "tau": "median",
            "DTE": "median",
            "dte_bucket": "first",
            "k": "median",
            "C_bid": "median",
            "C_ask": "median",
            "C_mid": "median",
            "intrinsic": "median",
            "source": lambda s: ";".join(sorted(set(map(str, s)))),
            "instrument_name": lambda s: ";".join(sorted(set(map(str, s)))),
            "synthetic_valid": "all",
            "relspread": "median",
        }
        panel = (
            panel.groupby(["currency", "date", "expiration_date", "strike"], as_index=False)
            .agg(agg_spec)
            .sort_values(["currency", "date", "expiration_date", "strike"])
        )
    return panel


def audit7_synthetic_call_panel(state: AuditState) -> None:
    config = state.config
    df = load_price_box(config)
    panel = build_synthetic_call_panel(df)
    keep_cols = [
        "currency",
        "date",
        "expiration_date",
        "strike",
        "forward_price",
        "underlyer_spot",
        "tau",
        "DTE",
        "dte_bucket",
        "k",
        "C_bid",
        "C_ask",
        "C_mid",
        "intrinsic",
        "source",
        "instrument_name",
        "synthetic_valid",
        "relspread",
    ]
    panel[keep_cols].to_parquet(config.output_root / "synthetic_call_panel.parquet", index=False)
    summary = panel.groupby(["currency", "source"]).agg(rows=("source", "size"), valid_rate=("synthetic_valid", "mean")).reset_index()
    write_csv(summary, config.tables_dir / "audit7_synthetic_call_panel.csv")
    state.record("Audit 7", "synthetic_call_rows", len(panel), len(panel) > 50_000)
    self_check_audit7(config.output_root / "synthetic_call_panel.parquet")


def self_check_audit7(path: Path) -> None:
    if not path.exists():
        raise AssertionError("Audit 7 did not write synthetic_call_panel.parquet")
    sample = pd.read_parquet(path, columns=["synthetic_valid"])
    if sample["synthetic_valid"].mean() < 0.95:
        raise AssertionError("Audit 7 synthetic call panel has too many invalid rows")


def load_synthetic(config: AuditConfig) -> pd.DataFrame:
    return pd.read_parquet(config.output_root / "synthetic_call_panel.parquet")


def audit8_strike_grid_coverage(state: AuditState) -> None:
    config = state.config
    df = load_synthetic(config)
    df = df[df["synthetic_valid"]].copy()
    df.sort_values(["currency", "date", "expiration_date", "strike"], inplace=True)
    df["h"] = df.groupby(["currency", "date", "expiration_date"])["k"].diff().abs()
    groups = df.groupby(["currency", "date", "expiration_date", "dte_bucket"]).agg(
        n_strikes=("strike", "nunique"),
        min_k=("k", "min"),
        max_k=("k", "max"),
        median_h=("h", "median"),
        DTE=("DTE", "median"),
    )
    groups = groups.reset_index()
    groups.to_parquet(config.parquet_dir / "audit8_slice_coverage.parquet", index=False)
    summary = (
        groups.groupby(["currency", "dte_bucket"])
        .agg(
            slice_count=("n_strikes", "size"),
            median_strikes=("n_strikes", "median"),
            p10_strikes=("n_strikes", lambda s: s.quantile(0.10)),
            slices_ge_7=("n_strikes", lambda s: (s >= 7).sum()),
            median_h=("median_h", "median"),
        )
        .reset_index()
    )
    write_csv(summary, config.tables_dir / "audit8_strike_grid_coverage.csv")
    state.record("Audit 8", "slices_ge_7", int((groups["n_strikes"] >= 7).sum()), (groups["n_strikes"] >= 7).mean() > 0.8)
    self_check_audit8(config.tables_dir / "audit8_strike_grid_coverage.csv")


def self_check_audit8(path: Path) -> None:
    self_check_table(path, min_rows=4)
    df = pd.read_csv(path)
    main = df[df["dte_bucket"].isin(["2-7", "8-14", "15-30", "31-45"])]
    if not (main["median_strikes"] >= 7).all():
        raise AssertionError("Audit 8 main DTE buckets do not have enough strikes")


def audit9_raw_noarb(state: AuditState) -> None:
    config = state.config
    df = load_synthetic(config)
    df = df[df["synthetic_valid"]].copy()
    rows = []
    for key, g in df.groupby(["currency", "date", "expiration_date"], sort=False):
        g = g.sort_values("strike")
        strikes = g["strike"].to_numpy(dtype=float)
        c_mid = g["C_mid"].to_numpy(dtype=float)
        spreads = (g["C_ask"] - g["C_bid"]).to_numpy(dtype=float)
        intrinsic = g["intrinsic"].to_numpy(dtype=float)
        mono = monotonicity_violations(c_mid)
        conv = convexity_violations(strikes, c_mid)
        rows.append(
            {
                "currency": key[0],
                "date": key[1],
                "expiration_date": key[2],
                "n_strikes": len(g),
                "DTE": float(g["DTE"].median()),
                "dte_bucket": classify_dte_bucket(float(g["DTE"].median())),
                "mono_violation_rate": float((mono > 1e-10).mean()) if len(mono) else 0.0,
                "convex_violation_rate": float((conv > 1e-10).mean()) if len(conv) else 0.0,
                "mono_pressure": float(mono.sum() / (np.nansum(spreads) + EPS)),
                "conv_pressure": float(conv.sum()),
                "intrinsic_violation_rate": float((c_mid < intrinsic - 1e-10).mean()),
                "upper_violation_rate": float((c_mid > 1.0 + 1e-10).mean()),
                "median_relspread": float(np.nanmedian(g["relspread"].to_numpy(dtype=float))),
                "median_abs_k": float(np.nanmedian(np.abs(g["k"].to_numpy(dtype=float)))),
                "strict_feasible": np.nan,
                "elastic_slack": np.nan,
                "feasibility_checked": False,
            }
        )
    noarb = pd.DataFrame(rows)
    noarb.to_parquet(config.output_root / "noarb_audit.parquet", index=False)
    summary = (
        noarb.groupby(["currency", "dte_bucket"])
        .agg(
            slice_count=("n_strikes", "size"),
            median_strikes=("n_strikes", "median"),
            mono_violation_rate=("mono_violation_rate", "mean"),
            convex_violation_rate=("convex_violation_rate", "mean"),
            intrinsic_violation_rate=("intrinsic_violation_rate", "mean"),
            upper_violation_rate=("upper_violation_rate", "mean"),
            median_mono_pressure=("mono_pressure", "median"),
            median_conv_pressure=("conv_pressure", "median"),
        )
        .reset_index()
    )
    write_csv(summary, config.tables_dir / "audit9_raw_noarb.csv")
    state.record("Audit 9", "noarb_slices", len(noarb), len(noarb) > 1_000)
    self_check_audit9(config.output_root / "noarb_audit.parquet")


def self_check_audit9(path: Path) -> None:
    if not path.exists():
        raise AssertionError("Audit 9 did not write noarb_audit.parquet")
    df = pd.read_parquet(path, columns=["n_strikes"])
    if len(df) == 0 or df["n_strikes"].median() < 7:
        raise AssertionError("Audit 9 has too few strikes per slice")


def audit10_box_feasibility(state: AuditState) -> None:
    config = state.config
    synthetic = load_synthetic(config)
    synthetic = synthetic[synthetic["synthetic_valid"]].copy()
    noarb = pd.read_parquet(config.output_root / "noarb_audit.parquet")
    # Raw-mid no-arb and bounds imply strict feasibility because the mid lies inside the box.
    raw_feasible = (
        (noarb["mono_violation_rate"] <= 1e-12)
        & (noarb["convex_violation_rate"] <= 1e-12)
        & (noarb["intrinsic_violation_rate"] <= 1e-12)
        & (noarb["upper_violation_rate"] <= 1e-12)
    )
    noarb.loc[raw_feasible, "strict_feasible"] = True
    noarb.loc[raw_feasible, "elastic_slack"] = 0.0
    noarb.loc[raw_feasible, "feasibility_checked"] = True

    needs_lp = noarb[~raw_feasible].copy()
    checked_parts = []
    for currency, g in needs_lp.groupby("currency"):
        checked_parts.append(g.head(config.feasibility_max_lp_per_currency))
    to_check = pd.concat(checked_parts, ignore_index=True) if checked_parts else pd.DataFrame()
    lookup = {
        (row.currency, row.date, row.expiration_date): idx
        for idx, row in noarb.iterrows()
    }
    for row in to_check.itertuples(index=False):
        g = synthetic[
            (synthetic["currency"] == row.currency)
            & (synthetic["date"] == row.date)
            & (synthetic["expiration_date"] == row.expiration_date)
        ].sort_values("strike")
        result = elastic_monotone_convex_slack(
            g["strike"].to_numpy(dtype=float),
            g["C_bid"].to_numpy(dtype=float),
            g["C_ask"].to_numpy(dtype=float),
            g["intrinsic"].to_numpy(dtype=float),
        )
        idx = lookup[(row.currency, row.date, row.expiration_date)]
        noarb.loc[idx, "strict_feasible"] = result.strict_feasible
        noarb.loc[idx, "elastic_slack"] = result.elastic_slack
        noarb.loc[idx, "feasibility_checked"] = True
    noarb.to_parquet(config.output_root / "noarb_audit.parquet", index=False)
    summary = (
        noarb[noarb["feasibility_checked"] == True]
        .groupby(["currency", "dte_bucket"])
        .agg(
            checked_slices=("strict_feasible", "size"),
            strict_feasible_rate=("strict_feasible", "mean"),
            median_elastic_slack=("elastic_slack", "median"),
            p90_elastic_slack=("elastic_slack", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )
    write_csv(summary, config.tables_dir / "audit10_box_feasibility.csv")
    state.record("Audit 10", "feasibility_checked_slices", int(noarb["feasibility_checked"].sum()), noarb["feasibility_checked"].sum() > 1_000)
    self_check_audit10(config.tables_dir / "audit10_box_feasibility.csv")


def self_check_audit10(path: Path) -> None:
    self_check_table(path, min_rows=2)
    df = pd.read_csv(path)
    if not (df["checked_slices"] > 0).all():
        raise AssertionError("Audit 10 did not check feasibility slices")


def audit11_binding_regimes(state: AuditState) -> None:
    config = state.config
    synthetic = load_synthetic(config)
    noarb = pd.read_parquet(config.output_root / "noarb_audit.parquet")
    synthetic = synthetic[synthetic["synthetic_valid"]].copy()
    synthetic.sort_values(["currency", "date", "expiration_date", "strike"], inplace=True)
    synthetic["h"] = synthetic.groupby(["currency", "date", "expiration_date"])["k"].diff().abs()
    synthetic["abs_k"] = synthetic["k"].abs()
    synthetic["short_dte"] = synthetic["DTE"].between(2, 7)
    synthetic["wing_005"] = synthetic["abs_k"] >= 0.05
    synthetic["wing_010"] = synthetic["abs_k"] >= 0.10
    out_rows = []
    for currency, g in synthetic.groupby("currency"):
        spread_q75 = g["relspread"].replace([np.inf, -np.inf], np.nan).quantile(0.75)
        h_q75 = g["h"].replace([np.inf, -np.inf], np.nan).quantile(0.75)
        gg = g.copy()
        gg["wide_spread"] = gg["relspread"] >= spread_q75
        gg["sparse_strike"] = gg["h"] >= h_q75
        gg["cell"] = (
            "short="
            + gg["short_dte"].astype(str)
            + "|wing05="
            + gg["wing_005"].astype(str)
            + "|wide="
            + gg["wide_spread"].astype(str)
            + "|sparse="
            + gg["sparse_strike"].astype(str)
        )
        merged = gg.merge(
            noarb[["currency", "date", "expiration_date", "mono_pressure", "conv_pressure"]],
            on=["currency", "date", "expiration_date"],
            how="left",
        )
        cell = (
            merged.groupby(["currency", "cell"])
            .agg(
                observations=("cell", "size"),
                median_spread=("relspread", "median"),
                median_h=("h", "median"),
                median_abs_k=("abs_k", "median"),
                median_mono_pressure=("mono_pressure", "median"),
                median_conv_pressure=("conv_pressure", "median"),
            )
            .reset_index()
        )
        out_rows.append(cell)
    out = pd.concat(out_rows, ignore_index=True)
    write_csv(out, config.tables_dir / "audit11_binding_regimes.csv")
    state.record("Audit 11", "regime_cells", len(out), len(out) >= 16)
    self_check_audit11(config.tables_dir / "audit11_binding_regimes.csv")


def self_check_audit11(path: Path) -> None:
    self_check_table(path, min_rows=8)


def audit12_next_day_continuity(state: AuditState) -> None:
    config = state.config
    lf = add_time_columns(scan_all(config)).filter(
        (pl.col("hour_utc") == config.main_snapshot_hour)
    )
    keys = lf.select(["currency", "date", "instrument_name"]).unique()
    current = (
        lf.filter(pl.col("DTE").is_between(config.main_dte_min, config.main_dte_max))
        .select(["currency", "date", "instrument_name", "DTE"])
        .with_columns((pl.col("date") + 86_400).alias("next_date"))
    )
    next_keys = keys.rename({"date": "next_date"}).with_columns(pl.lit(1).alias("next_available"))
    joined = current.join(next_keys, on=["currency", "next_date", "instrument_name"], how="left")
    out = (
        joined.with_columns(
            [
                pl.col("next_available").is_not_null().alias("next_available_bool"),
                pl.col("DTE")
                .map_elements(classify_dte_bucket, return_dtype=pl.String)
                .alias("dte_bucket"),
            ]
        )
        .group_by(["currency", "dte_bucket"])
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("next_available_bool").mean().alias("next_day_available_rate"),
            ]
        )
        .sort(["currency", "dte_bucket"])
        .collect()
    )
    path = config.tables_dir / "audit12_next_day_continuity.csv"
    write_csv(out, path)
    state.record("Audit 12", "nextday_rows", int(out["rows"].sum()), int(out["rows"].sum()) > 100_000)
    self_check_audit12(path)


def self_check_audit12(path: Path) -> None:
    self_check_table(path, min_rows=4)
    df = pd.read_csv(path)
    main = df[df["dte_bucket"].isin(["2-7", "8-14", "15-30", "31-45"])]
    if main["next_day_available_rate"].median() < 0.8:
        raise AssertionError("Audit 12 next-day continuity is too low")


def nearest_by_claim(g: pd.DataFrame, claim: str, target_k: float) -> tuple[bool, float, float, str]:
    sub = g[g["claim_type"].str.lower() == claim]
    if sub.empty:
        return False, float("nan"), float("nan"), ""
    idx = (sub["k"] - target_k).abs().idxmin()
    row = sub.loc[idx]
    return True, float(abs(row["k"] - target_k)), float(row["strike"]), str(row["instrument_name"])


def default_book_specs() -> dict[str, list[tuple[str, float]]]:
    return {
        "ATM_straddle": [("call", 0.0), ("put", 0.0)],
        "downside_put_spread": [("put", -0.05), ("put", -0.10)],
        "risk_reversal": [("put", -0.05), ("call", 0.05)],
        "wing_butterfly": [("put", -0.10), ("call", 0.0), ("call", 0.10)],
    }


def main_hour_instrument_keys(config: AuditConfig) -> set[tuple[str, int, str]]:
    lf = add_time_columns(scan_all(config)).filter(
        pl.col("hour_utc") == config.main_snapshot_hour
    )
    keys = lf.select(["currency", "date", "instrument_name"]).unique().collect().to_pandas()
    return set(keys.itertuples(index=False, name=None))


def compute_book_coverage(
    price: pd.DataFrame,
    config: AuditConfig,
    next_available_keys: set[tuple[str, int, str]],
) -> pd.DataFrame:
    if "usable_price_box" in price.columns:
        price = price[price["usable_price_box"]].copy()
    else:
        price = price.copy()
    rows = []
    specs = default_book_specs()
    for key, g in price.groupby(["currency", "date", "expiration_date"], sort=False):
        for book, legs in specs.items():
            dists = []
            strikes = []
            instruments = []
            found_all = True
            within_tolerance = True
            for claim, target in legs:
                found, dist, strike, instrument = nearest_by_claim(g, claim, target)
                found_all = found_all and found
                within_tolerance = within_tolerance and found and dist <= config.book_tolerance
                dists.append(dist)
                strikes.append(strike)
                instruments.append(instrument)
            constructible = bool(found_all and within_tolerance)
            next_date = int(key[1]) + 86_400
            if constructible:
                nextday_available = all((key[0], next_date, inst) in next_available_keys for inst in instruments)
            else:
                nextday_available = False
            rows.append(
                {
                    "currency": key[0],
                    "date": key[1],
                    "expiration_date": key[2],
                    "DTE": float(g["DTE"].median()),
                    "dte_bucket": classify_dte_bucket(float(g["DTE"].median())),
                    "book": book,
                    "constructible": constructible,
                    "nextday_available": bool(nextday_available),
                    "nextday_ready": bool(constructible and nextday_available),
                    "max_strike_distance": float(np.nanmax(dists)),
                    "median_strike_distance": float(np.nanmedian(dists)),
                    "selected_strikes": ";".join(map(lambda x: "" if math.isnan(x) else f"{x:.8g}", strikes)),
                    "selected_instruments": ";".join(instruments),
                }
            )
    return pd.DataFrame(rows)


def audit13_book_coverage(state: AuditState) -> None:
    config = state.config
    price = load_price_box(config)
    next_keys = main_hour_instrument_keys(config)
    out = compute_book_coverage(price, config, next_keys)
    out.to_parquet(config.output_root / "book_coverage.parquet", index=False)
    summary = (
        out.groupby(["currency", "book"])
        .agg(
            slices=("book", "size"),
            constructible_rate=("constructible", "mean"),
            nextday_available_rate=("nextday_available", "mean"),
            nextday_ready_rate=("nextday_ready", "mean"),
            median_strike_distance=("median_strike_distance", "median"),
            p90_strike_distance=("max_strike_distance", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )
    write_csv(summary, config.tables_dir / "audit13_book_coverage.csv")
    state.record("Audit 13", "book_rows", len(out), len(out) > 1_000)
    self_check_audit13(config.output_root / "book_coverage.parquet")


def self_check_audit13(path: Path) -> None:
    if not path.exists():
        raise AssertionError("Audit 13 did not write book_coverage.parquet")
    df = pd.read_parquet(path, columns=["constructible", "nextday_available", "nextday_ready"])
    if len(df) == 0:
        raise AssertionError("Audit 13 empty book coverage")
    if "nextday_available_placeholder" in pd.read_parquet(path).columns:
        raise AssertionError("Audit 13 still contains placeholder next-day availability")


def focused_greek_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["usable_price_box"])
        & (df["mark_price"] >= 0.0005)
        & (df["DTE"].between(2.0, 45.0))
        & (df["k"].abs() <= 0.03)
        & (df["relspread"] >= 0)
        & (df["relspread"] <= 0.25)
    ].copy()


def controlled_greek_slice_details(df: pd.DataFrame, iv_scale: float) -> pd.DataFrame:
    focused = focused_greek_sample(df)
    if focused.empty:
        return pd.DataFrame()
    groups = (
        focused.groupby(["currency", "date", "expiration_date"])
        .agg(
            sample_rows=("strike", "size"),
            median_abs_k=("k", lambda x: float(np.median(np.abs(x)))),
            median_relspread=("relspread", "median"),
            median_dte=("DTE", "median"),
        )
        .reset_index()
        .sort_values(["sample_rows", "median_relspread", "median_abs_k"], ascending=[False, True, True])
    )
    best = groups.iloc[0]
    sl = focused[
        (focused["date"] == best["date"])
        & (focused["expiration_date"] == best["expiration_date"])
        & (focused["currency"] == best["currency"])
    ].copy()

    sigma = sl["mark_iv"].to_numpy(dtype=float) / iv_scale
    spot = sl["underlyer_spot"].to_numpy(dtype=float)
    forward = sl["forward_price"].to_numpy(dtype=float)
    strike = sl["strike"].to_numpy(dtype=float)
    tau = sl["tau"].to_numpy(dtype=float)
    claim = sl["claim_type"].to_numpy()
    usd_delta, usd_gamma = vector_inverse_usd_greeks(claim, spot, forward, strike, tau, sigma)
    h = np.maximum(spot * 1e-4, 1e-6)
    up_spot = spot + h
    down_spot = np.maximum(spot - h, 1e-12)
    up_forward = forward * up_spot / spot
    down_forward = forward * down_spot / spot
    up_coin = vector_inverse_coin_price(claim, up_spot, up_forward, strike, tau, sigma)
    down_coin = vector_inverse_coin_price(claim, down_spot, down_forward, strike, tau, sigma)
    fd_usd_delta = (up_coin * up_spot - down_coin * down_spot) / (up_spot - down_spot)

    sl["datetime_utc"] = pd.to_datetime(sl["date"], unit="s", utc=True)
    sl["expiration_utc"] = pd.to_datetime(sl["expiration_date"], unit="s", utc=True)
    sl["controlled_slice_rows"] = int(best["sample_rows"])
    sl["usd_delta_model"] = usd_delta
    sl["fd_usd_delta"] = fd_usd_delta
    sl["usd_gamma_model"] = usd_gamma
    sl["delta_abs_diff"] = np.abs(sl["delta"].to_numpy(dtype=float) - usd_delta)
    sl["fd_delta_abs_diff"] = np.abs(sl["delta"].to_numpy(dtype=float) - fd_usd_delta)
    sl["gamma_abs_diff"] = np.abs(sl["gamma"].to_numpy(dtype=float) - usd_gamma)
    keep = [
        "currency",
        "datetime_utc",
        "expiration_utc",
        "controlled_slice_rows",
        "instrument_name",
        "claim_type",
        "strike",
        "DTE",
        "k",
        "relspread",
        "mark_price",
        "mark_iv",
        "delta",
        "usd_delta_model",
        "fd_usd_delta",
        "delta_abs_diff",
        "fd_delta_abs_diff",
        "gamma",
        "usd_gamma_model",
        "gamma_abs_diff",
    ]
    return sl[keep].sort_values(["currency", "strike", "claim_type"])


def audit14_greeks_sanity(state: AuditState) -> None:
    if state.iv_scale is None:
        raise AssertionError("Audit 14 requires IV scale")
    config = state.config
    rows = []
    details = []
    price = load_price_box(config)
    price = price[(price["mark_iv"] > 0) & (price["tau"] > 0)].copy()
    for currency in ["BTC", "ETH"]:
        df = price[price["currency"] == currency].copy()
        samples = {
            "all_main_price_box": df,
            "usable_floor": df[(df["usable_price_box"]) & (df["mark_price"] >= 0.0005)].copy(),
            "focused_atm_liquid": focused_greek_sample(df),
        }
        for sample_name, sample in samples.items():
            stats = greek_convention_stats(sample, state.iv_scale)
            rows.append({"currency": currency, "sample_name": sample_name, **stats})
        controlled = controlled_greek_slice_details(df, state.iv_scale)
        if not controlled.empty:
            details.append(controlled)
    out = pd.DataFrame(rows)
    write_csv(out, config.tables_dir / "audit14_greeks_sanity.csv")
    if details:
        write_csv(pd.concat(details, ignore_index=True), config.tables_dir / "audit14_greek_convention_controlled_slice.csv")
    focused_rows = int(out.loc[out["sample_name"] == "focused_atm_liquid", "sample_rows"].sum())
    state.record("Audit 14", "focused_atm_liquid_greek_rows", focused_rows, focused_rows > 10_000)
    self_check_audit14(config.tables_dir / "audit14_greeks_sanity.csv")


def self_check_audit14(path: Path) -> None:
    self_check_table(path, min_rows=6)
    df = pd.read_csv(path)
    for col in ["call_delta_positive_rate", "put_delta_negative_rate", "gamma_positive_rate", "vega_positive_rate"]:
        if df[col].median() < 0.5:
            raise AssertionError(f"Audit 14 sign sanity failed for {col}")
    focused = df[df["sample_name"] == "focused_atm_liquid"]
    if len(focused) < 2:
        raise AssertionError("Audit 14 did not write focused ATM/liquid Greek checks")
    if (focused["usd_delta_corr_with_deribit_delta"] < 0.99).any():
        raise AssertionError("Audit 14 USD delta convention check failed")
    if (focused["usd_delta_median_abs_diff"] > 0.01).any():
        raise AssertionError("Audit 14 USD delta median absolute difference is too large")
    if (focused["fd_usd_delta_corr_with_deribit_delta"] < 0.99).any():
        raise AssertionError("Audit 14 finite-difference USD delta convention check failed")
    if (focused["usd_gamma_corr_with_deribit_gamma"] < 0.99).any():
        raise AssertionError("Audit 14 USD gamma convention check failed")
    controlled_path = path.parent / "audit14_greek_convention_controlled_slice.csv"
    self_check_table(controlled_path, min_rows=10)


def write_field_units(state: AuditState) -> None:
    config = state.config
    field_units = {
        "iv_unit": "percent" if state.iv_scale == 100.0 else "decimal",
        "iv_scale_divisor": state.iv_scale,
        "price_unit": state.price_convention,
        "snapshot_hour_main": f"{config.main_snapshot_hour:02d}:00 UTC",
        "expiry_time": "08:00 UTC",
        "pricing_formula": "Deribit inverse Black-Scholes, base-currency premium = USD price / spot",
        "forward_field": "forward_price",
        "common_forward_field": "common_forward_price",
        "spot_field": "underlyer_spot",
        "common_spot_field": "common_underlyer_spot",
        "main_sample": f"{config.main_dte_min:g}<DTE<={config.main_dte_max:g}, selected UTC hour only",
        "bid_ask_price_source": "explicit best_bid_price/best_ask_price retained; IV-reconstructed prices audited separately",
        "downstream_curve_forward": "Synthetic calls, parity, moneyness, no-arbitrage, and book coverage use the slice-level median common_forward_price.",
        "greek_convention_sanity": "Deribit delta/gamma are checked against USD-premium spot Greeks with forward/spot carry fixed; coin-price finite differences are convention contrasts only.",
        "snapshot_hour_note": "The analysis uses 12:00 UTC unless 09:00 UTC is specified as a post-settlement anchor.",
    }
    path = config.output_root / "field_units.json"
    path.write_text(json.dumps(field_units, indent=2), encoding="utf-8")


def write_audit_summary(state: AuditState) -> None:
    config = state.config
    summary = pd.DataFrame(state.audit_summary or [])
    write_csv(summary, config.output_root / "audit_summary.csv")


AUDITS: list[tuple[str, Callable[[AuditState], None]]] = [
    ("Audit 0", audit0_integrity),
    ("Audit 1", audit1_time_axis),
    ("Audit 2", audit2_instrument_parsing),
    ("Audit 3", audit3_iv_units),
    ("Audit 4", audit4_price_convention),
    ("Audit 5", audit5_price_box),
    ("Audit 6", audit6_put_call_parity),
    ("Audit 7", audit7_synthetic_call_panel),
    ("Audit 8", audit8_strike_grid_coverage),
    ("Audit 9", audit9_raw_noarb),
    ("Audit 10", audit10_box_feasibility),
    ("Audit 11", audit11_binding_regimes),
    ("Audit 12", audit12_next_day_continuity),
    ("Audit 13", audit13_book_coverage),
    ("Audit 14", audit14_greeks_sanity),
]


def run_all_audits(config: AuditConfig | None = None) -> AuditState:
    config = config or AuditConfig()
    ensure_dirs(config)
    state = AuditState(config=config)
    (config.output_root / "run_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}, indent=2),
        encoding="utf-8",
    )
    log_lines = []
    for name, fn in AUDITS:
        log_lines.append(f"START {name}")
        print(f"START {name}", flush=True)
        fn(state)
        write_audit_summary(state)
        log_lines.append(f"PASS {name}")
        print(f"PASS {name}", flush=True)
    write_field_units(state)
    write_audit_summary(state)
    (config.logs_dir / "audit_run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return state
