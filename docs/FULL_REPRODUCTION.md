# Complete reproduction

## Requirements

Use Python 3.8 and the pinned packages in `requirements.txt`. The complete run
is CPU intensive and creates roughly 9 GB of intermediate output. Independent
evaluations are parallelized with `--jobs`; each worker is restricted to one
BLAS thread by the orchestration scripts.

The canonical input panel must match `data/input_panel.sha256`. The panel is a
Parquet file containing the hourly call-equivalent quote panel used by the
repair scripts. It is not included in this anonymous package.

## Stages

`python reproduce.py full --panel PANEL --jobs 4` executes, in order:

1. A full-constraint feasibility preflight.
2. The 108 classical evaluations and aggregation.
3. The 108 SVI-target evaluations and aggregation.
4. The 10 targeted learned-repair evaluations and aggregation.
5. The 180-refit, five-seed replication.
6. The selected two-maturity calendar screen.
7. LP ranges, downstream metrics, audit summaries, figures, and tests.

Use `--dry-run` to print every command and resolved output path before any
computation. Existing complete evaluations are reused unless their individual
runner receives its force/rerun option.

## Portable paths

`reproduce.py` sets these variables for all child processes:

```text
REPAIR_RISK_PANEL
REPAIR_RISK_RESULTS_ROOT
REPAIR_RISK_DERIVED_ROOT
REPAIR_RISK_FIGURES_ROOT
REPAIR_RISK_LOG_ROOT
```

They may also be set directly when running an individual script. No
machine-specific absolute path is required.

## Expected support counts

```text
Classical evaluations                 108
SVI-target evaluations                108
Targeted learned-repair evaluations    10
Targeted projection problems          500
Five-seed refits                       180
LP comparison observations          51,189
Common-scenario VaR-style cells        432
Repair-specific VaR-style cells        417
```

The final gate is `code/working/verify_full_rerun.py`, followed by the full
test suite. Exact reported values are listed in
`manifests/expected_headlines.json`.
