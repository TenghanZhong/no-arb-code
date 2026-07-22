# Repair-Risk Validation Artifact

Public replication artifact for *Quantifying Repair Risk in Crypto Option
Surfaces*. The repository contains the analysis code, compact source data for
the reported figures, numerical audit summaries, and a parameterized entry
point for the complete rerun.

## What is included

- The single-maturity repair implementation with effective price bounds,
  two-sided adjacent call-spread bounds, and discrete convexity.
- Classical, SVI-target, targeted learned-repair, five-seed, LP-range,
  calendar-screen, Greek, hedge, and VaR-style analysis code.
- Compact source data for Figures 3--6 and the reported summary statistics.
- Vector renderings of Figures 1--6.
- Tests for the constraint matrix, repair pipeline, Greek convention, audit
  accounting, pair scaling, seed design, and reported summaries.

The raw Deribit-derived panel and the approximately 9 GB of intermediate full
rerun files are not included. The raw panel's byte count and SHA-256 digest are
provided in `data/input_panel.sha256` so a local copy can be verified.

## Quick reproduction

Create the environment and rebuild the artifact outputs from the compact data:

```bash
conda env create -f environment.yml
conda activate repair-risk-artifact
python reproduce.py quick
python reproduce.py verify
```

`quick` writes regenerated figures, a compact Table 1 source, and a numerical
summary to `generated/`. Figure 2 is compiled from its standalone TikZ source
when `pdflatex` is available; otherwise the supplied vector rendering is copied
with that fallback recorded in the summary. `verify` also runs the test suite,
checks the reported quantities, and validates path portability.

## Complete rerun

Place the panel anywhere outside or inside the repository and first
inspect the commands without executing them:

```bash
python reproduce.py full --panel /path/to/call_equivalent_quote_panel.parquet --jobs 4 --dry-run
```

Then remove `--dry-run` to execute the complete pipeline. By default the panel
must match `data/input_panel.sha256`; use `--allow-panel-mismatch` only for an
explicit sensitivity run. Full outputs are written to ignored `results/`,
`derived/`, `figures/`, and `logs/` directories. See
`docs/FULL_REPRODUCTION.md` for stages and expected support counts.

## Repository layout

```text
code/                 Full-pipeline implementation
config/               Fixed analysis design and portable path conventions
data/                  Input instructions and canonical panel digest
docs/                  Full-rerun instructions and artifact scope
manifests/             Expected numerical checkpoints
outputs/               Compact reproduction data, audits, and expected figures
scripts/               Quick-output and artifact-verification utilities
tests/                 Unit and integration tests
reproduce.py           quick, verify, and full entry point
```

## Main numerical checkpoints

- Classical median evaluation-level p90 scaled spot-gamma gap: `0.0726`
  versus `0.0039` for the matched control.
- Calendar-month paired excess: positive in `18/18` months.
- LP range comparison: `51,189` observations, Spearman rank correlation
  `0.8722`.
- Five-seed comparison: positive excess in `17/18` calendar months.
- Risk-reversal hedge-notional difference: `18.62` basis points of fixed gross
  spot notional.

Exact machine-readable values are in `outputs/full_rerun_summary.json` and
`manifests/expected_results.json`.

## Artifact contents

This repository contains redistributable analysis inputs and compact outputs.
The raw market-data panel and large intermediate outputs are excluded; its
digest is supplied for input verification. See `docs/ARTIFACT_CONTENTS.md`.
