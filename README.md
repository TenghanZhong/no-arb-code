# Repair-Risk Validation Artifact

Anonymous review artifact for *Quantifying Repair Risk in Crypto Option
Surfaces*. The package contains the final analysis code, compact source data
for the reported figures, numerical audit summaries, and a parameterized entry
point for the complete rerun.

## What is included

- The final single-maturity repair implementation with effective price bounds,
  two-sided adjacent call-spread bounds, and discrete convexity.
- Classical, SVI-target, targeted learned-repair, five-seed, LP-range,
  calendar-screen, Greek, hedge, and VaR-style analysis code.
- Compact source data for Figures 3--6 and the final headline summary.
- Pre-rendered copies of Figures 1--6 used for visual comparison.
- Tests for the constraint matrix, repair pipeline, Greek convention, audit
  accounting, pair scaling, seed design, and reported summaries.

The raw Deribit-derived panel and the approximately 9 GB of intermediate full
rerun files are not included. The raw panel's byte count and SHA-256 digest are
provided in `data/input_panel.sha256` so an authorized copy can be verified.

## Quick reproduction

Create the environment and rebuild the review outputs from the compact data:

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
checks the headline quantities, and scans the artifact for secrets and local
identity traces.

## Complete rerun

Place the authorized panel anywhere outside or inside the repository and first
inspect the commands without executing them:

```bash
python reproduce.py full --panel /path/to/synthetic_call_panel.parquet --jobs 4 --dry-run
```

Then remove `--dry-run` to execute the complete pipeline. By default the panel
must match `data/input_panel.sha256`; use `--allow-panel-mismatch` only for an
explicit sensitivity run. Full outputs are written to ignored `results/`,
`derived/`, `figures/`, and `logs/` directories. See
`docs/FULL_REPRODUCTION.md` for stages and expected support counts.

## Repository layout

```text
code/                 Final full-pipeline implementation
config/               Frozen paper design and portable path conventions
data/                  Input instructions and canonical panel digest
docs/                  Full-rerun and anonymity notes
manifests/             Expected numerical checkpoints
outputs/               Compact review data, audits, and expected figures
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
`manifests/expected_headlines.json`.

## Anonymous-review hygiene

This repository is a clean export with one anonymous commit and no source
repository history. It intentionally omits author metadata, acknowledgments,
institutional paths, paper source, credentials, and a citation file during
double-blind review. Those items can be added to a public archival release
after review.
