# Release hygiene

The public artifact contains the executable analysis code, compact aggregate
data, numerical audits, figure sources, and verification tests. It excludes:

- usernames, home-directory paths, and machine inventories;
- API keys, tokens, credentials, and private remote URLs;
- the raw market-data panel and large intermediate results;
- caches, logs, temporary files, and local source-control history.

`python reproduce.py verify` checks the compact outputs, validates checksums,
runs the test suite, and scans text files for credentials and machine-local
paths. Add repository-level licensing and citation metadata before depositing a
versioned archival release.
