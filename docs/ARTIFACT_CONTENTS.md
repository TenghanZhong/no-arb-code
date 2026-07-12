# Artifact contents

The repository contains executable analysis code, compact aggregate data,
numerical audits, figure sources, and verification tests. It excludes:

- the raw market-data panel and large intermediate results;
- caches, logs, and temporary files.

`python reproduce.py verify` checks the compact outputs, validates checksums,
runs the test suite, and checks that paths remain portable.
