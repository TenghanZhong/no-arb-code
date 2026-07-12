# Anonymous review checklist

The review export intentionally excludes:

- author names, affiliations, acknowledgments, and citation metadata;
- source-repository history and branches;
- usernames, home-directory paths, and machine inventories;
- API keys, tokens, credentials, and remote data URLs;
- raw licensed/derived market data and large intermediate results;
- paper source files that could carry author metadata.

`python reproduce.py verify` scans text files for common identity and secret
patterns. Before a post-review archival release, add the license, authors,
citation metadata, permanent data-access statement, and archival DOI in a new
public release rather than rewriting this anonymous commit.
