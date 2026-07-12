# Input panel

The raw Deribit-derived panel is not redistributed in this artifact. To run the
complete pipeline, provide the Parquet file to
`reproduce.py full --panel ...`.

The canonical file used for the reported results has:

- file name: `synthetic_call_panel.parquet`
- byte count: `12,900,106`
- SHA-256: `c5805b34ad17bfafff0eb6f1ccde92d75c4434bcca8967155793452b46596499`

The pipeline checks both size and digest before starting. Compact aggregate
data sufficient to regenerate the reported data figures are included under
`outputs/figure_sources/`.
