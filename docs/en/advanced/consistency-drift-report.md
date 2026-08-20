# Consistency Drift Report

vime can turn the consistency audit artifacts already present in debug dumps
into a self-contained HTML report. The view is designed for post-training
debugging: it is organized like a profiler timeline, but it reports numerical
agreement and execution provenance rather than GPU kernel duration.

## Generate a report

The command accepts one or more train or rollout debug dumps. Passing several
rank dumps produces one combined view.

```bash
python tools/consistency_drift_report.py \
  /path/to/train_data/12_0.pt /path/to/train_data/12_1.pt \
  --output /path/to/consistency-drift.html
```

Open the generated HTML directly in a browser. It has no JavaScript package,
font, or network dependency.

## How to read the view

- **Training audit** covers the comparison represented by the result cube.
- **Rollout samples** shows the sample/replay order and lets you inspect the
  sample metadata from the dump.
- **Operator / backend** shows the actual backend when provenance is available;
  requested backend is only a fallback label.
- **Drift markers** identify the worst `|dlogp|` token and validation warnings or
  failures. Selecting any bar or marker opens its details below the timeline.

The report status is `PASS`, `WARN`, or `FAIL` and is also written as text, not
only as color. A report with no real timestamps uses the explicit
`ordinal_diagnostic` mode. Its horizontal positions are stable sample ordinals
and must not be interpreted as elapsed time.

This utility is intentionally read-only and does not add an Attention/FFN/logp
mismatch matrix to vime. The matrix and operator contracts remain owned by
RL-Kernel; vime only renders the audit metadata it actually received.
