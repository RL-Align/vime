# Consistency Drift Report

vime can turn the consistency audit artifacts already present in debug dumps
into a static PNG report. The image is designed for post-training debugging:
it uses a profiler-style track layout, but it reports numerical agreement and
execution provenance rather than GPU kernel duration.

## Generate a report

The command accepts one or more train or rollout debug dumps. Passing several
rank dumps produces one combined view.

```bash
python tools/consistency_drift_report.py \
  /path/to/train_data/12_0.pt /path/to/train_data/12_1.pt \
  --output /path/to/consistency-drift.png
```

The generated PNG is a single shareable image. It has no browser, JavaScript,
font, or network dependency at viewing time. JPEG output is also supported by
using a `.jpg` or `.jpeg` suffix.

## How to read the view

- **Training audit** covers the comparison represented by the result cube.
- **Rollout samples** shows the sample/replay order and the sample metadata
  captured in the dump.
- **Operator / backend** shows the actual backend when provenance is available;
  requested backend is only a fallback label.
- **Drift markers** identify the worst `|dlogp|` token and validation warnings or
  failures. The lower panels keep the selected anomaly, axes, and provenance
  visible in the image itself so the report can be pasted into a PR or issue.

The report status is `PASS`, `WARN`, or `FAIL` and is also written as text, not
only as color. A report with no real timestamps uses the explicit
`ordinal_diagnostic` mode. Its horizontal positions are stable sample ordinals
and must not be interpreted as elapsed time.

This utility is intentionally read-only and does not add an Attention/FFN/logp
mismatch matrix to vime. The matrix and operator contracts remain owned by
RL-Kernel; vime only renders the audit metadata it actually received.
