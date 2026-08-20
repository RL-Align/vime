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

## Generate an expandable trace

For an interactive, expandable timeline, export Chrome Trace Event JSON:

```bash
python tools/consistency_drift_trace.py \
  /path/to/train_data/12_0.pt /path/to/train_data/12_1.pt \
  --output /path/to/consistency-drift-trace.json
```

Open the generated `.json` file in [Perfetto UI](https://ui.perfetto.dev/).
The process and thread tracks can be expanded, zoomed, and selected to inspect
event arguments. This is a standard trace artifact, not an HTML report.
When the source dump has no timestamps, the trace keeps the explicit
`ordinal_diagnostic` mode and uses sample positions only for relative ordering.

This file is not an `.nsys-rep` file. Nsight Systems report files are generated
by the Nsight collector and are not a public interchange format; the JSON trace
is the portable equivalent for vime consistency diagnostics.

## Open a local expandable report

For day-to-day debugging, package the same report into one offline
`.vime-drift` file. The bundle contains the sanitized report, the expandable
trace, and (unless disabled) the PNG snapshot. It does not copy the original
training or rollout dumps.

Install the optional desktop viewer once:

```bash
pip install "vime[consistency-viewer]"
```

Build and open the bundle locally:

```bash
python tools/consistency_drift_bundle.py \
  /path/to/train_data/12_0.pt /path/to/train_data/12_1.pt \
  --output /path/to/consistency-drift.vime-drift
vime-consistency-view /path/to/consistency-drift.vime-drift
```

The viewer runs as a native desktop window with collapsible audit tracks,
timeline zoom/scroll, selectable events, and an event-details panel. It needs
no browser, local web server, or network connection at viewing time. The PNG
inside the bundle remains the compact artifact for a PR or issue; the bundle is
the richer artifact for interactive diagnosis. The equivalent module command
is also available:

```bash
python -m vime.utils.consistency_drift_viewer /path/to/consistency-drift.vime-drift
```

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
