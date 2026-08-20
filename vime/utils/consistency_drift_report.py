"""Build self-contained Nsight-style consistency drift reports.

The report is a diagnostic view over the consistency audit artifacts already
emitted by vime.  It intentionally does not collect profiler timestamps or
change the training/rollout path.  When artifacts do not contain timestamps,
the timeline uses stable sample ordinals and labels that mode explicitly.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = 1
_STATUS_LABELS = {"pass": "PASS", "warning": "WARN", "failure": "FAIL", "info": "INFO"}
_STATUS_COLORS = {
    "pass": "#39d98a",
    "warning": "#f4b942",
    "failure": "#ff5c77",
    "info": "#70a7ff",
}


def build_consistency_drift_report(
    *,
    replay_manifest: Mapping[str, Any] | None = None,
    result_cube: Mapping[str, Any] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Normalize existing audit artifacts into a visual diagnostic report."""

    manifest = _plain_mapping(replay_manifest)
    cube = _plain_mapping(result_cube)
    metrics = _plain_mapping(cube.get("metrics"))
    axes = _plain_mapping(cube.get("axes"))
    validation = _plain_mapping(cube.get("metadata_validation") or manifest.get("validation"))
    provenance = _plain_mapping(runtime_provenance or cube.get("runtime_provenance") or manifest.get("runtime_provenance"))
    samples = [_plain_mapping(item) for item in manifest.get("samples", []) if isinstance(item, Mapping)]
    warnings = [_plain_mapping(item) for item in validation.get("warnings", []) if isinstance(item, Mapping)]
    failures = [_plain_mapping(item) for item in validation.get("failures", []) if isinstance(item, Mapping)]

    max_abs_dlogp = _number(metrics.get("max_abs_dlogp"))
    warning_count = _number(metrics.get("warning_count"), default=0.0) or 0.0
    metadata_warning_count = _number(metrics.get("metadata_warning_count"), default=float(len(warnings))) or 0.0
    metadata_failure_count = _number(metrics.get("metadata_failure_count"), default=float(len(failures))) or 0.0
    runtime_fallback = bool(metrics.get("runtime_fallback") or provenance.get("fallback"))
    strict_failure = bool(metrics.get("runtime_strict_failure") or provenance.get("strict_failure"))

    if strict_failure or metadata_failure_count > 0:
        status = "failure"
    elif warning_count > 0 or metadata_warning_count > 0 or runtime_fallback or (max_abs_dlogp or 0.0) > 0.0:
        status = "warning"
    else:
        status = "pass"

    span = max(1.0, float(len(samples)))
    has_timestamps = any(_number(sample.get("start_ts")) is not None for sample in samples)
    timeline_mode = "timestamp" if has_timestamps else "ordinal_diagnostic"
    events: list[dict[str, Any]] = []

    train_status = "failure" if status == "failure" else "warning" if status == "warning" else "pass"
    events.append(
        {
            "id": "train-audit",
            "kind": "bar",
            "lane": "Training audit",
            "start": 0.0,
            "end": span,
            "label": "training-side audit",
            "status": train_status,
            "details": {"mode": manifest.get("mode", cube.get("mode", "unknown")), "rank": cube.get("rank")},
        }
    )
    for position, sample in enumerate(samples):
        sample_label = sample.get("sample_index")
        if sample_label is None:
            sample_label = sample.get("rollout_id")
        if sample_label is None:
            sample_label = position
        start = _number(sample.get("start_ts"), default=float(position))
        end = _number(sample.get("end_ts"), default=float(start or position) + 0.82)
        if end is None or start is None or end <= start:
            start, end = float(position), float(position) + 0.82
        events.append(
            {
                "id": f"rollout-{position}",
                "kind": "bar",
                "lane": "Rollout samples",
                "start": start,
                "end": end,
                "label": f"sample {sample_label}",
                "status": "info",
                "details": sample,
            }
        )

    actual_backend = _first_present(
        provenance.get("actual_backend"),
        provenance.get("backend_id"),
        cube.get("axes", {}).get("logp_backend") if isinstance(cube.get("axes"), Mapping) else None,
        provenance.get("requested_backend"),
        "unknown backend",
    )
    operator_status = "failure" if strict_failure else "warning" if runtime_fallback else "pass"
    events.append(
        {
            "id": "operator-backend",
            "kind": "bar",
            "lane": "Operator / backend",
            "start": 0.12,
            "end": max(0.94, span - 0.12),
            "label": str(actual_backend),
            "status": operator_status,
            "details": provenance,
        }
    )

    worst_token = _plain_mapping(cube.get("worst_token"))
    if worst_token:
        marker_position = _number(worst_token.get("sample_position"), default=max(0.0, span - 0.5)) or 0.0
        events.append(
            {
                "id": "worst-drift",
                "kind": "marker",
                "lane": "Drift markers",
                "start": marker_position + 0.41,
                "end": marker_position + 0.41,
                "label": f"|dlogp| {_format_number(worst_token.get('abs_dlogp'))}",
                "status": "failure" if status == "failure" else "warning" if status == "warning" else "pass",
                "details": worst_token,
            }
        )
    if warnings or failures or warning_count:
        marker_status = "failure" if failures else "warning"
        events.append(
            {
                "id": "validation-marker",
                "kind": "marker",
                "lane": "Drift markers",
                "start": max(0.2, span - 0.22),
                "end": max(0.2, span - 0.22),
                "label": f"{len(failures)} failures / {len(warnings)} warnings",
                "status": marker_status,
                "details": {"warnings": warnings, "failures": failures, "dlogp_warning_count": warning_count},
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "title": title or "Consistency drift report",
        "status": status,
        "status_label": _STATUS_LABELS[status],
        "timeline_mode": timeline_mode,
        "timeline_note": (
            "Real artifact timestamps are shown."
            if timeline_mode == "timestamp"
            else "No artifact timestamps were available; positions are stable sample ordinals, not elapsed time."
        ),
        "lanes": ["Training audit", "Rollout samples", "Operator / backend", "Drift markers"],
        "events": events,
        "axes": axes,
        "metrics": metrics,
        "worst_token": worst_token,
        "validation": validation,
        "runtime_provenance": provenance,
        "sample_count": len(samples),
        "replay_case_count": len(manifest.get("batch_invariance_cases", []) or []),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "cube_fingerprint": cube.get("fingerprint"),
    }


def render_consistency_drift_report(report: Mapping[str, Any]) -> str:
    """Render a report as a self-contained HTML document with an SVG timeline."""

    normalized = _plain_mapping(report)
    events = [_plain_mapping(event) for event in normalized.get("events", [])]
    lanes = [str(lane) for lane in normalized.get("lanes", [])]
    width = 1180
    left = 190
    right = 28
    top = 46
    row_height = 54
    timeline_width = width - left - right
    timeline_span = max(1.0, max((_number(event.get("end"), default=1.0) or 1.0 for event in events), default=1.0))
    svg_height = top + row_height * len(lanes) + 42
    event_map = {str(event.get("id")): event for event in events}

    def x(value: float) -> float:
        return left + max(0.0, min(timeline_span, value)) / timeline_span * timeline_width

    svg_parts = [
        f'<svg class="timeline" viewBox="0 0 {width} {svg_height}" role="img" aria-label="Consistency drift timeline">',
        f'<rect x="0" y="0" width="{width}" height="{svg_height}" rx="8" fill="#111827"/>',
    ]
    grid_steps = min(12, max(2, int(timeline_span) + 1))
    for index in range(grid_steps + 1):
        value = timeline_span * index / grid_steps
        xpos = x(value)
        svg_parts.append(f'<line x1="{xpos:.2f}" y1="{top - 18}" x2="{xpos:.2f}" y2="{svg_height - 38}" class="grid"/>')
        svg_parts.append(f'<text x="{xpos:.2f}" y="{svg_height - 16}" class="axis-label">{html.escape(_format_number(value))}</text>')
    for lane_index, lane in enumerate(lanes):
        ypos = top + lane_index * row_height
        svg_parts.append(f'<text x="18" y="{ypos + 13}" class="lane-label">{html.escape(lane)}</text>')
        svg_parts.append(f'<line x1="{left}" y1="{ypos + 27}" x2="{width - right}" y2="{ypos + 27}" class="lane-line"/>')

    for event in events:
        lane_index = lanes.index(str(event.get("lane"))) if str(event.get("lane")) in lanes else 0
        ypos = top + lane_index * row_height
        status = str(event.get("status", "info"))
        color = _STATUS_COLORS.get(status, _STATUS_COLORS["info"])
        event_id = html.escape(str(event.get("id")), quote=True)
        label = html.escape(_truncate(str(event.get("label", "event")), 28))
        title = html.escape(f"{event.get('label', 'event')} [{_STATUS_LABELS.get(status, status.upper())}]", quote=True)
        if event.get("kind") == "marker":
            xpos = x(_number(event.get("start"), default=0.0) or 0.0)
            points = f"{xpos:.2f},{ypos + 7} {xpos + 9:.2f},{ypos + 16} {xpos:.2f},{ypos + 25} {xpos - 9:.2f},{ypos + 16}"
            svg_parts.append(f'<polygon points="{points}" fill="{color}" class="event" data-event-id="{event_id}" tabindex="0"><title>{title}</title></polygon>')
            svg_parts.append(f'<text x="{xpos + 14:.2f}" y="{ypos + 20}" class="event-label">{label}</text>')
        else:
            start = _number(event.get("start"), default=0.0) or 0.0
            end = _number(event.get("end"), default=start + 0.5) or start + 0.5
            xpos = x(start)
            event_width = max(8.0, x(end) - xpos)
            svg_parts.append(f'<rect x="{xpos:.2f}" y="{ypos + 5}" width="{event_width:.2f}" height="24" rx="4" fill="{color}" fill-opacity="0.78" class="event" data-event-id="{event_id}" tabindex="0"><title>{title}</title></rect>')
            if event_width > 60:
                svg_parts.append(f'<text x="{xpos + 8:.2f}" y="{ypos + 21}" class="event-label event-label-on-bar">{label}</text>')
            else:
                svg_parts.append(f'<text x="{xpos + event_width + 8:.2f}" y="{ypos + 21}" class="event-label">{label}</text>')
    svg_parts.append("</svg>")
    svg = "".join(svg_parts)

    metrics = normalized.get("metrics") if isinstance(normalized.get("metrics"), Mapping) else {}
    metric_cards = [
        ("Max |dlogp|", _format_number(metrics.get("max_abs_dlogp"))),
        ("Active tokens", _format_number(metrics.get("active_token_count"))),
        ("Warnings", _format_number(metrics.get("warning_count"), default="0")),
        ("Replay cases", str(normalized.get("replay_case_count", 0))),
    ]
    cards_html = "".join(f'<div class="metric"><div class="metric-name">{html.escape(name)}</div><div class="metric-value">{html.escape(value)}</div></div>' for name, value in metric_cards)
    axes_html = _render_key_value_table(normalized.get("axes"), empty="No normalized axes were recorded.")
    provenance_html = _render_key_value_table(normalized.get("runtime_provenance"), empty="No runtime provenance was recorded.")
    validation_html = _render_validation(normalized.get("validation"))
    event_json = json.dumps(event_map, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    status = str(normalized.get("status", "info"))
    status_color = _STATUS_COLORS.get(status, _STATUS_COLORS["info"])
    title = html.escape(str(normalized.get("title", "Consistency drift report")))
    timeline_note = html.escape(str(normalized.get("timeline_note", "")))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#151d31; --line:#2a3854; --muted:#96a4bd; --text:#e9eef8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:13px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.shell {{ max-width:1220px; margin:0 auto; padding:28px 20px 42px; }} .top {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; }}
h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }} .subtitle {{ color:var(--muted); max-width:760px; }}
.status {{ border:1px solid {status_color}; color:{status_color}; border-radius:999px; padding:6px 12px; font-weight:700; letter-spacing:.06em; white-space:nowrap; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:24px 0 16px; }} .metric,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
.metric {{ padding:13px 15px; }} .metric-name {{ color:var(--muted); font-size:12px; }} .metric-value {{ font-size:20px; font-weight:700; margin-top:5px; }}
.panel {{ padding:16px; margin-top:16px; }} .panel h2 {{ margin:0 0 12px; font-size:15px; }} .timeline-wrap {{ overflow:auto; }} .timeline {{ min-width:900px; width:100%; height:auto; display:block; }}
.grid {{ stroke:#2c3c5d; stroke-width:1; stroke-dasharray:2 5; }} .lane-line {{ stroke:#25334d; stroke-width:1; }} .lane-label {{ fill:#cbd5e1; font-weight:600; }} .axis-label {{ fill:#8292ad; font-size:11px; text-anchor:middle; }}
.event {{ cursor:pointer; outline:none; }} .event:focus,.event:hover {{ filter:brightness(1.25); stroke:#fff; stroke-width:1.5; }} .event-label {{ fill:#dbe6f7; font-size:11px; pointer-events:none; }} .event-label-on-bar {{ fill:#07111f; font-weight:700; }}
.detail-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; vertical-align:top; padding:7px 8px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:500; width:32%; }} td {{ word-break:break-word; }} .empty {{ color:var(--muted); }} .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; }}
.warning {{ color:#f4b942; }} .failure {{ color:#ff5c77; }} .pass {{ color:#39d98a; }} .info {{ color:#70a7ff; }} pre {{ margin:0; white-space:pre-wrap; color:#cbd5e1; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
@media (max-width:760px) {{ .shell {{ padding:20px 12px 32px; }} .top {{ display:block; }} .status {{ display:inline-block; margin-top:12px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .detail-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main class="shell">
<div class="top"><div><h1>{title}</h1><div class="subtitle">{timeline_note}</div></div><div class="status">{html.escape(_STATUS_LABELS.get(status, status.upper()))}</div></div>
<section class="metrics">{cards_html}</section>
<section class="panel"><h2>Operator drift timeline <span class="pill {status}">{html.escape(str(normalized.get("timeline_mode", "diagnostic")))}</span></h2><div class="timeline-wrap">{svg}</div></section>
<section class="panel"><h2>Selected event</h2><div id="event-detail" class="empty">Select a bar or marker in the timeline.</div></section>
<div class="detail-grid"><section class="panel"><h2>Normalized axes</h2>{axes_html}</section><section class="panel"><h2>Runtime provenance</h2>{provenance_html}</section></div>
<section class="panel"><h2>Validation</h2>{validation_html}</section>
</main><script>
const EVENTS = {event_json};
const detail = document.getElementById('event-detail');
function renderValue(value) {{ if (value === null || value === undefined) return 'null'; if (typeof value === 'object') return JSON.stringify(value, null, 2); return String(value); }}
function selectEvent(id) {{
  const event = EVENTS[id];
  if (!event) return;
  const status = String(event.status || 'info');
  detail.replaceChildren();
  const pill = document.createElement('div');
  pill.className = 'pill ' + status;
  pill.textContent = status.toUpperCase();
  const heading = document.createElement('h3');
  heading.textContent = String(event.label || event.id);
  const pre = document.createElement('pre');
  pre.textContent = renderValue(event.details || {{}});
  detail.append(pill, heading, pre);
}}
document.querySelectorAll('[data-event-id]').forEach((node) => {{ node.addEventListener('click', () => selectEvent(node.dataset.eventId)); node.addEventListener('keydown', (e) => {{ if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); selectEvent(node.dataset.eventId); }} }}); }});
</script></body></html>"""


def write_consistency_drift_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a self-contained HTML report and return its resolved path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_consistency_drift_report(report), encoding="utf-8")
    return output


def _render_key_value_table(values: Any, *, empty: str) -> str:
    if not isinstance(values, Mapping) or not values:
        return f'<div class="empty">{html.escape(empty)}</div>'
    rows = []
    for key in sorted(values, key=str):
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format_value(values[key]))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _render_validation(validation: Any) -> str:
    if not isinstance(validation, Mapping):
        return '<div class="empty">No validation record was captured.</div>'
    warnings = validation.get("warnings") or []
    failures = validation.get("failures") or []
    if not warnings and not failures:
        return '<div class="pass">PASS: no metadata validation warnings or failures.</div>'
    rows = []
    for kind, items in (("failure", failures), ("warning", warnings)):
        for item in items:
            item = item if isinstance(item, Mapping) else {"message": item}
            rows.append(f'<tr><td class="{kind}">{kind.upper()}</td><td>{html.escape(str(item.get("code", "")))}</td><td>{html.escape(str(item.get("message", "")))}</td></tr>')
    return "<table><thead><tr><th>Status</th><th>Code</th><th>Message</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _plain_mapping(value: Any) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()} if isinstance(value, Mapping) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: Any, default: str = "-") -> str:
    number = _number(value)
    if number is None:
        return default
    if abs(number) >= 1000 or (abs(number) < 0.001 and number != 0):
        return f"{number:.3e}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(1, limit - 1)] + "…"
