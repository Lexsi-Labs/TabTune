"""Self-contained HTML rendering for the leaderboard.

The report is a single file with inlined CSS and SVG, no network access and no
JavaScript dependencies. That makes it committable to a repository, attachable
to a pull request, and servable from a static bucket - which is the difference
between a benchmark someone runs once and one a team keeps looking at.

The centrepiece is the quality-versus-latency scatter. For tabular foundation
models the most accurate model and the fastest model are rarely the same one,
and the license column adds a third axis that no accuracy table can express:
the best model on a benchmark may be one you are not permitted to deploy.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..evaluation.metrics import is_higher_better, primary_metric

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .leaderboard import LeaderboardEntry, TabularLeaderboard

__all__ = ["render_leaderboard_html", "render_pareto_svg"]

# Categorical hues chosen for colour-vision-deficiency separation and for
# adequate contrast on both the light and dark surfaces below.
_COMMERCIAL_COLORS = {
    "yes": "#1baf7a",
    "no": "#e34948",
    "unverified": "#eda100",
}

_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --accent: #2a78d6; --good: #1baf7a; --bad: #e34948; --warn: #eda100;
  --chip: rgba(11,11,11,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #0d0d0d;
    --ink: #fff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --accent: #3987e5; --good: #199e70; --bad: #e66767; --warn: #c98500;
    --chip: rgba(255,255,255,.07);
  }
}
* { box-sizing: border-box; }
body { margin:0; padding:32px 24px 64px; background:var(--plane); color:var(--ink);
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif; font-size:15px; line-height:1.55; }
.wrap { max-width:1100px; margin:0 auto; }
h1 { font-size:26px; font-weight:650; margin:0 0 4px; letter-spacing:-.015em; }
.sub { color:var(--ink-2); margin:0 0 6px; font-size:14px; }
.meta { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
h2 { font-size:12px; font-weight:650; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--border); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-top:18px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:13px 15px; }
.card .v { font-size:20px; font-weight:650; letter-spacing:-.02em; }
.card .l { font-size:12px; color:var(--ink-2); margin-top:2px; }
figure { margin:0; background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:16px 18px 12px; }
figcaption { font-size:13px; color:var(--ink-2); margin-bottom:12px; }
.legend { display:flex; flex-wrap:wrap; gap:16px; font-size:13px; color:var(--ink-2); margin-bottom:10px; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
svg { display:block; width:100%; height:auto; overflow:visible; }
table { width:100%; border-collapse:collapse; font-size:13.5px; background:var(--surface);
  border:1px solid var(--border); border-radius:10px; overflow:hidden; }
th { text-align:left; font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); font-weight:620; padding:9px 11px; border-bottom:1px solid var(--border);
  white-space:nowrap; }
td { padding:8px 11px; border-bottom:1px solid var(--grid); font-variant-numeric:tabular-nums; }
tr:last-child td { border-bottom:none; }
td.txt, th.txt { font-variant-numeric:normal; }
td.num, th.num { text-align:right; }
.rank { color:var(--muted); }
.rank.top { color:var(--ink); font-weight:650; }
.best { font-weight:650; }
.failed td { color:var(--muted); font-style:italic; }
.pill { display:inline-block; padding:1px 7px; border-radius:99px; font-size:11.5px;
  font-weight:600; background:var(--chip); }
.pill.yes { color:var(--good); } .pill.no { color:var(--bad); } .pill.unverified { color:var(--warn); }
.note { font-size:13px; color:var(--ink-2); background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:10px; padding:13px 16px; margin-top:12px; }
.note b { color:var(--ink); }
footer { margin-top:44px; padding-top:14px; border-top:1px solid var(--border);
  font-size:12.5px; color:var(--muted); }
code { font-family:ui-monospace,Menlo,monospace; font-size:12.5px;
  background:var(--chip); padding:1px 5px; border-radius:4px; }
"""


def _esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation."""
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric cell, rendering missing values as an em dash."""
    if value is None:
        return "&mdash;"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if math.isnan(number):
        return "&mdash;"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def render_pareto_svg(
    entries: Sequence[LeaderboardEntry],
    metric: str,
    *,
    width: int = 860,
    height: int = 380,
) -> str:
    """Render a quality-versus-latency scatter as inline SVG.

    Points are coloured by commercial-use status, so the reader can see at a
    glance whether the best-performing model is one they are allowed to ship.
    Frontier points are connected by a stepped line.

    Args:
        entries: Successful leaderboard entries.
        metric: Quality metric plotted on the y-axis.
        width: SVG viewBox width.
        height: SVG viewBox height.

    Returns:
        SVG markup, or an empty string when there is nothing plottable.
    """
    points = [
        e
        for e in entries
        if e.ok
        and metric in e.metrics
        and e.metrics[metric] == e.metrics[metric]
        and e.predict_seconds == e.predict_seconds
    ]
    if len(points) < 2:
        return ""

    left, right, top, bottom = 62, width - 24, 22, height - 52

    xs = [max(e.predict_seconds, 1e-4) for e in points]
    ys = [e.metrics[metric] for e in points]

    # Latency spans orders of magnitude across TFMs, so a linear axis collapses
    # every fast model into one pixel column. Log scale keeps them separable.
    log_xs = [math.log10(x) for x in xs]
    x_lo, x_hi = min(log_xs), max(log_xs)
    if x_hi - x_lo < 1e-9:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    pad_x = (x_hi - x_lo) * 0.08
    x_lo, x_hi = x_lo - pad_x, x_hi + pad_x

    y_lo, y_hi = min(ys), max(ys)
    if y_hi - y_lo < 1e-9:
        y_lo, y_hi = y_lo - 0.05, y_hi + 0.05
    pad_y = (y_hi - y_lo) * 0.12
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

    def sx(seconds: float) -> float:
        return left + (math.log10(max(seconds, 1e-4)) - x_lo) / (x_hi - x_lo) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_lo) / (y_hi - y_lo) * (bottom - top)

    parts: list[str] = []

    # Horizontal gridlines with value labels.
    for i in range(5):
        value = y_lo + (y_hi - y_lo) * i / 4
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--muted)">{value:.3f}</text>'
        )

    # Decade ticks on the log latency axis.
    decade = math.floor(x_lo)
    while decade <= x_hi:
        x = left + (decade - x_lo) / (x_hi - x_lo) * (right - left)
        if left <= x <= right:
            seconds = 10**decade
            label = f"{seconds:g}s" if seconds >= 1 else f"{seconds * 1000:g}ms"
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
                f'stroke="var(--grid)" stroke-width="1" stroke-dasharray="2 4"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" '
                f'font-size="11" fill="var(--muted)">{label}</text>'
            )
        decade += 1

    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{(left + right) / 2:.0f}" y="{bottom + 40}" text-anchor="middle" '
        f'font-size="12" fill="var(--ink-2)" font-weight="550">'
        f'Prediction latency (log scale)</text>'
    )
    parts.append(
        f'<text transform="translate(16,{(top + bottom) / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle" font-size="12" fill="var(--ink-2)" font-weight="550">'
        f'{_esc(metric)}</text>'
    )
    parts.append(
        f'<text x="{left + 8}" y="{top + 2}" font-size="11" font-style="italic" '
        f'fill="var(--muted)">better &#8598;</text>'
    )

    # Frontier line, drawn under the marks.
    sign = 1.0 if is_higher_better(metric) else -1.0
    front: list[LeaderboardEntry] = []
    for entry in sorted(points, key=lambda e: e.predict_seconds):
        score = sign * entry.metrics[metric]
        if all(sign * other.metrics[metric] < score for other in front):
            front.append(entry)
    if len(front) > 1:
        steps: list[str] = []
        for i, entry in enumerate(front):
            x, y = sx(entry.predict_seconds), sy(entry.metrics[metric])
            if i == 0:
                steps.append(f"M {x:.1f} {y:.1f}")
            else:
                prev_y = sy(front[i - 1].metrics[metric])
                steps.append(f"L {x:.1f} {prev_y:.1f} L {x:.1f} {y:.1f}")
        parts.append(
            f'<path d="{" ".join(steps)}" fill="none" stroke="var(--accent)" '
            f'stroke-width="2" stroke-dasharray="5 4" opacity="0.65"/>'
        )

    front_names = {id(e) for e in front}
    for entry in points:
        x, y = sx(entry.predict_seconds), sy(entry.metrics[metric])
        color = _COMMERCIAL_COLORS.get(entry.commercial_badge, _COMMERCIAL_COLORS["unverified"])
        radius = 7 if id(entry) in front_names else 5
        tooltip = (
            f"{entry.display_name}\n{metric}={entry.metrics[metric]:.4f}\n"
            f"predict={entry.predict_seconds:.3f}s\ncommercial: {entry.commercial_badge}"
        )
        parts.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="var(--surface)" stroke-width="2"><title>{_esc(tooltip)}</title></circle>'
            f'<text x="{x:.1f}" y="{y - radius - 5:.1f}" text-anchor="middle" '
            f'font-size="10.5" font-weight="600" fill="var(--ink-2)">'
            f'{_esc(entry.model_name)}</text></g>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Quality versus prediction latency. Full data in the table below.">'
        + "".join(parts)
        + "</svg>"
    )


def render_leaderboard_html(
    board: TabularLeaderboard,
    *,
    title: str = "TabTune Leaderboard",
    include_pareto: bool = True,
) -> str:
    """Render a complete leaderboard report as a self-contained HTML document.

    Args:
        board: A leaderboard that has been run.
        title: Report heading.
        include_pareto: Include the quality-versus-latency scatter.

    Returns:
        HTML source, ready to write to a file.
    """
    frame = board.results
    metric = board._rank_by or primary_metric(board.task_type)
    entries = board.entries
    succeeded = [e for e in entries if e.ok]
    best = board.best()

    # ---------------------------------------------------------------- summary
    cards = [
        ("Models run", f"{len(succeeded)}/{len(entries)}", "configurations that completed"),
        ("Ranked by", _esc(metric), "primary metric"),
        ("Train / test", f"{len(board.X_train):,} / {len(board.X_test):,}", "rows"),
    ]
    if best is not None:
        cards.append(("Best", _esc(best.model_name), f"{metric} = {best.metrics[metric]:.4f}"))
    deployable = [e for e in succeeded if e.commercial_use_ok is True]
    cards.append(
        ("Commercially deployable", f"{len(deployable)}/{len(succeeded)}", "by weight license")
    )

    cards_html = "".join(
        f'<div class="card"><div class="v">{value}</div>'
        f'<div class="l">{_esc(label)} &middot; {_esc(note)}</div></div>'
        for label, value, note in cards
    )

    # ------------------------------------------------------------------ table
    if frame.empty:
        table_html = "<p>No results. Call <code>run()</code> first.</p>"
    else:
        columns = list(frame.columns)
        header = "".join(
            f'<th class="{"txt" if _is_text(c) else "num"}">{_esc(c)}</th>' for c in columns
        )
        header = '<th class="num">#</th>' + header

        rows: list[str] = []
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            failed = str(row.get("Status", "ok")) == "failed"
            cells: list[str] = [
                f'<td class="num rank{" top" if rank <= 3 and not failed else ""}">{rank}</td>'
            ]
            for column in columns:
                value = row[column]
                if column == "Commercial":
                    badge = _esc(value)
                    cells.append(
                        f'<td class="txt"><span class="pill {badge}">{badge}</span></td>'
                    )
                elif _is_text(column):
                    cells.append(f'<td class="txt">{_esc(value)}</td>')
                else:
                    emphasis = ' class="num best"' if column == metric and rank == 1 and not failed else ' class="num"'
                    digits = 2 if column.endswith("_s") or column.endswith("_mb") else 4
                    cells.append(f"<td{emphasis}>{_fmt(value, digits)}</td>")
            rows.append(f'<tr class="{"failed" if failed else ""}">{"".join(cells)}</tr>')
        table_html = (
            f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    # ----------------------------------------------------------------- pareto
    pareto_html = ""
    if include_pareto:
        svg = render_pareto_svg(succeeded, metric)
        if svg:
            legend = "".join(
                f'<span><i class="dot" style="background:{color}"></i>'
                f"commercial use: {_esc(label)}</span>"
                for label, color in _COMMERCIAL_COLORS.items()
            )
            pareto_html = (
                "<h2>Quality vs latency</h2>"
                "<figure>"
                "<figcaption>Each point is one configuration. The dashed line marks the "
                "Pareto frontier: everything below and to the right of it is strictly "
                "worse on both axes. Colour encodes whether the model's weights permit "
                "commercial use.</figcaption>"
                f'<div class="legend">{legend}</div>'
                f"{svg}"
                "</figure>"
            )

    # ------------------------------------------------------------------ notes
    notes: list[str] = []
    front = board.pareto_front()
    if len(front) > 1:
        notes.append(
            "<b>Pareto front:</b> "
            + ", ".join(f"<code>{_esc(e.display_name)}</code>" for e in front)
            + ". These are the only configurations not beaten on both quality and speed."
        )
    if best is not None and best.commercial_use_ok is not True and deployable:
        sign = 1.0 if is_higher_better(metric) else -1.0
        top = max(
            (e for e in deployable if metric in e.metrics),
            key=lambda e: sign * e.metrics[metric],
            default=None,
        )
        if top is not None:
            notes.append(
                f"<b>Deployment note:</b> the top-ranked model "
                f"<code>{_esc(best.model_name)}</code> ships under "
                f"<code>{_esc(best.license_name)}</code>, which TabTune does not record as "
                f"cleared for commercial use. The best commercially deployable entry is "
                f"<code>{_esc(top.display_name)}</code> "
                f"({metric} = {top.metrics[metric]:.4f}). "
                f"Distillation is the other route: "
                f"<code>TabDistiller(teachers={best.model_name!r}, student='lgbm')</code>."
            )
    failures = [e for e in entries if not e.ok]
    if failures:
        notes.append(
            "<b>Failures:</b> "
            + "; ".join(f"<code>{_esc(e.display_name)}</code>: {_esc(e.error)}" for e in failures)
        )
    notes_html = "".join(f'<div class="note">{n}</div>' for n in notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(board.task_type.capitalize())} &middot; {len(entries)} configurations
     &middot; ranked by <code>{_esc(metric)}</code></p>
  <p class="meta">{len(board.X_train):,} training rows &middot; {len(board.X_test):,} test rows
     &middot; {board.X_train.shape[1] if hasattr(board.X_train, 'shape') else '?'} features</p>
  <div class="cards">{cards_html}</div>
  {pareto_html}
  <h2>Results</h2>
  {table_html}
  {notes_html}
  <footer>
    Generated by TabTune. Metrics come from <code>tabtune.evaluation.metrics</code>;
    license metadata comes from <code>tabtune.registry</code> and is a convenience,
    not legal advice - confirm terms upstream before deploying.
  </footer>
</div>
</body>
</html>
"""


def _is_text(column: str) -> bool:
    """Whether a leaderboard column holds text rather than a number."""
    return column in {"Model", "Strategy", "Mode", "Status", "License", "Commercial", "Error"}
