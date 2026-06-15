"""
tabtune.causal.reporter
=======================

Compose the outputs of identification, estimation, refutation, and
sensitive-attribute audits into a single compliance-ready artifact.

Two output modes are supported:

* ``rich`` -- logs a tidy, human-readable summary via the standard TabTune
  logger (``logger.info("[Causal] ...")``). Mirrors the
  :class:`TabularPipeline.evaluate(output_format='rich')` style so users
  who already know TabTune feel at home.
* ``html`` -- writes a self-contained HTML model card to ``output_path``.
  When :mod:`jinja2` is installed we use a templated render; otherwise a
  pure-Python fallback builds the file with no extra dependencies.

The HTML report is intended to be embedded directly in a model card or
attached to a regulatory submission.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _try_import_jinja():
    try:
        import jinja2  # noqa: F401
        return jinja2
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------
class Reporter:
    """
    Pretty-print and serialise a causal-analysis report.

    Parameters
    ----------
    report : dict
        Composed output dictionary produced by :class:`CausalAnalysis`.
        Expected (optional) keys: ``identification``, ``effect``,
        ``refutation``, ``proxy``, ``counterfactual_fairness``,
        ``meta`` (run metadata).
    """

    def __init__(self, report: dict):
        self.report = report

    # ------------------------------------------------------------------
    # rich text
    # ------------------------------------------------------------------
    def to_rich(self) -> None:
        """Pretty-print the report through the TabTune logger."""
        r = self.report
        logger.info("\n" + "=" * 72)
        logger.info("[Causal] Causal Analysis Report")
        logger.info("=" * 72)

        meta = r.get("meta", {})
        if meta:
            for k, v in meta.items():
                logger.info("[Causal]   %s: %s", k, v)

        if "identification" in r:
            ident = r["identification"]
            logger.info("\n[Causal] Identification (Step 1)")
            logger.info(
                "[Causal]   Identifiable: %s | Estimand: %s | Backend: %s",
                ident.get("identifiable"),
                ident.get("estimand_type"),
                ident.get("backend"),
            )
            if ident.get("adjustment_set"):
                logger.info(
                    "[Causal]   Adjustment set: %s",
                    ", ".join(map(str, ident["adjustment_set"])),
                )
            if ident.get("message"):
                logger.info("[Causal]   %s", ident["message"])

        if "effect" in r:
            eff = r["effect"]
            logger.info("\n[Causal] Effect (Step 2)")
            logger.info("[Causal]   Estimator: %s", eff.get("estimator"))
            logger.info(
                "[Causal]   ATE: %+.4f  (SE %.4f)",
                eff.get("ate", float("nan")),
                eff.get("std_error", float("nan")),
            )
            ci_lo = eff.get("ci_lower")
            ci_hi = eff.get("ci_upper")
            level = eff.get("confidence_level", 0.95)
            if ci_lo is not None and ci_hi is not None:
                logger.info(
                    "[Causal]   %.0f%% CI: [%+.4f, %+.4f]",
                    100 * level,
                    ci_lo,
                    ci_hi,
                )
            if eff.get("p_value") is not None:
                logger.info("[Causal]   p-value: %.4f", eff["p_value"])
            if "per_level" in eff:
                logger.info("[Causal]   Per-level contrasts:")
                for lvl, body in eff["per_level"].items():
                    logger.info(
                        "[Causal]     T=%s:  ATE=%+.4f  CI=[%+.4f, %+.4f]",
                        lvl,
                        body["ate"],
                        body["ci_lower"],
                        body["ci_upper"],
                    )

        if "refutation" in r:
            ref = r["refutation"]
            total = ref.get("refuters_passed", 0) + ref.get("refuters_failed", 0)
            logger.info("\n[Causal] Refutation (Step 3)")
            logger.info(
                "[Causal]   %d/%d refuters passed", ref.get("refuters_passed", 0), total
            )
            for name in (
                "placebo",
                "random_common_cause",
                "subset",
                "bootstrap",
                "sensitivity",
            ):
                if name not in ref:
                    continue
                body = ref[name]
                status = "PASS" if body.get("pass") else "FAIL"
                if "ate" in body and body["ate"] is not None:
                    logger.info(
                        "[Causal]     [%s] %s: ATE=%+.4f  rule=%s",
                        status,
                        name,
                        body["ate"],
                        body.get("rule", ""),
                    )
                elif "e_value" in body:
                    logger.info(
                        "[Causal]     [%s] %s: E-value=%.3f  rule=%s",
                        status,
                        name,
                        body["e_value"],
                        body.get("rule", ""),
                    )
                else:
                    logger.info(
                        "[Causal]     [%s] %s: rule=%s",
                        status,
                        name,
                        body.get("rule", ""),
                    )

        if "proxy" in r:
            prox = r["proxy"]
            logger.info("\n[Causal] Proxy audit")
            for s_col, body in prox.get("per_attribute", {}).items():
                flagged = body.get("flagged", [])
                if flagged:
                    logger.info(
                        "[Causal]   '%s': %d proxy feature(s) above threshold %.2f",
                        s_col,
                        len(flagged),
                        body.get("threshold", 0.5),
                    )
                    for f in flagged[:5]:
                        logger.info(
                            "[Causal]     - %s (score %.3f)", f["feature"], f["score"]
                        )
                else:
                    logger.info("[Causal]   '%s': no proxy features flagged.", s_col)

        if "counterfactual_fairness" in r:
            cf = r["counterfactual_fairness"]
            logger.info("\n[Causal] Counterfactual fairness")
            for s_col, body in cf.get("per_attribute", {}).items():
                logger.info(
                    "[Causal]   '%s': flip_rate=%.3f  mean_delta=%.4f",
                    s_col,
                    body.get("flip_rate", 0.0),
                    body.get("mean_delta", 0.0),
                )
        logger.info("=" * 72)

    # ------------------------------------------------------------------
    # json
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Return the report as a JSON string (printable / loggable)."""
        return json.dumps(self.report, indent=2, default=str)

    # ------------------------------------------------------------------
    # html
    # ------------------------------------------------------------------
    def to_html(self, output_path: str | Path) -> str:
        """Render an HTML model card and write it to ``output_path``."""
        jinja = _try_import_jinja()
        if jinja is not None:
            html = self._render_via_jinja(jinja)
        else:
            html = self._render_fallback()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.info("[Causal] HTML report written to %s", str(path))
        return str(path)

    # ------------------------------------------------------------------
    # rendering internals
    # ------------------------------------------------------------------
    _JINJA_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TabTune Causal Audit Report</title>
<style>
 body{font-family:'Inter',sans-serif;max-width:980px;margin:2em auto;color:#1f2937;padding:0 1em;}
 h1{border-bottom:2px solid #2563eb;padding-bottom:0.3em;}
 h2{margin-top:1.6em;color:#1e40af;}
 table{border-collapse:collapse;margin:0.6em 0;width:100%;}
 th,td{padding:8px 12px;border:1px solid #e5e7eb;}
 th{background:#f3f4f6;text-align:left;}
 .pass{color:#15803d;font-weight:600;}
 .fail{color:#b91c1c;font-weight:600;}
 .meta{color:#6b7280;font-size:0.9em;}
 code{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:0.92em;}
</style>
</head>
<body>
<h1>TabTune Causal Audit Report</h1>
{% if meta %}
<p class="meta">
 {% for k, v in meta.items() %}<b>{{ k }}</b>: {{ v }}{% if not loop.last %} &nbsp;|&nbsp; {% endif %}{% endfor %}
</p>
{% endif %}

{% if identification %}
<h2>Step 1 &mdash; Identification</h2>
<table>
 <tr><th>Identifiable</th><td class="{{ 'pass' if identification.identifiable else 'fail' }}">{{ identification.identifiable }}</td></tr>
 <tr><th>Estimand</th><td><code>{{ identification.estimand_type }}</code></td></tr>
 <tr><th>Adjustment set</th><td>{{ identification.adjustment_set|join(', ') }}</td></tr>
 <tr><th>Backend</th><td><code>{{ identification.backend }}</code></td></tr>
</table>
<p class="meta">{{ identification.message }}</p>
{% endif %}

{% if effect %}
<h2>Step 2 &mdash; Estimated Effect</h2>
<table>
 <tr><th>Estimator</th><td><code>{{ effect.estimator }}</code></td></tr>
 <tr><th>ATE</th><td>{{ "%+.4f"|format(effect.ate) }}</td></tr>
 <tr><th>Std error</th><td>{{ "%.4f"|format(effect.std_error) }}</td></tr>
 <tr><th>{{ (effect.confidence_level * 100)|int }}% CI</th>
  <td>[{{ "%+.4f"|format(effect.ci_lower) }}, {{ "%+.4f"|format(effect.ci_upper) }}]</td></tr>
 {% if effect.p_value is not none %}<tr><th>p-value</th><td>{{ "%.4f"|format(effect.p_value) }}</td></tr>{% endif %}
 <tr><th>n</th><td>{{ effect.n_samples }}</td></tr>
</table>
{% if effect.per_level %}
<h3>Per-level contrasts</h3>
<table>
 <tr><th>T</th><th>ATE</th><th>SE</th><th>CI lower</th><th>CI upper</th></tr>
{% for lvl, body in effect.per_level.items() %}
 <tr><td>{{ lvl }}</td><td>{{ "%+.4f"|format(body.ate) }}</td><td>{{ "%.4f"|format(body.std_error) }}</td>
     <td>{{ "%+.4f"|format(body.ci_lower) }}</td><td>{{ "%+.4f"|format(body.ci_upper) }}</td></tr>
{% endfor %}
</table>
{% endif %}
{% endif %}

{% if refutation %}
<h2>Step 3 &mdash; Refutation</h2>
<p>{{ refutation.refuters_passed }} / {{ refutation.refuters_passed + refutation.refuters_failed }} refuters passed</p>
<table>
 <tr><th>Check</th><th>Summary</th><th>Status</th><th>Rule</th></tr>
{% for name in ['placebo', 'random_common_cause', 'subset', 'bootstrap', 'sensitivity'] %}
 {% if name in refutation %}
 {% set body = refutation[name] %}
 <tr>
  <td><code>{{ name }}</code></td>
  <td>
   {% if body.ate is defined and body.ate is not none %}ATE={{ "%+.4f"|format(body.ate) }}
   {% elif body.e_value is defined %}E-value={{ "%.3f"|format(body.e_value) }}
   {% elif body.ci_lower is defined %}CI=[{{ "%+.4f"|format(body.ci_lower) }}, {{ "%+.4f"|format(body.ci_upper) }}]
   {% elif body.mean is defined %}mean={{ "%+.4f"|format(body.mean) }}, std={{ "%.4f"|format(body.std) }}
   {% else %}&mdash;{% endif %}
  </td>
  <td class="{{ 'pass' if body.pass else 'fail' }}">{{ 'PASS' if body.pass else 'FAIL' }}</td>
  <td class="meta">{{ body.rule }}</td>
 </tr>
 {% endif %}
{% endfor %}
</table>
{% endif %}

{% if proxy %}
<h2>Proxy Audit</h2>
{% for s_col, body in proxy.per_attribute.items() %}
<h3>Sensitive attribute: <code>{{ s_col }}</code></h3>
{% if body.flagged %}
<table>
 <tr><th>Feature</th><th>Proxy score</th></tr>
{% for f in body.flagged %}
 <tr><td><code>{{ f.feature }}</code></td><td class="fail">{{ "%.3f"|format(f.score) }}</td></tr>
{% endfor %}
</table>
{% else %}
<p class="pass">No proxy features flagged (threshold {{ "%.2f"|format(body.threshold) }}).</p>
{% endif %}
{% endfor %}
{% endif %}

{% if counterfactual_fairness %}
<h2>Counterfactual Fairness</h2>
<table>
 <tr><th>Sensitive</th><th>Flip rate</th><th>Mean delta</th><th>n rows audited</th></tr>
{% for s_col, body in counterfactual_fairness.per_attribute.items() %}
 <tr><td><code>{{ s_col }}</code></td>
     <td class="{{ 'fail' if body.flip_rate >= 0.1 else 'pass' }}">{{ "%.3f"|format(body.flip_rate) }}</td>
     <td>{{ "%.4f"|format(body.mean_delta) }}</td>
     <td>{{ body.n_rows_audited }}</td></tr>
{% endfor %}
</table>
{% endif %}

<p class="meta" style="margin-top:2em;">Generated by <code>tabtune.causal</code></p>
</body>
</html>
"""

    def _render_via_jinja(self, jinja2) -> str:
        env = jinja2.Environment(autoescape=True)
        template = env.from_string(self._JINJA_TEMPLATE)
        return template.render(
            meta=self.report.get("meta", {}),
            identification=self.report.get("identification"),
            effect=self.report.get("effect"),
            refutation=self.report.get("refutation"),
            proxy=self.report.get("proxy"),
            counterfactual_fairness=self.report.get("counterfactual_fairness"),
        )

    # ------------------------------------------------------------------
    # Dependency-free fallback renderer
    # ------------------------------------------------------------------
    def _render_fallback(self) -> str:
        r = self.report
        lines: list[str] = []
        lines.append("<!doctype html><html><head>")
        lines.append("<meta charset='utf-8'><title>TabTune Causal Audit Report</title>")
        lines.append(
            "<style>"
            "body{font-family:Inter,sans-serif;max-width:980px;margin:2em auto;"
            "color:#1f2937;padding:0 1em;}"
            "h1{border-bottom:2px solid #2563eb;padding-bottom:0.3em;}"
            "h2{margin-top:1.6em;color:#1e40af;}"
            "table{border-collapse:collapse;margin:.6em 0;width:100%;}"
            "th,td{padding:8px 12px;border:1px solid #e5e7eb;}"
            "th{background:#f3f4f6;text-align:left;}"
            ".pass{color:#15803d;font-weight:600;}"
            ".fail{color:#b91c1c;font-weight:600;}"
            ".meta{color:#6b7280;font-size:.9em;}"
            "code{background:#f3f4f6;padding:1px 5px;border-radius:3px;}"
            "</style></head><body>"
        )
        lines.append("<h1>TabTune Causal Audit Report</h1>")

        meta = r.get("meta") or {}
        if meta:
            meta_html = " | ".join(f"<b>{k}</b>: {v}" for k, v in meta.items())
            lines.append(f"<p class='meta'>{meta_html}</p>")

        if "identification" in r:
            i = r["identification"]
            lines.append("<h2>Step 1 — Identification</h2><table>")
            lines.append(
                f"<tr><th>Identifiable</th><td class='{'pass' if i.get('identifiable') else 'fail'}'>{i.get('identifiable')}</td></tr>"
            )
            lines.append(
                f"<tr><th>Estimand</th><td><code>{i.get('estimand_type')}</code></td></tr>"
            )
            lines.append(
                f"<tr><th>Adjustment set</th><td>{', '.join(map(str, i.get('adjustment_set') or []))}</td></tr>"
            )
            lines.append(
                f"<tr><th>Backend</th><td><code>{i.get('backend')}</code></td></tr>"
            )
            lines.append("</table>")
            if i.get("message"):
                lines.append(f"<p class='meta'>{i['message']}</p>")

        if "effect" in r:
            e = r["effect"]
            lines.append("<h2>Step 2 — Estimated Effect</h2><table>")
            lines.append(f"<tr><th>Estimator</th><td><code>{e.get('estimator')}</code></td></tr>")
            lines.append(f"<tr><th>ATE</th><td>{e.get('ate', float('nan')):+.4f}</td></tr>")
            lines.append(f"<tr><th>Std error</th><td>{e.get('std_error', float('nan')):.4f}</td></tr>")
            level_pct = int(round((e.get("confidence_level") or 0.95) * 100))
            lines.append(
                f"<tr><th>{level_pct}% CI</th>"
                f"<td>[{e.get('ci_lower', float('nan')):+.4f}, {e.get('ci_upper', float('nan')):+.4f}]</td></tr>"
            )
            if e.get("p_value") is not None:
                lines.append(f"<tr><th>p-value</th><td>{e['p_value']:.4f}</td></tr>")
            lines.append(f"<tr><th>n</th><td>{e.get('n_samples')}</td></tr></table>")
            if "per_level" in e:
                lines.append("<h3>Per-level contrasts</h3><table>")
                lines.append("<tr><th>T</th><th>ATE</th><th>SE</th><th>CI lower</th><th>CI upper</th></tr>")
                for lvl, body in e["per_level"].items():
                    lines.append(
                        f"<tr><td>{lvl}</td>"
                        f"<td>{body['ate']:+.4f}</td>"
                        f"<td>{body['std_error']:.4f}</td>"
                        f"<td>{body['ci_lower']:+.4f}</td>"
                        f"<td>{body['ci_upper']:+.4f}</td></tr>"
                    )
                lines.append("</table>")

        if "refutation" in r:
            ref = r["refutation"]
            total = ref.get("refuters_passed", 0) + ref.get("refuters_failed", 0)
            lines.append(
                f"<h2>Step 3 — Refutation</h2>"
                f"<p>{ref.get('refuters_passed', 0)} / {total} refuters passed</p>"
            )
            lines.append("<table><tr><th>Check</th><th>Summary</th><th>Status</th><th>Rule</th></tr>")
            for name in ("placebo", "random_common_cause", "subset", "bootstrap", "sensitivity"):
                if name not in ref:
                    continue
                body = ref[name]
                status_cls = "pass" if body.get("pass") else "fail"
                status_text = "PASS" if body.get("pass") else "FAIL"
                summary = ""
                if body.get("ate") is not None:
                    summary = f"ATE={body['ate']:+.4f}"
                elif "e_value" in body:
                    summary = f"E-value={body['e_value']:.3f}"
                elif "ci_lower" in body:
                    summary = f"CI=[{body['ci_lower']:+.4f}, {body['ci_upper']:+.4f}]"
                elif "mean" in body:
                    summary = f"mean={body['mean']:+.4f}, std={body['std']:.4f}"
                lines.append(
                    f"<tr><td><code>{name}</code></td><td>{summary}</td>"
                    f"<td class='{status_cls}'>{status_text}</td>"
                    f"<td class='meta'>{body.get('rule', '')}</td></tr>"
                )
            lines.append("</table>")

        if "proxy" in r:
            lines.append("<h2>Proxy Audit</h2>")
            for s_col, body in r["proxy"].get("per_attribute", {}).items():
                lines.append(f"<h3>Sensitive attribute: <code>{s_col}</code></h3>")
                if body.get("flagged"):
                    lines.append("<table><tr><th>Feature</th><th>Proxy score</th></tr>")
                    for f in body["flagged"]:
                        lines.append(
                            f"<tr><td><code>{f['feature']}</code></td>"
                            f"<td class='fail'>{f['score']:.3f}</td></tr>"
                        )
                    lines.append("</table>")
                else:
                    lines.append(
                        f"<p class='pass'>No proxy features flagged "
                        f"(threshold {body.get('threshold', 0.5):.2f}).</p>"
                    )

        if "counterfactual_fairness" in r:
            cf = r["counterfactual_fairness"]
            lines.append("<h2>Counterfactual Fairness</h2>")
            lines.append("<table><tr><th>Sensitive</th><th>Flip rate</th><th>Mean delta</th><th>n rows</th></tr>")
            for s_col, body in cf.get("per_attribute", {}).items():
                cls = "fail" if body.get("flip_rate", 0) >= 0.1 else "pass"
                lines.append(
                    f"<tr><td><code>{s_col}</code></td>"
                    f"<td class='{cls}'>{body.get('flip_rate', 0):.3f}</td>"
                    f"<td>{body.get('mean_delta', 0):.4f}</td>"
                    f"<td>{body.get('n_rows_audited', 0)}</td></tr>"
                )
            lines.append("</table>")

        lines.append("<p class='meta' style='margin-top:2em;'>Generated by <code>tabtune.causal</code></p>")
        lines.append("</body></html>")
        return "\n".join(lines)
