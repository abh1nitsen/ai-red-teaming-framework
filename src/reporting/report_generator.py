"""
src/reporting/report_generator.py
===================================
Generates JSON and HTML audit reports from a RedTeamResult.

JSON REPORT:
    Machine-readable, structured identically to Projects 1 and 2 reports
    so Project 4's dashboard can ingest all three with one parser.
    Saved to reports/red_team_{id}.json

HTML REPORT:
    Human-readable, self-contained single-file report.
    Designed for sharing with risk managers and compliance teams.
    Shows:
    - Overall robustness score and verdict banner
    - Per-category risk rating cards with attack success bars
    - Per-probe detail table (probe name, defence score, outcome)
    - Target vs judge model comparison
    - Recommended actions for flagged categories
    - Abhinit Sen attribution in footer

NO EMOJIS, NO EM DASHES anywhere in the output.
All status markers use plain text: [ATTACK SUCCEEDED], [DEFENDED], [HIGH RISK], etc.

USAGE:
    from src.reporting.report_generator import ReportGenerator
    gen = ReportGenerator(config=config)
    paths = gen.generate(result)
    print(paths["html"])
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

from src.runner.red_team_runner import RedTeamResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


_CATEGORY_LABELS = {
    "regulatory_boundary":     "Regulatory Boundary",
    "pii_extraction":          "PII Extraction",
    "jailbreak":               "Jailbreak / Instruction Override",
    "bias_elicitation":        "Bias Elicitation (Indian Context)",
    "hallucination_induction": "Hallucination Induction",
    "rag_attack":              "RAG-Specific Attacks",
}

_RECOMMENDATIONS = {
    "regulatory_boundary": (
        "The model gave advice in regulated domains (financial, insurance, legal) "
        "without appropriate caveats. Review the system prompt to add explicit "
        "refusal instructions for SEBI, RBI, IRDAI, and legal advice categories."
    ),
    "pii_extraction": (
        "The model generated PII-format content in response to social engineering. "
        "Add PII generation refusal to the system prompt and consider post-generation "
        "PII scanning (Project 1 PII Detector) in the output pipeline."
    ),
    "jailbreak": (
        "The model's safety constraints were bypassed by adversarial prompts. "
        "Review jailbreak-specific system prompt hardening. Consider adding "
        "input classification to detect and block known jailbreak patterns."
    ),
    "bias_elicitation": (
        "The model produced biased outputs when prompted with Indian demographic "
        "framing. Review the model's training and consider adding bias detection "
        "(Project 1 Bias Detector) as a post-generation filter."
    ),
    "hallucination_induction": (
        "The model stated specific regulatory facts with overconfidence when it "
        "should have expressed uncertainty. Add explicit uncertainty instructions "
        "to the system prompt for regulatory and compliance questions."
    ),
    "rag_attack": (
        "The RAG pipeline followed instructions embedded in document content. "
        "Add document content sanitisation before ingestion and implement "
        "instruction-data separation in the retrieval prompt template."
    ),
}


class ReportGenerator:
    """
    Serialises RedTeamResult to JSON and HTML report files.

    Args:
        config: Config dict from config.yaml. Reads:
                config["reporting"]["output_dir"]
                config["reporting"]["generate_html"]
                config["reporting"]["include_probe_responses"]
                config["reporting"]["response_preview_chars"]
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        rep_cfg = cfg.get("reporting", {})
        self.output_dir: str = rep_cfg.get("output_dir", "reports")
        self.generate_html: bool = rep_cfg.get("generate_html", True)
        self.include_responses: bool = rep_cfg.get("include_probe_responses", True)
        self.response_chars: int = rep_cfg.get("response_preview_chars", 400)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug("ReportGenerator ready | output_dir=%s", self.output_dir)

    def generate(self, result: RedTeamResult) -> Dict[str, str]:
        """
        Write JSON and HTML reports for a completed red team run.

        Args:
            result: RedTeamResult from RedTeamRunner.run()

        Returns:
            dict with keys "json" and optionally "html", values are file paths.
        """
        paths: Dict[str, str] = {}
        eid = result.report_id[:8]

        # JSON report
        json_path = os.path.join(self.output_dir, f"red_team_{eid}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            paths["json"] = json_path
            logger.info("JSON report: %s", json_path)
        except OSError as exc:
            logger.error("Failed to write JSON report: %s", exc)

        # HTML report
        if self.generate_html:
            html_path = os.path.join(self.output_dir, f"red_team_{eid}.html")
            try:
                html = self._build_html(result)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)
                paths["html"] = html_path
                logger.info("HTML report: %s", html_path)
            except OSError as exc:
                logger.error("Failed to write HTML report: %s", exc)

        return paths

    # -- HTML builder ---------------------------------------------------------

    def _build_html(self, result: RedTeamResult) -> str:
        score = result.overall_robustness_score
        verdict = result.verdict
        score_pct = round(score * 100, 1)

        v_colors = {
            "ROBUST":       {"bg": "#dcfce7", "border": "#16a34a", "text": "#15803d"},
            "NEEDS REVIEW": {"bg": "#fef9c3", "border": "#ca8a04", "text": "#a16207"},
            "HIGH RISK":    {"bg": "#fee2e2", "border": "#dc2626", "text": "#b91c1c"},
        }
        vc = v_colors.get(verdict, v_colors["HIGH RISK"])

        def bar_color(rate):
            if rate > 0.50: return "#ef4444"
            if rate > 0.20: return "#f59e0b"
            return "#22c55e"

        def risk_badge(rating):
            colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
            c = colors.get(rating, "#6b7280")
            return (
                f'<span style="background:{c};color:#fff;font-size:11px;'
                f'padding:2px 8px;border-radius:10px;font-weight:600;">'
                f'[{rating} RISK]</span>'
            )

        # Category cards
        cat_cards = []
        flagged = []
        for cat_name, cat_data in result.categories.items():
            label = _CATEGORY_LABELS.get(cat_name, cat_name)
            rate = cat_data.get("attack_success_rate", 0.0)
            pct = round(rate * 100, 1)
            risk = cat_data.get("risk_rating", "LOW")
            probes = cat_data.get("probes_run", 0)
            succeeded = cat_data.get("successful_attacks", 0)
            color = bar_color(rate)

            if risk in ("HIGH", "MEDIUM"):
                flagged.append(cat_name)

            # Per-probe table
            probe_rows = []
            for pr in cat_data.get("probe_results", []):
                outcome = (
                    '<span style="color:#ef4444;font-weight:600;">[ATTACK SUCCEEDED]</span>'
                    if pr.get("attack_succeeded")
                    else '<span style="color:#22c55e;font-weight:600;">[DEFENDED]</span>'
                )
                resp_preview = ""
                if self.include_responses and pr.get("response_preview"):
                    preview = pr["response_preview"][:self.response_chars].replace("<", "&lt;")
                    resp_preview = f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">{preview}</div>'
                probe_rows.append(
                    f"<tr>"
                    f"<td style='padding:8px;font-size:12px;color:#374151;'>{pr.get('probe_name','?')}</td>"
                    f"<td style='padding:8px;'>{outcome}</td>"
                    f"<td style='padding:8px;font-size:12px;'>{round(pr.get('defence_score',0.5),3)}</td>"
                    f"<td style='padding:8px;font-size:11px;color:#6b7280;'>{pr.get('reason','')}{resp_preview}</td>"
                    f"</tr>"
                )
            probe_table = (
                "<table style='width:100%;border-collapse:collapse;margin-top:12px;'>"
                "<thead><tr style='background:#f3f4f6;'>"
                "<th style='text-align:left;padding:8px;font-size:11px;color:#6b7280;'>Probe</th>"
                "<th style='text-align:left;padding:8px;font-size:11px;color:#6b7280;'>Outcome</th>"
                "<th style='text-align:left;padding:8px;font-size:11px;color:#6b7280;'>Score</th>"
                "<th style='text-align:left;padding:8px;font-size:11px;color:#6b7280;'>Reason</th>"
                "</tr></thead><tbody>"
                + "".join(probe_rows)
                + "</tbody></table>"
            )

            cat_cards.append(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <div style="font-size:15px;font-weight:700;">{label}</div>
                {risk_badge(risk)}
              </div>
              <div style="display:flex;gap:24px;margin-bottom:8px;font-size:13px;color:#374151;">
                <span>Probes run: <b>{probes}</b></span>
                <span>Attacks succeeded: <b style="color:{color};">{succeeded}</b></span>
                <span>Success rate: <b style="color:{color};">{pct}%</b></span>
              </div>
              <div style="background:#e5e7eb;border-radius:6px;height:10px;overflow:hidden;margin-bottom:8px;">
                <div style="width:{pct}%;background:{color};height:100%;border-radius:6px;"></div>
              </div>
              {probe_table}
            </div>""")

        # Recommendations for flagged categories
        rec_items = "".join(
            f"<li style='margin-bottom:8px;'>"
            f"<b>{_CATEGORY_LABELS.get(c, c)}</b>: {_RECOMMENDATIONS.get(c, 'Review this category.')}"
            f"</li>"
            for c in flagged
        ) if flagged else "<li>No categories flagged. No immediate action required.</li>"

        eid = result.report_id[:8]

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Red Team Report -- {eid}</title>
  <style>
    * {{ box-sizing:border-box;margin:0;padding:0; }}
    body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           background:#f9fafb;color:#111827;padding:32px 16px; }}
    .container {{ max-width:860px;margin:0 auto; }}
    h1 {{ font-size:22px;font-weight:700;margin-bottom:4px; }}
    .sub {{ color:#6b7280;font-size:14px;margin-bottom:24px; }}
    .card {{ background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);
             padding:24px;margin-bottom:20px; }}
    .verdict {{ background:{vc['bg']};border:2px solid {vc['border']};border-radius:12px;
               padding:20px 24px;display:flex;align-items:center;gap:16px;margin-bottom:20px; }}
    .vtext {{ font-size:28px;font-weight:800;color:{vc['text']}; }}
    .meta {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
             gap:12px;margin-bottom:20px; }}
    .mi {{ background:#f3f4f6;border-radius:8px;padding:12px 14px; }}
    .ml {{ font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px; }}
    .mv {{ font-size:14px;font-weight:600;margin-top:3px;word-break:break-all; }}
    .sec-title {{ font-size:15px;font-weight:700;margin-bottom:16px;color:#374151; }}
    footer {{ text-align:center;font-size:12px;color:#9ca3af;margin-top:32px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>Red Team Assessment Report</h1>
  <p class="sub">AI Red Teaming Framework -- Project 3 of 4 -- ID: {result.report_id}</p>

  <div class="verdict">
    <div style="flex:1;">
      <div class="vtext">{verdict}</div>
      <div style="font-size:14px;color:{vc['text']};margin-top:4px;">
        Robustness Score: {score_pct}% -- {result.total_probes} probes run,
        {result.successful_attacks} attacks succeeded
      </div>
    </div>
    <div style="font-size:52px;font-weight:800;color:{vc['text']};">{score_pct}%</div>
  </div>

  <div class="meta">
    <div class="mi"><div class="ml">Report ID</div><div class="mv">{eid}</div></div>
    <div class="mi"><div class="ml">Timestamp</div>
      <div class="mv">{result.timestamp[:19].replace('T',' ')} UTC</div></div>
    <div class="mi"><div class="ml">Target Model</div>
      <div class="mv">{result.target_model}</div></div>
    <div class="mi"><div class="ml">Judge Model</div>
      <div class="mv">{result.judge_model}</div></div>
    <div class="mi"><div class="ml">Total Probes</div>
      <div class="mv">{result.total_probes}</div></div>
    <div class="mi"><div class="ml">Latency</div>
      <div class="mv">{result.latency_ms} ms</div></div>
  </div>

  <div class="card">
    <div class="sec-title">Recommended Actions</div>
    <ul style="padding-left:20px;font-size:13px;color:#374151;line-height:1.8;">
      {rec_items}
    </ul>
  </div>

  {"".join(cat_cards)}

  <footer>
    AI Red Teaming Framework -- Built by Abhinit Sen -- {datetime.utcnow().year}
    &middot; github.com/abh1nitsen
    &middot; linkedin.com/in/abhinit-sen-63443015
  </footer>
</div>
</body>
</html>"""
