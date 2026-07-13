"""Serializes an evaluation report dict to JSON and a short human-readable
Markdown summary under data/reports/ or artifacts/metrics/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def render_markdown_summary(report: dict[str, Any], *, title: str = "Evaluation Report") -> str:
    cm = report.get("confusion_matrix", {})
    lines = [
        f"# {title}",
        "",
        f"- Samples evaluated: {report.get('n_samples')}",
        f"- Decision threshold: {report.get('threshold_used'):.4f}"
        if "threshold_used" in report
        else "",
        "",
        "## Ranking / calibration metrics",
        f"- ROC-AUC: {report.get('roc_auc'):.4f}",
        f"- PR-AUC (Average Precision): {report.get('pr_auc'):.4f}",
        f"- Brier score: {report.get('brier_score'):.4f}",
        f"- Log loss: {report.get('log_loss'):.4f}",
        f"- KS statistic: {report.get('ks_statistic'):.4f}",
        "",
        "## Threshold-dependent metrics",
        f"- Precision: {report.get('precision'):.4f}",
        f"- Recall: {report.get('recall'):.4f}",
        f"- F1: {report.get('f1_score'):.4f}",
        f"- Specificity: {report.get('specificity'):.4f}",
        f"- Balanced accuracy: {report.get('balanced_accuracy'):.4f}",
        "",
        "## Confusion matrix",
        f"- TP={cm.get('true_positive')} FP={cm.get('false_positive')} "
        f"FN={cm.get('false_negative')} TN={cm.get('true_negative')}",
    ]
    if "business_cost" in report:
        bc = report["business_cost"]
        lines += [
            "",
            "## Business cost",
            f"- Total cost: {bc.get('total_cost'):.2f}",
            f"- Average cost per decision: {bc.get('average_cost_per_decision'):.4f}",
        ]
    return "\n".join(line for line in lines if line is not None)


def write_markdown_report(report: dict[str, Any], path: Path, *, title: str = "Evaluation Report") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_summary(report, title=title))
    return path
