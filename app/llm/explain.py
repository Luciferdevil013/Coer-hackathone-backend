"""Explanations: the LLM writes words from evidence; it never changes numbers or actions.

Only the finding's evidence (already PII-free) is sent. Template fallback keeps the product
fully functional offline.
"""
from __future__ import annotations

import json

from app.llm.gateway import get_gateway
from app.models import Finding

SAFETY = {
    "DELETE": "Deleted files go to a restorable trash, and the action can be undone from the audit log.",
    "ARCHIVE": "Archived data moves to cold storage and can be restored at any time.",
    "SECURE": "Securing restricts access to the owner. It is reversible and does not change the data.",
    "CLEAN": "Cleaning writes a new file after a dry run and your approval. The original is never modified.",
    "KEEP": "Keeping these records ensures authentic customer reviews and sentiment distributions remain accurate without accidental data loss.",
}


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def template_explanation(f: Finding) -> str:
    parts = [f"Recommended action: {f.action} ({f.confidence:.0%} confidence). {f.summary}"]
    if f.savings_inr_year >= 1:
        parts.append(f"Acting on it saves an estimated {_inr(f.savings_inr_year)} per year in storage.")
    if f.kind == "sensitive_data" and f.details.get("past_retention"):
        parts.append("Under India's DPDP Act, personal data should be erased once its purpose is served. "
                     "Notify affected users (48 hours) before erasure.")
    elif f.category == "risk":
        parts.append(f"Risk score {f.risk_score:.0f}/100 is based on data sensitivity, how many people can access it, and volume.")
    parts.append(SAFETY.get(f.action, ""))
    return " ".join(p for p in parts if p)


SYSTEM = """You are DataGuardian AI's explanation writer. Write 2-3 plain-English sentences explaining a data
recommendation to a business user. Use ONLY the facts given: never invent numbers, files, users or dates,
and never change the recommended action or the confidence. Say why the action is safe or reversible. If
relevant, mention the India DPDP Act. No markdown, no bullet points, no preamble."""


def explain(f: Finding) -> tuple[str, str]:
    facts = {
        "recommended_action": f.action, "confidence": f"{f.confidence:.0%}", "title": f.title, "summary": f.summary,
        "severity": f.severity, "estimated_annual_saving_inr": round(f.savings_inr_year),
        "risk_score": f.risk_score, "evidence": f.evidence,
        "safety": SAFETY.get(f.action, ""),
    }
    text = get_gateway().text(purpose="explain_finding", system=SYSTEM,
                              user="<facts>\n" + json.dumps(facts, default=str) + "\n</facts>",
                              workspace_id=f.workspace_id, max_tokens=1500)
    if text:
        return text, "claude"
    return template_explanation(f), "template"
