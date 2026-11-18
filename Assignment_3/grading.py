"""Grading support. The final Pass/Fail is a human judgement (the assignment requires it); this module
only pre-fills a *suggested* grade and builds the sheet the human grader fills in."""
import os
import re
import unicodedata

import pandas as pd

NOT_IN_CONTEXT = "context does not contain the answer"


def _norm(text):
    return unicodedata.normalize("NFKC", text).lower().replace("’", "'")


def suggest_grade(answer, question, hits):
    """Keyword check: every keyword group must appear in the answer (any alternative in a group counts)."""
    a = _norm(answer)
    if NOT_IN_CONTEXT in a:
        return "F", "model said the context does not contain the answer"
    missing = [g[0] for g in question["keywords"] if not any(_norm(k) in a for k in g)]
    if missing:
        return "F", "answer is missing: " + ", ".join(missing)
    grounded = evidence_rank(question, hits) is not None
    return "P", "all key facts present" + ("; evidence passage was retrieved" if grounded
                                           else "; evidence passage NOT in retrieved chunks, check grounding")


def _squeeze(text):
    return re.sub(r"\s+", "", _norm(text))


def evidence_rank(question, hits):
    """Rank of the first retrieved chunk that contains the start of the evidence quote (None if not retrieved).

    Whitespace is ignored, and only the first 30 non-space characters are matched so that a quote split across
    two overlapping chunks still counts."""
    ev = _squeeze(question["evidence"])[:30]
    for h in hits:
        if ev in _squeeze(h["text"]):
            return h["rank"]
    return None


def write_grading_sheet(rows, path):
    """One row per (config, question). Greedy decoding makes repeats identical; any difference is flagged."""
    df = pd.DataFrame(rows)
    cols = ["config", "question_id", "question", "reference_answer", "answer", "answers_identical_across_repeats",
            "evidence_retrieved_at_rank", "retrieved_sources", "suggested_grade", "suggestion_reason",
            "final_grade", "grader_notes"]
    df["final_grade"] = ""
    df["grader_notes"] = ""
    df = df[cols]
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, sheet_name="grading", index=False)
        ws = xw.sheets["grading"]
        widths = {"A": 11, "B": 10, "C": 50, "D": 40, "E": 60, "F": 12, "G": 12, "H": 30, "I": 10, "J": 40,
                  "K": 12, "L": 30}
        from openpyxl.styles import Alignment, Font, PatternFill
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for cell in ws["K"][1:]:
            cell.fill = PatternFill("solid", fgColor="FFF2CC")
        ws.freeze_panes = "C2"
    return path


def read_final_grades(path):
    """Returns {(config, question_id): "P"/"F"} for rows a human has graded."""
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path, sheet_name="grading", dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        g = r["final_grade"].strip().upper()
        if g in ("P", "F"):
            out[(r["config"], r["question_id"])] = g
    return out
