"""Knowledge base: load the source PDFs page by page and clean the extracted text."""
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

from pypdf import PdfReader

from tracing import sha256_file

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
REFERENCES_HEADING = re.compile(r"^\s*(References|Bibliography)\s*$", re.I | re.M)


@dataclass
class Document:
    doc_id: str
    title: str
    file: str
    sha256: str
    text: str = ""
    page_starts: list = field(default_factory=list)   # (char offset, page number) for each kept page
    pages_total: int = 0
    pages_kept: int = 0
    references_start_page: int = None
    dropped_reference_pages: list = field(default_factory=list)

    def page_at(self, char_offset):
        page = self.page_starts[0][1]
        for start, p in self.page_starts:
            if start > char_offset:
                break
            page = p
        return page


def clean(text):
    text = unicodedata.normalize("NFKC", text)           # ligatures such as "ﬁ" -> "fi"
    lines = [l for l in text.split("\n") if l.strip()]
    while lines and re.fullmatch(r"\s*\d{1,4}\s*", lines[0]):    # page number in the header
        lines.pop(0)
    while lines and re.fullmatch(r"\s*\d{1,4}\s*", lines[-1]):   # page number in the footer
        lines.pop()
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)        # one paragraph per page
    return re.sub(r"\s{2,}", " ", text).strip()


def load_knowledge_base(drop_references=True):
    sources = json.load(open(os.path.join(KB_DIR, "sources.json")))
    docs = []
    for s in sources:
        path = os.path.join(KB_DIR, s["file"])
        reader = PdfReader(path)
        doc = Document(s["id"], s["title"], s["file"], sha256_file(path), pages_total=len(reader.pages))
        parts, offset, in_refs = [], 0, False
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            if drop_references and not in_refs:
                m = REFERENCES_HEADING.search(raw)
                if m:
                    raw, in_refs, doc.references_start_page = raw[:m.start()], True, i
            elif in_refs:
                doc.dropped_reference_pages.append(i)
                continue
            text = clean(raw)
            if not text:
                continue
            doc.page_starts.append((offset, i))
            parts.append(text)
            offset += len(text) + 1
        doc.text = " ".join(parts)
        doc.pages_kept = len(doc.page_starts)
        if s["sha256"] != doc.sha256:
            raise ValueError(f"{s['file']} does not match the hash in sources.json")
        docs.append(doc)
    return docs


def kb_stats(docs):
    return [{"doc_id": d.doc_id, "title": d.title, "file": d.file, "sha256": d.sha256,
             "pages_total": d.pages_total, "pages_kept": d.pages_kept,
             "references_start_page": d.references_start_page,
             "dropped_reference_pages": d.dropped_reference_pages,
             "chars": len(d.text), "words": len(d.text.split())} for d in docs]


def find_loose(text, quote):
    """Character offset of `quote` in `text`, ignoring whitespace and Unicode form (-1 if absent).

    PDF text extraction differs slightly between pypdf versions (e.g. "0 .954" vs "0.954"), so evidence
    quotes are matched with all whitespace removed.
    """
    norm_text = unicodedata.normalize("NFKC", text)
    positions = [i for i, ch in enumerate(norm_text) if not ch.isspace()]
    squeezed = "".join(norm_text[i] for i in positions)
    target = re.sub(r"\s+", "", unicodedata.normalize("NFKC", quote))
    j = squeezed.find(target)
    return positions[j] if j >= 0 else -1
