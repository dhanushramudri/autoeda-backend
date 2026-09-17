"""Renders a completed Auto EDA run into the REAL JMAN Word template.

This does not draw a look-alike cover or invent brand styling — it opens
the actual "JMAN Word Template v1.2 (Cover Page, No Appendix)" file (the
official corporate template, converted once from .dotx to a valid .docx —
see app/eda/assets/jman_template.docx and its README), fills in the
cover's title/subtitle placeholders and footer date, deletes the template's
demo/showcase body content, and inserts the run's real sections (headings,
tables, chart images, AI captions) using the template's OWN styles
("Heading 2", "Style1" table, "Normal") so formatting is pixel-identical
to a hand-authored JMAN document. Only the content changes per run — the
template file itself is never modified, only copied-in-memory and filled.

No LLM calls happen here; this is pure, deterministic formatting of data
the orchestrator already produced (parsed back out of the run's persisted
Markdown), so generating a report costs no extra AI tokens.
"""
import io
import re
from base64 import b64decode
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm

_TEMPLATE_PATH = Path(__file__).parent / "assets" / "jman_template.docx"

_TITLE_PLACEHOLDER = "Document Title"
_SUBTITLE_PLACEHOLDER = "Document Subtitle"
_DATE_PLACEHOLDERS = ("Month yyyy", "December 2024")
_VERSION_PLACEHOLDER = "Version x.x"


def _replace_text_everywhere(element, replacements: dict[str, str]):
    """Swaps exact-match placeholder text in every w:t node under `element`
    — covers body content plus header/footer floating text boxes, and
    both the modern-DrawingML and legacy-VML copies Word keeps of each
    floating shape (so the substitution shows correctly regardless of
    which one Word ends up rendering)."""
    for t in element.iter(qn("w:t")):
        if t.text in replacements:
            t.text = replacements[t.text]


def _fill_cover(doc: Document, title: str, subtitle: str, generated_at: str, version: str):
    replacements = {
        _TITLE_PLACEHOLDER: title,
        _SUBTITLE_PLACEHOLDER: subtitle,
        _VERSION_PLACEHOLDER: version,
    }
    date_replacements = {ph: generated_at for ph in _DATE_PLACEHOLDERS}

    _replace_text_everywhere(doc.element.body, replacements)
    for section in doc.sections:
        _replace_text_everywhere(section.footer._element, {**replacements, **date_replacements})
        _replace_text_everywhere(section.header._element, replacements)
        if section.different_first_page_header_footer:
            _replace_text_everywhere(section.first_page_footer._element, {**replacements, **date_replacements})
            _replace_text_everywhere(section.first_page_header._element, replacements)


def _clear_demo_body(doc: Document):
    """The template's body is: [cover spacer, cover section-break paragraph
    (carries the floating title/subtitle box)] + [demo showcase content:
    Heading 1..4, bullets, a sample table] + [bare trailing sectPr for the
    content section]. Strip the middle part only — the cover and the
    section break must survive untouched."""
    body = doc.element.body
    children = list(body)

    cover_break_idx = next(
        i for i, el in enumerate(children)
        if el.tag == qn("w:p") and el.find(f".//{qn('w:sectPr')}") is not None
    )
    final_sectpr_idx = len(children) - 1
    assert children[final_sectpr_idx].tag == qn("w:sectPr")

    for el in children[cover_break_idx + 1: final_sectpr_idx]:
        body.remove(el)

    # Detach the trailing sectPr so doc.add_paragraph()/add_table()/etc
    # (which always append at the current end of body) land BEFORE it —
    # re-appended by the caller once all real content has been added.
    body.remove(children[final_sectpr_idx])
    return children[final_sectpr_idx]


_TABLE_STYLE = "Style1"
_IMG_RE = re.compile(r"!\[[^\]]*\]\(data:image/png;base64,([A-Za-z0-9+/=]+)\)")


def _parse_section(body: str) -> dict:
    img_match = _IMG_RE.search(body)
    image_bytes = b64decode(img_match.group(1)) if img_match else None
    without_image = _IMG_RE.sub("", body)

    table = None
    table_match = re.search(r"(\|.+\|(?:\n\|.+\|)+)", without_image)
    if table_match:
        rows = [r.strip() for r in table_match.group(1).strip().split("\n")]
        rows = [r for r in rows if not re.match(r"^\|[\s\-:|]+\|$", r)]
        parsed_rows = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
        if parsed_rows:
            table = parsed_rows
        without_image = without_image[:table_match.start()] + without_image[table_match.end():]

    # Defensive: strip any ```fenced``` code block (e.g. a raw JSON dump from
    # an older run generated before custom_python results were rendered as
    # tables) — a wall of raw JSON reads badly in a finished report, and the
    # AI caption alongside it already explains the finding in prose.
    without_image = re.sub(r"```.*?```", "", without_image, flags=re.DOTALL)

    caption = "\n".join(
        line.strip() for line in without_image.split("\n")
        if line.strip() and not line.strip().startswith("|")
    ).strip()

    return {"image_bytes": image_bytes, "table": table, "caption": caption}


def _parse_markdown(markdown: str) -> list[dict]:
    sections = []
    parts = re.split(r"^## (.+)$", markdown, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append({"title": title, **_parse_section(body)})
    return sections


def build_docx_report(*, title: str, markdown: str, generated_at: str, business_context: str | None = None) -> bytes:
    """Returns the finished .docx as bytes, built from the real JMAN
    template. `title` is the AutoEdaRun's own persisted `title` (e.g.
    "Automated EDA — <workspace name>", already computed by
    auto_eda_orchestrator so the cover and body headings stay consistent
    with the live report view). `markdown` is the run's persisted
    `markdown` field; `generated_at` is a pre-formatted "Month YYYY"
    display string."""
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"JMAN template not found at {_TEMPLATE_PATH} — see app/eda/assets/README.md"
        )

    doc = Document(str(_TEMPLATE_PATH))
    _fill_cover(
        doc,
        title=title,
        subtitle="Autonomous exploratory data analysis, generated by AutoEDA",
        generated_at=generated_at,
        version="Auto-generated",
    )
    trailing_sectpr = _clear_demo_body(doc)

    table_style_names = {s.name for s in doc.styles}

    # Heading levels in the JMAN template carry fixed colors (see
    # report_builder module docstring / README): Heading 1 is dark indigo,
    # Heading 2 primary indigo, Heading 3 rose/pink — matching the
    # reference doc's "1.1 Section" (indigo) / "1.1.1 Subsection" (pink)
    # pattern. Each Auto EDA run is one H1 "chapter" containing one H2
    # umbrella section, with every individual analysis as a pink H3 below it.
    doc.add_heading(title, level=1)
    doc.add_heading("Exploratory Data Analysis", level=2)

    if business_context and business_context.strip():
        p = doc.add_paragraph()
        run = p.add_run(f"Business context: {business_context.strip()}")
        run.italic = True

    for item in _parse_markdown(markdown):
        doc.add_heading(item["title"], level=3)

        if item["table"]:
            rows = item["table"]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            if _TABLE_STYLE in table_style_names:
                table.style = _TABLE_STYLE
            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    if c_idx >= len(table.columns):
                        continue
                    table.cell(r_idx, c_idx).text = cell_text.replace("**", "")

        if item["image_bytes"]:
            doc.add_picture(io.BytesIO(item["image_bytes"]), width=Cm(15))

        if item["caption"]:
            doc.add_paragraph(item["caption"])

    doc.element.body.append(trailing_sectpr)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
