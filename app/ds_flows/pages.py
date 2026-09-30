"""Page specs: how a flow stage describes its results page. One generic renderer on the frontend draws these, so every
solution (forecasting, pricing, revenue growth...) gets the same look without bespoke UI per stage.

A page is an ordered list of blocks:
  kpis | chart | table | note | bullets | checks | downloads | markdown
"""

from __future__ import annotations

from typing import Any, Sequence

from .common import clean


def kpis(*items: Sequence) -> dict:
    """items: (title, value, sub=None, pink=False)"""
    out = []
    for it in items:
        title, value = it[0], it[1]
        out.append({"title": title, "value": value, "sub": it[2] if len(it) > 2 else None, "pink": bool(it[3]) if len(it) > 3 else False})
    return {"type": "kpis", "items": out}


def chart(kind: str, title: str, data: list[dict], x: str, series: list[tuple], subtitle: str | None = None,
          fmt: str = "num", horizontal: bool = False, height: int = 260, band: bool = False) -> dict:
    """kind: line | bar | area. series: (key, name[, color]) with color in brand|pink|soft|muted.
    band=True draws the 'band' field ([low, high]) of each row as a shaded interval behind the lines."""
    return {"type": "chart", "kind": kind, "title": title, "subtitle": subtitle, "data": data, "x": x, "fmt": fmt,
            "horizontal": horizontal, "height": height, "band": band,
            "series": [{"key": s[0], "name": s[1], "color": s[2] if len(s) > 2 else None} for s in series]}


def table(title: str, head: list[str], rows: list[list[Any]], subtitle: str | None = None) -> dict:
    return {"type": "table", "title": title, "subtitle": subtitle, "head": head, "rows": rows}


def note(text: str, tone: str = "pink") -> dict:
    return {"type": "note", "text": text, "tone": tone}


def bullets(title: str | None, items: list[str]) -> dict:
    return {"type": "bullets", "title": title, "items": items}


def checks(items: list[dict]) -> dict:
    return {"type": "checks", "items": items}


def downloads(items: list[tuple], subtitle: str | None = None) -> dict:
    """items: (kind, label, format, primary=False)  kind in enriched|accounts|dictionary|model|report"""
    return {"type": "downloads", "subtitle": subtitle,
            "items": [{"kind": i[0], "label": i[1], "format": i[2], "primary": bool(i[3]) if len(i) > 3 else False} for i in items]}


def markdown(text: str, title: str | None = None) -> dict:
    return {"type": "markdown", "title": title, "text": text}


def page(*blocks: dict) -> dict:
    return clean({"blocks": [b for b in blocks if b]})


def fmt_num(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:.2f}M"
    if a >= 1e4:
        return f"{x / 1e3:.1f}K"
    if a >= 100:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"
