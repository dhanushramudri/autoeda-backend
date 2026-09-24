"""Server-side chart rendering for Auto EDA reports.

Every other chart in the product renders client-side in React (Recharts/
Plotly/D3) — there's no server-side visualization anywhere else. Auto EDA
needs actual image files it can embed into a portable, self-contained
Markdown/Word report, so this renders the same shapes of data as static PNGs
via matplotlib's headless ('Agg') backend and returns them as base64 data
URIs — embedded directly in the Markdown, not stored anywhere, so the
report never depends on an external URL that could expire or break.

Colors are JMAN's actual brand palette (extracted from the JMAN_1 Office
theme shared by the reference EDA doc and slide deck), not arbitrary picks:
  dk2/primary   #3411A3  deep indigo — brand color, used for single-series
                         charts and as the "positive" end of diverging scales
  lt2           #19105B  darker indigo — cover page gradient only
  accent1       #FF6196  pink
  accent2       #71EAE1  mint
  accent3       #26D4F0  cyan
  accent4       #A16BDB  lavender
  accent5       #A6265E  wine — "negative"/warning end of diverging scales
  accent6       #16978E  teal
  text          #1D1C1C
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless — no display server on the backend host
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Real JMAN brand hexes (see module docstring) — not a placeholder palette.
BRAND_PRIMARY = "#3411A3"
BRAND_TEXT = "#1D1C1C"
BRAND_PANEL_BG = "#FAFAFA"
ACCENT_PINK = "#FF6196"
ACCENT_MINT = "#71EAE1"
ACCENT_CYAN = "#26D4F0"
ACCENT_LAVENDER = "#A16BDB"
ACCENT_WINE = "#A6265E"
ACCENT_TEAL = "#16978E"

# One qualitative palette, used in this fixed order wherever a chart needs
# more than one series/category color (e.g. a future grouped-bar chart).
QUALITATIVE_PALETTE = [BRAND_PRIMARY, ACCENT_PINK, ACCENT_TEAL, ACCENT_LAVENDER, ACCENT_CYAN, ACCENT_WINE, ACCENT_MINT]

# Diverging scale for correlation-style matrices: wine (negative) -> white -> indigo (positive).
_DIVERGING_CMAP = LinearSegmentedColormap.from_list("jman_diverging", [ACCENT_WINE, "#FFFFFF", BRAND_PRIMARY])

_FIGSIZE = (7, 4)
_DPI = 110

matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
matplotlib.rcParams["text.color"] = BRAND_TEXT
matplotlib.rcParams["axes.labelcolor"] = BRAND_TEXT
matplotlib.rcParams["xtick.color"] = BRAND_TEXT
matplotlib.rcParams["ytick.color"] = BRAND_TEXT
matplotlib.rcParams["axes.edgecolor"] = "#CCCCCC"


def _to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _new_fig(figsize=_FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BRAND_PANEL_BG)
    return fig, ax


def render_bar(labels: list[str], values: list[float], title: str, ylabel: str = "") -> str:
    # Each bar is a distinct category (a column, a segment, a value) —
    # give each its own brand-toned color, cycling the palette, rather than
    # one flat color for the whole chart (matches the reference EDA doc's
    # own bar charts, e.g. "Customers by Population Segment").
    colors = [QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i in range(len(labels))]
    fig, ax = _new_fig()
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:,.2f}" if isinstance(v, float) and not v.is_integer() else f"{v:,.0f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return _to_data_uri(fig)


def render_histogram(values: list[float], title: str, xlabel: str = "") -> str:
    fig, ax = _new_fig()
    ax.hist(values, bins=min(30, max(5, len(set(values)))), color=BRAND_PRIMARY, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _to_data_uri(fig)


def render_heatmap(cols: list[str], matrix: list[list[float | None]], title: str) -> str:
    fig, ax = plt.subplots(figsize=(max(5, len(cols) * 0.9), max(4, len(cols) * 0.8)))
    fig.patch.set_facecolor("white")
    data = [[v if v is not None else 0.0 for v in row] for row in matrix]
    im = ax.imshow(data, cmap=_DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = matrix[i][j]
            if v is not None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(v) > 0.5 else BRAND_TEXT)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _to_data_uri(fig)


def render_line(x: list, y: list, title: str, ylabel: str = "") -> str:
    fig, ax = _new_fig()
    ax.plot(x, y, color=BRAND_PRIMARY, linewidth=1.75)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _to_data_uri(fig)


def render_rate_bar(
    categories: list[str], rates: list[float], title: str,
    counts: list[int] | None = None, total_n: int | None = None, rate_label: str = "Rate",
) -> str:
    """Horizontal bar of a target rate (e.g. % churn) per category bucket —
    the single-feature "deep dive" shape used throughout the reference EDA
    deck (e.g. churn rate by connection count, by membership band). When
    counts + total_n are given, each y-tick also shows that bucket's share
    of all accounts, so a big bar on a tiny bucket doesn't read as more
    important than a small bar covering most of the base."""
    if counts is not None and total_n:
        labels = [f"{c}  ({100 * n / total_n:.0f}% of accounts)" for c, n in zip(categories, counts)]
    else:
        labels = categories
    fig, ax = _new_fig((7, max(3.2, 0.55 * len(categories))))
    y = range(len(categories))
    bars = ax.barh(list(y), rates, color=ACCENT_PINK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # first category at the top, matching the source table order
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.set_xlabel(rate_label, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for b, v in zip(bars, rates):
        ax.annotate(f"{v:.0f}%", (b.get_width(), b.get_y() + b.get_height() / 2),
                    ha="left", va="center", fontsize=8, xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    return _to_data_uri(fig)


def render_rate_line(x: list[float], rates: list[float], title: str, xlabel: str, ylabel: str = "Rate", trend: bool = False) -> str:
    """Smoothed target-rate-vs-continuous-feature line — e.g. churn rate by
    tenure — the other deep-dive shape in the reference deck, distinct from
    render_rate_bar because the feature is numeric/ordered rather than a
    small set of named buckets. An optional dotted linear trend line names
    the direction in one glance, same as the reference deck's tenure slide."""
    fig, ax = _new_fig()
    ax.plot(x, rates, color=ACCENT_PINK, linewidth=2.2, label=ylabel)
    if trend and len(x) >= 2:
        import numpy as np
        coeffs = np.polyfit(x, rates, 1)
        trend_y = [coeffs[0] * xi + coeffs[1] for xi in x]
        ax.plot(x, trend_y, color="#999999", linewidth=1.2, linestyle=":", label=f"Linear ({ylabel})")
        ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _to_data_uri(fig)


def render_rate_heatmap(
    row_labels: list[str], col_labels: list[str],
    rate_matrix: list[list[float | None]], count_matrix: list[list[int | None]], title: str,
) -> str:
    """Two-way interaction matrix — a target rate for every (row category,
    column category) pair, each cell annotated with both the rate and the
    underlying count (a 40% rate on 3 accounts and a 40% rate on 3,000
    accounts are not equally trustworthy). Matches the reference deck's
    "Connections & Bands" style matrices — the highest-value chart a real
    analyst reaches for once two categorical drivers are both known to
    matter individually."""
    n_rows, n_cols = len(row_labels), len(col_labels)
    fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.1), max(3.5, n_rows * 0.7)))
    fig.patch.set_facecolor("white")
    cmap = LinearSegmentedColormap.from_list("jman_sequential", ["#FFFFFF", ACCENT_TEAL])
    data = [[v if v is not None else 0.0 for v in row] for row in rate_matrix]
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=max(1.0, max((v for row in data for v in row), default=1.0)))
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(n_rows):
        for j in range(n_cols):
            rate, count = rate_matrix[i][j], count_matrix[i][j]
            if rate is None:
                continue
            color = "white" if rate > (im.get_clim()[1] * 0.6) else BRAND_TEXT
            ax.text(j, i, f"{rate:.0f}%\n{count:,}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rate")
    fig.tight_layout()
    return _to_data_uri(fig)


def render_box(labels: list[str], data: list[list[float]], title: str) -> str:
    fig, ax = _new_fig()
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color="white", linewidth=1.5),
                     whiskerprops=dict(color=BRAND_TEXT),
                     capprops=dict(color=BRAND_TEXT))
    # One color per category, same convention as render_bar (matches the
    # reference doc's own multi-colored box plots).
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)])
        box.set_edgecolor(BRAND_TEXT)
    ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _to_data_uri(fig)
