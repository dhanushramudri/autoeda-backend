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
