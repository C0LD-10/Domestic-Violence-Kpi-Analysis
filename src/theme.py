"""
src/theme.py

One shared visual design system, driven by config.yaml's `theme:` section,
so the notebook, the pipeline's saved PNGs, and the Streamlit app all render
with the same restrained, non-alarmist palette (see README for rationale).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.config import AppConfig


def sequential_palette(theme: dict[str, str]) -> list[str]:
    """An ordered color list for plots needing >1 series (e.g. multi-year overlays)."""
    return [theme["ink"], theme["teal"], theme["amber"], theme["rose"], theme["grey"]]


def apply_mpl_rcparams(theme: dict[str, str]) -> None:
    """Set global matplotlib defaults so every figure inherits the theme automatically."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = theme["paper"]
    plt.rcParams["axes.facecolor"] = theme["paper"]
    plt.rcParams["savefig.facecolor"] = theme["paper"]


def apply_theme(ax: Axes, theme: dict[str, str], title: str = None,
                 xlabel: str = None, ylabel: str = None) -> Axes:
    """
    Apply the shared visual theme to a single matplotlib Axes.

    This is the one function every chart in the project (pipeline figures,
    the notebook, the app) should route through, so a palette change in
    config.yaml propagates everywhere without touching plotting code.
    """
    ax.set_facecolor(theme["paper"])
    ax.figure.set_facecolor(theme["paper"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grey"])
    ax.tick_params(colors=theme["text"], labelsize=10)
    if title:
        ax.set_title(title, fontsize=13, color=theme["ink"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=theme["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=theme["text"])
    return ax


def new_fig(figsize: tuple[float, float] = (9, 4.5), dpi: int = 130):
    """Create a figure/axes pair at a consistent DPI (130 = crisp PNGs for reports/app)."""
    return plt.subplots(figsize=figsize, dpi=dpi)


def configure(config: AppConfig) -> dict[str, str]:
    """Convenience one-liner: set rcParams and return the theme dict, given an AppConfig."""
    apply_mpl_rcparams(config.theme)
    return config.theme
