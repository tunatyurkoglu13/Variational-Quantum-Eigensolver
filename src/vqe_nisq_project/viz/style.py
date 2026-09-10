"""Publication-quality matplotlib styling shared across all figures in this repo.

Palette validated for color-vision-deficiency safety with the dataviz skill's
validator (OKLab CVD Delta-E, adjacent-pair check): worst adjacent pair Delta E
9.1 (light surface). Do not reorder CATEGORICAL_COLORS/MARKERS without re-running
that validation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

STYLE_PATH = Path(__file__).parent / "publication.mplstyle"

# Keep in sync with publication.mplstyle's axes.prop_cycle.
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Colors below this index validate as pairwise-distinguishable even when ALL
# shown together at once (e.g. every point on one scatter plot, as opposed to
# an ordered sequence of adjacent lines/bars). Beyond 3, pair with MARKERS too.
ALL_PAIRS_SAFE_COUNT = 3

# Sequential single-hue ramp (blue, light -> dark) for magnitude data
# (e.g. barren-plateau gradient-variance heatmaps). Steps 100->700 from the
# dataviz skill's reference palette.
_SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#1c5cab", "#104281", "#0d366b",
]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "vqe_seq_blue", _SEQUENTIAL_BLUE_STEPS
)


def apply_style() -> None:
    """Apply the shared publication style to all subsequent matplotlib figures."""
    plt.style.use(str(STYLE_PATH))
    if "vqe_seq_blue" in plt.colormaps:
        plt.colormaps.unregister("vqe_seq_blue")
    plt.colormaps.register(SEQUENTIAL_BLUE, name="vqe_seq_blue")


def savefig_both(fig: Figure, out_dir: Path, name: str) -> tuple[Path, Path]:
    """Save fig as both name.pdf (report) and name.png (README/notebooks).

    Returns the (pdf_path, png_path) written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    return pdf_path, png_path
