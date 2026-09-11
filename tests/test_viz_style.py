"""Verify the publication style actually applies and figures get written to disk."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI/test environments

import matplotlib.pyplot as plt

from vqe_nisq_project.viz import CATEGORICAL_COLORS, MARKERS, apply_style, savefig_both


def test_categorical_colors_and_markers_are_paired_one_to_one() -> None:
    assert len(CATEGORICAL_COLORS) == len(MARKERS)
    assert len(set(CATEGORICAL_COLORS)) == len(CATEGORICAL_COLORS)  # no duplicate hues


def test_categorical_colors_are_valid_hex() -> None:
    for color in CATEGORICAL_COLORS:
        assert color.startswith("#")
        assert len(color) == 7
        int(color[1:], 16)  # raises ValueError if not valid hex


def test_savefig_both_writes_nonempty_pdf_and_png(tmp_path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])

    pdf_path, png_path = savefig_both(fig, tmp_path, "unit_test_figure")
    plt.close(fig)

    assert pdf_path.exists()
    assert png_path.exists()
    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
    assert pdf_path.suffix == ".pdf"
    assert png_path.suffix == ".png"


def test_sequential_colormap_is_registered_and_monotonic_in_lightness() -> None:
    apply_style()
    cmap = plt.get_cmap("vqe_seq_blue")
    low = cmap(0.0)
    high = cmap(1.0)
    # Sequential ramp: high end must be darker (lower luminance) than the low end.
    luminance_low = 0.299 * low[0] + 0.587 * low[1] + 0.114 * low[2]
    luminance_high = 0.299 * high[0] + 0.587 * high[1] + 0.114 * high[2]
    assert luminance_high < luminance_low
