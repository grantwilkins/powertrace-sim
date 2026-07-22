"""Paper figures must be byte-stable across identical regenerations.

A wrong renderer can preserve every plotted value while embedding wall-clock
PDF metadata, invalidating the paper manifest on every run.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paper.render import save_figure


def _render(path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])
    save_figure(figure, path)
    plt.close(figure)


def test_pdf_render_is_byte_stable(tmp_path: Path) -> None:
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    _render(first)
    _render(second)
    assert first.read_bytes() == second.read_bytes()
