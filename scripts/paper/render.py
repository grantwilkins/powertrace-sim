"""Deterministic figure output for maintained paper producers."""
from pathlib import Path

PDF_METADATA = {"CreationDate": None, "ModDate": None}


def save_figure(figure, path: str | Path, **kwargs) -> None:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        kwargs["metadata"] = PDF_METADATA
    figure.savefig(path, **kwargs)
