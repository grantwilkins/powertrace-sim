#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.release import DEFAULT_ARTIFACT
from model.training.release_fit import fit_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the frozen timing and clean v4 power equations."
    )
    parser.add_argument(
        "--prepared-manifest",
        default="results/clean_model/prepared_dataset.json",
    )
    parser.add_argument("--out-artifact", required=True)
    parser.add_argument("--template-artifact", default=str(DEFAULT_ARTIFACT))
    args = parser.parse_args()
    fit_release(
        args.prepared_manifest, args.out_artifact,
        template_artifact=args.template_artifact,
    )
    print(f"[train] wrote {args.out_artifact}")


if __name__ == "__main__":
    main()

