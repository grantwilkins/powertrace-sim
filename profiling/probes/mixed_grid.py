"""CLI: mixed prefill x decode grid probe (interaction term)."""

from _cli import base_parser, execute
from schedule import build_mixed_grid


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-points", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    schedule = build_mixed_grid(
        n_points=args.n_points, seed=args.seed, hold_s=args.hold_s,
        decode_range=(1, args.max_num_seqs),
    )
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
