"""CLI: transient steps probe (lag-filter constant alpha)."""

from _cli import base_parser, execute
from schedule import build_transients


def main():
    p = base_parser(__doc__)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--on-s", type=float, default=20.0)
    p.add_argument("--off-s", type=float, default=20.0)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    schedule = build_transients(
        concurrency=args.concurrency, on_s=args.on_s,
        off_s=args.off_s, repeats=args.repeats,
    )
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
