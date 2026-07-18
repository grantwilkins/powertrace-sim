"""CLI: idle hold probe (idle power anchor)."""

from _cli import base_parser, execute
from schedule import build_idle_hold


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    schedule = build_idle_hold(hold_s=args.hold_s)
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
