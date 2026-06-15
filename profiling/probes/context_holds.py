"""CLI: context-decode holds probe (e_kv / KV-read energy)."""

from _cli import base_parser, execute
from schedule import build_context_holds


def main():
    p = base_parser(__doc__)
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[2048, 8192, 32768, 131072])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output-len", type=int, default=256)
    args = p.parse_args()
    schedule = build_context_holds(
        contexts=tuple(args.contexts), batch=args.batch,
        hold_s=args.hold_s, output_len=args.output_len,
    )
    # honor the probe's max_model_len requirement
    args.max_model_len = max(args.max_model_len,
                             schedule.server_overrides["max_model_len"])
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
