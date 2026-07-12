"""CLI: fixed-batch x context decode grid for up-range response shape."""

from _cli import base_parser, execute
from schedule import build_decode_context_grid


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--output-len", type=int, default=512)
    args = parser.parse_args()
    schedule = build_decode_context_grid(
        batches=tuple(args.batches), contexts=tuple(args.contexts),
        hold_s=args.hold_s, output_len=args.output_len,
    )
    args.max_model_len = max(
        args.max_model_len, schedule.server_overrides["max_model_len"]
    )
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
