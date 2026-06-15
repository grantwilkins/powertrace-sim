"""CLI: prefill staircase probe (e_f_prefill, attention-L^2, TTFT-vs-length).

Disables chunked prefill so prefill occupies clean, pure-prefill dwells.
"""

from _cli import base_parser, execute
from schedule import build_prefill_staircase


def main():
    p = base_parser(__doc__)
    p.add_argument("--input-lens", type=int, nargs="+",
                   default=[256, 1024, 4096, 16384, 65536])
    args = p.parse_args()
    args.enable_chunked_prefill = False  # probe requires it OFF
    schedule = build_prefill_staircase(input_lens=tuple(args.input_lens),
                                       hold_s=args.hold_s)
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
