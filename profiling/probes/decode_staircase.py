"""CLI: decode staircase probe (decode saturation, e_w_decode, power cap)."""

from _cli import base_parser, execute
from schedule import build_decode_staircase


def main():
    p = base_parser(__doc__)
    p.add_argument("--prompt-len", type=int, default=8)
    p.add_argument("--output-len", type=int, default=2048)
    args = p.parse_args()
    schedule = build_decode_staircase(
        args.max_num_seqs, hold_s=args.hold_s,
        prompt_len=args.prompt_len, output_len=args.output_len,
    )
    print(execute(schedule, args))


if __name__ == "__main__":
    main()
