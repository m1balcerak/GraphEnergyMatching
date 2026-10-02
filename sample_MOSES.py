#!/usr/bin/env python3
"""Sample molecules from a released MOSES checkpoint."""

import argparse
from pathlib import Path
import sys


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate MOSES molecules with 225 transport steps followed by configurable mixing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a MOSES checkpoint")
    parser.add_argument("--num-samples", type=positive_int, default=100, help="number of generation attempts")
    parser.add_argument("--mixing-steps", type=nonnegative_int, default=1000, help="MCMC steps per molecule after transport")
    parser.add_argument("--batch-size", type=positive_int, default=64, help="molecules sampled together")
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="random seed (0 to 4294967295)")
    parser.add_argument("--output", type=Path, default=Path("samples.csv"), help="new CSV file for SMILES and validity flags")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.seed >= 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if output.exists():
        parser.error(f"output already exists; choose a new --output path: {output}")

    # Keep --help and argument validation usable without importing the ML stack.
    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo / "src"))
    from gem.sample_moses import sample_checkpoint

    sample_checkpoint(
        checkpoint, output, num_samples=args.num_samples,
        mixing_steps=args.mixing_steps, batch_size=args.batch_size,
        seed=args.seed, config_dir=repo / "configs",
    )


if __name__ == "__main__":
    main()
