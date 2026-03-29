#!/usr/bin/env python3
"""Unified entry point for 21cm interferometer constraints."""
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Run 21cm interferometer constraint calculations"
    )
    parser.add_argument(
        "scenario",
        choices=["single", "double", "FarView"],
        help="Which interferometer configuration to use"
    )
    args = parser.parse_args()

    if args.scenario == "single":
        from dark_ages_21cm_single_array import run
    elif args.scenario == "double":
        from dark_ages_21cm_double_array import run
    elif args.scenario == 'FarView':
        from dark_ages_21cm_FarView_array import run

    run()

if __name__ == "__main__":
    main()