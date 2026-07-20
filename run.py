#!/usr/bin/env python
"""
run.py — single entry point for the RF Localization Simulator.

Instead of remembering long module paths like
`python -m src.experiments.clustering_main`, everything is reachable through
one command with subcommands:

    python run.py single       [main.py args]          # random per-run sampling study
    python run.py sweep        [sweep args]            # structured P0 x beta x sigma grid
    python run.py clustering   [clustering args]        # anchor-geometry (GDOP) study
    python run.py compare      [comparison args]        # cross-solver accuracy/runtime
    python run.py report-crlb        [--input ... --outdir ...]
    python run.py report-solvers     [--input ... --outdir ...]
    python run.py report-clustering  [--input ... --outdir ...]
    python run.py app                                   # launch the Streamlit dashboard
    python run.py tree                                  # print a clean project tree

All arguments after the subcommand are forwarded verbatim to the underlying
script, so e.g.:

    python run.py sweep --ple 2 3 4 --sigma 1 2 4 8 --replications 20
    python run.py compare --solvers ipopt ampl_ipopt ampl_bonmin

is identical to calling the old module directly, just shorter and discoverable.
Run `python run.py <subcommand> --help` to see that script's own options.
"""
import argparse
import runpy
import sys

# subcommand -> module that exposes a main() / is runnable as __main__
SUBCOMMANDS = {
    "single": "src.experiments.main",
    "sweep": "src.experiments.sweep_main",
    "clustering": "src.experiments.clustering_main",
    "compare": "src.experiments.solver_comparison_main",
    "report-crlb": "src.evaluation.crlb_report",
    "report-solvers": "src.evaluation.solver_comparison_report",
    "report-clustering": "src.evaluation.clustering_report",
}


def _print_help():
    print(__doc__)
    print("Available subcommands:")
    for name in list(SUBCOMMANDS) + ["app", "tree"]:
        print(f"  {name}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return

    subcommand = sys.argv[1]
    forwarded = sys.argv[2:]

    if subcommand == "app":
        # Launch Streamlit. Streamlit needs to be invoked as its own process.
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py", *forwarded]))

    if subcommand == "tree":
        _print_tree()
        return

    module = SUBCOMMANDS.get(subcommand)
    if module is None:
        print(f"Unknown subcommand: {subcommand!r}\n")
        _print_help()
        raise SystemExit(2)

    # Rebuild argv so the delegated script's argparse sees a sensible prog name
    # and only its own args, then run it as if it were __main__.
    sys.argv = [f"run.py {subcommand}", *forwarded]
    runpy.run_module(module, run_name="__main__")


def _print_tree(root="."):
    import os

    def walk(path, prefix=""):
        try:
            entries = [
                e for e in os.listdir(path)
                if e != "__pycache__"
                and not e.startswith(".")
                and not e.endswith((".pyc", ".html", ".csv"))
                and e not in ("venv", ".venv", "env")
            ]
        except PermissionError:
            return
        entries.sort()
        for i, entry in enumerate(entries):
            full = os.path.join(path, entry)
            last = i == len(entries) - 1
            # ASCII connectors so this prints cleanly on Windows consoles
            # (cp1252) as well as UTF-8 terminals.
            print(f"{prefix}{'`-- ' if last else '|-- '}{entry}")
            if os.path.isdir(full):
                walk(full, prefix + ("    " if last else "|   "))

    print(f"{os.path.basename(os.path.abspath(root))}/")
    walk(root)


if __name__ == "__main__":
    main()
