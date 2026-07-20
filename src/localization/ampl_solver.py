"""
src/localization/ampl_solver.py

Generic "AMPL -> Tools optimization" comparison harness from the
professor's notes: the same RSSI-domain nonlinear least-squares model
(identical objective to IPOPTSolver's RSSIDomainProblem) is posed once in
AMPL, and any AMPL-registered solver can be swapped in via `solver_name`.

SOLVER_CAPABILITIES below is load-bearing, not decorative: this model is a
smooth, nonconvex, CONTINUOUS 2-variable NLP. CBC and NVIDIA cuOpt are
LP/MIP(/routing)-oriented solvers that cannot represent a nonlinear
objective at all — calling them here raises a clear error rather than
silently returning a meaningless result. See the class docstring for where
they'd actually be a good fit instead.
"""
import contextlib
import io
import os
import sys

import numpy as np
from .base_solver import BaseSolver
from .multistart import generate_starting_points


@contextlib.contextmanager
def _suppress_native_stdout():
    """Silence the very chatty native solver banners (Ipopt/Bonmin print
    directly to the C-level stdout/stderr file descriptors, so a plain
    contextlib.redirect_stdout is not enough — we redirect the OS-level fds).

    Without this, a sweep of a few hundred AMPL solves floods the terminal
    with thousands of lines of solver banners, drowning out the actual
    experiment progress/results. Errors are still raised normally; only the
    solvers' own stdout/stderr chatter is dropped.
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # No real fds (e.g. captured stdout in some test runners) — nothing to do.
        yield
        return

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


SOLVER_CAPABILITIES = {
    "ipopt": {
        "continuous_nlp": True,
        "notes": "Local interior-point NLP solver.",
    },
    "bonmin": {
        "continuous_nlp": True,
        "notes": "MINLP solver; with zero integer variables it reduces to an NLP "
                 "solve (internally via Ipopt's B-BB algorithm).",
    },
    "scip": {
        "continuous_nlp": True,
        "notes": "Global MINLP/NLP solver via spatial branch-and-bound. Slower than "
                 "IPOPT/BonMin, but reports an optimality gap rather than only a "
                 "local solution — useful as a 'how far from global optimum' check.",
    },
    "cbc": {
        "continuous_nlp": False,
        "notes": "Pure LP/MIP solver — cannot represent the nonlinear log-distance "
                 "RSSI objective at all. Not applicable to per-target localization "
                 "as posed; would need the problem reformulated as a discretized "
                 "candidate-position MIP, which is a different (lossy) model. A "
                 "better fit for CBC in this project is anchor-placement/clustering "
                 "optimization (choosing anchor locations is genuinely combinatorial).",
    },
    "cuopt": {
        "continuous_nlp": False,
        "notes": "GPU-accelerated LP/MIP/routing solver — same limitation as CBC for "
                 "this smooth nonconvex NLP. NVIDIA markets it for large-scale "
                 "LP/MIP/vehicle-routing, not continuous NLP.",
    },
}


class AMPLSolver(BaseSolver):
    """Solves the RSSI-domain NLP through AMPL, routed to whichever solver
    is named. Only solvers with continuous_nlp=True in SOLVER_CAPABILITIES
    can solve this model as posed; others raise a ValueError explaining why,
    rather than being silently called and returning nonsense.

    multi_start / starting_points behave identically to IPOPTSolver (same
    generate_starting_points helper), so comparisons between "ipopt"
    (via IPOPTSolver/cyipopt) and "ipopt" (via this AMPL path) or "bonmin"/
    "scip" isolate the solver's own behavior rather than incidental
    initialization differences.
    """

    MODEL_TEMPLATE = """
    param n_anchors integer > 0;
    set ANCHORS := 1..n_anchors;
    param ax {ANCHORS};
    param ay {ANCHORS};
    param rssi_meas {ANCHORS};
    param w {ANCHORS} default 1;
    param P0;
    param ple;
    param x_min; param x_max; param y_min; param y_max;

    var x >= x_min, <= x_max;
    var y >= y_min, <= y_max;

    minimize rssi_sse:
        sum {i in ANCHORS} w[i] *
            (rssi_meas[i] - (P0 - 10*ple*log10(sqrt((x-ax[i])^2 + (y-ay[i])^2 + 1e-9))))^2;
    """

    def solve(self, anchors, distances, x0=None, ref_power=-40, ple=2.2, weights=None,
              x_range=(0, 1000), y_range=(0, 1000), solver_name="ipopt",
              solver_options: str = "", multi_start=True,
              starting_points=("warm_start", "anchor_centroid", "random_point"),
              verbose=False):
        """
        solver_name: one of SOLVER_CAPABILITIES' keys ("ipopt", "bonmin", "scip"
            are usable here; "cbc"/"cuopt" raise a ValueError — see module docstring).
        solver_options: raw AMPL solver-options string, e.g. "tol=1e-6 max_iter=500"
            for ipopt, passed straight through as `ampl.set_option(f"{solver_name}_options", ...)`.
            Left as a raw passthrough rather than a per-solver dataclass (like
            IPOPTHyperparams) because option names/semantics differ across
            solvers and this project can't independently verify every AMPL
            solver's exact option grammar — check that solver's AMPL
            documentation before relying on a specific flag.
        """
        capability = SOLVER_CAPABILITIES.get(solver_name)
        if capability is None:
            raise ValueError(
                f"Unknown solver_name '{solver_name}'. Known solvers: {list(SOLVER_CAPABILITIES)}"
            )
        if not capability["continuous_nlp"]:
            raise ValueError(
                f"Solver '{solver_name}' cannot solve this problem as posed: {capability['notes']}"
            )

        try:
            from amplpy import AMPL
        except ImportError as exc:
            raise RuntimeError(
                "amplpy is not installed in this environment. Install with "
                "`pip install amplpy`, then register solvers with "
                "`python -m amplpy.modules install <solver>` "
                "(requires an AMPL license; a limited free/community license "
                "is available from ampl.com)."
            ) from exc

        anchors = np.asarray(anchors, dtype=float)
        distances = np.asarray(distances, dtype=float)
        rssi_meas = ref_power - 10.0 * ple * np.log10(distances + 1e-9)
        if weights is None:
            weights = np.ones(len(anchors))
        else:
            weights = np.asarray(weights, dtype=float)

        starts = generate_starting_points(anchors, x0, x_range, y_range, starting_points)
        labels_to_run = starting_points if multi_start else (starting_points[0],)

        # Silence the native solver banners unless the caller explicitly wants
        # the per-candidate verbose trace. See _suppress_native_stdout.
        silence = _suppress_native_stdout() if not verbose else contextlib.nullcontext()

        candidates = []
        with silence:
          for label in labels_to_run:
            start = starts[label]
            start_clipped = np.array([
                np.clip(start[0], x_range[0], x_range[1]),
                np.clip(start[1], y_range[0], y_range[1]),
            ])

            ampl = AMPL()
            ampl.eval(self.MODEL_TEMPLATE)

            n = len(anchors)
            ampl.param["n_anchors"] = n
            ampl.param["ax"] = {i + 1: float(anchors[i, 0]) for i in range(n)}
            ampl.param["ay"] = {i + 1: float(anchors[i, 1]) for i in range(n)}
            ampl.param["rssi_meas"] = {i + 1: float(rssi_meas[i]) for i in range(n)}
            ampl.param["w"] = {i + 1: float(weights[i]) for i in range(n)}
            ampl.param["P0"] = float(ref_power)
            ampl.param["ple"] = float(ple)
            ampl.param["x_min"] = float(x_range[0])
            ampl.param["x_max"] = float(x_range[1])
            ampl.param["y_min"] = float(y_range[0])
            ampl.param["y_max"] = float(y_range[1])

            ampl.get_variable("x").set_value(float(start_clipped[0]))
            ampl.get_variable("y").set_value(float(start_clipped[1]))

            ampl.set_option("solver", solver_name)
            if solver_options:
                ampl.set_option(f"{solver_name}_options", solver_options)

            ampl.solve()
            solve_result = ampl.get_value("solve_result")
            x_opt = ampl.get_variable("x").value()
            y_opt = ampl.get_variable("y").value()
            objective_val = ampl.get_objective("rssi_sse").value()

            candidates.append({
                "label": label,
                "start": start_clipped,
                "solution": np.array([x_opt, y_opt]),
                "objective": objective_val,
                "solve_result": solve_result,
            })

            if verbose:
                print(
                    f"  [ampl:{solver_name}] {label:<15} "
                    f"start=({start_clipped[0]:7.2f},{start_clipped[1]:7.2f})  "
                    f"-> solution=({x_opt:7.2f},{y_opt:7.2f})  "
                    f"objective={objective_val:.4f}  result={solve_result}"
                )

            ampl.close()

        best = min(candidates, key=lambda c: c["objective"])

        if verbose:
            print(f"  [ampl:{solver_name}] BEST: {best['label']} (objective={best['objective']:.4f})")

        return {
            "solution": best["solution"],
            "success": best["solve_result"] == "solved",
            "solver_name": solver_name,
            "chosen_start": best["label"],
            "objective": best["objective"],
            "multi_start_candidates": candidates,
        }
