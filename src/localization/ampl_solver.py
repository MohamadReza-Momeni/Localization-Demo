import numpy as np
from amplpy import AMPL
from src.localization.base_solver import BaseSolver
from src.experiments.context import RunContext

class AMPLSolver(BaseSolver):
    def __init__(self, solver_name="bonmin"):
        self.solver_name = solver_name

    def solve(self, ctx: RunContext) -> dict:
        ampl = AMPL()
        
        # 1. Define the mathematical model in AMPL syntax
        # We add 1e-6 inside the sqrt to prevent derivative explosion at zero distance
        ampl.eval("""
            param N;
            param ax{1..N};
            param ay{1..N};
            param dist{1..N};
            
            param x_min;
            param x_max;
            param y_min;
            param y_max;
            
            param x_guess;
            param y_guess;

            var x >= x_min, <= x_max, := x_guess;
            var y >= y_min, <= y_max, := y_guess;

            minimize Target_Error:
                sum{i in 1..N} (sqrt((x - ax[i])^2 + (y - ay[i])^2 + 1e-6) - dist[i])^2;
        """)

        # 2. Map Python data to AMPL parameters
        N = len(ctx.anchors)
        ampl.param['N'] = N
        
        # amplpy expects 1-indexed dictionaries for array parameters
        ampl.param['ax'] = {i+1: float(ctx.anchors[i, 0]) for i in range(N)}
        ampl.param['ay'] = {i+1: float(ctx.anchors[i, 1]) for i in range(N)}
        ampl.param['dist'] = {i+1: float(ctx.distances[i]) for i in range(N)}
        
        ampl.param['x_min'] = ctx.x_range[0]
        ampl.param['x_max'] = ctx.x_range[1]
        ampl.param['y_min'] = ctx.y_range[0]
        ampl.param['y_max'] = ctx.y_range[1]

        # Warm start
        if ctx.baseline_guess is not None:
            ampl.param['x_guess'] = ctx.baseline_guess[0]
            ampl.param['y_guess'] = ctx.baseline_guess[1]
        else:
            ampl.param['x_guess'] = (ctx.x_range[0] + ctx.x_range[1]) / 2.0
            ampl.param['y_guess'] = (ctx.y_range[0] + ctx.y_range[1]) / 2.0

        # 3. Configure and execute the solver
        ampl.set_option('solver', self.solver_name)
        ampl.set_option('outlev', 0) # Suppress console output for batch runs
        
        try:
            ampl.solve()
            
            # 4. Extract results
            solve_result = ampl.get_value('solve_result')
            success = solve_result == 'solved'
            
            x_est = ampl.get_variable('x').value()
            y_est = ampl.get_variable('y').value()
            
            return {
                "solution": np.array([x_est, y_est]),
                "success": success
            }
            
        except Exception:
            # Fallback if the solver crashes
            fallback = ctx.baseline_guess if ctx.baseline_guess is not None else np.array([0.0, 0.0])
            return {"solution": fallback, "success": False}