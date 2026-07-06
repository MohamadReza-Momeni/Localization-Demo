Here is a comprehensive, professional `README.md` for your project. It reflects the new Clean Code architecture, the parallel processing capabilities, the Streamlit dashboard, and the full suite of localization algorithms we implemented.

You can copy and paste this directly into a `README.md` file in the root of your project.

---

# RF Localization Simulator

A robust, highly modular Python framework for simulating, evaluating, and visualizing RF-based (Radio Frequency) localization algorithms. This project simulates Received Signal Strength Indicator (RSSI) measurements under high-noise, non-linear conditions and tests multiple optimization strategies to accurately track hidden targets.

## Features

* **Interactive Web Dashboard:** Built with Streamlit for dynamic parameter adjustment, real-time data preview, and interactive Folium map rendering.
* **Parallel Processing:** Utilizes Python's `ProcessPoolExecutor` to distribute large batches of Monte Carlo simulations across all available CPU cores.
* **Clean Code Architecture:** Designed using SOLID principles and the Strategy Pattern, making it incredibly easy to add new localization algorithms without modifying core logic.
* **Advanced Algorithmic Pipelines:** Features "Warm Starting", passing initial robust estimations from SciPy into high-precision solvers like IPOPT to prevent logarithmic gradient explosions.
* **Geospatial Mapping:** Integrates `pyproj` and `folium` to project simulated East-North-Up (ENU) coordinates onto real-world maps.

## Project Structure

```text
localization_demo/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── src/
│   ├── experiments/            # Core Simulation Orchestration
│   │   ├── config.py           # Experiment parameters and bounds
│   │   ├── context.py          # State/Data passed to solvers
│   │   ├── executor.py         # Multi-core batch processing
│   │   ├── exporter.py         # Data persistence (CSV saving)
│   │   ├── strategies.py       # Strategy Pattern registry for solvers
│   │   └── task.py             # Physics and pipeline orchestration
│   ├── evaluation/
│   │   └── metrics.py          # Euclidean error calculations
│   ├── geo/
│   │   └── enu.py              # WGS84 to ENU geospatial conversions
│   ├── localization/           # The Algorithms
│   │   ├── base_solver.py      # Abstract base class for all solvers
│   │   ├── ipopt_solver.py     # CyIPOPT RSSI-domain solver
│   │   ├── particle_filter_solver.py # Probabilistic particle filter
│   │   └── scipy_solver.py     # Levenberg-Marquardt distance solver
│   ├── scenario/               # Map Generation
│   │   ├── anchors.py          # Anchor placement logic
│   │   ├── point_generator.py  # Base random coordinate generator
│   │   └── targets.py          # Target placement logic
│   ├── signal/                 # Physics Engine
│   │   ├── distance.py         # RSSI-to-distance conversion
│   │   └── rssi.py             # Log-Distance Path Loss model
│   └── visualization/
│       └── map_generator.py    # Folium HTML map rendering

```

## Implemented Algorithms

This framework tests targets against several distinct mathematical approaches:

1. **Vanilla (SciPy):** A robust Least-Squares optimizer (`Levenberg-Marquardt`) that operates in the linear distance domain. Excellent for fast, reliable baseline guesses.
2. **Weighted (SciPy):** Applies geometric weighting to the Vanilla solver, prioritizing anchors closer to the center of the grid.
3. **IPOPT:** An interior-point non-linear optimizer operating directly in the logarithmic RSSI (decibel) domain. **Note:** Because RSSI gradients can explode under high noise, this solver utilizes a *Warm Start* pipeline, inheriting its initial `x0` guess from the Vanilla solver.
4. **Weighted IPOPT:** Combines the RSSI-domain optimization of IPOPT with geometric anchor weighting.
5. **Particle Filter:** A brute-force probabilistic solver that scatters thousands of particles and resamples them based on Gaussian measurement likelihoods. Highly resilient to the non-linear noise that confuses gradient-based solvers.

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/localization_demo.git
cd localization_demo

```


2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


*Note: The IPOPT solvers require the `cyipopt` C-extension. Depending on your OS, you may need to install the IPOPT binaries first (e.g., `conda install -c conda-forge cyipopt` or `apt-get install coinor-libipopt-dev`).*

## Usage

Launch the interactive web dashboard using Streamlit:

```bash
streamlit run app.py

```

1. Open your browser to the local URL provided (usually `http://localhost:8501`).
2. Use the left sidebar to configure your environment (number of runs, anchors, map size, and noise variables).
3. Select the algorithms you wish to benchmark.
4. Click **Run Simulation**.
5. Review the statistical outputs and interact with the Folium map generated at the bottom of the page.

## Adding a New Solver

Thanks to the Open/Closed Principle (OCP), adding a new algorithm (e.g., an Extended Kalman Filter) is trivial:

1. Create a new file in `src/localization/` (e.g., `ekf_solver.py`) that inherits from `BaseSolver`.
2. Add the solver to the dictionary inside `src/experiments/strategies.py`.
3. The Streamlit UI and parallel processor will automatically detect and execute it!