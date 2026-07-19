import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from src.experiments.config import ExperimentConfig
from src.experiments.task import SimulationTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.visualization.map_generator import LocalizationVisualizer
from src.localization.ipopt_params import IPOPTHyperparams

st.set_page_config(page_title="Localization Simulator", layout="wide")
st.title("Localization Algorithm Simulator")

# --- 1. SIDEBAR INPUTS ---
st.sidebar.header("Simulation Parameters")
anchors = st.sidebar.number_input("Number of Anchors", min_value=3, max_value=20, value=6)
targets = st.sidebar.number_input("Number of Targets", min_value=1, max_value=50, value=3)
runs = st.sidebar.number_input("Number of Runs", min_value=1, max_value=1000, value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Map Dimensions & Location")
map_width = st.sidebar.number_input("Map Width (X meters)", min_value=100, max_value=10000, value=1000)
map_height = st.sidebar.number_input("Map Height (Y meters)", min_value=100, max_value=10000, value=1000)
lat0 = st.sidebar.number_input("Origin Latitude", value=35.7152, format="%.6f")
lon0 = st.sidebar.number_input("Origin Longitude", value=51.4043, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.subheader("Randomized Signal Environment")
p0_range = st.sidebar.slider("P0 Range (dBm)", min_value=-50.0, max_value=50.0, value=(-50.0, 50.0), step=1.0)
ple_range = st.sidebar.slider("Path Loss Exponent (n / beta) Range", min_value=2.0, max_value=8.0, value=(2.0, 8.0), step=0.1)
noise_range = st.sidebar.slider("Noise Std Dev (sigma) Range", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)

het_factor = st.sidebar.slider(
    "Heterogeneous noise (het_factor)", min_value=0.0, max_value=5.0, value=0.0, step=0.1,
    help="0 = homogeneous (every anchor same sigma). >0 makes far anchors noisier, "
         "which is when the weighted solvers start to help."
)
het_reference_distance = st.sidebar.number_input(
    "Het reference distance (m)", min_value=1.0, max_value=10000.0, value=100.0,
    help="Distance at which het noise adds one het_factor of sigma."
)
use_seed = st.sidebar.checkbox(
    "Reproducible run (fixed seed)", value=False,
    help="Off = fresh randomness each run. On = anchors/targets/noise are reproducible."
)
base_seed = st.sidebar.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0) if use_seed else None

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithms")
available_solvers = ["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"]
selected_solvers = st.sidebar.multiselect(
    "Select solvers to evaluate:", options=available_solvers, default=available_solvers
)

# --- IPOPT INTERNAL PARAMETERS (now first-class project parameters) ---
st.sidebar.markdown("---")
st.sidebar.subheader("IPOPT Internal Parameters")
with st.sidebar.expander("Tune IPOPT internals", expanded=False):
    ipopt_tol_exp = st.slider(
        "tol (10^-x)", min_value=2, max_value=10, value=6,
        help="Main convergence tolerance (epsilon). Higher x = stricter."
    )
    ipopt_acceptable_tol_exp = st.slider(
        "acceptable_tol (10^-x)", min_value=1, max_value=8, value=4,
        help="Looser fallback tolerance. Must correspond to a value >= tol, "
             "i.e. this x must be <= the tol exponent above."
    )
    ipopt_max_iter = st.number_input("max_iter", min_value=10, max_value=5000, value=500)
    ipopt_hessian = st.selectbox("hessian_approximation", ["limited-memory", "exact"], index=0)
    ipopt_mu_strategy = st.selectbox("mu_strategy", ["monotone", "adaptive"], index=0)

    st.markdown("**Starting point strategy**")
    ipopt_starting_points = st.multiselect(
        "Candidates to run (best objective wins)",
        options=["warm_start", "anchor_centroid", "random_point"],
        default=["warm_start", "anchor_centroid", "random_point"],
    )
    ipopt_multi_start = st.checkbox("Enable multi-start", value=True,
                                     help="If off, only the first candidate above runs.")

    use_fixed_init = st.checkbox("Override warm start with a fixed initial point", value=False,
                                  help="Useful for reproducing a specific MATLAB-side initial point.")
    fixed_init_point = None
    if use_fixed_init:
        fx = st.number_input("Fixed initial X", value=float(map_width) / 2)
        fy = st.number_input("Fixed initial Y", value=float(map_height) / 2)
        fixed_init_point = (fx, fy)

# --- 2. EXECUTION ---
if st.sidebar.button("Run Simulation", type="primary"):
    if not selected_solvers:
        st.sidebar.error("Please select at least one solver to run.")
        st.stop()
    if not ipopt_starting_points:
        st.sidebar.error("Select at least one IPOPT starting point strategy.")
        st.stop()
    if ipopt_acceptable_tol_exp > ipopt_tol_exp:
        st.sidebar.error("acceptable_tol must be looser (larger) than tol — lower the acceptable_tol exponent.")
        st.stop()

    with st.spinner(f"Running {runs} batch experiments across {len(selected_solvers)} solvers..."):

        ipopt_params = IPOPTHyperparams(
            tol=10 ** (-ipopt_tol_exp),
            acceptable_tol=10 ** (-ipopt_acceptable_tol_exp),
            max_iter=ipopt_max_iter,
            hessian_approximation=ipopt_hessian,
            mu_strategy=ipopt_mu_strategy,
            starting_points=tuple(ipopt_starting_points),
            multi_start=ipopt_multi_start,
            fixed_initial_point=fixed_init_point,
        )

        config = ExperimentConfig(
            anchor_count=anchors,
            target_count=targets,
            x_range=(0, map_width),
            y_range=(0, map_height),
            lat0=lat0,
            lon0=lon0,
            p0_range=p0_range,
            ple_range=ple_range,
            noise_range=noise_range,
            het_factor=het_factor,
            het_reference_distance=het_reference_distance,
            base_seed=int(base_seed) if base_seed is not None else None,
            solvers=tuple(selected_solvers),
            ipopt_params=ipopt_params,
        )

        task = SimulationTask(config)
        executor = BatchExecutor(task)

        results = executor.run(run_count=runs)
        ResultExporter.save_csv(results)

        st.session_state["results"] = results
        st.session_state["total_runs"] = runs

    st.success("Simulation Complete!")


# --- DISPLAY SECTION ---
if "results" in st.session_state:
    results = st.session_state["results"]
    total_runs = st.session_state["total_runs"]

    # --- 4. DATA PREVIEW & STATISTICS ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(results.head(10), use_container_width=True)
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Results (CSV)", data=csv_data,
            file_name="localization_results.csv", mime="text/csv", use_container_width=True
        )

    with col2:
        st.subheader("Error Statistics (meters) by Solver")
        stats = results.groupby("solver")["error"].describe()
        st.dataframe(stats, use_container_width=True)

    if "ipopt_chosen_start" in results.columns:
        st.markdown("---")
        st.subheader("IPOPT Starting Point Behavior")
        ipopt_rows = results.dropna(subset=["ipopt_chosen_start"])
        if not ipopt_rows.empty:
            st.dataframe(
                ipopt_rows.groupby(["solver", "ipopt_chosen_start"]).size().rename("count").reset_index(),
                use_container_width=True,
            )

    # --- 3. DYNAMIC MAP VISUALIZATION (MOVED TO TOP) ---
    st.markdown("---")
    st.subheader("Geospatial Map Visualization")

    selected_run = st.slider(
        "Select Run ID to Visualize",
        min_value=0,
        max_value=total_runs - 1,
        value=0,
        help="Slide to see how the anchors and targets randomized for different runs!"
    )

    visualizer = LocalizationVisualizer(
        lat0=lat0, lon0=lon0, x_range=(0, map_width), y_range=(0, map_height)
    )

    map_filename = f"web_localization_map_run_{selected_run}.html"
    visualizer.generate_run_map(run_id=selected_run, output_filename=map_filename)

    with open(map_filename, "r") as f:
        html_data = f.read()
        components.html(html_data, height=600)

    # --- 5. ADVANCED PLOTS ---
    st.markdown("---")
    st.subheader("Solver Performance Analysis")

    fig_heatmap = px.density_heatmap(
        results,
        x="true_x",
        y="true_y",
        z="error",
        histfunc="avg",
        facet_col="solver",
        facet_col_wrap=3,
        title="Spatial Error Heatmap (Red = Higher Average Error)",
        labels={"true_x": "Map X (m)", "true_y": "Map Y (m)", "error": "Avg Error (m)"},
        color_continuous_scale="Reds",
        range_x=[0, map_width],
        range_y=[0, map_height],
        nbinsx=15,
        nbinsy=15
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    fig_box = px.box(
        results,
        x="solver",
        y="error",
        color="solver",
        title="Distribution of Localization Errors by Algorithm",
        labels={"error": "Error Distance (meters)", "solver": "Algorithm"},
        points="outliers"
    )
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
