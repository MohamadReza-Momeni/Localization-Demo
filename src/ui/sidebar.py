import streamlit as st
from src.experiments.config import ExperimentConfig

def render_sidebar() -> tuple[ExperimentConfig, int, int, bool]:
    """Renders the sidebar and returns the configuration and execution flags."""
    st.sidebar.header("Simulation Parameters")
    anchors = st.sidebar.number_input("Number of Anchors", min_value=3, max_value=20, value=6)
    samples_per_anchor = st.sidebar.number_input("Measurements per Anchor", min_value=1, max_value=1000, value=1, help="Simulates taking a burst of RSSI packets and averaging them to reduce noise.")
    targets = st.sidebar.number_input("Number of Targets", min_value=1, max_value=50, value=3)
    runs = st.sidebar.number_input("Number of Runs", min_value=1, max_value=1000, value=10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Map Dimensions & Location")
    map_width = st.sidebar.number_input("Map Width (X meters)", min_value=100, max_value=10000, value=1000)
    map_height = st.sidebar.number_input("Map Height (Y meters)", min_value=100, max_value=10000, value=1000)
    lat0 = st.sidebar.number_input("Origin Latitude", value=35.7152, format="%.6f")
    lon0 = st.sidebar.number_input("Origin Longitude", value=51.4043, format="%.6f")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter Sweep Grid")

    p0_input = st.sidebar.text_input("P0 Values (dBm, comma-separated)", value="-40.0, -30.0")
    ple_input = st.sidebar.text_input("Path Loss Exponent (n) (comma-separated)", value="2.2, 3.0")
    noise_input = st.sidebar.text_input("Noise Std Dev (sigma) (comma-separated)", value="1.0, 2.0, 4.0")

    # --- Parse Text Inputs into Floats ---
    try:
        p0_values = tuple(float(x.strip()) for x in p0_input.split(",") if x.strip())
        ple_values = tuple(float(x.strip()) for x in ple_input.split(",") if x.strip())
        noise_values = tuple(float(x.strip()) for x in noise_input.split(",") if x.strip())
        
        if not (p0_values and ple_values and noise_values):
            raise ValueError()
    except ValueError:
        st.sidebar.error("Please ensure all parameter sweep boxes have valid comma-separated numbers.")
        st.stop()

    total_combos = len(p0_values) * len(ple_values) * len(noise_values)
    total_iterations = total_combos * runs * targets
    st.sidebar.info(
        f"**Grid Summary:**\n"
        f"*   Combos: **{total_combos}**\n"
        f"*   Runs per combo: **{runs}**\n"
        f"*   Total target tests: **{total_iterations}**"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithms")
    available_solvers = [
        "vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter", "ampl_bonmin" # NEW
    ]
    selected_solvers = st.sidebar.multiselect(
        "Select solvers to evaluate:", options=available_solvers, default=available_solvers
    )

    is_ready = st.sidebar.button("Run Simulation Sweep", type="primary")

    if is_ready and not selected_solvers:
        st.sidebar.error("Please select at least one solver to run.")
        st.stop()

    config = ExperimentConfig(
        anchor_count=anchors,
        samples_per_anchor=samples_per_anchor,
        target_count=targets,
        x_range=(0, map_width),   
        y_range=(0, map_height),  
        lat0=lat0,
        lon0=lon0,
        p0_values=p0_values,
        ple_values=ple_values,
        noise_values=noise_values,
        solvers=tuple(selected_solvers) 
    )

    return config, runs, total_combos, is_ready