import streamlit as st
import streamlit.components.v1 as components
from src.experiments.runner import ExperimentRunner
from src.visualization.map_generator import LocalizationVisualizer

st.set_page_config(page_title="Localization Simulator", layout="wide")
st.title("Localization Algorithm Simulator")

# --- 1. SIDEBAR INPUTS ---
st.sidebar.header("Simulation Parameters")
anchors = st.sidebar.number_input("Number of Anchors", min_value=3, max_value=20, value=6)
targets = st.sidebar.number_input("Number of Targets", min_value=1, max_value=50, value=3)
runs = st.sidebar.number_input("Number of Runs", min_value=1, max_value=1000, value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Map Dimensions & Location")
# ADDED: Dynamic Map Width, Height, and Origin Coordinates
map_width = st.sidebar.number_input("Map Width (X meters)", min_value=100, max_value=10000, value=1000)
map_height = st.sidebar.number_input("Map Height (Y meters)", min_value=100, max_value=10000, value=1000)
lat0 = st.sidebar.number_input("Origin Latitude", value=35.7152, format="%.6f")
lon0 = st.sidebar.number_input("Origin Longitude", value=51.4043, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.subheader("Randomized Signal Environment")
p0_range = st.sidebar.slider("P0 Range (dBm)", min_value=-50.0, max_value=50.0, value=(-50.0, 50.0), step=1.0)
ple_range = st.sidebar.slider("Path Loss Exponent (n) Range", min_value=2.0, max_value=8.0, value=(2.0, 8.0), step=0.1)
noise_range = st.sidebar.slider("Noise Std Dev (sigma) Range", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithms")
available_solvers = ["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"]
selected_solvers = st.sidebar.multiselect(
    "Select solvers to evaluate:", options=available_solvers, default=available_solvers
)

# --- 2. EXECUTION ---
if st.sidebar.button("Run Simulation", type="primary"):
    if not selected_solvers:
        st.sidebar.error("Please select at least one solver to run.")
        st.stop()
    
    with st.spinner(f"Running {runs} batch experiments across {len(selected_solvers)} solvers..."):
        runner = ExperimentRunner(
            anchor_count=anchors,
            target_count=targets,
            x_range=(0, map_width),   # PASSED HERE
            y_range=(0, map_height),  # PASSED HERE
            p0_range=p0_range,
            ple_range=ple_range,
            noise_range=noise_range,
            solvers=selected_solvers 
        )
        
        results = runner.run_batch(run_count=runs)
        runner.save(results) 
        
    st.success("Simulation Complete!")
    
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

    st.markdown("---")
    st.subheader("Geospatial Map Visualization (Run 0)")
    
    # ADDED: Passing custom lat/lon and bounds to the Visualizer
    visualizer = LocalizationVisualizer(
        lat0=lat0, lon0=lon0, x_range=(0, map_width), y_range=(0, map_height)
    )
    map_filename = "web_localization_map.html"
    visualizer.generate_run_map(run_id=0, output_filename=map_filename)
    
    with open(map_filename, "r") as f:
        html_data = f.read()
        components.html(html_data, height=600)
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")