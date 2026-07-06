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
st.sidebar.subheader("Randomized Signal Environment")
p0_range = st.sidebar.slider("P0 Range (dBm)", min_value=-50.0, max_value=50.0, value=(-50.0, 50.0), step=1.0)
ple_range = st.sidebar.slider("Path Loss Exponent (n) Range", min_value=2.0, max_value=8.0, value=(2.0, 8.0), step=0.1)
noise_range = st.sidebar.slider("Noise Std Dev (sigma) Range", min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithms")
# ADDED: Multi-select box for solvers
available_solvers = ["vanilla", "weighted", "ipopt", "weighted_ipopt"]
selected_solvers = st.sidebar.multiselect(
    "Select solvers to evaluate:",
    options=available_solvers,
    default=available_solvers # By default, all are selected
)

lat0 = 35.7152
lon0 = 51.4043

# --- 2. EXECUTION ---
if st.sidebar.button("Run Simulation", type="primary"):
    
    # ADDED: Prevent running if no solvers are selected
    if not selected_solvers:
        st.sidebar.error("❌ Please select at least one solver to run.")
        st.stop()
    
    with st.spinner(f"Running {runs} batch experiments across {len(selected_solvers)} solvers..."):
        
        runner = ExperimentRunner(
            anchor_count=anchors,
            target_count=targets,
            p0_range=p0_range,
            ple_range=ple_range,
            noise_range=noise_range,
            solvers=selected_solvers # ADDED: Pass the selected list
        )
        
        results = runner.run_batch(run_count=runs)
        runner.save(results) 
        
    st.success("Simulation Complete!")
    
    # --- 3. DISPLAY RESULTS & DOWNLOAD ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(results.head(10), use_container_width=True)
        
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name="localization_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col2:
        st.subheader("Error Statistics (meters) by Solver")
        stats = results.groupby("solver")["error"].describe()
        st.dataframe(stats, use_container_width=True)

    # --- 4. DISPLAY MAP ---
    st.markdown("---")
    st.subheader("Geospatial Map Visualization (Run 0)")
    
    visualizer = LocalizationVisualizer(lat0=lat0, lon0=lon0)
    map_filename = "web_localization_map.html"
    visualizer.generate_run_map(run_id=0, output_filename=map_filename)
    
    with open(map_filename, "r") as f:
        html_data = f.read()
        components.html(html_data, height=600)
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")