import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from src.experiments.config import ExperimentConfig
from src.experiments.task import SimulationTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.visualization.map_generator import LocalizationVisualizer

st.set_page_config(page_title="Localization Simulator", layout="wide")
st.title("Localization Parameter-Sweep Simulator")

# --- 1. SIDEBAR INPUTS ---
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

# Comma-separated inputs for our parameter grid
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

# Show total permutations summary
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
available_solvers = ["vanilla", "weighted", "ipopt", "weighted_ipopt", "particle_filter"]
selected_solvers = st.sidebar.multiselect(
    "Select solvers to evaluate:", options=available_solvers, default=available_solvers
)

# --- 2. EXECUTION ---
if st.sidebar.button("Run Simulation Sweep", type="primary"):
    if not selected_solvers:
        st.sidebar.error("Please select at least one solver to run.")
        st.stop()
    
    with st.spinner(f"Running grid sweep across {total_combos} environment combinations..."):
        
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
        
        task = SimulationTask(config)
        executor = BatchExecutor(task)
        
        # Note: we pass the config parameter combinations downward through executor
        results = executor.run(run_count=runs)
        ResultExporter.save_csv(results) 
        
        st.session_state["results"] = results
        st.session_state["total_runs"] = runs
        
    st.success("Simulation Complete!")


# --- DISPLAY SECTION ---
if "results" in st.session_state:
    results = st.session_state["results"]
    total_runs = st.session_state["total_runs"]

    # --- 3. DYNAMIC MAP VISUALIZATION ---
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

    # --- 5. ADVANCED PLOTS ---
    st.markdown("---")
    st.subheader("Solver Performance Analysis")
    
    # Spatial Error Heatmap
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

    # Box Plot
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
    st.info("Adjust parameter lists in the sidebar and click 'Run Simulation Sweep' to start.")