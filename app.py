import streamlit as st
import streamlit.components.v1 as components

from src.experiments.task import SimulationTask
from src.experiments.executor import BatchExecutor
from src.experiments.exporter import ResultExporter
from src.visualization.map_generator import LocalizationVisualizer

# --- IMPORT OUR NEW HELPER MODULES ---
from src.ui.sidebar import render_sidebar
from src.visualization.plots import render_data_preview, render_heatmap, render_boxplot

st.set_page_config(page_title="Localization Simulator", layout="wide")
st.title("Localization Parameter-Sweep Simulator")

# --- 1. UI DELEGATION ---
config, runs_count, total_combos, is_ready = render_sidebar()

# --- 2. EXECUTION ---
if is_ready:
    with st.spinner(f"Running grid sweep across {total_combos} environment combinations..."):
        task = SimulationTask(config)
        executor = BatchExecutor(task)
        
        # UPDATED: We now catch both DataFrames returned by the Executor
        solver_results, measurement_results = executor.run(run_count=runs_count)
        
        # Save the primary solver results
        ResultExporter.save_csv(solver_results) 
        
        # Save both to session state
        st.session_state["results"] = solver_results
        st.session_state["measurements"] = measurement_results
        st.session_state["config"] = config
        st.session_state["total_runs"] = runs_count
        
    st.success("Simulation Complete!")

# --- 3. DISPLAY DELEGATION ---
if "results" in st.session_state:
    results = st.session_state["results"]
    measurements = st.session_state["measurements"]
    saved_config = st.session_state["config"]
    total_runs = st.session_state["total_runs"]

    st.markdown("---")
    st.subheader("Geospatial Map Visualization")
    
    if total_runs > 1:
        selected_run = st.slider(
            "Select Run ID to Visualize", 
            min_value=0, 
            max_value=total_runs - 1, 
            value=0,
            help="Slide to see how the anchors and targets randomized for different runs!"
        )
    else:
        selected_run = 0
        st.info("Displaying the map for Run 0.")
    
    visualizer = LocalizationVisualizer(
        lat0=saved_config.lat0, lon0=saved_config.lon0, 
        x_range=saved_config.x_range, y_range=saved_config.y_range
    )
    
    map_filename = f"web_localization_map_run_{selected_run}.html"
    visualizer.generate_run_map(run_id=selected_run, output_filename=map_filename)
    
    with open(map_filename, "r") as f:
        html_data = f.read()
        components.html(html_data, height=600)

    # --- Call Plotting Helpers for the Main Table ---
    render_data_preview(results)
    
    # --- NEW: Raw Measurements Section ---
    st.markdown("---")
    st.subheader("Raw Measurement Data")
    st.dataframe(measurements.head(10), use_container_width=True)
    st.download_button(
        label="Download Raw Measurements (CSV)", 
        data=measurements.to_csv(index=False).encode('utf-8'),
        file_name="raw_measurements.csv", 
        mime="text/csv", 
        use_container_width=True
    )

    render_heatmap(results, saved_config.x_range[1], saved_config.y_range[1])
    render_boxplot(results)

else:
    st.info("Adjust parameter lists in the sidebar and click 'Run Simulation Sweep' to start.")