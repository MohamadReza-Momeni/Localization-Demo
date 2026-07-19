import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

def render_data_preview(results):
    st.markdown("---")
    
    # --- Calculate True RMSE vs CRLB ---
    # We copy the dataframe to avoid modifying the original data in session_state
    results_copy = results.copy()
    
    # Ensure CRLB is numeric (replace inf with a large number or drop for calculations)
    results_copy = results_copy[results_copy['crlb'] != float('inf')]
    
    # Square the errors to convert standard deviations into variances
    results_copy['error_sq'] = results_copy['error'] ** 2
    results_copy['crlb_sq'] = results_copy['crlb'] ** 2
    
    # Group by the environment parameters to find the average variance
    group_cols = ['solver', 'p0', 'ple', 'sigma']
    rmse_stats = results_copy.groupby(group_cols)[['error_sq', 'crlb_sq']].mean().reset_index()
    
    # Take the square root to get back to meters (RMSE)
    rmse_stats['RMSE (m)'] = np.sqrt(rmse_stats['error_sq'])
    rmse_stats['CRLB (m)'] = np.sqrt(rmse_stats['crlb_sq'])
    
    # Calculate how close the solver is to the mathematical limit (1.0 = perfect)
    rmse_stats['Efficiency'] = rmse_stats['CRLB (m)'] / rmse_stats['RMSE (m)']
    
    # Clean up the table for display
    rmse_stats = rmse_stats.drop(columns=['error_sq', 'crlb_sq']).round(3)

    # --- Render UI ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(results.head(10), use_container_width=True)
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Results (CSV)", data=csv_data,
            file_name="localization_results.csv", mime="text/csv", use_container_width=True
        )
        
    with col2:
        st.subheader("True RMSE vs. Theoretical Limit")
        st.dataframe(rmse_stats, use_container_width=True)

def render_heatmap(results, map_width, map_height):
    st.markdown("---")
    st.subheader("Spatial Error Heatmap")
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

def render_boxplot(results):
    st.markdown("---")
    st.subheader("Solver Stability Analysis")
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