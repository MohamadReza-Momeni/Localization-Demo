import streamlit as st
import plotly.express as px

def render_data_preview(results):
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