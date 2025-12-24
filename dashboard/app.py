"""
Day-Ahead Wind Curtailment Forecasting Dashboard

Displays 48-period curtailment predictions with interactive visualizations.
Automatically loads latest predictions from predictions/ folder.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import helper functions
from utils import (
    load_latest_predictions,
    load_prediction_by_date,
    get_available_dates,
    format_timestamp,
    create_probability_chart,
    create_heatmap,
    load_metadata
)

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Day-Ahead Wind Curtailment Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .status-green {
        color: #28a745;
        font-weight: bold;
    }
    .status-red {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=NESO+Logo", width=150)  # Replace with actual logo
    st.markdown("### 🌬️ Wind Curtailment Forecast")
    st.markdown("---")
    
    # Date selector
    st.markdown("#### 📅 Select Date")
    available_dates = get_available_dates()
    
    if not available_dates:
        st.error("No predictions found in predictions/ folder")
        st.stop()
    
    # Default to latest date
    default_date = max(available_dates)
    
    # Date picker
    selected_date = st.selectbox(
        "Prediction Date",
        options=sorted(available_dates, reverse=True),
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)"),
        index=0
    )
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Info panel
    st.markdown("#### ℹ️ About")
    st.info("""
    **Day-Ahead Forecasting System**
    
    This dashboard displays wind curtailment predictions for the next day (48 half-hourly periods).
    
    **Updated:** Daily at 3:30 PM
    
    **Model:** XGBoost Classifier
    """)

# -----------------------
# Main Content
# -----------------------

# Load predictions for selected date
predictions, metadata = load_prediction_by_date(selected_date)

if predictions is None:
    st.error(f"No predictions found for {selected_date}")
    st.stop()

# -----------------------
# Header
# -----------------------
st.markdown('<div class="main-header">🌬️ Day-Ahead Wind Curtailment Forecast</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Predictions for {selected_date.strftime("%A, %B %d, %Y")}</div>', unsafe_allow_html=True)

# Last updated info
if metadata:
    last_updated = datetime.fromisoformat(metadata['prediction_timestamp'])
    time_ago = datetime.now() - last_updated
    
    if time_ago.days > 0:
        time_str = f"{time_ago.days} day(s) ago"
    elif time_ago.seconds > 3600:
        time_str = f"{time_ago.seconds // 3600} hour(s) ago"
    else:
        time_str = f"{time_ago.seconds // 60} minute(s) ago"
    
    st.markdown(f"""
        <div class="info-box">
        📊 <strong>Last Updated:</strong> {last_updated.strftime('%Y-%m-%d %H:%M:%S')} ({time_str})
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Key Metrics
# -----------------------
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
total_curtailment = predictions['curtailment_prediction'].sum()
avg_probability = predictions['curtailment_probability'].mean()
max_probability = predictions['curtailment_probability'].max()
high_risk_periods = (predictions['curtailment_probability'] > 0.5).sum()

with col1:
    st.metric(
        label="Curtailment Periods",
        value=f"{total_curtailment}/48",
        delta=f"{(total_curtailment/48)*100:.1f}%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Average Probability",
        value=f"{avg_probability:.1%}",
        delta=None
    )

with col3:
    st.metric(
        label="Maximum Probability",
        value=f"{max_probability:.1%}",
        delta=None
    )

with col4:
    st.metric(
        label="High Risk Periods (>50%)",
        value=high_risk_periods,
        delta=None,
        delta_color="inverse"
    )

st.markdown("---")

# -----------------------
# Curtailment Status Banner
# -----------------------
if total_curtailment == 0:
    st.success("✅ **LOW RISK:** No curtailment expected for this day")
elif total_curtailment < 10:
    st.warning(f"⚠️ **MODERATE RISK:** {total_curtailment} period(s) with curtailment expected")
else:
    st.error(f"🚨 **HIGH RISK:** {total_curtailment} period(s) with curtailment expected")

st.markdown("---")

# -----------------------
# Probability Chart
# -----------------------
st.markdown("### 📈 Curtailment Probability (48 Half-Hourly Periods)")

fig_probability = create_probability_chart(predictions, metadata)
st.plotly_chart(fig_probability, use_container_width=True)

st.markdown("---")

# -----------------------
# Two-column layout: Heatmap + Data Table
# -----------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🔥 Curtailment Heatmap")
    fig_heatmap = create_heatmap(predictions)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col_right:
    st.markdown("### 📋 Detailed Predictions")
    
    # Format dataframe for display
    display_df = predictions.copy()
    display_df['settlement_period_time'] = pd.to_datetime(display_df['settlement_period_time'])
    display_df['Time'] = display_df['settlement_period_time'].dt.strftime('%H:%M')
    display_df['Probability'] = (display_df['curtailment_probability'] * 100).round(2).astype(str) + '%'
    display_df['Curtailment'] = display_df['curtailment_prediction'].map({0: '❌ No', 1: '✅ Yes'})
    
    # Select columns for display
    display_cols = ['Time', 'Probability', 'Curtailment']
    
    st.dataframe(
        display_df[display_cols],
        height=400,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# -----------------------
# Download Section
# -----------------------
st.markdown("### 💾 Download Predictions")

col_csv, col_json = st.columns(2)

with col_csv:
    csv = predictions.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{selected_date.strftime('%Y%m%d')}_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_json:
    if metadata:
        json_str = json.dumps(metadata, indent=2)
        st.download_button(
            label="📥 Download Metadata (JSON)",
            data=json_str,
            file_name=f"{selected_date.strftime('%Y%m%d')}_metadata.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")

# -----------------------
# Metadata Panel
# -----------------------
with st.expander("🔍 **Technical Details & Metadata**", expanded=False):
    if metadata:
        col_meta1, col_meta2 = st.columns(2)
        
        with col_meta1:
            st.markdown("**Model Information**")
            st.write(f"- **Model:** XGBoost Classifier")
            st.write(f"- **Version:** {metadata.get('xgboost_version', 'N/A')}")
            st.write(f"- **Threshold:** {metadata.get('threshold', 0):.4f}")
            st.write(f"- **Features:** {metadata.get('n_features', 0)}")
        
        with col_meta2:
            st.markdown("**Data Sources**")
            sources = metadata.get('data_sources', {})
            st.write(f"- **Wind Forecast:** {sources.get('wind_forecast', 'N/A')}")
            st.write(f"- **Demand Forecast:** {sources.get('demand_forecast', 'N/A')}")
            st.write(f"- **Constraints:** {sources.get('constraints', 'N/A')}")
            st.write(f"- **BMUs:** {sources.get('bmus', 'N/A')}")
        
        st.markdown("**Prediction Statistics**")
        st.write(f"- **Total Periods:** {metadata.get('n_periods', 0)}")
        st.write(f"- **Curtailment Periods:** {metadata.get('n_curtailment_predicted', 0)}")
        st.write(f"- **Avg Probability:** {metadata.get('avg_curtailment_probability', 0):.4f}")
        st.write(f"- **Max Probability:** {metadata.get('max_curtailment_probability', 0):.4f}")
    else:
        st.info("Metadata not available for this prediction")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with Streamlit • Powered by XGBoost • © 2025 MOBI Analytics
    </div>
""", unsafe_allow_html=True)