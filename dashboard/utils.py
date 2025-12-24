"""
Helper functions for Streamlit dashboard
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import json
import streamlit as st

# Path to predictions folder (relative to dashboard/)
PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_dates():
    """Get list of dates with available predictions."""
    if not PREDICTIONS_DIR.exists():
        return []
    
    dates = []
    for file in PREDICTIONS_DIR.glob("*_predictions.csv"):
        try:
            date_str = file.stem.split('_')[0]  # Extract YYYYMMDD
            date = datetime.strptime(date_str, '%Y%m%d').date()
            dates.append(date)
        except:
            continue
    
    return sorted(dates, reverse=True)

@st.cache_data(ttl=300)
def load_prediction_by_date(date):
    """Load predictions and metadata for a specific date."""
    date_str = date.strftime('%Y%m%d')
    
    # Load predictions CSV
    csv_path = PREDICTIONS_DIR / f"{date_str}_predictions.csv"
    if not csv_path.exists():
        return None, None
    
    predictions = pd.read_csv(csv_path)
    predictions['settlement_period_time'] = pd.to_datetime(predictions['settlement_period_time'])
    
    # Load metadata JSON
    json_path = PREDICTIONS_DIR / f"{date_str}_metadata.json"
    metadata = None
    if json_path.exists():
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    
    return predictions, metadata

def load_latest_predictions():
    """Load the most recent predictions."""
    available_dates = get_available_dates()
    if not available_dates:
        return None, None
    
    latest_date = max(available_dates)
    return load_prediction_by_date(latest_date)

def load_metadata(date):
    """Load metadata for a specific date."""
    date_str = date.strftime('%Y%m%d')
    json_path = PREDICTIONS_DIR / f"{date_str}_metadata.json"
    
    if not json_path.exists():
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)

def format_timestamp(dt):
    """Format datetime for display."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def create_probability_chart(predictions, metadata):
    """Create interactive probability bar chart."""
    df = predictions.copy()
    df['Time'] = pd.to_datetime(df['settlement_period_time']).dt.strftime('%H:%M')
    df['Curtailment'] = df['curtailment_prediction'].map({0: 'No', 1: 'Yes'})
    df['Probability_pct'] = df['curtailment_probability'] * 100
    
    # Get threshold from metadata
    threshold = metadata.get('threshold', 0.3032) if metadata else 0.3032
    threshold_pct = threshold * 100
    
    # Color based on prediction
    colors = ['#28a745' if pred == 0 else '#dc3545' for pred in df['curtailment_prediction']]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df['Time'],
        y=df['Probability_pct'],
        marker_color=colors,
        text=df['Probability_pct'].round(1).astype(str) + '%',
        textposition='outside',
        hovertemplate='<b>Time:</b> %{x}<br>' +
                      '<b>Probability:</b> %{y:.2f}%<br>' +
                      '<b>Curtailment:</b> %{customdata}<br>' +
                      '<extra></extra>',
        customdata=df['Curtailment']
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold_pct,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({threshold_pct:.2f}%)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Curtailment Probability by Settlement Period",
        xaxis_title="Time (Half-Hourly Periods)",
        yaxis_title="Curtailment Probability (%)",
        height=500,
        hovermode='x unified',
        showlegend=False,
        template="plotly_white"
    )
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_heatmap(predictions):
    """Create heatmap visualization of curtailment predictions."""
    df = predictions.copy()
    df['Hour'] = pd.to_datetime(df['settlement_period_time']).dt.hour
    df['Period'] = pd.to_datetime(df['settlement_period_time']).dt.strftime('%H:%M')
    
    # Reshape for heatmap (24 hours x 2 periods per hour)
    heatmap_data = []
    for hour in range(24):
        hour_data = df[df['Hour'] == hour]
        if len(hour_data) == 2:
            heatmap_data.append([
                hour_data.iloc[0]['curtailment_probability'],
                hour_data.iloc[1]['curtailment_probability']
            ])
        else:
            heatmap_data.append([0, 0])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['00', '30'],
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale='RdYlGn_r',
        text=[[f"{val:.1%}" for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title="Curtailment Risk Heatmap",
        xaxis_title="Minutes",
        yaxis_title="Hour",
        height=600,
        template="plotly_white"
    )
    
    return fig