import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gpxpy
import streamlit as st
import re

# --- CONFIGURATION ---
# Based on UTA Miler testing, 10 is the "Goldilocks" zone.
SMOOTHING_WINDOW = 10 
# ---------------------

def process_gpx(uploaded_file):
    """Parses GPX from Streamlit uploader and returns binned summary."""
    gpx = gpxpy.parse(uploaded_file)

    cumulative_dist = [0.0]
    elevations = []

    # Flatten tracks and segments into points
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            points.extend(segment.points)

    for i, point in enumerate(points):
        elevations.append(point.elevation or 0.0)
        if i == 0:
            continue
        p_prev = points[i-1]
        dist = p_prev.distance_2d(point)
        cumulative_dist.append(cumulative_dist[-1] + dist)

    df = pd.DataFrame({'cum_dist': cumulative_dist, 'ele': elevations})

    # Smooth the elevation to fix GPS jitter
    df['ele_smooth'] = df['ele'].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()

    # Calculate gradients
    df['dist_diff'] = df['cum_dist'].diff()
    df['ele_diff'] = df['ele_smooth'].diff()

    # Apply 1m threshold to avoid gradient spikes from stationary GPS jitter
    df['gradient'] = np.where(df['dist_diff'] > 1.0, (df['ele_diff'] / df['dist_diff']) * 100, 0)
    df['gradient'] = df['gradient'].replace([np.inf, -np.inf], 0).fillna(0)

    # Replicate VeloViewer Bins
    bins = [-np.inf, -22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5, np.inf]
    labels = [
        "-27.5 to -22.5 %", "-22.5 to -17.5 %", "-17.5 to -12.5 %",
        "-12.5 to -7.5 %", "-7.5 to -2.5 %", "-2.5 to 2.5 %",
        "2.5 to 7.5 %", "7.5 to 12.5 %", "12.5 to 17.5 %",
        "17.5 to 22.5 %", "22.5 to 27.5 %"
    ]

    df['Bin'] = pd.cut(df['gradient'], bins=bins, labels=labels, right=False)
    summary = df.groupby('Bin', observed=False)['dist_diff'].sum().reset_index()
    summary.rename(columns={'dist_diff': 'Distance_km'}, inplace=True)
    summary['Distance_km'] = summary['Distance_km'] / 1000.0 
    
    return summary, df['cum_dist'].max() / 1000.0

# --- STREAMLIT UI ---
st.set_page_config(page_title="mkUltra.run | GPX Analyser", layout="wide")

st.title("🏃‍♂️ mkUltra.run GPX Vert Analyser")
st.markdown("Upload a GPX file to see your distance breakdown by gradient. Optimized for Australian trail and ultra-running.")

uploaded_file = st.file_uploader("Choose a GPX file", type="gpx")

if uploaded_file is not None:
    # Process the file
    race_name = uploaded_file.name.replace(".gpx", "")
    summary, total_dist = process_gpx(uploaded_file)

    # Prepare Data for Charting
    summary['Perc'] = (summary['Distance_km'] / total_dist * 100) if total_dist > 0 else 0
    summary['sort'] = summary['Bin'].apply(lambda x: float(re.findall(r"[-+]?\d*\.?\d+", str(x))[0]))
    summary = summary.sort_values('sort', ascending=True).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    custom_colors = ["#803131", "#d63131", "#e77d31", "#efe331", "#d3d388", "#31d431", "#7fd1af", "#4ee6e6", "#3197e9", "#3b36db", "#682d94"]
    cmap = LinearSegmentedColormap.from_list("vv", custom_colors[::-1], N=len(summary))
    
    bars = ax.barh(summary['Bin'], summary['Distance_km'], color=[cmap(i) for i in range(len(summary))])

    # Add labels to bars
    max_val = summary['Distance_km'].max()
    for i, bar in enumerate(bars):
        dist = summary.loc[i, 'Distance_km']
        perc = summary.loc[i, 'Perc']
        if dist > 0:
            ax.text(bar.get_width() + (max_val * 0.02), bar.get_y() + bar.get_height()/2, 
                    f"{dist:.2f}km ({perc:.1f}%)", va='center', fontweight='bold')

    plt.title(f"{race_name}\nTotal Distance: {total_dist:.2f} km", fontsize=18, fontweight='bold')
    plt.xlabel("Distance (km)")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Display in Streamlit
    ax.set_xlim(0, max_val * 1.3)
    st.pyplot(fig)

    # Download Option
    csv = summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Analysis as CSV",
        data=csv,
        file_name=f"{race_name}_analysis.csv",
        mime='text/csv',
    )
