"""
app.py - Streamlit dashboard for the Edge-AI Geospatial Infrastructure &
Asset Reporter.

Provides:
  * A sidebar for tuning per-class confidence thresholds and triggering an
    inference run against the simulated feed folder
  * A live-updating dataframe of stored detections
  * A PyDeck map pinpointing every anomaly, color-coded by type, plus a
    Folium fallback map
  * A button that triggers reporter.py to build (and attempt to email) a
    PDF summary, with an in-app download link
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import folium
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium

import config
import database
import inference
import reporter

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Color palette for anomaly types (used by both PyDeck and Folium maps and
# the breakdown chart)
# --------------------------------------------------------------------------
ANOMALY_COLORS = {
    "pothole": [230, 126, 34],
    "crack": [149, 165, 166],
    "encroachment": [231, 76, 60],
    "debris": [155, 89, 182],
    "flooding": [52, 152, 219],
    "vegetation_overgrowth": [39, 174, 96],
    "damaged_pole": [241, 196, 15],
    "exposed_wiring": [192, 57, 43],
}
DEFAULT_COLOR = [127, 140, 141]


def _color_for(anomaly_type: str):
    return ANOMALY_COLORS.get(anomaly_type, DEFAULT_COLOR)


@st.cache_data(ttl=config.DATAFRAME_REFRESH_SECONDS, show_spinner=False)
def load_detections(_cache_key: int) -> pd.DataFrame:
    """`_cache_key` forces Streamlit's cache to invalidate on the configured
    cadence, giving a 'live-updating' dataframe without hammering SQLite on
    every widget interaction."""
    records = database.get_all_detections()
    if not records:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "anomaly_type",
                "confidence",
                "latitude",
                "longitude",
                "source_image",
                "notes",
            ]
        )
    df = pd.DataFrame(records)
    df["anomaly_label"] = df["anomaly_type"].map(lambda t: config.ANOMALY_CLASSES.get(t, t))
    return df


def render_sidebar() -> dict:
    st.sidebar.title(f"{config.APP_ICON} Control Panel")
    st.sidebar.caption(
        f"Compute device: **{config.DEVICE.upper()}**"
        f"{'  (fp16 autocast)' if config.USE_FP16 else '  (CPU multi-threaded)'}"
    )

    st.sidebar.subheader("Detection Thresholds")
    st.sidebar.write("Minimum confidence required per anomaly class.")

    thresholds = {}
    for anomaly_type, label in config.ANOMALY_CLASSES.items():
        default = config.CLASS_CONFIDENCE_THRESHOLDS.get(
            anomaly_type, config.DEFAULT_CONFIDENCE_THRESHOLD
        )
        thresholds[anomaly_type] = st.sidebar.slider(
            label,
            min_value=0.05,
            max_value=0.95,
            value=float(default),
            step=0.05,
            key=f"threshold_{anomaly_type}",
        )

    st.sidebar.divider()
    st.sidebar.subheader("Simulated Feed")
    st.sidebar.write(f"Images folder: `{config.IMAGES_DIR}`")
    image_count = len(inference.discover_feed_images())
    st.sidebar.write(f"Images currently queued: **{image_count}**")

    run_clicked = st.sidebar.button(
        "\u25B6 Run Detection Pipeline", use_container_width=True, type="primary"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Reporting")
    generate_clicked = st.sidebar.button("\U0001F4C4 Generate PDF Report", use_container_width=True)

    st.sidebar.divider()
    if st.sidebar.button("\U0001F5D1\uFE0F Clear All Detections", use_container_width=True):
        deleted = database.clear_all_detections()
        st.sidebar.success(f"Deleted {deleted} records.")
        st.cache_data.clear()

    return {
        "thresholds": thresholds,
        "run_clicked": run_clicked,
        "generate_clicked": generate_clicked,
    }


def render_kpis(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", len(df))
    if not df.empty:
        col2.metric("Avg. Confidence", f"{df['confidence'].mean() * 100:.1f}%")
        col3.metric("Distinct Anomaly Types", df["anomaly_type"].nunique())
        latest = pd.to_datetime(df["timestamp"]).max()
        col4.metric("Latest Detection", latest.strftime("%H:%M:%S"))
    else:
        col2.metric("Avg. Confidence", "N/A")
        col3.metric("Distinct Anomaly Types", 0)
        col4.metric("Latest Detection", "N/A")


def render_dataframe(df: pd.DataFrame) -> None:
    st.subheader("\U0001F4CB Live Detection Log")
    if df.empty:
        st.info(
            "No detections stored yet. Run the detection pipeline from the "
            "sidebar to populate this table."
        )
        return

    display_df = df.copy()
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
    display_df = display_df[
        ["id", "timestamp", "anomaly_label", "confidence", "latitude", "longitude", "source_image"]
    ].rename(columns={"anomaly_label": "anomaly_type"})

    st.dataframe(
        display_df.sort_values("timestamp", ascending=False),
        use_container_width=True,
        height=360,
        hide_index=True,
    )


def render_pydeck_map(df: pd.DataFrame) -> None:
    st.subheader("\U0001F5FA\uFE0F Anomaly Map (PyDeck)")
    if df.empty:
        st.info("No geolocated anomalies to display yet.")
        return

    plot_df = df.copy()
    plot_df["color"] = plot_df["anomaly_type"].apply(_color_for)
    plot_df["radius"] = 25 + (plot_df["confidence"] * 40)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.75,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(
        latitude=float(plot_df["latitude"].mean()),
        longitude=float(plot_df["longitude"].mean()),
        zoom=config.MAP_DEFAULT_ZOOM,
        pitch=30,
    )

    tooltip = {
        "html": "<b>{anomaly_label}</b><br/>Confidence: {confidence}<br/>Lat/Lon: {latitude}, {longitude}",
        "style": {"backgroundColor": "#0B3D91", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v10",
        )
    )


def render_folium_map(df: pd.DataFrame) -> None:
    with st.expander("\U0001F5FA\uFE0F Anomaly Map (Folium alternative view)"):
        if df.empty:
            st.info("No geolocated anomalies to display yet.")
            return

        center_lat = float(df["latitude"].mean())
        center_lon = float(df["longitude"].mean())
        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=config.MAP_DEFAULT_ZOOM,
            tiles="cartodbpositron",
        )

        for _, row in df.iterrows():
            color = _color_for(row["anomaly_type"])
            hex_color = "#%02x%02x%02x" % tuple(color)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6 + row["confidence"] * 6,
                color=hex_color,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.8,
                popup=folium.Popup(
                    html=(
                        f"<b>{row['anomaly_label']}</b><br>"
                        f"Confidence: {row['confidence'] * 100:.1f}%<br>"
                        f"Time: {row['timestamp']}"
                    ),
                    max_width=250,
                ),
            ).add_to(fmap)

        st_folium(fmap, width=None, height=420, key="folium_map")


def render_breakdown_chart(df: pd.DataFrame) -> None:
    st.subheader("\U0001F4CA Detections by Anomaly Type")
    if df.empty:
        st.info("Nothing to chart yet.")
        return
    counts = df["anomaly_label"].value_counts()
    st.bar_chart(counts)


def handle_pipeline_run(thresholds: dict) -> None:
    with st.spinner(f"Running inference on {config.DEVICE.upper()}..."):
        progress_bar = st.progress(0.0, text="Starting inference run...")

        def _progress(done: int, total: int) -> None:
            frac = done / total if total else 1.0
            progress_bar.progress(frac, text=f"Processed {done}/{total} frames")

        summary = inference.run_inference_pipeline(thresholds=thresholds, progress_callback=_progress)
        progress_bar.empty()

    st.success(
        f"Pipeline complete: {summary['images_processed']} frames processed, "
        f"{summary['detections_found']} anomalies stored "
        f"({summary['device_used'].upper()}, {summary['elapsed_seconds']}s)."
    )
    st.cache_data.clear()


def handle_report_generation() -> None:
    with st.spinner("Building PDF summary report..."):
        result = reporter.generate_and_send_report()

    st.success(f"Report generated: {Path(result['pdf_path']).name}")
    if result["email_configured"]:
        if result["emailed"]:
            st.info("Report emailed to configured recipients.")
        else:
            st.warning("Report generated but email delivery failed. Check SMTP settings/logs.")
    else:
        st.info("SMTP not configured (see config.py) - skipping email delivery.")

    with open(result["pdf_path"], "rb") as pdf_file:
        st.download_button(
            label="\u2B07 Download PDF Report",
            data=pdf_file.read(),
            file_name=Path(result["pdf_path"]).name,
            mime="application/pdf",
        )


def main() -> None:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption(
        "Local, edge-deployable computer-vision monitoring for infrastructure "
        "and asset anomalies - optimized for 6GB-VRAM GPUs with automatic CPU fallback."
    )

    sidebar_state = render_sidebar()

    if sidebar_state["run_clicked"]:
        handle_pipeline_run(sidebar_state["thresholds"])

    if sidebar_state["generate_clicked"]:
        handle_report_generation()

    # Bucket the cache key by the configured refresh interval so the
    # dataframe/map "live update" without a full page rerun loop.
    cache_bucket = int(dt.datetime.now().timestamp() // config.DATAFRAME_REFRESH_SECONDS)
    df = load_detections(cache_bucket)

    render_kpis(df)
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        render_dataframe(df)
    with right:
        render_breakdown_chart(df)

    st.divider()
    render_pydeck_map(df)
    render_folium_map(df)

    st.caption(
        f"Auto-refreshing every {config.DATAFRAME_REFRESH_SECONDS}s (cache-bucketed). "
        "Use the sidebar to run new detections or generate a report on demand."
    )


if __name__ == "__main__":
    main()
