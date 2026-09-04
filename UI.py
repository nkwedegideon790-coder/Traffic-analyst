import sqlite3

import streamlit as st

st.set_page_config(page_title="Traffic Analysis Dashboard", page_icon=":car:", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0b1220 0%, #111827 100%);
            color: #e5eefb;
        }

        .metric-box {
            background: #111827;
            border: 1px solid #243244;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.18);
            height: 100%;
        }

        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.8rem;
        }

        .low { background: #1d4ed8; color: #e0ecff; }
        .medium { background: #0ea5e9; color: #e0f2fe; }
        .high { background: #b91c1c; color: #fee2e2; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Traffic Analysis Dashboard")


@st.cache_data(show_spinner=False)
def get_recent_logs(limit: int = 8):
    try:
        conn = sqlite3.connect("traffic.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, time, traffic_density, traffic_count, car_count, motorcycle_count, bus_count, truck_count FROM traffic_data ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except sqlite3.Error:
        return []


rows = get_recent_logs(8)
if rows:
    latest_status = rows[0][2]
    latest_count = rows[0][3]
    latest_time = rows[0][1]
else:
    latest_status = "No data"
    latest_count = 0
    latest_time = "Waiting"

if rows:
    average_vehicles = round(sum(row[3] for row in rows) / len(rows), 1)
    peak_vehicles = max(row[3] for row in rows)
else:
    average_vehicles = 0
    peak_vehicles = 0

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="metric-box"><div style="color:#6b7280; font-size:0.85rem;">Status</div><h3>{latest_status}</h3></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-box"><div style="color:#6b7280; font-size:0.85rem;">Avg vehicles</div><h3>{average_vehicles}</h3></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-box"><div style="color:#6b7280; font-size:0.85rem;">Peak</div><h3>{peak_vehicles}</h3></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Live Camera")
    try:
        st.components.v1.iframe("http://localhost:8000/video_feed", height=430)
    except Exception:
        st.info("Run the backend server to view the traffic feed.")
        st.image("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1200&q=80", use_container_width=True)

with right_col:
    st.subheader("Traffic Status")
    if latest_status == "Low":
        badge_class = "low"
    elif latest_status == "Medium":
        badge_class = "medium"
    elif latest_status == "High":
        badge_class = "high"
    else:
        badge_class = "low"

    st.markdown(f'<div class="status-badge {badge_class}">{latest_status}</div>', unsafe_allow_html=True)
    st.metric("Detected vehicles", latest_count)
    st.metric("Last update", latest_time)

    st.markdown("---")
    if rows:
        density_data = {"Low": 0, "Medium": 0, "High": 0}
        for row in rows:
            density = row[2]
            count = row[3]
            if density in density_data:
                density_data[density] += count
        st.bar_chart(density_data)
    else:
        st.write("No data yet")

st.markdown("---")

st.subheader("Recent Logs")
if rows:
    table_data = [{"Time": row[1], "Density": row[2], "Car": row[4], "Motorcycle": row[4], "Bus": row[5], "Truck": row[6], "Vehicles": row[7]} for row in rows]
    st.table(table_data)
else:
    st.info("No traffic data has been recorded yet.")
