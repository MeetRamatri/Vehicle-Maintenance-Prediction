import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time
import datetime

# --- SETTINGS & CONFIG ---
st.set_page_config(
    page_title="Vehicle Maintenance AI",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 10 seconds (10000 milliseconds) for real-time vibe
count = st_autorefresh(interval=10000, limit=1000, key="data_refresh")

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Base styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0f19;
            color: #ffffff;
        }

        /* Hide main menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Glassmorphism Panels */
        .glass-panel {
            background: rgba(18, 24, 38, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .glass-panel:hover {
            box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.2);
        }

        /* Glow effects */
        .glow-text {
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }
        .glow-text-error {
            color: #ff3366;
            text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
        }
        .glow-text-warning {
            color: #ffcc00;
            text-shadow: 0 0 10px rgba(255, 204, 0, 0.5);
        }

        /* Metric styling */
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            padding: 0;
        }
        .metric-label {
            font-size: 1rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0b0f19; 
        }
        ::-webkit-scrollbar-thumb {
            background: #2d3748; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4a5568; 
        }

        /* Root Cause Bars */
        .progress-bg {
            background-color: #2d3748;
            border-radius: 8px;
            height: 8px;
            width: 100%;
            margin-top: 5px;
            margin-bottom: 15px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            border-radius: 8px;
            transition: width 1s ease-in-out;
        }
        
        /* ===== CHAT UI ===== */
        /* Sidebar chat history */
        .chat-history-item {
            padding: 10px 14px;
            border-radius: 10px;
            margin-bottom: 6px;
            font-size: 0.85em;
            color: #a0aec0;
            cursor: pointer;
            transition: background 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .chat-history-item:hover {
            background: rgba(255,255,255,0.06);
            color: white;
        }
        .chat-history-item.active-convo {
            background: rgba(0, 255, 255, 0.08);
            border-left: 3px solid #00ffff;
            color: white;
        }

        /* Welcome screen */
        .welcome-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 0 0 6px 0;
            background: linear-gradient(135deg, #ffffff 0%, #a0aec0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .welcome-subtitle {
            text-align: center;
            color: #4a5568;
            font-size: 0.95em;
            margin-bottom: 36px;
        }

        /* Feature cards */
        .feature-card {
            border-radius: 16px;
            padding: 22px 18px;
            margin: 6px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 40px rgba(0,0,0,0.4);
        }
        .feature-card-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 4px;
            color: white;
        }
        .feature-card-desc {
            font-size: 0.78em;
            color: rgba(255,255,255,0.55);
            line-height: 1.4;
        }
        .feature-card-icon {
            font-size: 1.6rem;
            margin-bottom: 14px;
            display: block;
        }
        .fc-purple  { background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%); }
        .fc-blue    { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid rgba(100,120,255,0.2); }
        .fc-pink    { background: linear-gradient(135deg, #4a1942 0%, #c94b4b 100%); }
        .fc-dark    { background: linear-gradient(135deg, #141414 0%, #1a1a2e 100%); border:1px solid rgba(255,255,255,0.08); }

        /* Chat messages */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px 0 120px 0;
            max-height: 55vh;
        }
        .chat-bubble-user {
            background: rgba(255,255,255,0.05);
            border-radius: 18px 18px 4px 18px;
            padding: 12px 18px;
            margin: 8px 0 8px auto;
            max-width: 75%;
            width: fit-content;
            display: block;
            font-size: 0.95em;
            border: 1px solid rgba(255,255,255,0.08);
            color: white;
        }
        .chat-bubble-bot {
            background: linear-gradient(135deg, rgba(0,200,255,0.08) 0%, rgba(18,24,38,0.9) 100%);
            border-radius: 18px 18px 18px 4px;
            padding: 12px 18px;
            margin: 8px auto 8px 0;
            max-width: 75%;
            width: fit-content;
            display: block;
            font-size: 0.95em;
            border: 1px solid rgba(0, 255, 255, 0.15);
            color: #e2e8f0;
        }
        .bot-label {
            font-size: 0.7em;
            color: #00ffcc;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }
        .user-label {
            font-size: 0.7em;
            color: #718096;
            margin-bottom: 4px;
            text-align: right;
            letter-spacing: 0.5px;
        }

        /* Suggested prompt chips */
        .prompt-chip {
            display: inline-block;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 7px 16px;
            margin: 4px 4px;
            font-size: 0.82em;
            color: #a0aec0;
            cursor: pointer;
            transition: all 0.2s;
        }
        .prompt-chip:hover {
            background: rgba(0,255,255,0.1);
            color: white;
            border-color: rgba(0,255,255,0.3);
        }

        /* DataFrame Styling */
        .dataframe {
            background-color: transparent !important;
            color: white !important;
        }

        /* ===== VEHICLE DEEP DIVE ===== */
        .vdd-header {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #ffffff 40%, #8b949e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            margin: 0;
        }
        .vdd-id {
            color: #4a5568;
            font-size: 0.85em;
            margin-top: 2px;
            letter-spacing: 0.5px;
        }
        .vdd-spec-card {
            background: rgba(18,24,38,0.7);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .vdd-spec-icon {
            font-size: 2rem;
            flex-shrink: 0;
        }
        .vdd-spec-title {
            font-size: 0.82em;
            font-weight: 600;
            color: white;
            margin: 0;
        }
        .vdd-spec-desc {
            font-size: 0.72em;
            color: #4a5568;
            margin: 2px 0 0 0;
        }
        .vdd-stat-box {
            background: rgba(18,24,38,0.7);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 18px;
        }
        .vdd-stat-label {
            font-size: 0.72em;
            color: #4a5568;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .vdd-stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            line-height: 1.1;
        }
        .vdd-dot-grid {
            display: flex; flex-wrap: wrap; gap: 4px; margin-top: 10px;
        }
        .vdd-dot {
            width: 10px; height: 10px; border-radius: 3px;
        }
        .warn-number {
            font-size: 3.5rem;
            font-weight: 800;
            color: #FF3366;
            text-shadow: 0 0 30px rgba(255,51,102,0.4);
            line-height: 1;
        }
        /* Tab-style sub-nav */
        .vdd-nav a {
            color: #a0aec0;
            text-decoration: none;
            font-size: 0.9em;
            padding: 6px 14px;
            border-radius: 8px;
            margin-right: 4px;
        }
        .vdd-nav a.active {
            background: rgba(255,255,255,0.07);
            color: white;
        }
        /* AI Explanation Panel */
        .ai-panel {
            background: linear-gradient(135deg, rgba(0,200,255,0.06) 0%, rgba(18,24,38,0.95) 100%);
            border: 1px solid rgba(0,200,255,0.18);
            border-radius: 16px;
            padding: 22px 24px;
        }
        .ai-reasoning-text {
            font-size: 0.92em;
            color: #cbd5e0;
            line-height: 1.8;
        }
        .ai-tag {
            display: inline-block;
            background: rgba(0,255,204,0.08);
            border: 1px solid rgba(0,255,204,0.2);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.75em;
            color: #00ffcc;
            margin: 3px 3px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- MOCK DATA GENERATOR ---
def get_mock_fleet_data(refresh_count=0):
    # Base data
    base_data = {
        'Vehicle_ID': [f"V-{1000+i}" for i in range(20)],
        'Vehicle_Model': np.random.choice(["Ford F-150", "Toyota Hilux", "Volvo FH16", "Mercedes Actros", "Scania R450"], 20),
        'Fuel_Type': np.random.choice(["Diesel", "Electric", "Hybrid"], 20),
        'Age_Years': np.random.uniform(1, 12, 20).round(1),
        'Mileage_km': np.random.uniform(10000, 500000, 20).round(0),
        'Region': np.random.choice(["North", "South", "East", "West"], 20),
    }
    df = pd.DataFrame(base_data)
    
    # Introduce real-time dynamism based on autorefresh count
    np.random.seed(refresh_count)
    df['Risk_Score'] = (
        (df['Age_Years'] / 12) * 40 + 
        (df['Mileage_km'] / 500000) * 40 + 
        np.random.uniform(0, 20, 20)
    ).clip(0, 100).round(1)
    
    # Categorize Risk
    conditions = [
        (df['Risk_Score'] >= 75),
        (df['Risk_Score'] >= 50),
        (df['Risk_Score'] < 50)
    ]
    choices = ['High', 'Medium', 'Low']
    df['Risk_Level'] = np.select(conditions, choices, default='Low')
    
    return df.sort_values('Risk_Score', ascending=False)

def get_trend_data(refresh_count=0):
    np.random.seed(42) # fixed seed for base trend
    dates = pd.date_range(end=datetime.datetime.today(), periods=30)
    base_trend = np.linspace(40, 65, 30) + np.random.normal(0, 5, 30)
    
    # Add a slight active wiggle
    wiggle = np.random.normal(0, 2, 30) * np.sin(refresh_count / 2)
    current_trend = (base_trend + wiggle).clip(0, 100)
    
    return pd.DataFrame({'Date': dates, 'Average_Fleet_Risk': current_trend})

# --- COMPONENT HELPERS ---
def render_metric(label, value, color_class="glow-text", icon=""):
    st.markdown("""
        <div class="glass-panel" style="text-align: center;">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def render_gauge(score):
    color = "#FF3366" if score >= 75 else "#FFCC00" if score >= 50 else "#00FFCC"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fleet Risk Index", 'font': {'color': 'white'}},
        number = {'font': {'color': color, 'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 255, 204, 0.1)"},
                {'range': [50, 75], 'color': "rgba(255, 204, 0, 0.1)"},
                {'range': [75, 100], 'color': "rgba(255, 51, 102, 0.1)"}],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- MAIN APP ---
def main():
    inject_custom_css()
    
    # Generate interactive data
    df_fleet = get_mock_fleet_data(count)
    df_trend = get_trend_data(count)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 class='glow-text'>⚙️ Quantico API</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: gray; font-size: 0.9em;'>ID: CMP-1006</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        page = st.radio("Navigation", ["📊 Dashboard", "🚘 Vehicle Deep Dive", "💬 AI Assistant (Chat)"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<b style='color:#00ffff;'>Filters</b>", unsafe_allow_html=True)
        selected_fuel = st.multiselect("Fuel Type", options=["All"] + list(df_fleet['Fuel_Type'].unique()), default="All")
        selected_region = st.multiselect("Region", options=["All"] + list(df_fleet['Region'].unique()), default="All")
        age_range = st.slider("Age Range (Years)", 0.0, 15.0, (0.0, 15.0))
        
        # Apply filters
        filtered_df = df_fleet.copy()
        if "All" not in selected_fuel and len(selected_fuel) > 0:
            filtered_df = filtered_df[filtered_df['Fuel_Type'].isin(selected_fuel)]
        if "All" not in selected_region and len(selected_region) > 0:
            filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]
        filtered_df = filtered_df[(filtered_df['Age_Years'] >= age_range[0]) & (filtered_df['Age_Years'] <= age_range[1])]

    # --- DASHBOARD PAGE ---
    if "Dashboard" in page:
        st.markdown("<h2>Fleet Operations Overview</h2>", unsafe_allow_html=True)
        
        # TOP ROW: METRICS
        col1, col2, col3, col4 = st.columns(4)
        avg_risk = filtered_df['Risk_Score'].mean() if not filtered_df.empty else 0
        high_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'High'])
        
        with col1:
            render_metric("Avg Risk Score", f"{avg_risk:.1f}", color_class="glow-text" if avg_risk < 50 else "glow-text-warning" if avg_risk < 75 else "glow-text-error")
        with col2:
            render_metric("High Risk Vehicles", f"{high_risk_count}", color_class="glow-text-error" if high_risk_count > 0 else "glow-text", icon="⚠️")
        with col3:
            render_metric("Total Vehicles", f"{len(filtered_df)}")
        with col4:
            render_metric("Maintenance Overdue", f"{np.random.randint(2, 8)}", color_class="glow-text-warning")

        # MIDDLE ROW: GAUGE & TREND
        col_m1, col_m2 = st.columns([1, 2])
        
        with col_m1:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.plotly_chart(render_gauge(avg_risk), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white;'>Trend Over Time (Average Risk)</b>", unsafe_allow_html=True)
            fig_trend = px.line(df_trend, x='Date', y='Average_Fleet_Risk', template="plotly_dark")
            fig_trend.update_traces(line_color='#00ffcc', line_width=3, fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)')
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # BOTTOM ROW: CHARTS & ROOT CAUSE
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white;'>Fleet Risk Distribution</b>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig_pie = px.pie(filtered_df, names='Risk_Level', color='Risk_Level', hole=0.7, 
                                 color_discrete_map={'High': '#FF3366', 'Medium': '#FFCC00', 'Low': '#00FFCC'})
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=220, margin=dict(l=0, r=0, t=10, b=0))
                
                # Add total label in center
                fig_pie.add_annotation(text=f"{len(filtered_df)}<br>Total", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("No data for current filters")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_b2:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white;'>Maintenance Risk Histogram</b>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig_hist = px.histogram(filtered_df, x='Risk_Score', nbins=10, template="plotly_dark", color_discrete_sequence=['#ffcc00'])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_b3:
            st.markdown("<div class='glass-panel' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white;'>Root Cause Breakdown</b>", unsafe_allow_html=True)
            
            # Simulated root causes
            causes = [("Aging impact", 32, "#FFcc00"), ("Thermal risk", 21, "#FF3366"), ("Lubrication", 14, "#00FFcc")]
            for name, val, color in causes:
                st.markdown(f"<div style='display: flex; justify-content: space-between;'><span style='color: #a0aec0; font-size: 0.9em;'>{name}</span> <span style='color: white;'>{val}%</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='progress-bg'><div class='progress-bar' style='width: {val}%; background-color: {color};'></div></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # High Risk Alert Panel & Table
        st.markdown("<h3 style='margin-top: 20px;'>Top 5 High-Risk Vehicles Alert</h3>", unsafe_allow_html=True)
        top5 = filtered_df.head(5)
        
        if len(top5[top5['Risk_Level'] == 'High']) > 0:
            st.error(f"⚠️ Warning: {len(top5[top5['Risk_Level'] == 'High'])} severe alerts found. Immediate action recommended.", icon="🚨")
        
        # Display styled dataframe
        st.dataframe(
            top5[['Vehicle_ID', 'Vehicle_Model', 'Fuel_Type', 'Age_Years', 'Mileage_km', 'Risk_Score', 'Risk_Level']],
            use_container_width=True,
            hide_index=True
        )
        
        # SHAP graph equivalent (Feature importance)
        st.markdown("<div class='glass-panel' style='margin-top: 20px;'>", unsafe_allow_html=True)
        st.markdown("<b style='color:white;'>Model Explanation: SHAP Feature Importance</b>", unsafe_allow_html=True)
        feature_data = pd.DataFrame({
            "Feature": ["Engine Age", "Mileage", "Avg Temp", "Last Service Days", "Oil Quality"],
            "Importance": [0.35, 0.28, 0.15, 0.12, 0.10]
        }).sort_values('Importance', ascending=True)
        
        fig_shap = px.bar(feature_data, x="Importance", y="Feature", orientation='h', template="plotly_dark")
        fig_shap.update_traces(marker_color='#00ffcc')
        fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== VEHICLE DEEP DIVE PAGE =====
    elif "Deep Dive" in page:

        # Vehicle selector (simulate a fleet)
        vehicles = {
            "V-1001 · Ford F-150 (Diesel)":    {"id": "133755", "model": "Ford F-150 Platinum", "engine": "5.0L V8 Ti-VCT", "fuel": "Diesel", "age": 8.4, "mileage": 320000, "risk": 82.5, "region": "North"},
            "V-1004 · Volvo FH16 (Hybrid)":    {"id": "133759", "model": "Volvo FH16 750", "engine": "16L D16K Euro 6", "fuel": "Hybrid", "age": 5.1, "mileage": 180000, "risk": 54.2, "region": "East"},
            "V-1007 · Toyota Hilux (Diesel)":  {"id": "133762", "model": "Toyota Hilux GR Sport", "engine": "2.8L 1GD-FTV Turbo", "fuel": "Diesel", "age": 3.2, "mileage": 95000,  "risk": 28.7, "region": "South"},
            "V-1012 · Mercedes Actros (EV)":   {"id": "133771", "model": "Mercedes-Benz Actros L", "engine": "Electric 400kW", "fuel": "Electric", "age": 2.0, "mileage": 62000,  "risk": 19.3, "region": "West"},
        }

        selected_vehicle_key = st.selectbox("Select Vehicle", list(vehicles.keys()), label_visibility="collapsed")
        v = vehicles[selected_vehicle_key]
        risk = v["risk"]
        risk_color = "#FF3366" if risk >= 75 else "#FFCC00" if risk >= 50 else "#00FFCC"

        # ── TOP HEADER ──────────────────────────────────────────
        h_left, h_right = st.columns([3, 1])
        with h_left:
            st.markdown("""
                <p class='vdd-header'>{v['model']}</p>
                <p class='vdd-id'>ID: {v['id']} &nbsp;·&nbsp; {v['fuel']} &nbsp;·&nbsp; Region: {v['region']}</p>
            """, unsafe_allow_html=True)

        with h_right:
            st.markdown("""
                <div style='text-align:right; padding-top:8px;'>
                    <div class='vdd-nav'>
                        <a href='#' class='active'>Overview</a>
                        <a href='#'>Analytics</a>
                        <a href='#'>Monitoring</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border:1px solid rgba(255,255,255,0.05); margin:10px 0 20px 0;'>", unsafe_allow_html=True)

        # ── ROW 1: Spec Cards | Visual Banner | Colors ──────────
        row1_left, row1_center, row1_right = st.columns([1.2, 2.5, 1])

        with row1_left:
            st.markdown("""
                <div class='vdd-spec-card'>
                    <span class='vdd-spec-icon'>⚙️</span>
                    <div>
                        <p class='vdd-spec-title'>{v['engine']}</p>
                        <p class='vdd-spec-desc'>Primary Powertrain · Registered {v['age']}yr ago</p>
                    </div>
                </div>
                <div class='vdd-spec-card'>
                    <span class='vdd-spec-icon'>🔩</span>
                    <div>
                        <p class='vdd-spec-title'>Bodywork Grade A</p>
                        <p class='vdd-spec-desc'>Carbon composite reinforced frame</p>
                    </div>
                </div>
                <div class='vdd-spec-card'>
                    <span class='vdd-spec-icon'>🛞</span>
                    <div>
                        <p class='vdd-spec-title'>Tyre Set 4/4</p>
                        <p class='vdd-spec-desc'>Last replaced: 3 months ago</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with row1_center:
            # Display real car image
            import base64
            import pathlib
            car_img_path = pathlib.Path("static/supercar.png")
            if car_img_path.exists():
                b64 = base64.b64encode(car_img_path.read_bytes()).decode()
                st.markdown(f"""
                    <div style='
                        background: radial-gradient(ellipse at center, rgba(180,180,200,0.08) 0%, rgba(0,0,0,0) 70%);
                        border-radius: 20px;
                        padding: 10px 0;
                        text-align: center;
                        position: relative;
                    '>
                        <!-- Connector lines decorations -->
                        <div style='position:absolute; top:40%; left:0; width:18%; height:1px; background:rgba(255,255,255,0.15);'></div>
                        <div style='position:absolute; top:40%; right:0; width:18%; height:1px; background:rgba(255,255,255,0.15);'></div>
                        <img src='data:image/png;base64,{b64}'
                            style='width:100%; max-width:460px; height:220px; object-fit:contain;
                                   filter: drop-shadow(0 0 40px rgba(180,200,255,0.25));
                                   border-radius: 12px;'/>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center; font-size:6rem; padding:30px 0;'>🚗</div>", unsafe_allow_html=True)

        with row1_right:
            st.markdown("""
                <div style='padding-top:10px;'>
                    <div style='font-size:0.72em; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Status Colors</div>
                    <div style='display:flex; gap:10px; flex-wrap:wrap;'>
                        <div style='width:24px; height:24px; border-radius:50%; background:#00ffcc; box-shadow: 0 0 12px #00ffcc66;'></div>
                        <div style='width:24px; height:24px; border-radius:50%; background:#ffcc00; box-shadow: 0 0 12px #ffcc0066;'></div>
                        <div style='width:24px; height:24px; border-radius:50%; background:#ff3366; box-shadow: 0 0 12px #ff336666;'></div>
                        <div style='width:24px; height:24px; border-radius:50%; background:#4488ff; box-shadow: 0 0 12px #4488ff55;'></div>
                        <div style='width:24px; height:24px; border-radius:50%; background:#aaaaaa;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ── ROW 2: Risk Gauge | Maintenance Cards | System Warnings ──
        r2_col1, r2_col2, r2_col3 = st.columns([1.2, 2, 1.2])

        with r2_col1:
            # Animated risk gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk,
                delta={"reference": 50, "valueformat": ".1f"},
                number={"font": {"color": risk_color, "size": 48}, "suffix": ""},
                title={"text": "Risk Score", "font": {"color": "#8b949e", "size": 11}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#2d3748"},
                    "bar": {"color": risk_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],   "color": "rgba(0,255,204,0.07)"},
                        {"range": [50, 75],  "color": "rgba(255,204,0,0.07)"},
                        {"range": [75, 100], "color": "rgba(255,51,102,0.1)"},
                    ],
                    "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.75, "value": risk}
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=220,
                font={"color": "white", "family": "Inter"},
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Consumption mini stats
            dots_green  = "".join([f"<div class='vdd-dot' style='background:#00ffcc; opacity:{0.3+0.07*i};'></div>" for i in range(12)])
            dots_yellow = "".join([f"<div class='vdd-dot' style='background:#ffcc00; opacity:{0.3+0.07*i};'></div>" for i in range(9)])
            dots_pink   = "".join([f"<div class='vdd-dot' style='background:#ff3366; opacity:{0.3+0.07*i};'></div>" for i in range(4)])
            mileage_k = v['mileage'] // 1000
            st.markdown(f"""
                <div class='vdd-stat-box' style='margin-top:10px;'>
                    <div class='vdd-stat-label'>Mileage · km</div>
                    <div class='vdd-stat-value'>{mileage_k}k</div>
                    <div style='font-size:0.75em; color:#4a5568; margin-top:6px;'>
                        🟢 Drive 41% &nbsp; 🟡 Eco 53% &nbsp; 🔴 Idle 6%
                    </div>
                    <div class='vdd-dot-grid'>{dots_green}{dots_yellow}{dots_pink}</div>
                </div>
            """, unsafe_allow_html=True)

        with r2_col2:
            # Maintenance Forecast cards
            st.markdown("""
                <div style='font-size:0.72em; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;'>Maintenance Forecast</div>
            """, unsafe_allow_html=True)

            forecasts = [
                ("🛢️", "Oil Change",         "1300 km",  65,  "#ff3366", "Due soon"),
                ("🛑", "Brake Pad Life",    "65%",      65,  "#ffcc00", "Monitor"),
                ("🔵", "Tyre Pressure",     "Overdue",  100, "#ff3366", "Overdue"),
                ("⚡", "Battery Health",    "87%",      87,  "#00ffcc", "Good"),
            ]
            for icon, label, value_text, bar_val, bar_color, status in forecasts:
                st.markdown("""
                    <div style='margin-bottom:14px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span style='font-size:0.85em; color:#a0aec0;'>{icon} {label}</span>
                            <span style='font-size:0.85em; color:white; font-weight:600;'>{value_text}</span>
                        </div>
                        <div class='progress-bg'>
                            <div class='progress-bar' style='width:{bar_val}%; background:{bar_color};'></div>
                        </div>
                        <div style='font-size:0.72em; color:{bar_color};'>{status}</div>
                    </div>
                """, unsafe_allow_html=True)

        with r2_col3:
            # System warnings panel
            error_count = 3 if risk >= 75 else 1 if risk >= 50 else 0
            warn_color = "#FF3366" if error_count >= 3 else "#FFCC00" if error_count == 1 else "#00FFCC"
            error_msg = "errors found" if error_count > 0 else "All systems OK"
            st.markdown(f"""
                <div class='vdd-stat-box'>
                    <div class='vdd-stat-label'>System Warnings</div>
                    <div class='warn-number' style='color:{warn_color}; text-shadow: 0 0 30px {warn_color}66;'>{error_count}</div>
                    <div style='font-size:0.82em; color:#a0aec0; margin-top:4px;'>{error_msg}</div>
                </div>
            """, unsafe_allow_html=True)

            # Error bar chart
            errors_df = pd.DataFrame({
                "System":  ["Engine", "Wheels", "Gearbox"],
                "Errors":  [2 if risk >= 75 else 0, 1, 0 if risk < 50 else 1],
                "Color":   ["#FF3366", "#FFCC00", "#8b7355"]
            })
            fig_err = px.bar(errors_df, x="Errors", y="System", orientation="h",
                             color="System", color_discrete_sequence=["#FF3366", "#FFCC00", "#8b7355"],
                             template="plotly_dark")
            fig_err.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=180, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("<hr style='border:1px solid rgba(255,255,255,0.05); margin:20px 0;'>", unsafe_allow_html=True)

        # ── ROW 3: SHAP | Root Cause | AI Panel ─────────────────
        r3_col1, r3_col2, r3_col3 = st.columns([1.5, 1, 1.5])

        with r3_col1:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white; font-size:0.9em;'>🔬 SHAP Feature Importance</b>", unsafe_allow_html=True)
            st.markdown("<p style='color:#4a5568; font-size:0.75em; margin-bottom:10px;'>Visual explanation of risk drivers</p>", unsafe_allow_html=True)

            feature_data = pd.DataFrame({
                "Feature": ["Engine Age", "Mileage", "Avg Temp", "Last Service", "Oil Quality"],
                "Importance": [0.35, 0.28, 0.15, 0.12, 0.10],
                "Direction": ["Increases Risk", "Increases Risk", "Increases Risk", "Increases Risk", "Decreases Risk"]
            }).sort_values("Importance", ascending=True)

            fig_shap = px.bar(feature_data, x="Importance", y="Feature", orientation="h",
                              color="Importance", color_continuous_scale=[[0, "#00FFCC"], [0.5, "#FFCC00"], [1, "#FF3366"]],
                              template="plotly_dark")
            fig_shap.update_coloraxes(showscale=False)
            fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   height=260, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r3_col2:
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            st.markdown("<b style='color:white; font-size:0.9em;'>⚠️ Root Cause Breakdown</b>", unsafe_allow_html=True)
            st.markdown("<p style='color:#4a5568; font-size:0.75em; margin-bottom:14px;'>Weighted contribution to risk</p>", unsafe_allow_html=True)

            causes = [
                ("🔥 Aging impact",  32, "#FFcc00"),
                ("🌡️ Thermal risk",  21, "#FF3366"),
                ("💧 Lubrication",   14, "#00FFcc"),
                ("⚙️ Wear & Tear",   18, "#ff8c00"),
                ("🔋 Battery load",  10, "#4488ff"),
                ("💨 Exhaust cycle",  5, "#888888"),
            ]
            for name, val, color in causes:
                st.markdown("""
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:6px;'>
                        <span style='color:#a0aec0; font-size:0.8em;'>{name}</span>
                        <span style='color:white; font-size:0.8em; font-weight:600;'>{val}%</span>
                    </div>
                    <div class='progress-bg'>
                        <div class='progress-bar' style='width:{val}%; background:{color};'></div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with r3_col3:
            # AI Explanation Panel
            recommendation = "Immediate Service Needed" if risk >= 75 else "Schedule Within 30 Days" if risk >= 50 else "Routine Monitoring"
            rec_color = "#FF3366" if risk >= 75 else "#FFCC00" if risk >= 50 else "#00FFCC"

            aging_impact = int(v['age'] / 12 * 40)
            mileage_impact = int(v['mileage'] / 500000 * 35)
            risk_level = "High" if risk >= 75 else "Medium" if risk >= 50 else "Low"
            days_range = "30–60" if risk >= 75 else "60–90" if risk >= 50 else "90+"
            downtime_cost = int(risk * 48)

            ai_text = f"""
                This vehicle has a <b style='color:{risk_color};'>Risk Score of {risk:.1f}/100</b>, placing it in the
                <b>{risk_level} Risk</b> category.<br><br>

                The primary risk contributors are <b>engine aging</b> ({aging_impact}% impact) and <b>high mileage</b>
                ({mileage_impact}% impact), with secondary contributions from thermal stress and lubrication degradation.
                These factors collectively indicate an elevated probability of component failure within the next
                <b>{days_range} days</b> without intervention.<br><br>

                <b>AI Recommendation:</b> <span style='color:{rec_color};'>{recommendation}</span>. Prioritize oil change
                and brake inspection. Estimated downtime cost if delayed: <b>${downtime_cost:,}</b>.
            """

            st.markdown(f"""
                <div class='ai-panel'>
                    <div style='display:flex; align-items:center; gap:8px; margin-bottom:14px;'>
                        <span style='font-size:1.4rem;'>🤖</span>
                        <div>
                            <div style='font-weight:600; font-size:0.9em;'>AI Diagnostic Reasoning</div>
                            <div style='font-size:0.7em; color:#4a5568;'>Powered by Fleet Intelligence Engine</div>
                        </div>
                    </div>
                    <div class='ai-reasoning-text'>{ai_text}</div>
                    <div style='margin-top:16px;'>
                        <span class='ai-tag'>🔥 Aging</span>
                        <span class='ai-tag'>🌡️ Thermal</span>
                        <span class='ai-tag'>💧 Lubrication</span>
                        <span class='ai-tag'>📍 {v["region"]} Region</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ===== CHAT PAGE =====
    elif "Chat" in page:

        # --- AI Responses engine ---
        def ai_respond(user_msg: str) -> str:
            msg = user_msg.lower()
            if "compare" in msg:
                vehicles = [w for w in user_msg.split() if w.startswith('V-')]
                if len(vehicles) >= 2:
                    return (f"🔍 **Comparison: {vehicles[0]} vs {vehicles[1]}**\n\n"
                            f"- **{vehicles[0]}**: Risk Score 78.4 | Age 9yr | 320k km | Diesel\n"
                            f"- **{vehicles[1]}**: Risk Score 52.1 | Age 5yr | 180k km | Hybrid\n\n"
                            f"**Verdict:** {vehicles[0]} is in a significantly higher risk category and needs **immediate maintenance**.")
                return "Please specify two vehicles to compare, e.g., *Compare V-1001 and V-1005*."
            elif "delay" in msg and "maintenance" in msg:
                return ("⚠️ **Delay Impact Analysis**\n\n"
                        "Delaying maintenance by **3 months** for a high-risk vehicle increases:\n"
                        "- Critical failure risk: **+18%**\n"
                        "- Estimated cost overhead: **$3,500–$6,200**\n"
                        "- Downtime risk: **+2.4 days average**\n\n"
                        "*Recommendation: Schedule service within the next 2 weeks.*")
            elif "cost" in msg or "estimate" in msg:
                return ("💰 **Estimated Cost Impact**\n\n"
                        "Based on current fleet risk profile:\n"
                        "- Preventive maintenance cost: **$1,200/vehicle**\n"
                        "- Reactive repair (if delayed): **$4,800–$9,000**\n"
                        "- Fleet-wide savings potential: **$38,000/quarter** \n\n"
                        "*Invest in prevention to avoid escalating repair costs.*")
            elif "risk" in msg or "score" in msg:
                return ("📊 **Fleet Risk Summary (Live)**\n\n"
                        "- High Risk: 4 vehicles (immediate action needed)\n"
                        "- Medium Risk: 8 vehicles (schedule within 30 days)\n"
                        "- Low Risk: 8 vehicles (routine monitoring)\n\n"
                        "*Average fleet risk index: 58.4 / 100*")
            elif "hello" in msg or "hi" in msg:
                return "👋 Hello! I'm your AI Fleet Maintenance Assistant. Ask me about vehicle risk, cost estimates, or comparisons."
            else:
                return ("🤖 I understand your query. Based on current fleet data and maintenance logs, "
                        "I recommend reviewing the top 5 high-risk vehicles flagged on the Dashboard. "
                        "You can also ask me to **compare** vehicles, **estimate cost impact**, or analyze **delay scenarios**.")

        # Initialize chat state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "show_welcome" not in st.session_state:
            st.session_state.show_welcome = True

        # Layout: Sidebar history | Main chat area
        chat_sidebar, chat_main = st.columns([1, 3.2])

        # ---- LEFT: Chat History Sidebar ----
        with chat_sidebar:
            st.markdown("""
                <div class='glass-panel' style='min-height: 72vh; padding: 18px 12px;'>
                    <div style='display:flex; align-items:center; gap:8px; margin-bottom:18px;'>
                        <span style='font-size:1.2rem;'>🤖</span>
                        <span style='font-weight:600; font-size:0.95em;'>Fleet AI</span>
                    </div>
                    <div style='font-size:0.7em; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Conversations</div>
            """, unsafe_allow_html=True)

            history_items = [
                ("🔴", "V-1002 Engine Alert"),
                ("🟡", "Fleet Risk Overview"),
                ("🟢", "Q4 Maintenance Plan"),
                ("⚪", "Cost Estimation Run"),
                ("⚪", "Region East Analysis"),
            ]
            active_idx = 0
            for i, (dot, label) in enumerate(history_items):
                cls = 'chat-history-item active-convo' if i == active_idx else 'chat-history-item'
                st.markdown(f"<div class='{cls}'>{dot} {label}</div>", unsafe_allow_html=True)

            st.markdown("""
                    <div style='margin-top:24px; border-top:1px solid rgba(255,255,255,0.05); padding-top:14px;'>
                        <div style='font-size:0.7em; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Quick Actions</div>
                        <div class='chat-history-item'>📊 Fleet Report</div>
                        <div class='chat-history-item'>⚙️ Settings</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ---- RIGHT: Main Chat Area ----
        with chat_main:
            # WELCOME SCREEN (shown when no messages)
            if st.session_state.show_welcome and len(st.session_state.messages) == 0:
                st.markdown("""
                    <div style='padding: 40px 10px 20px 10px;'>
                        <p class='welcome-title'>Welcome to Fleet AI</p>
                        <p class='welcome-subtitle'>Your intelligent assistant for vehicle maintenance analysis and fleet insights</p>
                    </div>
                """, unsafe_allow_html=True)

                # Feature Cards Grid
                c1, c2, c3, c4 = st.columns(4)
                cards = [
                    (c1, "fc-purple", "⚡", "Risk Analysis",      "Deep-dive into vehicle risk scores and health metrics"),
                    (c2, "fc-blue",   "🌐", "Fleet Comparison",    "Compare vehicles side-by-side across any parameter"),
                    (c3, "fc-pink",   "🔮", "Predictive Insights", "AI-powered failure forecasting and cost estimation"),
                    (c4, "fc-dark",   "</>","Maintenance Plan",   "Generate structured service schedules automatically"),
                ]
                for col, fc_class, icon, title, desc in cards:
                    with col:
                        st.markdown("""
                            <div class='feature-card {fc_class}'>
                                <span class='feature-card-icon'>{icon}</span>
                                <div class='feature-card-title'>{title}</div>
                                <div class='feature-card-desc'>{desc}</div>
                            </div>
                        """, unsafe_allow_html=True)

                # Suggested Prompt Chips
                st.markdown("""
                    <div style='text-align:center; margin-top: 30px; margin-bottom: 10px;'>
                        <span class='prompt-chip'>🚗 Tell me a bad fault</span>
                        <span class='prompt-chip'>📽 Recommend a review to watch</span>
                        <span class='prompt-chip'>🤔 How do I plan care plan?</span>
                        <span class='prompt-chip'>💡 What's new?</span>
                    </div>
                """, unsafe_allow_html=True)

            # CHAT MESSAGES (shown when conversation started)
            else:
                st.markdown("<div class='chat-area'>", unsafe_allow_html=True)
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown("""
                            <div style='text-align:right;'>
                                <div class='user-label'>You</div>
                                <div class='chat-bubble-user'>{msg['content']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div>
                                <div class='bot-label'>🤖 Fleet AI</div>
                                <div class='chat-bubble-bot'>{msg['content']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Typing animation when awaiting response
            if st.session_state.get("is_typing", False):
                st.markdown("""
                    <div>
                        <div class='bot-label'>🤖 Fleet AI</div>
                        <div class='chat-bubble-bot'>
                            <span style='letter-spacing: 3px; animation: blink 1s infinite;'>● ● ●</span>
                        </div>
                    </div>
                    <style>
                    @keyframes blink {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.3; }
                    }
                    </style>
                """, unsafe_allow_html=True)

            # --- Bottom Input Bar ---
            st.markdown("<div style='margin-top: 16px;'>", unsafe_allow_html=True)

            # Suggested prompts above input
            if len(st.session_state.messages) == 0:
                c_s1, c_s2, c_s3 = st.columns(3)
                suggested = [
                    (c_s1, "Compare Vehicle A and B"),
                    (c_s2, "What if I delay maintenance by 3 months?"),
                    (c_s3, "Estimate cost impact"),
                ]
                for col, text in suggested:
                    with col:
                        if st.button(f"💬 {text}", key=f"suggest_{text[:10]}", use_container_width=True):
                            st.session_state.show_welcome = False
                            st.session_state.is_typing = True
                            st.session_state.messages.append({"role": "user", "content": text})
                            st.rerun()

            prompt = st.chat_input("Ask anything... e.g. 'Compare V-1001 and V-1005'")
            if prompt:
                st.session_state.show_welcome = False
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.is_typing = True
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Process AI response after rerun
        if st.session_state.get("is_typing", False) and len(st.session_state.messages) > 0:
            if st.session_state.messages[-1]["role"] == "user":
                last_user_msg = st.session_state.messages[-1]["content"]
                time.sleep(0.8)  # Short typing delay
                response = ai_respond(last_user_msg)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.is_typing = False
                st.rerun()

if __name__ == "__main__":
    main()
