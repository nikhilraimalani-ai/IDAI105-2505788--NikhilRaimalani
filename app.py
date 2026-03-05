import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #07090f; }
.block-container { padding: 2rem 2.5rem 4rem; }

[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
[data-testid="stSidebar"] label { color: #6b7a99 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

.page-hero { margin-bottom: 2rem; }
.page-hero h1 { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#e8edf5; letter-spacing:-0.02em; line-height:1.1; margin:0 0 0.4rem; }
.page-hero p  { font-size:0.95rem; color:#5a6a8a; margin:0; }

.divider { height:1px; background:#1a2236; margin:1.5rem 0; }

.section-hd {
    font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700;
    color:#c5cfe0; margin:2rem 0 0.8rem;
    display:flex; align-items:center; gap:8px;
}
.section-hd::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1a2236,transparent); }

.insight-card {
    background:#0a1628; border:1px solid #1a2e4a; border-left:3px solid #4da3ff;
    border-radius:8px; padding:14px 16px; margin:8px 0;
    font-size:0.87rem; color:#8ba8cc; line-height:1.6;
}
.insight-card strong { color:#c5d8f0; }

[data-testid="metric-container"] {
    background:#0d1117 !important; border:1px solid #1a2236 !important;
    border-radius:12px !important; padding:16px 20px !important;
}
[data-testid="metric-container"] label { color:#4a5a7a !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e8edf5 !important; font-family:'Syne',sans-serif; font-size:1.8rem !important; }
div[data-testid="stHorizontalBlock"] { gap:14px; }
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; color:#e8edf5 !important; }
p, li { color:#8a9ab8; }
.stDataFrame { border-radius:10px; overflow:hidden; }
.stCaption { color:#3a4a6a !important; font-size:0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ALTAIR DARK THEME
# ─────────────────────────────────────────────
DARK_THEME = {
    "config": {
        "background": "#0d1117",
        "view": {"fill": "#0d1117", "stroke": "transparent"},
        "axis": {
            "domainColor": "#1a2236", "gridColor": "#1a2236",
            "tickColor": "#1a2236", "labelColor": "#8a9ab8",
            "titleColor": "#8a9ab8", "labelFont": "DM Sans", "titleFont": "DM Sans"
        },
        "legend": {
            "labelColor": "#8a9ab8", "titleColor": "#8a9ab8",
            "labelFont": "DM Sans", "titleFont": "DM Sans",
            "fillColor": "#0d1117", "strokeColor": "#1a2236"
        },
        "title": {"color": "#c5cfe0", "font": "Syne", "fontSize": 14},
        "mark": {"tooltip": True}
    }
}
alt.themes.register("dark", lambda: DARK_THEME)
alt.themes.enable("dark")

COLORS = ["#4da3ff","#00d4c8","#a78bfa","#fb923c","#34d399","#f472b6","#facc15","#60a5fa"]

# ─────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("detailed_ev_charging_stations.csv")

@st.cache_data
def run_clustering(n_clusters):
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    features = ["Usage Stats (avg users/day)","Charging Capacity (kW)",
                "Cost (USD/kWh)","Reviews (Rating)","Distance to City (km)","Parking Spots"]
    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(2,11)]
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
    result_df = df.loc[X.index].copy()
    result_df["Cluster"] = labels.astype(str)
    return result_df, inertias, features

@st.cache_data
def compute_rules(min_support, min_confidence):
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        return pd.DataFrame()
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    df["Usage Level"]    = pd.cut(df["Usage Stats (avg users/day)"], bins=[0,30,60,100,999],
                                   labels=["Low Usage","Medium Usage","High Usage","Very High Usage"])
    df["Cost Level"]     = pd.cut(df["Cost (USD/kWh)"],              bins=[0,0.2,0.35,0.5,999],
                                   labels=["Cheap","Moderate Cost","Expensive","Very Expensive"])
    df["Rating Level"]   = pd.cut(df["Reviews (Rating)"],            bins=[0,3,4,5],
                                   labels=["Low Rated","Mid Rated","High Rated"])
    df["Capacity Level"] = pd.cut(df["Charging Capacity (kW)"],      bins=[0,50,150,350,9999],
                                   labels=["Low Capacity","Medium Capacity","High Capacity","Ultra Capacity"])
    tcols = ["Charger Type","Station Operator","Renewable Energy Source",
             "Maintenance Frequency","Usage Level","Cost Level","Rating Level","Capacity Level"]
    te = TransactionEncoder()
    te_array = te.fit_transform(df[tcols].astype(str).values.tolist())
    te_df = pd.DataFrame(te_array, columns=te.columns_)
    freq = apriori(te_df, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules[["antecedents","consequents","support","confidence","lift"]].sort_values("lift", ascending=False).reset_index(drop=True)

@st.cache_data
def detect_anomalies(method, threshold):
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    df["anomaly"] = False
    df["anomaly_reason"] = ""
    for col in ["Usage Stats (avg users/day)","Cost (USD/kWh)","Reviews (Rating)","Charging Capacity (kW)"]:
        if method == "Z-Score":
            flag = ((df[col] - df[col].mean()) / df[col].std()).abs() > threshold
        else:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            flag = (df[col] < Q1 - threshold*(Q3-Q1)) | (df[col] > Q3 + threshold*(Q3-Q1))
        df.loc[flag & ~df["anomaly"], "anomaly_reason"] = col
        df.loc[flag, "anomaly"] = True
    return df

df_raw = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 4px 24px;border-bottom:1px solid #1a2236;margin-bottom:16px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;color:#4da3ff;'>⚡ SmartCharging</div>
        <div style='font-size:0.7rem;color:#3a4a6a;margin-top:3px;text-transform:uppercase;letter-spacing:0.1em;'>EV Analytics Platform</div>
    </div>""", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for pname, icon in [("Home","🏠"),("EDA","📊"),("Map","🗺️"),
                         ("Clustering","🔵"),("Association","🔗"),("Anomalies","🚨")]:
        if st.button(f"{icon}  {pname}", key=f"nav_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    st.markdown("<div style='height:1px;background:#1a2236;margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#2a3a5a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Global Filters</div>", unsafe_allow_html=True)

    sel_charger  = st.selectbox("Charger Type", ["All"]+sorted(df_raw["Charger Type"].dropna().unique().tolist()))
    sel_operator = st.selectbox("Operator",     ["All"]+sorted(df_raw["Station Operator"].dropna().unique().tolist()))
    sel_renewable= st.selectbox("Renewable",    ["All","Yes","No"])

    df = df_raw.copy()
    if sel_charger   != "All": df = df[df["Charger Type"]==sel_charger]
    if sel_operator  != "All": df = df[df["Station Operator"]==sel_operator]
    if sel_renewable != "All": df = df[df["Renewable Energy Source"]==sel_renewable]

    st.markdown(f"<div style='font-size:0.72rem;color:#2a3a5a;margin-top:8px;'>{len(df):,} stations loaded</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1a2236;margin:16px 0 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#1a2a4a;text-align:center;'>SmartCharging Analytics · 2025</div>", unsafe_allow_html=True)

page = st.session_state.page

def sh(label):
    st.markdown(f"<div class='section-hd'>{label}</div>", unsafe_allow_html=True)

def pdk_map(data, color_cols=("r","g","b"), radius=35000, tooltip_html="<b>{Station ID}</b>"):
    layer = pdk.Layer("ScatterplotLayer", data=data,
                       get_position=["Longitude","Latitude"],
                       get_color=[*[f"{c}" for c in color_cols], 200],
                       get_radius=radius, pickable=True, auto_highlight=True)
    return pdk.Deck(layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.2),
                    tooltip={"html": tooltip_html,
                             "style":{"backgroundColor":"#0d1117","color":"#c9d1e0",
                                      "fontSize":"12px","padding":"8px","borderRadius":"8px"}},
                    map_style="mapbox://styles/mapbox/dark-v10")

# ═════════════════════════════════════════════  HOME
# ═════════════════════════════════════════════
if page == "Home":
    st.markdown("<div class='page-hero'><h1>SmartCharging Analytics</h1><p>Global EV charging intelligence — clusters, associations, anomalies, and live maps.</p></div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Stations",   f"{len(df):,}")
    with c2: st.metric("Avg Users / Day",  f"{df['Usage Stats (avg users/day)'].mean():.1f}")
    with c3: st.metric("Avg Rating",       f"{df['Reviews (Rating)'].mean():.2f} / 5")
    with c4: st.metric("Renewable",        f"{(df['Renewable Energy Source']=='Yes').mean()*100:.1f}%")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c5,c6 = st.columns(2)
    with c5:
        sh("Charger Type Distribution")
        ct = df["Charger Type"].value_counts().reset_index(); ct.columns=["Type","Count"]
        st.altair_chart(alt.Chart(ct).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Type:N",axis=alt.Axis(labelAngle=0)), y="Count:Q",
            color=alt.Color("Type:N",scale=alt.Scale(range=COLORS),legend=None),
            tooltip=["Type","Count"]).properties(height=280), use_container_width=True)
    with c6:
        sh("Top Station Operators")
        op = df["Station Operator"].value_counts().head(6).reset_index(); op.columns=["Operator","Count"]
        st.altair_chart(alt.Chart(op).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Operator:N",axis=alt.Axis(labelAngle=-15)), y="Count:Q",
            color=alt.Color("Operator:N",scale=alt.Scale(range=COLORS),legend=None),
            tooltip=["Operator","Count"]).properties(height=280), use_container_width=True)

    sh("Stations Installed Over Time")
    yr = df["Installation Year"].value_counts().sort_index().reset_index(); yr.columns=["Year","Count"]
    area = alt.Chart(yr).mark_area(
        line={"color":"#4da3ff","strokeWidth":2},
        color=alt.Gradient("linear",stops=[
            alt.GradientStop(color="rgba(77,163,255,0.3)",offset=0),
            alt.GradientStop(color="rgba(77,163,255,0.02)",offset=1)],x1=1,x2=1,y1=1,y2=0)
    ).encode(x=alt.X("Year:O",axis=alt.Axis(labelAngle=0)),y="Count:Q",tooltip=["Year","Count"]).properties(height=220)
    st.altair_chart(area, use_container_width=True)

    sh("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, height=300)
    st.caption(f"Showing first 50 of {len(df):,} records")

# ═════════════════════════════════════════════  EDA
# ═════════════════════════════════════════════
elif page == "EDA":
    st.markdown("<div class='page-hero'><h1>Exploratory Analysis</h1><p>Distributions, correlations, and behavioral patterns.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        sh("Usage Stats Distribution")
        st.altair_chart(alt.Chart(df).mark_bar(color="#4da3ff",opacity=0.85,cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
            x=alt.X("Usage Stats (avg users/day):Q",bin=alt.Bin(maxbins=40),title="Users/Day"),
            y="count()",tooltip=["count()"]).properties(height=260), use_container_width=True)
    with c2:
        sh("Cost Distribution (USD/kWh)")
        st.altair_chart(alt.Chart(df).mark_bar(color="#00d4c8",opacity=0.85,cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
            x=alt.X("Cost (USD/kWh):Q",bin=alt.Bin(maxbins=40)),
            y="count()",tooltip=["count()"]).properties(height=260), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        sh("Ratings Distribution")
        st.altair_chart(alt.Chart(df).mark_bar(color="#a78bfa",opacity=0.85,cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
            x=alt.X("Reviews (Rating):Q",bin=alt.Bin(maxbins=25)),
            y="count()",tooltip=["count()"]).properties(height=260), use_container_width=True)
    with c4:
        sh("Charging Capacity by Type")
        st.altair_chart(alt.Chart(df).mark_boxplot(extent="min-max").encode(
            x=alt.X("Charger Type:N",axis=alt.Axis(labelAngle=0)),
            y="Charging Capacity (kW):Q",
            color=alt.Color("Charger Type:N",scale=alt.Scale(range=COLORS),legend=None)
        ).properties(height=260), use_container_width=True)

    sh("Average Daily Usage by Operator")
    op_u = df.groupby("Station Operator")["Usage Stats (avg users/day)"].mean().reset_index()
    op_u.columns = ["Operator","Avg Usage"]
    st.altair_chart(alt.Chart(op_u).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X("Operator:N",sort="-y",axis=alt.Axis(labelAngle=-15)),
        y="Avg Usage:Q",
        color=alt.Color("Avg Usage:Q",scale=alt.Scale(scheme="blues"),legend=None),
        tooltip=["Operator","Avg Usage"]).properties(height=260), use_container_width=True)

    sh("Usage vs Cost — Interactive Scatter")
    samp = df.sample(min(800,len(df)),random_state=42)
    st.altair_chart(alt.Chart(samp).mark_circle(opacity=0.6,size=40).encode(
        x="Cost (USD/kWh):Q", y="Usage Stats (avg users/day):Q",
        color=alt.Color("Charger Type:N",scale=alt.Scale(range=COLORS)),
        tooltip=["Station ID","Charger Type","Cost (USD/kWh)","Usage Stats (avg users/day)","Reviews (Rating)"]
    ).properties(height=300).interactive(), use_container_width=True)

    c5,c6 = st.columns(2)
    with c5:
        sh("Renewable Energy Split")
        ren = df["Renewable Energy Source"].value_counts().reset_index(); ren.columns=["Renewable","Count"]
        st.altair_chart(alt.Chart(ren).mark_arc(innerRadius=60).encode(
            theta="Count:Q",
            color=alt.Color("Renewable:N",scale=alt.Scale(domain=["Yes","No"],range=["#34d399","#f87171"])),
            tooltip=["Renewable","Count"]).properties(height=260), use_container_width=True)
    with c6:
        sh("Maintenance Frequency")
        mf = df["Maintenance Frequency"].value_counts().reset_index(); mf.columns=["Frequency","Count"]
        st.altair_chart(alt.Chart(mf).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Frequency:N",axis=alt.Axis(labelAngle=0)),y="Count:Q",
            color=alt.Color("Frequency:N",scale=alt.Scale(range=COLORS),legend=None),
            tooltip=["Frequency","Count"]).properties(height=260), use_container_width=True)

    sh("Correlation Heatmap")
    num_cols = ["Cost (USD/kWh)","Distance to City (km)","Usage Stats (avg users/day)",
                "Charging Capacity (kW)","Reviews (Rating)","Parking Spots"]
    corr = df[num_cols].corr().round(2).reset_index().melt("index")
    corr.columns = ["Var1","Var2","Correlation"]
    hm = alt.Chart(corr).mark_rect().encode(
        x=alt.X("Var1:N",title=None,axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Var2:N",title=None),
        color=alt.Color("Correlation:Q",scale=alt.Scale(scheme="redblue",domain=[-1,1])),
        tooltip=["Var1","Var2","Correlation"])
    txt = alt.Chart(corr).mark_text(fontSize=11).encode(
        x="Var1:N",y="Var2:N",
        text=alt.Text("Correlation:Q",format=".2f"),
        color=alt.condition(alt.datum.Correlation>0.5,alt.value("white"),alt.value("#c9d1e0")))
    st.altair_chart((hm+txt).properties(height=340), use_container_width=True)

# ═════════════════════════════════════════════  MAP
# ═════════════════════════════════════════════
elif page == "Map":
    st.markdown("<div class='page-hero'><h1>Station Map</h1><p>Every EV charging station on the globe. Hover for details.</p></div>", unsafe_allow_html=True)

    color_by = st.selectbox("Color by", ["Charger Type","Renewable Energy Source","Station Operator"])
    smap = df.dropna(subset=["Latitude","Longitude"]).sample(min(1500,len(df)),random_state=42).copy()

    palette = [[77,163,255],[0,212,200],[167,139,250],[251,146,60],[52,211,153],[244,114,182],[250,204,21],[96,165,250]]
    cat_map = {c: palette[i%len(palette)] for i,c in enumerate(smap[color_by].unique())}
    smap["r"] = smap[color_by].map(lambda c: cat_map[c][0])
    smap["g"] = smap[color_by].map(lambda c: cat_map[c][1])
    smap["b"] = smap[color_by].map(lambda c: cat_map[c][2])

    st.pydeck_chart(pdk_map(smap, radius=35000,
        tooltip_html="<b>{Station ID}</b><br/>{Address}<br/>Type: {Charger Type}<br/>Operator: {Station Operator}<br/>Usage: {Usage Stats (avg users/day)} users/day<br/>Rating: ⭐ {Reviews (Rating)}<br/>Cost: ${Cost (USD/kWh)}/kWh"), use_container_width=True)
    st.caption(f"Showing {len(smap):,} of {len(df):,} stations · Colored by {color_by}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    s2 = df.sample(min(600,len(df)),random_state=1)
    with c1:
        sh("Usage vs Distance to City")
        st.altair_chart(alt.Chart(s2).mark_circle(opacity=0.55,size=35).encode(
            x="Distance to City (km):Q", y="Usage Stats (avg users/day):Q",
            color=alt.Color("Charger Type:N",scale=alt.Scale(range=COLORS)),
            tooltip=["Station ID","Distance to City (km)","Usage Stats (avg users/day)","Charger Type"]
        ).properties(height=280).interactive(), use_container_width=True)
    with c2:
        sh("Cost vs Rating by Operator")
        st.altair_chart(alt.Chart(s2).mark_circle(opacity=0.55,size=35).encode(
            x="Cost (USD/kWh):Q", y="Reviews (Rating):Q",
            color=alt.Color("Station Operator:N",scale=alt.Scale(range=COLORS)),
            tooltip=["Station ID","Cost (USD/kWh)","Reviews (Rating)","Station Operator"]
        ).properties(height=280).interactive(), use_container_width=True)

# ═════════════════════════════════════════════  CLUSTERING
# ═════════════════════════════════════════════
elif page == "Clustering":
    st.markdown("<div class='page-hero'><h1>Cluster Analysis</h1><p>Group stations by usage behavior, cost, capacity, and performance.</p></div>", unsafe_allow_html=True)

    n_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
    result_df, inertias, features = run_clustering(n_clusters)

    sh("Elbow Method — Choosing Optimal K")
    elbow_df = pd.DataFrame({"K":list(range(2,11)),"Inertia":inertias})
    st.altair_chart(alt.Chart(elbow_df).mark_line(color="#4da3ff",strokeWidth=2.5,
        point=alt.OverlayMarkDef(color="#4da3ff",size=60)).encode(
        x=alt.X("K:O",axis=alt.Axis(labelAngle=0)),y="Inertia:Q",tooltip=["K","Inertia"]
    ).properties(height=240), use_container_width=True)

    sh("Cluster Scatter")
    c1,c2 = st.columns(2)
    with c1: x_axis = st.selectbox("X Axis", features, index=0)
    with c2: y_axis = st.selectbox("Y Axis", features, index=1)
    samp = result_df.sample(min(800,len(result_df)),random_state=1)
    st.altair_chart(alt.Chart(samp).mark_circle(opacity=0.7,size=45).encode(
        x=alt.X(f"{x_axis}:Q"), y=alt.Y(f"{y_axis}:Q"),
        color=alt.Color("Cluster:N",scale=alt.Scale(range=COLORS)),
        tooltip=["Station ID","Charger Type","Station Operator","Cluster",x_axis,y_axis]
    ).properties(height=340).interactive(), use_container_width=True)

    sh("Cluster Profiles")
    summary = result_df.groupby("Cluster")[features].mean().round(2)
    lm = {}
    for idx, row in summary.iterrows():
        if row["Usage Stats (avg users/day)"]>summary["Usage Stats (avg users/day)"].median() and row["Cost (USD/kWh)"]<summary["Cost (USD/kWh)"].median():
            lm[idx]="🟢 High Demand · Low Cost"
        elif row["Usage Stats (avg users/day)"]>summary["Usage Stats (avg users/day)"].median():
            lm[idx]="🟡 High Demand · Premium"
        elif row["Reviews (Rating)"]>summary["Reviews (Rating)"].median():
            lm[idx]="🔵 Low Traffic · High Rated"
        else:
            lm[idx]="🔴 Underperforming"
    summary.insert(0,"Profile",[lm[i] for i in summary.index])
    summary.index = [f"Cluster {i}" for i in summary.index]
    st.dataframe(summary, use_container_width=True)

    sh("Cluster World Map")
    mdata = result_df.dropna(subset=["Latitude","Longitude"]).sample(min(1200,len(result_df)),random_state=42).copy()
    cp = [[77,163,255],[0,212,200],[167,139,250],[251,146,60],[52,211,153],[244,114,182],[250,204,21]]
    mdata["r"] = mdata["Cluster"].apply(lambda c: cp[int(c)%len(cp)][0])
    mdata["g"] = mdata["Cluster"].apply(lambda c: cp[int(c)%len(cp)][1])
    mdata["b"] = mdata["Cluster"].apply(lambda c: cp[int(c)%len(cp)][2])
    st.pydeck_chart(pdk_map(mdata, radius=35000,
        tooltip_html="<b>{Station ID}</b><br/>Cluster: {Cluster}<br/>{Charger Type}"), use_container_width=True)

    sh("Cluster Composition")
    c3,c4 = st.columns(2)
    with c3:
        comp = result_df.groupby(["Cluster","Charger Type"]).size().reset_index(name="Count")
        st.altair_chart(alt.Chart(comp).mark_bar().encode(
            x=alt.X("Cluster:N",axis=alt.Axis(labelAngle=0)),y="Count:Q",
            color=alt.Color("Charger Type:N",scale=alt.Scale(range=COLORS)),
            tooltip=["Cluster","Charger Type","Count"]).properties(height=260), use_container_width=True)
    with c4:
        comp2 = result_df.groupby(["Cluster","Renewable Energy Source"]).size().reset_index(name="Count")
        st.altair_chart(alt.Chart(comp2).mark_bar().encode(
            x=alt.X("Cluster:N",axis=alt.Axis(labelAngle=0)),y="Count:Q",
            color=alt.Color("Renewable Energy Source:N",scale=alt.Scale(domain=["Yes","No"],range=["#34d399","#f87171"])),
            tooltip=["Cluster","Renewable Energy Source","Count"]).properties(height=260), use_container_width=True)

# ═════════════════════════════════════════════  ASSOCIATION
# ═════════════════════════════════════════════
elif page == "Association":
    st.markdown("<div class='page-hero'><h1>Association Rule Mining</h1><p>Hidden connections between station features, usage levels, and performance.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: min_support    = st.slider("Min Support",    0.05,0.5, 0.1, 0.01)
    with c2: min_confidence = st.slider("Min Confidence", 0.3, 1.0, 0.5, 0.05)
    st.markdown("""<div class='insight-card'>
        <strong>Support</strong> — how often the pattern appears &nbsp;·&nbsp;
        <strong>Confidence</strong> — reliability of the rule &nbsp;·&nbsp;
        <strong>Lift</strong> — how much better than random chance (>1 = meaningful)
    </div>""", unsafe_allow_html=True)

    with st.spinner("Mining rules..."):
        rules = compute_rules(min_support, min_confidence)

    if rules.empty:
        st.warning("No rules found. Lower the thresholds.")
    else:
        st.success(f"Found **{len(rules)}** rules")
        sh("Top Rules by Lift")
        disp = rules.head(20).copy()
        disp[["support","confidence","lift"]] = disp[["support","confidence","lift"]].round(3)
        disp.index = range(1,len(disp)+1)
        st.dataframe(disp, use_container_width=True)

        sh("Strongest 10 Rules")
        top10 = rules.head(10).copy()
        top10["rule"] = top10["antecedents"] + " → " + top10["consequents"]
        st.altair_chart(alt.Chart(top10).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("lift:Q",title="Lift"),
            y=alt.Y("rule:N",sort="-x",title=None),
            color=alt.Color("confidence:Q",scale=alt.Scale(scheme="blues")),
            tooltip=["antecedents","consequents","support","confidence","lift"]
        ).properties(height=320), use_container_width=True)

        sh("Support vs Confidence (bubble = lift)")
        st.altair_chart(alt.Chart(rules.head(60)).mark_circle().encode(
            x="support:Q", y="confidence:Q",
            size=alt.Size("lift:Q",scale=alt.Scale(range=[40,500])),
            color=alt.Color("lift:Q",scale=alt.Scale(scheme="viridis")),
            tooltip=["antecedents","consequents","support","confidence","lift"]
        ).properties(height=320).interactive(), use_container_width=True)

        sh("💬 Plain English Insights")
        for _, row in rules.head(6).iterrows():
            st.markdown(f"""<div class='insight-card'>
                Stations with <strong>{row['antecedents']}</strong> are
                {"strongly" if row['lift']>2 else "moderately"} associated with
                <strong>{row['consequents']}</strong> —
                confidence <strong>{row['confidence']:.0%}</strong>,
                lift <strong>{row['lift']:.2f}×</strong>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════  ANOMALIES
# ═════════════════════════════════════════════
elif page == "Anomalies":
    st.markdown("<div class='page-hero'><h1>Anomaly Detection</h1><p>Stations with unusual usage, cost, rating, or capacity behavior.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: method = st.selectbox("Detection Method", ["Z-Score","IQR"])
    with c2:
        if method=="Z-Score": threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
        else:                 threshold = st.slider("IQR Multiplier",     1.0, 3.0, 1.5, 0.1)

    result    = detect_anomalies(method, threshold)
    anomalies = result[result["anomaly"]].copy()
    normal    = result[~result["anomaly"]].copy()
    if sel_charger  != "All": anomalies = anomalies[anomalies["Charger Type"]==sel_charger]
    if sel_operator != "All": anomalies = anomalies[anomalies["Station Operator"]==sel_operator]

    c3,c4,c5 = st.columns(3)
    with c3: st.metric("Anomalous Stations", f"{len(anomalies):,}")
    with c4: st.metric("Anomaly Rate",        f"{len(anomalies)/len(result)*100:.1f}%")
    with c5: st.metric("Normal Stations",     f"{len(normal):,}")

    sh("Distribution — Normal vs Anomaly")
    sel_feat = st.selectbox("Feature to inspect", [
        "Usage Stats (avg users/day)","Cost (USD/kWh)","Reviews (Rating)","Charging Capacity (kW)"])
    nd = normal.sample(min(800,len(normal)),random_state=1)[[sel_feat]].assign(Type="Normal")
    ad = anomalies[[sel_feat]].assign(Type="Anomaly") if len(anomalies)>0 else pd.DataFrame(columns=[sel_feat,"Type"])
    hist_df = pd.concat([nd,ad],ignore_index=True)
    st.altair_chart(alt.Chart(hist_df).mark_bar(opacity=0.75,cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
        x=alt.X(f"{sel_feat}:Q",bin=alt.Bin(maxbins=50),title=sel_feat),
        y=alt.Y("count()",stack=None),
        color=alt.Color("Type:N",scale=alt.Scale(domain=["Normal","Anomaly"],range=["#4da3ff","#f87171"])),
        tooltip=["Type","count()"]).properties(height=280), use_container_width=True)

    sh("Anomaly World Map")
    sn = normal.dropna(subset=["Latitude","Longitude"]).sample(min(400,len(normal)),random_state=42).copy()
    sn[["r","g","b"]] = [77,163,255]
    if len(anomalies)>0:
        sa = anomalies.dropna(subset=["Latitude","Longitude"]).copy()
        sa[["r","g","b"]] = [248,113,113]
        map_df = pd.concat([sn,sa])
    else:
        map_df = sn
    st.pydeck_chart(pdk_map(map_df, radius=40000,
        tooltip_html="<b>{Station ID}</b><br/>{Address}<br/>{Charger Type}<br/>Usage: {Usage Stats (avg users/day)}/day"), use_container_width=True)
    st.caption("🔵 Normal &nbsp;&nbsp; 🔴 Anomaly")

    sh("Anomaly Breakdown by Feature")
    bd = anomalies["anomaly_reason"].value_counts().reset_index(); bd.columns=["Feature","Count"]
    st.altair_chart(alt.Chart(bd).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X("Feature:N",axis=alt.Axis(labelAngle=0)),y="Count:Q",
        color=alt.Color("Feature:N",scale=alt.Scale(range=["#f87171","#fb923c","#a78bfa","#34d399"]),legend=None),
        tooltip=["Feature","Count"]).properties(height=240), use_container_width=True)

    sh("Anomalous Stations — Full Table")
    disp_cols = ["Station ID","Address","Charger Type","Station Operator",
                 "Usage Stats (avg users/day)","Cost (USD/kWh)",
                 "Reviews (Rating)","Charging Capacity (kW)","anomaly_reason"]
    st.dataframe(anomalies[disp_cols].rename(columns={"anomaly_reason":"Flagged For"}).reset_index(drop=True),
                 use_container_width=True)
    st.caption(f"{len(anomalies):,} anomalous stations · {method} threshold: {threshold}")
