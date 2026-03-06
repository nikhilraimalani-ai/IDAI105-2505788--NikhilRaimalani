import streamlit as st
import pandas as pd
import numpy as np
import math
import random
from collections import defaultdict
from itertools import combinations

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
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #07090f; }
.block-container { padding: 2rem 2.5rem 4rem; }
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
[data-testid="stSidebar"] label { color: #6b7a99 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.page-hero { margin-bottom: 2rem; }
.page-hero h1 { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#e8edf5; letter-spacing:-0.02em; line-height:1.1; margin:0 0 0.4rem; }
.page-hero p { font-size:0.95rem; color:#5a6a8a; margin:0; }
.divider { height:1px; background:#1a2236; margin:1.5rem 0; }
.sh { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#c5cfe0; margin:1.8rem 0 0.6rem; border-bottom:1px solid #1a2236; padding-bottom:6px; }
.kcard { background:#0d1117; border:1px solid #1a2236; border-radius:12px; padding:18px 20px; position:relative; overflow:hidden; }
.kcard::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#1b4fff,#00d4ff); }
.kcard .lbl { font-size:0.68rem; color:#4a5a7a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px; }
.kcard .val { font-family:'Syne',sans-serif; font-size:1.85rem; font-weight:700; color:#e8edf5; line-height:1; }
.kcard .sub { font-size:0.73rem; color:#3a5a8a; margin-top:4px; }
.icard { background:#0a1628; border:1px solid #1a2e4a; border-left:3px solid #4da3ff; border-radius:8px; padding:14px 16px; margin:8px 0; font-size:0.87rem; color:#8ba8cc; line-height:1.6; }
.icard strong { color:#c5d8f0; }
.atag { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; margin:2px; }
[data-testid="metric-container"] { background:#0d1117 !important; border:1px solid #1a2236 !important; border-radius:12px !important; padding:16px 20px !important; }
[data-testid="metric-container"] label { color:#4a5a7a !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e8edf5 !important; font-family:'Syne',sans-serif; font-size:1.8rem !important; }
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; color:#e8edf5 !important; }
p, li { color:#8a9ab8; }
.stDataFrame { border-radius:10px; overflow:hidden; }
.stCaption { color:#3a4a6a !important; font-size:0.75rem !important; }
div[data-testid="stHorizontalBlock"] { gap:14px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def sh(t): st.markdown(f"<div class='sh'>{t}</div>", unsafe_allow_html=True)
def kcard(label, value, sub=""):
    st.markdown(f"<div class='kcard'><div class='lbl'>{label}</div><div class='val'>{value}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  K-MEANS (pure numpy — no sklearn)
# ─────────────────────────────────────────────
def kmeans_scratch(X, k, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any(): centers[j] = X[mask].mean(axis=0)
    inertia = sum(np.sum((X[labels==j] - centers[j])**2) for j in range(k))
    return labels, inertia

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

# ─────────────────────────────────────────────
#  ASSOCIATION RULES (pure python)
# ─────────────────────────────────────────────
def apriori_scratch(transactions, min_support, min_confidence):
    n = len(transactions)
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] += 1
    freq = {k: v/n for k, v in item_counts.items() if v/n >= min_support}
    all_freq = dict(freq)
    prev = list(freq.keys())
    size = 2
    while prev:
        candidates = []
        items = sorted(set(i for fs in prev for i in fs))
        for combo in combinations(items, size):
            fs = frozenset(combo)
            if all(frozenset(combo[:size-1]), frozenset(combo[1:])):
                candidates.append(fs)
        counts = defaultdict(int)
        for t in transactions:
            ts = set(t)
            for c in candidates:
                if c <= ts: counts[c] += 1
        new_freq = {k: v/n for k,v in counts.items() if v/n >= min_support}
        all_freq.update(new_freq)
        prev = list(new_freq.keys())
        size += 1
        if size > 4: break
    rules = []
    for fs in all_freq:
        if len(fs) < 2: continue
        items = list(fs)
        for r in range(1, len(items)):
            for ant_items in combinations(items, r):
                ant = frozenset(ant_items)
                cons = fs - ant
                if not cons: continue
                sup = all_freq[fs]
                ant_sup = all_freq.get(ant, 0)
                cons_sup = all_freq.get(cons, 0)
                if ant_sup == 0 or cons_sup == 0: continue
                conf = sup / ant_sup
                lift = conf / cons_sup if cons_sup > 0 else 0
                if conf >= min_confidence:
                    rules.append({
                        "antecedents": ", ".join(sorted(ant)),
                        "consequents": ", ".join(sorted(cons)),
                        "support": round(sup, 3),
                        "confidence": round(conf, 3),
                        "lift": round(lift, 3)
                    })
    return sorted(rules, key=lambda x: -x["lift"])

# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("detailed_ev_charging_stations.csv")

@st.cache_data
def get_clusters(n_clusters):
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    feats = ["Usage Stats (avg users/day)", "Charging Capacity (kW)",
             "Cost (USD/kWh)", "Reviews (Rating)", "Distance to City (km)", "Parking Spots"]
    sub = df[feats].dropna()
    X = sub.values.copy().astype(float)
    for i in range(X.shape[1]):
        mn, mx = X[:,i].min(), X[:,i].max()
        X[:,i] = (X[:,i]-mn)/(mx-mn+1e-9)
    inertias = []
    for k in range(2, 10):
        _, ine = kmeans_scratch(X, k)
        inertias.append((k, ine))
    labels, _ = kmeans_scratch(X, n_clusters)
    result = df.loc[sub.index].copy()
    result["Cluster"] = labels.astype(str)
    return result, inertias, feats

@st.cache_data
def get_rules(min_support, min_confidence):
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    df["Usage Level"] = pd.cut(df["Usage Stats (avg users/day)"], bins=[0,30,60,100,999],
        labels=["LowUsage","MedUsage","HighUsage","VeryHighUsage"])
    df["Cost Level"] = pd.cut(df["Cost (USD/kWh)"], bins=[0,0.2,0.35,0.5,999],
        labels=["Cheap","ModerateCost","Expensive","VeryExpensive"])
    df["Rating Level"] = pd.cut(df["Reviews (Rating)"], bins=[0,3,4,5],
        labels=["LowRated","MidRated","HighRated"])
    df["Capacity Level"] = pd.cut(df["Charging Capacity (kW)"], bins=[0,50,150,350,9999],
        labels=["LowCap","MedCap","HighCap","UltraCap"])
    tcols = ["Charger Type","Renewable Energy Source","Maintenance Frequency",
             "Usage Level","Cost Level","Rating Level","Capacity Level"]
    transactions = df[tcols].astype(str).values.tolist()
    sample = random.sample(transactions, min(1000, len(transactions)))
    return apriori_scratch(sample, min_support, min_confidence)

@st.cache_data
def get_anomalies(method, threshold):
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    cols = ["Usage Stats (avg users/day)","Cost (USD/kWh)","Reviews (Rating)","Charging Capacity (kW)"]
    df["anomaly"] = False
    df["anomaly_reason"] = ""
    for col in cols:
        if method == "Z-Score":
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            flag = z.abs() > threshold
        else:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            flag = (df[col] < Q1 - threshold*IQR) | (df[col] > Q3 + threshold*IQR)
        df.loc[flag & ~df["anomaly"], "anomaly_reason"] = col
        df.loc[flag, "anomaly"] = True
    return df

df_raw = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 4px 20px;border-bottom:1px solid #1a2236;margin-bottom:14px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#4da3ff;'>⚡ SmartCharging</div>
        <div style='font-size:0.68rem;color:#3a4a6a;margin-top:3px;text-transform:uppercase;letter-spacing:0.1em;'>EV Analytics Platform</div>
    </div>""", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for pname, icon in [("Home","🏠"),("EDA","📊"),("Map","🗺️"),
                         ("Clustering","🔵"),("Association","🔗"),("Anomalies","🚨")]:
        if st.button(f"{icon}  {pname}", key=f"nav_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    st.markdown("<div style='height:1px;background:#1a2236;margin:14px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#2a3a5a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)

    sel_charger  = st.selectbox("Charger Type", ["All"]+sorted(df_raw["Charger Type"].dropna().unique().tolist()))
    sel_operator = st.selectbox("Operator",     ["All"]+sorted(df_raw["Station Operator"].dropna().unique().tolist()))
    sel_renewable= st.selectbox("Renewable",    ["All","Yes","No"])

    df = df_raw.copy()
    if sel_charger   != "All": df = df[df["Charger Type"]==sel_charger]
    if sel_operator  != "All": df = df[df["Station Operator"]==sel_operator]
    if sel_renewable != "All": df = df[df["Renewable Energy Source"]==sel_renewable]

    st.markdown(f"<div style='font-size:0.72rem;color:#2a3a5a;margin-top:8px;'>{len(df):,} stations loaded</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1a2236;margin:14px 0 6px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.66rem;color:#1a2a4a;text-align:center;'>SmartCharging Analytics · 2025</div>", unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════
if page == "Home":
    st.markdown("<div class='page-hero'><h1>SmartCharging Analytics</h1><p>Global EV charging intelligence — clusters, associations, anomalies, and live maps.</p></div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kcard("Total Stations",   f"{len(df):,}")
    with c2: kcard("Avg Users / Day",  f"{df['Usage Stats (avg users/day)'].mean():.1f}")
    with c3: kcard("Avg Rating",       f"{df['Reviews (Rating)'].mean():.2f} / 5")
    with c4: kcard("Renewable",        f"{(df['Renewable Energy Source']=='Yes').mean()*100:.1f}%")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c5,c6 = st.columns(2)
    with c5:
        sh("Charger Type Distribution")
        ct = df["Charger Type"].value_counts()
        st.bar_chart(ct, color="#4da3ff")
    with c6:
        sh("Top Station Operators")
        op = df["Station Operator"].value_counts().head(6)
        st.bar_chart(op, color="#00d4c8")

    sh("Stations Installed Over Time")
    yr = df["Installation Year"].value_counts().sort_index()
    st.area_chart(yr, color="#4da3ff")

    sh("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, height=300)
    st.caption(f"Showing first 50 of {len(df):,} records")

# ═══════════════════════════════════════════
#  EDA
# ═══════════════════════════════════════════
elif page == "EDA":
    st.markdown("<div class='page-hero'><h1>Exploratory Analysis</h1><p>Distributions, correlations, and behavioral patterns.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        sh("Usage Stats — Histogram")
        bins = pd.cut(df["Usage Stats (avg users/day)"], bins=20).value_counts().sort_index()
        st.bar_chart(bins, color="#4da3ff")
    with c2:
        sh("Cost Distribution (USD/kWh)")
        bins2 = pd.cut(df["Cost (USD/kWh)"], bins=20).value_counts().sort_index()
        st.bar_chart(bins2, color="#00d4c8")

    c3,c4 = st.columns(2)
    with c3:
        sh("Ratings Distribution")
        bins3 = pd.cut(df["Reviews (Rating)"], bins=15).value_counts().sort_index()
        st.bar_chart(bins3, color="#a78bfa")
    with c4:
        sh("Avg Charging Capacity by Type")
        cap = df.groupby("Charger Type")["Charging Capacity (kW)"].mean()
        st.bar_chart(cap, color="#fb923c")

    sh("Avg Daily Usage by Operator")
    op_u = df.groupby("Station Operator")["Usage Stats (avg users/day)"].mean().sort_values(ascending=False)
    st.bar_chart(op_u, color="#34d399")

    c5,c6 = st.columns(2)
    with c5:
        sh("Renewable Energy Split")
        ren = df["Renewable Energy Source"].value_counts()
        st.bar_chart(ren, color="#34d399")
    with c6:
        sh("Maintenance Frequency")
        mf = df["Maintenance Frequency"].value_counts()
        st.bar_chart(mf, color="#f472b6")

    sh("Avg Rating by Operator")
    rat = df.groupby("Station Operator")["Reviews (Rating)"].mean().sort_values(ascending=False)
    st.bar_chart(rat, color="#facc15")

    sh("Correlation — Numeric Features")
    num_cols = ["Cost (USD/kWh)","Distance to City (km)","Usage Stats (avg users/day)",
                "Charging Capacity (kW)","Reviews (Rating)","Parking Spots"]
    corr = df[num_cols].corr().round(2)
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)

    sh("Usage vs Cost — Sample Scatter")
    samp = df[["Usage Stats (avg users/day)","Cost (USD/kWh)","Latitude","Longitude"]].dropna().sample(min(500,len(df)),random_state=42)
    samp = samp.rename(columns={"Usage Stats (avg users/day)":"Usage","Cost (USD/kWh)":"Cost"})
    st.scatter_chart(samp, x="Cost", y="Usage", color="#4da3ff")

# ═══════════════════════════════════════════
#  MAP
# ═══════════════════════════════════════════
elif page == "Map":
    st.markdown("<div class='page-hero'><h1>Station Map</h1><p>Every EV charging station on the globe. Pan and zoom to explore.</p></div>", unsafe_allow_html=True)

    map_df = df.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Charger Type","Station Operator",
        "Usage Stats (avg users/day)","Reviews (Rating)","Cost (USD/kWh)","Renewable Energy Source"]].copy()
    map_df = map_df.sample(min(2000,len(map_df)),random_state=42)
    map_df = map_df.rename(columns={"Latitude":"lat","Longitude":"lon"})

    st.map(map_df, latitude="lat", longitude="lon", size=500, color="#4da3ff")
    st.caption(f"Showing {len(map_df):,} of {len(df):,} stations")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        sh("Usage by Distance Bracket")
        df2 = df.copy()
        df2["Distance Bracket"] = pd.cut(df2["Distance to City (km)"], bins=5)
        dist_usage = df2.groupby("Distance Bracket")["Usage Stats (avg users/day)"].mean()
        st.bar_chart(dist_usage, color="#4da3ff")
    with c2:
        sh("Avg Cost by Operator")
        cost_op = df.groupby("Station Operator")["Cost (USD/kWh)"].mean().sort_values(ascending=False)
        st.bar_chart(cost_op, color="#fb923c")

    sh("Usage vs Rating — Scatter")
    s2 = df[["Usage Stats (avg users/day)","Reviews (Rating)"]].dropna().sample(min(600,len(df)),random_state=1)
    s2 = s2.rename(columns={"Usage Stats (avg users/day)":"Usage","Reviews (Rating)":"Rating"})
    st.scatter_chart(s2, x="Rating", y="Usage", color="#a78bfa")

# ═══════════════════════════════════════════
#  CLUSTERING
# ═══════════════════════════════════════════
elif page == "Clustering":
    st.markdown("<div class='page-hero'><h1>Cluster Analysis</h1><p>Group stations by usage behavior, cost, capacity, and performance.</p></div>", unsafe_allow_html=True)

    n_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)

    with st.spinner("Running K-Means..."):
        result_df, inertias, features = get_clusters(n_clusters)

    sh("Elbow Method — Inertia by K")
    elbow = pd.DataFrame(inertias, columns=["K","Inertia"]).set_index("K")
    st.line_chart(elbow, color="#4da3ff")

    sh("Cluster Scatter")
    c1,c2 = st.columns(2)
    with c1: x_axis = st.selectbox("X Axis", features, index=0)
    with c2: y_axis = st.selectbox("Y Axis", features, index=1)
    samp = result_df[[x_axis, y_axis, "Cluster"]].dropna().sample(min(600,len(result_df)),random_state=1)
    st.scatter_chart(samp, x=x_axis, y=y_axis, color="Cluster")

    sh("Cluster Profiles (Avg Values)")
    summary = result_df.groupby("Cluster")[features].mean().round(2)
    profiles = []
    for idx, row in summary.iterrows():
        u_med = summary["Usage Stats (avg users/day)"].median()
        c_med = summary["Cost (USD/kWh)"].median()
        r_med = summary["Reviews (Rating)"].median()
        if row["Usage Stats (avg users/day)"] > u_med and row["Cost (USD/kWh)"] < c_med:
            p = "🟢 High Demand · Low Cost"
        elif row["Usage Stats (avg users/day)"] > u_med:
            p = "🟡 High Demand · Premium"
        elif row["Reviews (Rating)"] > r_med:
            p = "🔵 Low Traffic · High Rated"
        else:
            p = "🔴 Underperforming"
        profiles.append(p)
    summary.insert(0, "Profile", profiles)
    summary.index = [f"Cluster {i}" for i in summary.index]
    st.dataframe(summary, use_container_width=True)

    sh("Cluster Distribution — Charger Type")
    comp = result_df.groupby(["Cluster","Charger Type"]).size().unstack(fill_value=0)
    st.bar_chart(comp)

    sh("Cluster Map")
    mdata = result_df.dropna(subset=["Latitude","Longitude"]).sample(min(1000,len(result_df)),random_state=42)
    mdata = mdata.rename(columns={"Latitude":"lat","Longitude":"lon"})
    st.map(mdata, latitude="lat", longitude="lon", size=400)
    st.caption(f"Showing {len(mdata):,} clustered stations")

# ═══════════════════════════════════════════
#  ASSOCIATION
# ═══════════════════════════════════════════
elif page == "Association":
    st.markdown("<div class='page-hero'><h1>Association Rule Mining</h1><p>Hidden connections between station features, usage levels, and performance.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: min_support    = st.slider("Min Support",    0.05, 0.5,  0.15, 0.01)
    with c2: min_confidence = st.slider("Min Confidence", 0.3,  1.0,  0.5,  0.05)

    st.markdown("""<div class='icard'>
        <strong>Support</strong> — how often the pattern appears &nbsp;·&nbsp;
        <strong>Confidence</strong> — reliability of the rule &nbsp;·&nbsp;
        <strong>Lift</strong> — how much better than random (&gt;1 = meaningful)
    </div>""", unsafe_allow_html=True)

    with st.spinner("Mining rules..."):
        rules = get_rules(min_support, min_confidence)

    if not rules:
        st.warning("No rules found. Try lowering the thresholds.")
    else:
        st.success(f"Found **{len(rules)}** association rules")
        rules_df = pd.DataFrame(rules)

        sh("Top Rules by Lift")
        st.dataframe(rules_df.head(20), use_container_width=True)

        sh("Lift of Top 10 Rules")
        top10 = rules_df.head(10).copy()
        top10["rule"] = top10["antecedents"].str[:30] + " → " + top10["consequents"].str[:20]
        top10 = top10.set_index("rule")[["lift","confidence","support"]]
        st.bar_chart(top10[["lift"]], color="#4da3ff")

        sh("💬 Plain English Insights")
        for row in rules[:6]:
            strength = "strongly" if row["lift"] > 2 else "moderately"
            st.markdown(f"""<div class='icard'>
                Stations with <strong>{row['antecedents']}</strong> are {strength} associated
                with <strong>{row['consequents']}</strong> —
                confidence <strong>{row['confidence']:.0%}</strong>,
                lift <strong>{row['lift']:.2f}×</strong>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  ANOMALIES
# ═══════════════════════════════════════════
elif page == "Anomalies":
    st.markdown("<div class='page-hero'><h1>Anomaly Detection</h1><p>Stations with unusual usage, cost, rating, or capacity.</p></div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1: method = st.selectbox("Detection Method", ["Z-Score","IQR"])
    with c2:
        if method == "Z-Score": threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
        else:                   threshold = st.slider("IQR Multiplier",     1.0, 3.0, 1.5, 0.1)

    result    = get_anomalies(method, threshold)
    anomalies = result[result["anomaly"]].copy()
    normal    = result[~result["anomaly"]].copy()
    if sel_charger  != "All": anomalies = anomalies[anomalies["Charger Type"]==sel_charger]
    if sel_operator != "All": anomalies = anomalies[anomalies["Station Operator"]==sel_operator]

    c3,c4,c5 = st.columns(3)
    with c3: kcard("Anomalous Stations", f"{len(anomalies):,}")
    with c4: kcard("Anomaly Rate",       f"{len(anomalies)/len(result)*100:.1f}%")
    with c5: kcard("Normal Stations",    f"{len(normal):,}")

    sh("Feature to Inspect")
    sel_feat = st.selectbox("Select feature", [
        "Usage Stats (avg users/day)","Cost (USD/kWh)","Reviews (Rating)","Charging Capacity (kW)"])

    c6,c7 = st.columns(2)
    with c6:
        sh("Normal — Distribution")
        nb = pd.cut(normal[sel_feat], bins=20).value_counts().sort_index()
        st.bar_chart(nb, color="#4da3ff")
    with c7:
        sh("Anomalies — Distribution")
        if len(anomalies) > 0:
            ab = pd.cut(anomalies[sel_feat], bins=20).value_counts().sort_index()
            st.bar_chart(ab, color="#f87171")
        else:
            st.info("No anomalies detected with current settings.")

    sh("Anomaly Breakdown by Feature")
    bd = anomalies["anomaly_reason"].value_counts()
    if len(bd): st.bar_chart(bd, color="#fb923c")

    sh("Anomalous Stations — Map")
    if len(anomalies) > 0:
        amap = anomalies.dropna(subset=["Latitude","Longitude"]).rename(
            columns={"Latitude":"lat","Longitude":"lon"})
        st.map(amap, latitude="lat", longitude="lon", size=600, color="#f87171")
    else:
        st.info("No anomalies to map.")

    sh("Anomalous Stations — Full Table")
    disp_cols = ["Station ID","Address","Charger Type","Station Operator",
                 "Usage Stats (avg users/day)","Cost (USD/kWh)",
                 "Reviews (Rating)","Charging Capacity (kW)","anomaly_reason"]
    st.dataframe(
        anomalies[disp_cols].rename(columns={"anomaly_reason":"Flagged For"}).reset_index(drop=True),
        use_container_width=True)
    st.caption(f"{len(anomalies):,} anomalous stations · {method} @ {threshold}")
