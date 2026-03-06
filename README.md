# ⚡ SmartCharging Analytics

Interactive dashboard analyzing 5,000 global EV charging stations to uncover usage patterns, cluster behaviors, discover associations, and flag anomalies.


## 📊 Key Findings

### Overview
- **5,000 stations** analyzed across North America, Europe, Asia, South America, and Oceania
- Stations average **55.5 users per day** with a cost of **$0.30/kWh**
- **51.3%** of stations are powered by renewable energy
- Average user rating across all stations is **3.99 / 5**
- Network has grown steadily from **2010 to 2023**, with consistent year-on-year additions


### Charger Type Distribution
- **AC Level 2** is the most common type (1,764 stations — 35%)
- **AC Level 1** follows closely (1,638 stations — 33%)
- **DC Fast Chargers** make up 32% of the network (1,598 stations)
- DC Fast Chargers score the highest average rating (**4.02**) despite similar usage to other types
- AC Level 2 stations are the most likely to be renewable-powered (**53.7%**)


### Station Operators
- Five major operators share the network nearly equally: **Tesla, ChargePoint, EVgo, Greenlots, Ionity**
- **EVgo** leads in average daily usage (**56.7 users/day**)
- **Tesla** has the lowest average usage (**54.1 users/day**) despite the largest station count
- Renewable energy adoption is consistent across all operators, ranging from **48.8% (Ionity)** to **52.8% (EVgo)**


### Clustering Insights
The K-Means algorithm (Elbow Method suggests K=3) reveals distinct station profiles:

- 🟢 **High Demand · Low Cost** — busy stations offering competitive pricing; ideal candidates for expansion
- 🟡 **High Demand · Premium** — high-traffic stations with above-average cost, often DC Fast Chargers
- 🔵 **Low Traffic · High Rated** — underutilized but well-maintained stations with strong reviews
- 🔴 **Underperforming** — low usage and low ratings; candidates for review or relocation


### Association Rule Findings
Frequent patterns discovered in the data:

- Stations with **high charging capacity** are strongly associated with **DC Fast Charger** type and **high daily usage**
- **Renewable-powered** stations consistently appear alongside **higher user ratings**
- **Monthly maintenance** is associated with **higher usage stats** compared to annually serviced stations
- **Cheap pricing** stations tend to cluster with **medium-to-high usage** levels, confirming price sensitivity in EV adoption


### Anomaly Detection
Using IQR and Z-Score methods, the app flags stations that deviate significantly from the network norm:

- Stations with **extremely high or low usage** relative to their charger type and location
- Stations with **unusually high cost** given their capacity and operator
- **Poorly rated stations** despite high maintenance frequency — indicating possible service quality issues
- **High-capacity stations** with very low usage — potential infrastructure overinvestment or poor placement


### Infrastructure Observations
- Stations range from **0.5 km to 20 km** from city centers
- Charging capacity spans **22 kW to 350 kW** across the network
- Average of **5.5 parking spots** per station
- Maintenance is split evenly between **Monthly, Quarterly, and Annual** schedules

- LINK: https://idai105-2505788--nikhilraimalani.streamlit.app/
