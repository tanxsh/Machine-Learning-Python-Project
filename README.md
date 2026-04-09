# Understanding Airbnb Market Demand and Listing Segments in California

**Demand Driver Prediction, Listing Clustering, and Cluster-Enhanced Forecasting Across 8 California Markets**

Course: ISBA 2403 — Machine Learning with Python | Santa Clara University

---

## Overview

This project analyzes Airbnb listing data across 8 California markets to answer three core questions:

1. **What drives demand for Airbnb listings, and does it vary by market?**
2. **What distinct listing segments exist, and what characterizes each?**
3. **Does clustering-based market segmentation improve demand prediction accuracy?**

Using regression modeling, unsupervised clustering, and a cluster-enhanced prediction framework, the analysis delivers market-specific insights and actionable recommendations for Airbnb hosts and platform strategists.

---

## Dataset

- **Source:** Inside Airbnb (publicly available Airbnb listing data)
- **Markets:** Los Angeles, Oakland, Pacific Grove, San Diego, San Francisco, San Mateo County, Santa Clara County, Santa Cruz County
- **Features:** ~31 features covering host attributes, property details, pricing, availability, and review scores
- **Target Variable (Demand):** `reviews_per_month` — the industry proxy for demand since Airbnb does not publicly disclose booking counts

---

## Project Structure

```
├── ML Project.ipynb    # Full analysis: EDA, demand modeling, clustering, predictive question
└── README.md
```

---

## Methodology

### Data Preprocessing
- Merged 8 separate market CSV files into a single combined dataset
- Cleaned price fields (removed `$` and `,`), standardized boolean columns, imputed missing review scores with 0
- Removed top 1% price outliers
- One-hot encoded categorical variables (`room_type`, `property_type`, `host_response_time`)

### Exploratory Data Analysis
- Listing count distribution across 8 markets
- Room type breakdown and price distribution per night
- Price distribution by room type (boxplots)

---

### Part 1 — Demand Driver Analysis
**Research Question:** How do the key drivers of Airbnb listing demand vary across different locations in California?

**Models compared:** Linear Regression, Random Forest, XGBoost  
**Evaluation:** R², RMSE, MAE with 5-fold cross-validation and GridSearchCV hyperparameter tuning

- Best model: **Tuned XGBoost** (`learning_rate=0.2`, `max_depth=6`, `n_estimators=200`)
- Trained separate XGBoost models per market to extract market-specific feature importances
- Generated a **feature importance heatmap** across all 8 markets
- Deep-dived into Santa Clara County's top 5 demand drivers
- Compared Los Angeles vs. Pacific Grove to illustrate how demand drivers diverge between high-density urban and boutique coastal markets

**Key Recommendations:**
- Treat host responsiveness as a core utility — first to respond often wins the booking regardless of location
- Standardize amenities and privacy features; California travelers show strong preference for professional-grade predictability

---

### Part 2 — Clustering Analysis
**Research Question:** Can Airbnb listings be segmented into distinct market groups based on property characteristics, demand, and host behavior?

**Algorithms:** K-Means, DBSCAN  
**Optimal k selection:** Elbow Method (suggested k=2) + Silhouette Score (confirmed k=3 with score of 0.4027)

- Scaled features using StandardScaler; imputed missing values with median
- Chose k=3 over k=2 — a two-cluster split provides no actionable segmentation beyond "generic" vs. "other"
- Visualized clusters using PCA (2D projection) and price distribution boxplots
- DBSCAN was evaluated but found unsuitable — unable to find meaningful cluster boundaries in this dataset
- Identified 3 distinct listing tiers: budget/entry, mid-tier standard, and premium/professional

**Key Recommendations:**
- Hosts with low ratings should audit cleanliness, value, location, and communication scores first
- Standardize cleaning procedures and implement faster response systems to move listings into higher-performing clusters

---

### Part 3 — Predictive Question
**Research Question:** Does clustering-based market segmentation improve the prediction of Airbnb listing demand across California?

**Approach:** Compared a single **Global Tuned XGBoost model** against three **cluster-specific XGBoost models** trained independently on each K-Means segment.

| Model | R² |
|---|---|
| Global Baseline (Tuned XGBoost) | 0.4830 |
| Cluster 0 — XGBoost | **0.5194** |
| Cluster 1 — XGBoost | 0.3736 |
| Cluster 2 — XGBoost | 0.4783 |

**Conclusion:** Yes — clustering improves prediction, but with important nuances:

- **Cluster 0** outperformed the global baseline, confirming that specialized models capture segment-specific demand behaviors that a global model misses
- **Cluster 1** (professional/niche listings) proved highly unpredictable (R²=0.3736), likely relying on external marketing or corporate contracts not captured in Airbnb's public data — the global model masks this chaos
- **Cluster 2** performed comparably to the global baseline

Airbnb is not a single unified market. Hosts seeking accurate demand predictions should build models optimized for their specific listing tier rather than relying on a one-size-fits-all approach.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python (pandas, numpy) | Data cleaning, feature engineering |
| scikit-learn | Linear Regression, Random Forest, K-Means, DBSCAN, PCA, GridSearchCV |
| XGBoost | Primary demand driver and cluster-specific prediction model |
| matplotlib, seaborn | EDA and results visualization |
| kneed | Automated elbow point detection for K-Means |
