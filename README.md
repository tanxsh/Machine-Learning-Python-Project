# Understanding Airbnb Market Demand and Listing Segments in California

**Demand Driver Prediction and Listing Clustering Across 8 California Markets**

Course: ISBA 2403 — Machine Learning with Python | Santa Clara University | Winter 2026  

---

## Overview

This project analyzes Airbnb listing data across 8 California markets to answer two core questions:

1. **What drives demand for Airbnb listings, and does it vary by market?**
2. **What distinct listing segments exist, and what characterizes each?**

Using a combination of regression modeling and unsupervised clustering, the analysis delivers market-specific insights and actionable recommendations for Airbnb hosts and platform strategists.

---

## Dataset

- **Source:** Inside Airbnb (publicly available Airbnb listing data)
- **Markets:** Los Angeles, Oakland, Pacific Grove, San Diego, San Francisco, San Mateo County, Santa Clara County, Santa Cruz County
- **Features:** ~31 features covering host attributes, property details, pricing, availability, and review scores
- **Target Variable (Demand):** `reviews_per_month` — the industry proxy for demand since Airbnb does not publicly disclose booking counts

---

## Project Structure

```
├── ML_Project_Final_Group_15.ipynb    # Full analysis: Data cleaning, EDA, feature engineering, modeling, clustering
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

### Part 1 — Demand Driver Analysis
**Models compared:** Linear Regression, Random Forest, XGBoost  
**Evaluation:** R², RMSE, MAE with cross-validation and GridSearchCV hyperparameter tuning

- Trained a market-specific XGBoost model for each of the 8 California markets
- Generated a **feature importance heatmap** across all markets to identify which drivers are universal vs. market-specific
- Deep-dived into Santa Clara County's top 5 demand drivers
- Compared Los Angeles vs. Pacific Grove to illustrate how demand drivers differ between high-density urban and boutique coastal markets

### Part 2 — Clustering Analysis
**Algorithms:** K-Means, DBSCAN  
**Optimal k selection:** Elbow Method + Silhouette Score

- Scaled features using StandardScaler
- Identified optimal number of clusters via KneeLocator
- Visualized clusters using PCA (2D projection)
- Profiled each cluster by price distribution and listing characteristics

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python (pandas, numpy) | Data cleaning, feature engineering |
| scikit-learn | Linear Regression, Random Forest, K-Means, DBSCAN, PCA, GridSearchCV |
| XGBoost | Primary demand driver model |
| matplotlib, seaborn | EDA and results visualization |
| kneed | Automated elbow point detection for K-Means |

---

## Key Findings

**Demand Drivers**
- `number_of_reviews` and `availability_365` are the strongest universal demand signals across all 8 markets
- Market-specific drivers diverge significantly — Santa Clara County demand is driven heavily by host responsiveness and acceptance rate, while Pacific Grove listings are more sensitive to price and property type
- XGBoost outperformed Linear Regression and Random Forest across most markets after tuning

**Clustering**
- K-Means identified distinct listing segments (budget, mid-tier, premium) with clear differences in pricing, availability, and review activity
- DBSCAN successfully flagged anomalous high-price outlier listings that K-Means absorbed into clusters

---

*ISBA 2403: Machine Learning with Python — Santa Clara University, Winter 2026*
