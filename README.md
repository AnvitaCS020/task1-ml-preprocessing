#  Task 1 - Data Cleaning & Preprocessing


---

## 📌 Objective
Clean and prepare the **Ames House Prices Dataset** for Machine Learning by applying standard preprocessing techniques including handling missing values, encoding, scaling, and outlier removal.

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots |
| Scikit-learn | Scaling & Encoding |

---

## 📂 Dataset

**Ames Housing Dataset (train.csv)**
A professional, real-world dataset describing residential homes in Ames, Iowa, USA.

| Property | Value |
|----------|-------|
| Rows | 1460 |
| Columns | 81 |
| Target Column | SalePrice |
| Source | Kaggle - House Prices Competition |
| Link | https://www.kaggle.com/datasets/lespin/house-prices-dataset |

---

## ⚙️ Preprocessing Steps Performed

### ✅ Step 1 — Load & Explore Dataset
- Loaded `train.csv` using Pandas
- Checked shape, column names, data types
- Identified numerical vs categorical columns
- Analyzed missing values per column

### ✅ Step 2 — Handle Missing Values
- Dropped columns with **more than 40% missing data**
  - Examples: `Alley`, `PoolQC`, `MiscFeature`, `Fence`
- Filled **numerical** missing values with **median**
- Filled **categorical** missing values with **mode**
- Result: **0 missing values** after handling

### ✅ Step 3 — Encode Categorical Features
- **Label Encoding** → columns with more than 10 unique values
- **One-Hot Encoding** → columns with 10 or fewer unique values
- Used `drop_first=True` to avoid multicollinearity

### ✅ Step 4 — Standardize Numerical Features
- Applied **StandardScaler** on key numerical columns:
  - `LotArea`, `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`
- Result: mean ≈ 0, standard deviation ≈ 1

### ✅ Step 5 — Detect & Remove Outliers
- Visualized outliers using **Boxplots**
- Removed outliers using **IQR (Interquartile Range)** method
  - Lower bound = Q1 - 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR

### ✅ Step 6 — Correlation Heatmap
- Generated heatmap of top 12 numerical features
- Helps identify strongly correlated features before model building

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `house_prices_cleaned.csv` | Final cleaned & preprocessed dataset |
| `missing_values.png` | Bar chart showing % of missing values per column |
| `boxplots.png` | Boxplots for outlier detection |
| `correlation_heatmap.png` | Heatmap of feature correlations |

---

## 📁 Repository Structure

```
task1-ml-preprocessing/
│
├── preprocessing.py            # Main Python script
├── train.csv                   # Original raw dataset
├── house_prices_cleaned.csv    # Final cleaned dataset
├── missing_values.png          # Missing value visualization
├── boxplots.png                # Outlier visualization
├── correlation_heatmap.png     # Correlation heatmap
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Place `train.csv` in the same folder as the script

### 3. Run the script
```bash
python preprocessing.py
```

---

## 📈 Results Summary

| Metric | Before | After |
|--------|--------|-------|
| Total Rows | 1460 | ~1380 |
| Total Columns | 81 | ~120 (after encoding) |
| Missing Values | 6965 | 0 |
| Outliers Removed | — | ~80 rows |

---


