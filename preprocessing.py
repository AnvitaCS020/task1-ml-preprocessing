# ============================================================
#  Task 1 - Data Cleaning & Preprocessing
#  Dataset: House Prices (train.csv)

# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 60)
print("  STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv('train.csv')

print(f"Dataset Loaded Successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 2: EXPLORE DATASET
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: EXPLORING DATASET")
print("=" * 60)

print(f"\nData Types:\n{df.dtypes.value_counts()}")
print(f"\nBasic Statistics:")
print(df.describe())

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nColumns with Missing Values ({len(missing)} total):")
print(missing.head(15))

# Plot missing values
plt.figure(figsize=(12, 5))
missing_pct = (missing / len(df)) * 100
missing_pct.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 15 Columns with Missing Values (%)', fontsize=14, fontweight='bold')
plt.xlabel('Columns')
plt.ylabel('Missing %')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.show()
print("Saved: missing_values.png")

# ============================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: HANDLING MISSING VALUES")
print("=" * 60)

cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > 0.4]
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns with >40% missing: {cols_to_drop}")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\nMissing values after handling: {df.isnull().sum().sum()}")

# ============================================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: ENCODING CATEGORICAL FEATURES")
print("=" * 60)

le = LabelEncoder()
label_encoded_cols = []
onehot_cols = []

for col in cat_cols:
    if col in df.columns:
        if df[col].nunique() > 10:
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoded_cols.append(col)
        else:
            onehot_cols.append(col)

print(f"Label Encoded: {len(label_encoded_cols)} columns")
print(f"One-Hot Encoding: {len(onehot_cols)} columns")

if onehot_cols:
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

print(f"\nShape after encoding: {df.shape}")

# ============================================================
# STEP 5: STANDARDIZE NUMERICAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: STANDARDIZING NUMERICAL FEATURES")
print("=" * 60)

scale_cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']
scale_cols = [col for col in scale_cols if col in df.columns]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print(f"Standardized: {scale_cols}")
print(f"\nAfter Standardization (mean ~ 0, std ~ 1):")
print(df[scale_cols].describe().loc[['mean', 'std']].round(3))

# ============================================================
# STEP 6: VISUALIZE & REMOVE OUTLIERS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: OUTLIER DETECTION & REMOVAL")
print("=" * 60)

outlier_cols = [col for col in ['LotArea', 'GrLivArea', 'GarageArea'] if col in df.columns]

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(15, 5))
fig.suptitle('Outlier Detection - Boxplots', fontsize=15, fontweight='bold')

for i, col in enumerate(outlier_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(col)

plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()
print("Saved: boxplots.png")

rows_before = len(df)
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

rows_after = len(df)
print(f"\nRows before: {rows_before} | After: {rows_after} | Removed: {rows_before - rows_after}")

# ============================================================
# STEP 7: CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: CORRELATION HEATMAP")
print("=" * 60)

top_cols = df.select_dtypes(include=[np.number]).columns[:12].tolist()

plt.figure(figsize=(12, 8))
corr = df[top_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Saved: correlation_heatmap.png")

# ============================================================
# STEP 8: SAVE CLEANED DATASET
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: SAVING CLEANED DATASET")
print("=" * 60)

df.to_csv('house_prices_cleaned.csv', index=False)
print(f"Saved: house_prices_cleaned.csv")
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

print("\n" + "=" * 60)
print("  ALL PREPROCESSING STEPS COMPLETED!")
print("=" * 60)
print("""
  1. Loaded House Prices Dataset (train.csv)
  2. Explored data (shape, dtypes, missing values)
  3. Handled missing values (median / mode / drop)
  4. Encoded categorical features (Label + One-Hot)
  5. Standardized numerical features (StandardScaler)
  6. Detected & removed outliers (IQR method)
  7. Visualized correlations (Heatmap)
  8. Saved cleaned dataset

  Output Files:
  - house_prices_cleaned.csv
  - missing_values.png
  - boxplots.png
  - correlation_heatmap.png
""")