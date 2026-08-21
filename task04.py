import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("Set2")

# ==========================================
# 1. DATA INGESTION & SYNTHETIC GENERATION
# ==========================================
print("--- Step 1: Loading Data ---")
# Generates realistic synthetic e-commerce data (resembling Online Retail dataset)
np.random.seed(42)
n_records = 5000

start_date = pd.to_datetime("2025-01-01")
dates = [start_date + pd.Timedelta(days=int(x)) for x in np.random.randint(0, 365, n_records)]
customer_ids = np.random.choice(range(10001, 10500), size=n_records)
quantities = np.random.randint(-2, 20, size=n_records)  # Includes negative for returns
unit_prices = np.round(np.random.uniform(2.0, 150.0, size=n_records), 2)

df = pd.DataFrame({
    'InvoiceNo': [f"INV{10000+i}" if q > 0 else f"C{10000+i}" for i, q in enumerate(quantities)],
    'CustomerID': customer_ids,
    'InvoiceDate': dates,
    'Quantity': quantities,
    'UnitPrice': unit_prices
})

print(f"Raw Data Shape: {df.shape}")
print(df.head())

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
print("\n--- Step 2: Cleaning Data ---")
# Drop missing customer IDs
df = df.dropna(subset=['CustomerID'])

# Remove cancellations/returns (negative quantities) and zero prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Calculate Total Sales per transaction line
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

print(f"Cleaned Data Shape: {df.shape}")

# ==========================================
# 3. FEATURE ENGINEERING (RFM METRICS)
# ==========================================
print("\n--- Step 3: Feature Engineering (RFM Metrics) ---")
# Set snapshot date to one day after max date in dataset
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by CustomerID to extract Recency, Frequency, Monetary
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalSales': ['sum', 'mean']                            # Monetary Value
}).reset_index()

# Flatten MultiIndex columns
rfm.columns = ['CustomerID', 'Recency_Days', 'Frequency', 'Monetary_Total', 'Monetary_Avg']

# Feature Engineering: Average Days Between Purchases
rfm['Avg_Days_Between_Purchases'] = rfm['Recency_Days'] / np.maximum(rfm['Frequency'], 1)

print(rfm.head())

# ==========================================
# 4. PREDICTIVE MODELING (TARGET: FUTURE CLV)
# ==========================================
print("\n--- Step 4: Building Predictor Model ---")

# Define features (X) and target (y)
# In real workflow: X = historical RFM, y = future 3-month total sales
X = rfm[['Recency_Days', 'Frequency', 'Monetary_Avg', 'Avg_Days_Between_Purchases']]
y = rfm['Monetary_Total']  # Predicting overall Customer Lifetime Value

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f" - RMSE : ${rmse:.2f}")
print(f" - MAE  : ${mae:.2f}")
print(f" - R²   : {r2:.4f}")

# ==========================================
# 5. VISUALIZATIONS & INSIGHTS
# ==========================================
print("\n--- Step 5: Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Recency vs Monetary Scatter
sns.scatterplot(data=rfm, x='Recency_Days', y='Monetary_Total', hue='Frequency', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title("Customer Recency vs. Total Spend (Color = Frequency)")
axes[0, 0].set_xlabel("Days Since Last Purchase")
axes[0, 0].set_ylabel("Total Monetary Value ($)")

# 2. RFM Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', ax=axes[0, 1], color='teal')
axes[0, 1].set_title("Random Forest Feature Importance for CLV")
axes[0, 1].set_xlabel("Relative Importance")

# 3. Predicted vs Actual CLV
sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 0], alpha=0.7, color='indigo')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title("Actual vs. Predicted CLV")
axes[1, 0].set_xlabel("Actual Monetary Total ($)")
axes[1, 0].set_ylabel("Predicted CLV ($)")

# 4. Customer Segmentation (Quantile-based)
rfm['Customer_Segment'] = pd.qcut(rfm['Monetary_Total'], q=3, labels=['Low Value', 'Mid Value', 'High Value'])
sns.boxplot(data=rfm, x='Customer_Segment', y='Recency_Days', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title("Recency across Customer Value Segments")
axes[1, 1].set_xlabel("Customer Segment")
axes[1, 1].set_ylabel("Days Since Last Purchase")

plt.tight_layout()
plt.show()

print("\nProject pipeline execution complete!")