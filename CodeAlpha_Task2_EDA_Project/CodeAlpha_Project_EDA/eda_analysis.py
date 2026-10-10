"""
CodeAlpha Task 2 — Exploratory Data Analysis (EDA)
Run: python eda_analysis.py
Dataset: data/retail_sales_dataset.csv (synthetic learning dataset)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "retail_sales_dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

print("\n=== 1. Dataset Overview ===")
print("Rows, columns:", df.shape)
print("\nFirst five rows:\n", df.head().to_string(index=False))
print("\nColumn types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df.describe(include="all").to_string())

print("\n=== 2. Data Quality ===")
missing = df.isna().sum().rename("Missing_Count").to_frame()
print("\nMissing values:\n", missing)
print("\nDuplicate Order_ID values:", df["Order_ID"].duplicated().sum())
print("Invalid dates:", df["Order_Date"].isna().sum())
print("Non-positive quantities:", (df["Quantity"] <= 0).sum())

print("\n=== 3. Explore Important Questions ===")
print("Total orders:", len(df))
print(f"Total sales (INR): {df['Sales_INR'].sum():,.2f}")
print(f"Total profit (INR): {df['Profit_INR'].sum():,.2f}")
print(f"Average sales per order (INR): {df['Sales_INR'].mean():,.2f}")

category = df.groupby("Category", as_index=False).agg(
    Orders=("Order_ID", "count"),
    Total_Sales_INR=("Sales_INR", "sum"),
    Total_Profit_INR=("Profit_INR", "sum"),
    Average_Sales_INR=("Sales_INR", "mean")
).sort_values("Total_Sales_INR", ascending=False)

region = df.groupby("Region", as_index=False).agg(
    Orders=("Order_ID", "count"),
    Total_Sales_INR=("Sales_INR", "sum"),
    Total_Profit_INR=("Profit_INR", "sum")
).sort_values("Total_Sales_INR", ascending=False)

monthly = df.assign(Month=df["Order_Date"].dt.to_period("M").astype(str)).groupby(
    "Month", as_index=False
).agg(Orders=("Order_ID", "count"), Total_Sales_INR=("Sales_INR", "sum"),
      Total_Profit_INR=("Profit_INR", "sum"))

category.to_csv(OUT / "eda_category_summary.csv", index=False)
region.to_csv(OUT / "eda_region_summary.csv", index=False)
monthly.to_csv(OUT / "eda_monthly_summary.csv", index=False)
missing.to_csv(OUT / "eda_missing_values.csv")

print("\nSales by category:\n", category.to_string(index=False))
print("\nSales by region:\n", region.to_string(index=False))
print("\nMonthly summary (first 12 rows):\n", monthly.head(12).to_string(index=False))
print("\nEDA tables saved to:", OUT)
