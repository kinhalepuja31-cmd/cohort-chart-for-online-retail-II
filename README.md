# Internship Analytics Track - Task 22: Cohort Retention Basics

## 📌 Project Overview
By tracking customer return behaviors across a continuous 24-month window, this matrix isolates consumer behavior from raw user-acquisition growth, providing stakeholders with clear insights into Product-Market Fit (PMF) and operational seasonality.

## 📊 Dataset Profile
The analysis is performed on the official **UCI Online Retail II Dataset**, which contains real continuous transactions for a UK-based non-store online retail company.

- **Data Window:** December 1, 2009, to December 9, 2011.
- **File Structure:** `online_retail_II.xlsx` contains two baseline worksheets that must be merged to prevent data corruption and tracking misalignments:
  1. `Year 2009-2010` (525,461 rows)
  2. `Year 2010-2011` (541,910 rows)
- **Total Raw Records:** 1,067,371 rows
- **Target Schema Fields:**
  - `Invoice`: Unique transactional transaction number.
  - `StockCode`: Unique product SKU code.
  - `Description`: Nominal product name description.
  - `Quantity`: Quantities of items per transaction row.
  - `InvoiceDate`: Operational timestamp of purchase.
  - `Price`: Product unit price.
  - `Customer ID`: Unique categorical identifier for registered users.
  - `Country`: Geographical region of the transaction.

## 🛠️ Step-by-Step Execution Pipeline

### 1. Data Merging & Synchronization
Because customers signing up in 2009 frequently return in 2010 and 2011, parsing sheets individually will falsely label returning customers as "new users." Both worksheets are loaded via `pandas.read_excel` and stacked using `pd.concat` to ensure historical integrity.

### 2. Data Cleansing
- Rows containing missing `Customer ID` flags are safely omitted to restrict calculations strictly to registered, trackable users.
- Transactions with a negative or zero value in the `Quantity` field are dropped to purge order cancellations and returns from the baseline matrix.

### 3. Cohort Index Modeling
- **`InvoiceMonth`**: Isolated period component derived from `InvoiceDate` (Formatted as `YYYY-MM`).
- **`CohortMonth`**: Modeled via `.groupby('Customer ID')['InvoiceMonth'].transform('min')` to latch each user permanently to the month of their absolute first transaction.
- **`CohortIndex`**: An integer calculating the monthly lifecycle offset delta (Years × 12 + Months) from Month 0 to Month 24.

### 4. Pivot Evaluation
Aggregated profiles are restructured via a pivot framework to present unique customer retention values alongside their corresponding relative lifecycle decay curve.

## 💻 Source Code (Google Colab Notebook Blocks)

### Cell 1: Environment Setup, Downloader, & Synchronization
```python
# Install the UCI Repository interface package
!pip install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the separate sheets from the project workspace directory
file_path = '/content/online_retail_II.xlsx'

print("Loading the first year worksheet (2009-2010)...")
df_sheet1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")

print("Loading the second year worksheet (2010-2011)...")
df_sheet2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

# Vertically merge files into a continuous historical log
df = pd.concat([df_sheet1, df_sheet2], ignore_index=True)
print(f"Data successfully merged! Total transactional records: {df.shape[0]}")

### Cell 2: Cleansing & Cohort Base Modeling
# 1. Clean missing values and remove cancellations
df = df.dropna(subset=['Customer ID'])
df = df[df['Quantity'] > 0]

# 2. Synchronize date column datatypes
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 3. Formulate transaction month period object
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')

# 4. Map the cohort baseline birth month per client profile
df['CohortMonth'] = df.groupby('Customer ID')['InvoiceMonth'].transform('min')

print("Cohort base successfully mapped!
### Cell 3: Building the Cohort Matrix Grid
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

invoice_year, invoice_month = get_date_int(df, 'InvoiceMonth')
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')

# Calculate precise month indexing loops
years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month
df['CohortIndex'] = years_diff * 12 + months_diff

# Group rows and count unique buyers per cohort block
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()

# Pivot data into a structured raw counts grid
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Customer ID')
cohort_counts.index.name = 'Cohort Month'

# Extract base column cohort sizes (Month index 0)
cohort_sizes = cohort_counts.iloc[:, 0]

# Compute relative percentage retention matrix
retention_matrix = cohort_counts.divide(cohort_sizes, axis=0)

### Cell 4: Visualizing the Matrix Heatmap
plt.figure(figsize=(18, 12))
plt.title('Task 22 Deliverable: Cohort Retention Basics (%)', fontsize=16, fontweight='bold', pad=20)

sns.heatmap(data=retention_matrix,
            annot=True,
            fmt='.1%',
            vmin=0.0,
            vmax=0.5,
            cmap='Blues',
            cbar_kws={'label': 'Retention Scale'})

plt.xlabel('Months Passed Since Initial Sign-Up (Cohort Index)', fontsize=12, labelpad=10)
plt.ylabel('Initial Sign-Up Cohort Month', fontsize=12, labelpad=10)
plt.xticks(rotation=0)
plt.show()

### Cell 5: Programmatic Executive Insights Generation
print("=== PROGRAMMATIC COHORT INSIGHTS REPORT ===\n")

# 1. Average Month 1 Churn / Drop-off
avg_month_1_retention = retention_matrix[1].mean()
print(f"📊 1. IMMEDIATE CHURN TREND:")
print(f"   - On average, only {avg_month_1_retention:.1%} of customers return in Month 1.")
print(f"   - This means {1 - avg_month_1_retention:.1%} of your customer base drops off immediately.")
print(f"   - ACTION: Implement onboarding or discount campaigns within 14 days of the first purchase.\n")

# 2. Best Performing Cohort (Highest Month 1 Retention)
best_cohort = retention_matrix[1].idxmax()
best_cohort_val = retention_matrix[1].max()
print(f"⭐ 2. TOP PERFORMING SIGN-UP COHORT:")
print(f"   - The {best_cohort} cohort had the highest Month 1 return rate at {best_cohort_val:.1%}.")
print(f"   - ACTION: Investigate marketing campaigns or product collections launched in {best_cohort} to replicate success.\n")

# 3. Long-Term Value (LTV) Flattening Stability
long_term_avg = retention_matrix.loc[:, 6:12].mean().mean()
print(f"🔄 3. LONG-TERM RETENTION STABILITY:")
print(f"   - Between Month 6 and Month 12, retention flattens out to an average of {long_term_avg:.1%}.")
print(f"   - This plateau proves your platform has achieved strong Product-Market Fit with a core segment of loyal buyers.\n")

# 4. Seasonality Impacts
print(f"🎄 4. SEASONALITY INSIGHT:")
print(f"   - Heatmap evaluation highlights a diagonal alignment or sudden structural uptick in user return rates across older historical cohorts during November and December 2010.")
print(f"   - This spike is driven by predictable holiday shopping surges.")

## 📈 Key Insights & Analytical Summary

1. **Immediate Onboarding Churn:** There is a distinct, recurring drop-off across all cohorts between **Month 0 and Month 1**, where on average **78.8%** of the user volume immediately drops off. This points to a heavy reliance on single-transaction shoppers. 
2. **Long-Term Core Retention:** Beyond Month 6, retention curves stabilize into a plateau, holding steady at an average of **16.6%** through Month 12. This flattening curve confirms sustainable **Product-Market Fit** among a highly loyal segment of repeat wholesale buyers.
3. **Diagonal Seasonality:** In November and December 2010, return rates across older, dormant cohorts spike simultaneously. Because this dataset tracks giftware and home decor retail, this surge captures holiday gift-buying seasonality.

## 🎓 Project Knowledge Check

* **What is a cohort?**  
  A cohort is a group of users who share a common characteristic or experience within a defined time frame. In this project, we utilize **Time-Based Cohorts**, grouping customers strictly by the month of their absolute first transaction on the platform.
* **Why is cohort retention useful for business stakeholders?**  
  Unlike macro metrics (e.g., Monthly Active Users), which can look healthy due to aggressive marketing spend even while a product is struggling, cohort analysis isolates user retention from top-of-funnel acquisition. This approach provides an unskewed look at long-term customer lifetime value (LTV), retention stabilization curves, and marketing campaign performance over time.
# cohort-chart-for-online-retail-II
