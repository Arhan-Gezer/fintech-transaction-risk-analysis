# Fintech-transaction-data-cleaning & Risk Analysis
* This project performs exploratory data analysis (EDA), data cleaning, and risk-focused KPI analysis on a 20,000-row financial transactions dataset using Python (Pandas).

* The goal is to improve data quality, reduce missing values, and analyze transaction-level risk patterns and revenue metrics.

## Dataset Overview

Each row represents a financial transaction from a payment platform.

Key columns:
- transaction_id: unique transaction identifier
- amount & currency: transaction value
- status: captured, failed, refunded, disputed
- risk_score: fraud risk indicator (0–100)
- fee_amount: platform commission
- net_revenue: platform earnings
- chargeback_loss_est: estimated dispute loss



## Data Dictionary

| Column | Description |
|--------|------------|
| transaction_id | Unique transaction identifier |
| timestamp | Transaction date and time |
| customer_id | Unique customer ID |
| customer_age | Customer age |
| merchant_id | Merchant identifier |
| merchant_category | Merchant business category |
| country | Customer country |
| ip_country | IP-based country |
| currency | Transaction currency |
| payment_method | Payment method used |
| device_type | Device used for transaction |
| amount | Transaction amount |
| risk_score | Fraud risk score (0–100) |
| status | Transaction outcome |
| fee_amount | Platform commission |
| net_revenue | Platform earnings |
| chargeback_loss_est | Estimated loss from disputes |


## Data Cleaning

- Identified missing values across multiple categorical and numerical columns
- Applied rule-based imputation:
  - Filled missing `ip_country` using `country`
  - Used mode for categorical columns
  - Applied median-based imputation for numerical features
- Capped outliers in transaction amounts
- Removed duplicate transaction IDs

  
## KPI & Risk Analysis

- Calculated capture, refund, and dispute rates
- Analyzed revenue distribution by merchant category
- Evaluated risk_score patterns across transaction status
- Investigated correlation between risk_score and dispute probability

  ## Technologies

- Python
- Pandas
- NumPy


## Project Summary (CV Version)

Performed end-to-end data preprocessing and risk-oriented analysis on a 20,000-row FinTech transaction dataset. Reduced missing values through rule-based imputation and analyzed transaction-level KPIs and fraud risk indicators.
