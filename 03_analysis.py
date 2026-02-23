import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1) KPI (Key Performance Indicator)

CLEAN_PATH = "transactions_clean.csv"

df = pd.read_csv(CLEAN_PATH)

print("\n===== GENERAL KPI SUMMARY =====\n")

#total volume
total_transactions = len(df)
print(f"Total Transactions: {total_transactions}")

#Status Distribution
print("\nStatus Count:",df["status"].value_counts())
status_percentage = df["status"].value_counts(normalize=True) * 100
print("\nStatus Distribution (%):",status_percentage)

plt.figure()
status_percentage.plot(kind="bar")
plt.title("Transaction Status Distribution (%)")
plt.xlabel("Status")
plt.ylabel("Percentage")
plt.show()

#REVENUE KPI ANALYSİS
##Platform comission
total_fee = df["fee_amount"].sum()
##actual earnings
total_net_revenue = df["net_revenue"].sum()
##Total chargeback loss estimate
total_chargeback_loss = df["chargeback_loss_est"].sum()
##Estimated net profit after chargebacks
estimated_net_profit = total_net_revenue - total_chargeback_loss

print("\n=== REVENUE KPI SUMMARY ===\n")
print(f"Total Fee Revenue: {total_fee:,.2f}")
print(f"Total Net Revenue: {total_net_revenue:,.2f}")
print(f"Total Chargeback Loss Estimate: {total_chargeback_loss:,.2f}")
print(f"Estimated Net Profit After Chargebacks: {estimated_net_profit:,.2f}")

#Revenue visualization
revenue_values = [
    total_net_revenue,
    total_chargeback_loss
]

revenue_labels = [
    "Net Revenue",
    "Chargeback Loss"
]

plt.figure()
plt.bar(revenue_labels, revenue_values)
plt.title("Revenue vs Chargeback Loss")
plt.ylabel("Amount")
plt.show()


# REVENUE BY STATUS ANALYSİS
status_revenue_summary = df.groupby("status").agg(
    total_transactions=("transaction_id", "count"),
    total_fee=("fee_amount", "sum"),
    total_net_revenue=("net_revenue", "sum"),
    total_chargeback_loss=("chargeback_loss_est", "sum")
)

print("\n===== REVENUE BY STATUS =====\n")
print(status_revenue_summary)


status_revenue_summary["estimated_profit"] = (
    status_revenue_summary["total_net_revenue"]
    - status_revenue_summary["total_chargeback_loss"]
)

print("\n===== ESTIMATED PROFIT BY STATUS =====\n")
print(status_revenue_summary)  #Number of disputed transactions are low but they cause huge loss per transaction

#RİSK SEGMENTATİON  #qcut because there is no high in cut
df["risk_segment"] = pd.qcut(         
    df["risk_score"],
    q = 3,
    labels=["Low","Medium","High"]
)
print("\n===== RISK SEGMENT DISTRIBUTION =====\n")
print(df["risk_segment"].value_counts(dropna=False))

## RISK SEGMENT KPI & PROFIT ANALYSIS
risk_summary = df.groupby("risk_segment").agg(
    total_transactions=("transaction_id", "count"),
    dispute_rate_percantage=("status", lambda x: (x == "disputed").mean() * 100),
    total_net_revenue=("net_revenue", "sum"),
    total_chargeback_loss=("chargeback_loss_est", "sum")
)
risk_summary["estimated_profit"] = (
    risk_summary["total_net_revenue"] - risk_summary["total_chargeback_loss"]
)
risk_summary["avg_profit_per_transaction"] = (
    risk_summary["estimated_profit"] / risk_summary["total_transactions"]
)
print("\n===== RISK SEGMENT SUMMARY =====\n")
print(risk_summary)  #High risk segment has high dispute however higher profit per transaction so high risk transaction may be risky but they are profitable

plt.figure()
risk_summary["total_chargeback_loss"].plot(kind="bar")
plt.title("Total_chargeback_loss by segment")
plt.xlabel("Risk Segment")
plt.ylabel("Total_chargeback_loss")
plt.show()  

## RISK SEGMENT - COUNTRY MISMATCH ANALYSIS
risk_mismatch_summary = df.groupby(
    ["risk_segment", "country_mismatch"]
).agg(
    total_transactions=("transaction_id", "count"),
    dispute_rate_percantage=("status", lambda x: (x == "disputed").mean() * 100),
    total_chargeback_loss=("chargeback_loss_est", "sum")
)

print("\n===== RISK + COUNTRY MISMATCH SUMMARY =====\n")
print(risk_mismatch_summary) #mismatch is not a fraud signal it does not effect


# HIGH RISK - MERCHANT CATEGORY ANALYSIS
high_risk_df = df[df["risk_segment"] == "High"]

merchant_risk_summary = high_risk_df.groupby("merchant_category").agg(
    total_transactions=("transaction_id", "count"),
    dispute_rate_percantage=("status", lambda x: (x == "disputed").mean() * 100),
    total_chargeback_loss=("chargeback_loss_est", "sum"),
    total_net_revenue=("net_revenue", "sum")
)
merchant_risk_summary["estimated_profit"] = (
    merchant_risk_summary["total_net_revenue"]
    - merchant_risk_summary["total_chargeback_loss"]
)
merchant_risk_summary["estimated_profit_per_transaction"] = (
    merchant_risk_summary["estimated_profit"]
    / merchant_risk_summary["total_transactions"]
)
merchant_risk_summary = merchant_risk_summary.sort_values(
    by="total_chargeback_loss",
    ascending=False
)
print("\n===== HIGH RISK - MERCHANT CATEGORY SUMMARY =====\n")
print(merchant_risk_summary)