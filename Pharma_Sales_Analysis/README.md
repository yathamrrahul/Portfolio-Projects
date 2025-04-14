# Pharma Sales Analysis

An end-to-end data analytics and forecasting project that analyzes pharmaceutical sales at multiple granularities (hourly, daily, weekly, and monthly) to derive actionable insights for stakeholders and guide marketing budget allocation. The analysis combines Python-based data processing, machine learning forecasting, and Power BI for dynamic, stakeholder-friendly reporting.

## Background & Purpose

This was a sample dataset found on Kaggle for Pharmaceutical Sales Data collected between 2014 and 2019. THe sales data ranges for 8 different drugs and was collected with a POS system. 

Source: https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data

---

## Objectives

- Analyze trends across pharmaceutical sales datasets
- Perform year-over-year (YoY) comparisons
- Identify seasonal patterns and campaign timing windows
- Forecast future sales (2020–2024) using advanced ML (Prophet)
- Recommend data-driven decisions for marketing and stakeholder communication

---

## Tools & Technologies

| Tool/Library      | Purpose |
|------------------|---------|
| **Python**        | Data cleaning, analysis, forecasting |
| `pandas`, `numpy`| Data manipulation |
| `matplotlib`, `seaborn`, `plotly` | Visualizations |
| `Prophet`         | Time series forecasting |
| `scikit-learn`    | Correlation & segmentation analysis |
| `Power BI`        | Interactive dashboard creation |
| `Jupyter Notebook` | EDA and analysis |
| `Git & GitHub`    | Version control |

---

## Directory Structure

```bash
Pharma_Sales_Analysis/
│
├── Pharma Sales Analysis.pbix          # Power BI dashboard
├── PharmaSalesAnalysis.ipynb           # Python analysis notebook
├── prophet_forecast_2020_2024.csv      # Forecasted data using Prophet
├── requirements.txt                    # Python dependencies
├── Pharma_Sales_Dataset/               # Raw CSVs (hourly, daily, etc.)
│   ├── saleshourly.csv
│   ├── salesdaily.csv
│   ├── salesweekly.csv
│   └── salesmonthly.csv
└── README.md
```

## Project Highlights

**Granular Analysis**: Data broken down by hour, day, week, and month for detailed insights

**Forecasting**: Sales predicted for each drug using Facebook Prophet through 2024

**Segmentation**: Drug performance categorized using correlation, classification, and ranking

**YoY & Seasonality**: Annual trends and seasonal patterns identified for budget planning

**Pareto Analysis**: Identified top-performing drugs (80/20 rule) to focus efforts

**Anomaly Detection**: Detected sales spikes/drops for investigative follow-up


## Sample Visualizations

-Prophet forecast (per drug)

-Top 10 Drugs by Sales Contribution

-Heatmaps of daily and hourly sales

-YoY growth charts


## Use Cases

**Marketing teams**: Allocate ad spend based on seasonal peaks

**Executives**: Understand YoY growth for key drugs

**Analysts**: Repurpose the code structure for other product lines

**Stakeholders**: View summary dashboards in Power BI with intuitive filters
