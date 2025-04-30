# Pharma Sales Analysis

An end-to-end data analytics and forecasting project that analyzes pharmaceutical sales to derive actionable insights for stakeholders and guide marketing budget allocation. The analysis combines Python-based data processing, machine learning forecasting, and Power BI for dynamic, stakeholder-friendly reporting.

#### Project Notes: This is a dynamic dashboard. As sales data is updated, the dashboard, machine learning forecasts, and other metrics are updated automatically. Data is also archived as a precaution before perfoming any changes. The automater does take quite a while to run by design. The frequency of the automater was thought to be set to once a month using simple automation tools.

## Background & Purpose

This was a sample dataset found on Kaggle for Pharmaceutical Sales Data collected between 2014 and 2019. THe sales data ranges for 8 different drug classes and was collected with a POS system as per the website. This project is built around a fictional pharmaceutical company.

The entire project is built on the following goals in mind:

1. Given a sales dataset, what kind of data insights can I extract?
2. Can we use this sales data to guide the marketing team to precisely tailor marketing strategies toward specific quarters, seasons, months, etc?
3. Can we use this sales data to guide purchasing and inventory?
4. Can we use this sales data to provide insights to stakeholders in the overall health of the company?

****Overall, with a combination of the objectives above, can we prioritize spending in a way where we can minimize purchasing waste and maximize our value?****


Source: https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data

---

## Objectives

- Analyze trends across pharmaceutical sales datasets
- Perform year-over-year (YoY) comparisons
- Identify seasonal patterns
- Forecast future sales (2020–2024) using Machine Learning techniques
- Recommend data-driven decisions for marketing and stakeholder communication
- Automate the process and dash.
---

## Tools & Technologies

| Tool/Library            | Purpose                                     |
|-------------------------|---------------------------------------------|
| **Python**              | Data analysis and machine learning          |
| `pandas`, `numpy`       | Data manipulation                           |
| `matplotlib`, `seaborn`, `plotly` | Visualization libraries        |
| `prophet`               | Univariate time series forecasting          |
| `scikit-learn`          | Regression, correlation & segmentation      |
| `xgboost`, `lightgbm`   | Advanced tree-based forecasting models      |
| `statsmodels`           | Statistical analysis                        |
| `Power BI`              | Interactive dashboarding and reporting      |
| `Jupyter Notebook`      | Exploratory Data Analysis (EDA)             |
| `Git & GitHub`          | Version control                             |

---

## Directory Structure

```bash
PHARMA_SALES_ANALYSIS/
│
├── .vscode/                             # VS Code settings
├── Archives/
│   ├── Archived Forecasts/
│   │   ├── Forecasting Data_archive_2025-04-29.xlsx
│   ├── Archived Metrics/
│   │   └── best_model_summary_metrics_archive_2025-04-29.csv
│   └── Archived Sales Data/
│       ├── salesdaily_archive_2025-04-29.csv
│       ├── saleshourly_archive_2025-04-29.csv
│       ├── salesmonthly_archive_2025-04-29.csv
│       └── salesweekly_archive_2025-04-29.csv
│
├── Forecasts/
│   ├── best_model_summary_metrics.csv
│   ├── Forecasting Data.xlsx
│   ├── lstm_forecast_daily_36_months.csv
│   ├── prophet_forecast.csv
│   ├── random_forest_forecast_daily_36_months.csv
│   ├── seasonality_strength.csv
│   └── xgboost_forecast_daily_36_months.csv
│
├── Pharma_Sales_Dataset/
│   ├── Sales_Combined.xlsx
│   ├── salesdaily.csv
│   ├── saleshourly.csv
│   ├── salesmonthly.csv
│   └── salesweekly.csv
│
├── Drug Descriptions.xlsx               # Supplemental drug metadata
├── Pharma Sales Analysis.pbix          # Power BI dashboard
├── Pharma_Sales_Automater.ipynb        # Automated forecasting pipeline
├── PharmaSalesAnalysis.ipynb           # Primary analysis notebook
├── README.md
└── requirements.txt                    # Python dependencies

```

## Project Highlights

**Granular Analysis**: Data broken down by hour, day, week, and month for detailed insights

**Multi-Model Forecasting**: Prophet, Random Forest, XGBoost, and LSTM used to forecast for 36 months.

**Segmentation**: Drug performance categorized using correlation, classification, and ranking

**YoY & Seasonality**: Annual trends and seasonal patterns identified for budget planning


## Sample Visualizations

-Prophet forecast (per drug)

-Machine Learning Forecast if available

-Top 10 Drugs by Sales Contribution

-Heatmaps of daily and hourly sales

-YoY growth charts


## Use Cases

**Marketing teams**: Allocate ad spend based on seasonal peaks

**Executives**: Understand YoY growth for key drugs

**Analysts**: Repurpose the code structure for other product lines

**Stakeholders**: View summary dashboards in Power BI with intuitive filters
