# Rainfall Time Series Analysis

Python reconstruction of an academic time-series project on rainfall patterns in India.

## Original project scope

The report compares two regions:

- Delhi, Haryana & Chandigarh
- Assam & Meghalaya

The original aim was to isolate trend, seasonal, cyclic and random components and then use Box-Jenkins/model comparison methods to forecast rainfall through 2025.

## What is reproducible from the files currently supplied

The uploaded CSV contains **600 monthly observations from January 1966 to December 2015 for Delhi, Haryana & Chandigarh**. The Python code in this repository reconstructs this region from the raw monthly observations only; it does not use the spreadsheet's precomputed formula columns.

The Assam-Meghalaya raw monthly dataset was not included with the uploaded files, so its full Python reconstruction is intentionally not claimed.

## Python analysis

`src/rainfall_time_series_analysis.py` performs:

1. Data cleaning and monthly-date parsing
2. Time plot and annual-average linear trend
3. Multiplicative decomposition
4. Seasonal indices and deseasonalization
5. Harmonic analysis using 20 trial periods
6. Cyclic and random component estimation
7. Variate-difference calculations
8. Augmented Dickey-Fuller test
9. ACF/PACF diagnostics
10. Model comparison:
   - Simple Seasonal exponential smoothing
   - SARIMA(0,0,0)(0,1,1)[12]
   - ARIMA(2,0,2)
11. Forecast for 2016-2025

## Important reproduction checks

The code reproduces several key results from the original report very closely:

- Maximum recorded rainfall: **405.3**
- ADF statistic with lag 8 and trend: approximately **-17.806**
- Dominant long cyclic trial period: approximately **100 months (8.33 years)**
- Simple Seasonal model is the best of the three compared models for Delhi/Haryana/Chandigarh
- The seasonal component is dominant, with the largest seasonal indices in **July-August**

One rainfall cell (November 1970) is blank in the supplied CSV. It is treated as `0.0` because doing so reproduces the report's 1970 annual average of **44.85**.

## Model comparison

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Simple Seasonal | 0.658 | 38.166 | 22.816 |
| SARIMA(0,0,0)(0,1,1)[12] | 0.616 | 40.084 | 24.086 |
| ARIMA(2,0,2) | 0.330 | 53.398 | 36.851 |

The **Simple Seasonal** model performs best among these three models for the supplied Delhi/Haryana/Chandigarh series.

## Run

```bash
pip install -r requirements.txt
python src/rainfall_time_series_analysis.py
```

Generated outputs are saved under `data/processed/`, `results/`, and `figures/`.

## Original report

The original academic report is preserved in:

`report/original_academic_report.pdf`

The report also contains the Assam-Meghalaya analysis; that part is kept as the original academic work because its raw monthly data are not currently available for independent Python reproduction.
