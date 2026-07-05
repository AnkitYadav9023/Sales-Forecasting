# SalesPulse — Sales Forecasting Dashboard 📊

An interactive **Streamlit** dashboard for retail sales forecasting. Upload your own sales data (or use the built-in Rossmann Store Sales demo dataset) and generate forward-looking forecasts using **ARIMA**, **Prophet**, or **XGBoost**, complete with accuracy metrics, validation charts, and historical trend analysis.

## Features

- **Multiple forecasting models**
  - **ARIMA** — classical time-series forecasting (order 5,1,2)
  - **Prophet** — trend + seasonality forecasting with 80% confidence intervals
  - **XGBoost** — gradient-boosted regression using lag/rolling-window/trend features
- **Flexible data input** — use the bundled demo dataset or upload your own `train.csv`
- **Store-level filtering** and custom date-range selection
- **KPI cards** — total historical sales, predicted sales for the selected horizon, model accuracy, and average daily sales
- **Three analysis tabs**
  1. **Future Forecast** — forecast chart, prediction table, and summary metrics
  2. **Actual vs Predicted** — model validation against the last 60 days (MAPE, RMSE, R²)
  3. **Historical Trends** — monthly sales trend (with 30-day rolling average) and weekly sales pattern
- Dark-themed, custom-styled UI built with Streamlit + Plotly

## Project Structure
Sales Forecasting/
├── app.py                # Main Streamlit application
├── train.csv             # Demo dataset — Rossmann daily sales records
├── store.csv             # Demo dataset — store metadata (not tracked in git)
├── arima_model.pkl        # Cached/pre-trained ARIMA model artifact
├── prophet_model.pkl      # Cached/pre-trained Prophet model artifact
├── xgb_model.pkl          # Cached/pre-trained XGBoost model artifact
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version pin (for deployment)
├── render.yaml             # Render.com deployment configuration
└── .gitignore

## Usage

1. In the sidebar, choose a data source:
   - **Use Demo Dataset** — loads `train.csv` (Rossmann Store Sales) automatically
   - **Upload Custom Dataset** — upload your own CSV. It must contain a date column (name containing "date") and a sales column (name containing "sale"); an optional `Open`/`Store` column enables store filtering
2. Select a **forecast horizon** (7, 30, 60, or 90 days)
3. Choose a **model** (XGBoost is recommended for short horizons, Prophet for longer ones)
4. Optionally filter by **store** and **date range**
5. Review the KPI cards and explore the three tabs for forecasts, validation metrics, and historical trends

> **Note:** At least 60 days of data are required to generate a forecast.

🔗 **Live Demo:** [sales-forecasting-2uxo.onrender.com](https://sales-forecasting-2uxo.onrender.com/)
