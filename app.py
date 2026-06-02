import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SalesPulse — Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D0F14;
    color: #E8EAF0;
}
section[data-testid="stSidebar"] {
    background: #13161E !important;
    border-right: 1px solid #1E2130;
}
section[data-testid="stSidebar"] * { color: #C8CADE !important; }
.main .block-container { background: #0D0F14; padding: 2rem 2.5rem; max-width: 1400px; }
.kpi-card {
    background: linear-gradient(135deg, #13161E 0%, #1A1D28 100%);
    border: 1px solid #1E2130; border-radius: 16px;
    padding: 1.4rem 1.6rem; position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent);
}
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin-bottom: 8px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #F0F2FF; line-height: 1; margin-bottom: 6px; }
.kpi-delta { font-size: 12px; font-weight: 500; padding: 2px 8px; border-radius: 999px; display: inline-block; }
.kpi-delta.up   { background: #0D2B1E; color: #34D399; }
.kpi-delta.down { background: #2B0D0D; color: #F87171; }
.kpi-icon { position: absolute; right: 1.2rem; top: 50%; transform: translateY(-50%); font-size: 32px; opacity: 0.12; }
.page-title { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; color: #F0F2FF; margin-bottom: 4px; }
.page-subtitle { font-size: 14px; color: #6B7280; margin-bottom: 2rem; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700;
    color: #A5B4FC; text-transform: uppercase; letter-spacing: 0.06em;
    margin: 1.5rem 0 1rem; display: flex; align-items: center; gap: 8px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: #1E2130; }
.badge { font-size: 11px; padding: 3px 10px; border-radius: 999px; font-weight: 600; letter-spacing: 0.04em; }
.badge-green  { background: #0D2B1E; color: #34D399; }
.badge-blue   { background: #0D1B3E; color: #60A5FA; }
.badge-purple { background: #1E0D3E; color: #A78BFA; }
.stTabs [data-baseweb="tab-list"] { background: #13161E; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #6B7280; font-weight: 500; font-size: 13px; }
.stTabs [aria-selected="true"] { background: #1E2130 !important; color: #A5B4FC !important; }
[data-testid="stMetric"] { background: #13161E; border: 1px solid #1E2130; border-radius: 12px; padding: 1rem; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #F0F2FF !important; font-family: 'Syne', sans-serif !important; }
hr { border-color: #1E2130 !important; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor='#0D0F14', plot_bgcolor='#0D0F14',
    font=dict(color='#9CA3AF', family='DM Sans'),
    xaxis=dict(gridcolor='#1E2130', showgrid=True, zeroline=False, title=None),
    yaxis=dict(gridcolor='#1E2130', showgrid=True, zeroline=False,
               tickprefix='₹', tickformat=',.0f', title=None),
    legend=dict(bgcolor='#13161E', bordercolor='#1E2130', borderwidth=1, font=dict(size=12)),
    hovermode='x unified',
    margin=dict(l=10, r=10, t=30, b=10)
)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def format_inr(v):
    if v >= 1_000_000: return f"₹{v/1_000_000:.1f}M"
    if v >= 1_000:     return f"₹{v/1_000:.0f}K"
    return f"₹{v:.0f}"

def calc_mape(actual, pred):
    a, p = np.array(actual, dtype=float), np.array(pred, dtype=float)
    mask = a != 0
    return round(float(np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100), 2)

def calc_rmse(actual, pred):
    diff = np.array(actual, dtype=float) - np.array(pred, dtype=float)
    return round(float(np.sqrt(np.mean(diff**2))), 0)

def kpi_card(label, value, delta=None, icon="📦", accent="#A5B4FC"):
    delta_html = ""
    if delta is not None:
        cls  = "up" if delta >= 0 else "down"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{sign} {abs(delta):.1f}%</div>'
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        <div class="kpi-icon">{icon}</div>
    </div>"""

# ─────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────
@st.cache_data
def make_demo_data():
    """Generate realistic demo sales data — no CSV needed"""
    np.random.seed(42)
    dates = pd.date_range('2013-01-01', '2015-07-31', freq='D')
    n     = len(dates)
    # Trend + seasonality + noise
    trend    = np.linspace(4_000_000, 6_000_000, n)
    weekly   = 500_000 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly   = 800_000 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise    = np.random.normal(0, 200_000, n)
    sales    = np.maximum(trend + weekly + yearly + noise, 500_000)
    # Remove Sundays (low sales day)
    df = pd.DataFrame({'Date': dates, 'Sales': sales.astype(int)})
    df = df[df['Date'].dt.dayofweek != 6].reset_index(drop=True)
    return df

# ─────────────────────────────────────────
# DATA PROCESSOR
# ─────────────────────────────────────────
@st.cache_data
def process_uploaded(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()

    # Find date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: 'Date'})

    # Find sales column
    sales_col = next((c for c in df.columns if 'sale' in c.lower()), None)
    if sales_col is None:
        return None, [], f"No sales column found. Columns: {df.columns.tolist()}"
    df = df.rename(columns={sales_col: 'Sales'})

    # Filter open stores if column exists
    if 'Open' in df.columns:
        df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]

    store_list = sorted(df['Store'].unique().tolist()) if 'Store' in df.columns else []
    daily = df.groupby('Date')['Sales'].sum().reset_index()
    daily = daily.sort_values('Date').reset_index(drop=True)
    return daily, store_list, None

# ─────────────────────────────────────────
# MODEL TRAINING + FORECAST
# ─────────────────────────────────────────
@st.cache_data
def run_forecast(sales_arr, dates_arr, n_days, model_name):
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    import os
    os.environ['STAN_BACKEND'] = 'CMDSTANPY'
    from prophet import Prophet as _Prophet
    import xgboost as _xgb

    sales = np.array(sales_arr, dtype=float)
    dates = pd.to_datetime(dates_arr)
    n     = len(sales)
    split = max(n - 60, int(n * 0.8))

    train_s, test_s = sales[:split], sales[split:]
    train_d, test_d = dates[:split], dates[split:]

    last_date    = dates[-1]
    # FIX: Use timedelta instead of freq arithmetic
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days,
        freq='D'
    )

    pred_test   = np.zeros(len(test_s))
    pred_future = np.zeros(n_days)
    fut_lo = fut_hi = None

    # ── ARIMA ──────────────────────────────
    if model_name == "ARIMA":
        try:
            m  = _ARIMA(train_s, order=(5,1,2)).fit()
            pred_test = m.forecast(steps=len(test_s))
            m2 = _ARIMA(sales, order=(5,1,2)).fit()
            pred_future = m2.forecast(steps=n_days)
        except Exception as e:
            st.error(f"ARIMA error: {e}")

    # ── Prophet ────────────────────────────
    elif model_name == "Prophet":
        df_tr = pd.DataFrame({'ds': train_d, 'y': train_s})
        m     = _Prophet(yearly_seasonality=True, weekly_seasonality=True,
                         seasonality_mode='multiplicative',
                         interval_width=0.80)
        m.fit(df_tr)
        pred_test = m.predict(pd.DataFrame({'ds': test_d}))['yhat'].values

        df_all = pd.DataFrame({'ds': dates, 'y': sales})
        m2     = _Prophet(yearly_seasonality=True, weekly_seasonality=True,
                          seasonality_mode='multiplicative',
                          interval_width=0.80)
        m2.fit(df_all)
        fc2         = m2.predict(pd.DataFrame({'ds': future_dates}))
        pred_future = fc2['yhat'].values
        fut_lo      = fc2['yhat_lower'].values
        fut_hi      = fc2['yhat_upper'].values

    # ── XGBoost ────────────────────────────
    elif model_name in ("XGBoost", "SARIMA"):
        def make_feats(d, s):
            df_ = pd.DataFrame({'Date': pd.to_datetime(d), 'Sales': np.array(s, dtype=float)})
            df_['day']        = df_['Date'].dt.day
            df_['month']      = df_['Date'].dt.month
            df_['dow']        = df_['Date'].dt.dayofweek
            df_['dayofyear']  = df_['Date'].dt.dayofyear
            df_['year']       = df_['Date'].dt.year
            df_['lag_7']      = df_['Sales'].shift(7)
            df_['lag_14']     = df_['Sales'].shift(14)
            df_['lag_30']     = df_['Sales'].shift(30)
            df_['roll_7']     = df_['Sales'].shift(1).rolling(7).mean()
            df_['roll_30']    = df_['Sales'].shift(1).rolling(30).mean()
            # Trend feature — slope of last 7 days
            df_['trend_7']    = df_['Sales'].shift(1).rolling(7).apply(
                lambda x: float(np.polyfit(range(len(x)), x, 1)[0]), raw=True)
            return df_.dropna()

        full_f = make_feats(dates.values, sales)
        feats  = ['day','month','dow','dayofyear','year',
                  'lag_7','lag_14','lag_30','roll_7','roll_30','trend_7']
        n_test = len(test_s)
        tr_f   = full_f.iloc[:-n_test]
        te_f   = full_f.iloc[-n_test:]

        xm = _xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03,
            max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=42)
        xm.fit(tr_f[feats], tr_f['Sales'])
        pred_test = xm.predict(te_f[feats])

        # Iterative future prediction with trend
        history_s = list(sales)
        preds     = []
        for i in range(n_days):
            nd     = future_dates[i]
            lag7   = history_s[-7]
            lag14  = history_s[-14]
            lag30  = history_s[-30]
            roll7  = float(np.mean(history_s[-7:]))
            roll30 = float(np.mean(history_s[-30:]))
            # Trend = slope — KEY FIX: passes growth signal forward
            trend7 = float(np.polyfit(range(7), history_s[-7:], 1)[0])
            row = pd.DataFrame([{
                'day': nd.day, 'month': nd.month, 'dow': nd.dayofweek,
                'dayofyear': nd.timetuple().tm_yday, 'year': nd.year,
                'lag_7': lag7, 'lag_14': lag14, 'lag_30': lag30,
                'roll_7': roll7, 'roll_30': roll30, 'trend_7': trend7
            }])
            p = float(xm.predict(row)[0])
            preds.append(p)
            history_s.append(p)
        pred_future = np.array(preds)

    return (
        test_d.values, test_s, np.array(pred_test),
        future_dates.values, np.array(pred_future),
        fut_lo, fut_hi
    )

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 SalesPulse")
    st.markdown("---")

    data_mode = st.radio(
        "Data Source",
        ["Use Demo Dataset", "Upload Custom Dataset"],
        index=0
    )

    uploaded   = None
    daily_data = None
    store_list = []

    if data_mode == "Upload Custom Dataset":
        uploaded = st.file_uploader("Upload train.csv", type=["csv"])
        if uploaded:
            raw = uploaded.read()
            daily_data, store_list, err = process_uploaded(raw)
            if err:
                st.error(err)
                daily_data = None
            else:
                st.success(f"✅ {len(daily_data)} days loaded")
    else:
        daily_data, _, _ = process_uploaded(open('train.csv', 'rb').read())
        st.success("✅ Demo data loaded (Rossmann Store Sales)")

    st.markdown("---")

    forecast_days = st.selectbox(
        "📅 Forecast Days",
        [7, 30, 60, 90],
        index=1,
        format_func=lambda x: {
            7 : "7 days (1 week)",
            30: "30 days (1 month)",
            60: "60 days (2 months)",
            90: "90 days (3 months)"
        }[x]
    )

    # Smart model recommendation based on forecast days
    if forecast_days <= 30:
        rec_reason  = "Best for short-term (≤30 days)"
        model_opts  = ["Prophet", "XGBoost", "ARIMA"]
    else:
        rec_reason  = "Best for long-term (>30 days)"
        model_opts  = ["Prophet", "XGBoost", "ARIMA"]

    st.caption(f"💡 Recommended: **{model_opts[0]}** — {rec_reason}")

    model_choice = st.selectbox(
        "🤖 Select Model",
        model_opts,
        index=0
    )

    # Warning for bad combinations
    if model_choice == "XGBoost" and forecast_days >= 60:
        st.warning("⚠️ XGBoost accuracy drops for 60+ days. Use Prophet instead.")

    if store_list:
        sel_store = st.selectbox("🏪 Store Filter",
                                  ["All Stores"] + [f"Store {s}" for s in store_list[:50]])
    else:
        sel_store = "All Stores"

    if daily_data is not None:
        min_d = daily_data['Date'].min().date()
        max_d = daily_data['Date'].max().date()
        date_range = st.date_input("📆 Date Range",
                                   value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    
    st.markdown("---")
    run_btn = st.button("🚀 Generate Forecast", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.markdown('<div class="page-title">SalesPulse Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Retail Sales Intelligence · Forecast · Analyze · Decide</div>', unsafe_allow_html=True)

if daily_data is None:
    st.info("👈 Select 'Use Demo Dataset' or upload your train.csv from the sidebar.")
    st.stop()

# Date filter
if 'date_range' in dir() and len(date_range) == 2:
    mask = ((daily_data['Date'].dt.date >= date_range[0]) &
            (daily_data['Date'].dt.date <= date_range[1]))
    df_filtered = daily_data[mask].reset_index(drop=True)
else:
    df_filtered = daily_data.copy()

if len(df_filtered) < 60:
    st.warning("⚠️ Data too short — please select a wider date range (min 60 days)")
    st.stop()

# ─────────────────────────────────────────
# RUN FORECAST
# ─────────────────────────────────────────
with st.spinner(f"⏳ Training {model_choice} model — please wait..."):
    result = run_forecast(
        df_filtered['Sales'].values,
        df_filtered['Date'].values,
        forecast_days,
        model_choice
    )

test_dates, test_actual, test_pred, fut_dates, fut_pred, fut_lo, fut_hi = result
fut_dates_dt = pd.to_datetime(fut_dates)
test_dates_dt = pd.to_datetime(test_dates)

# ─────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────
total_hist    = float(df_filtered['Sales'].sum())
total_fut     = float(np.sum(fut_pred))
avg_daily     = float(df_filtered['Sales'].mean())
model_mape    = calc_mape(test_actual, test_pred)
model_acc     = round(100 - model_mape, 1)
last30_actual = float(df_filtered['Sales'].tail(30).sum())
growth        = ((total_fut - last30_actual) / last30_actual * 100) if last30_actual else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Total Historical Sales", format_inr(total_hist),
                         icon="💰", accent="#A5B4FC"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(f"Predicted Next {forecast_days}d", format_inr(total_fut),
                         delta=growth, icon="🔮", accent="#34D399"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Model Accuracy", f"{model_acc}%",
                         icon="🎯", accent="#FBBF24"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Avg Daily Sales", format_inr(avg_daily),
                         icon="📅", accent="#F472B6"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Future Forecast", "📊  Actual vs Predicted", "📈  Historical Trends"])

# ─── TAB 1: FUTURE FORECAST ──────────────
with tab1:
    st.markdown('<div class="section-header">Future Sales Forecast</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.caption(
            f"**{model_choice}** forecast · "
            f"{fut_dates_dt[0].strftime('%d %b %Y')} → {fut_dates_dt[-1].strftime('%d %b %Y')}"
        )
    with col_b:
        bmap = {"ARIMA":"badge-blue","Prophet":"badge-purple",
                "XGBoost":"badge-green","SARIMA":"badge-blue"}
        st.markdown(f'<span class="badge {bmap[model_choice]}">{model_choice}</span>',
                    unsafe_allow_html=True)

    fig1 = go.Figure()

    # Historical last 90 days
    hist90 = df_filtered.tail(90)
    fig1.add_trace(go.Scatter(
        x=hist90['Date'], y=hist90['Sales'],
        name='Historical', mode='lines',
        line=dict(color='#6B7280', width=1.5),
        hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra>Historical</extra>'
    ))

    # Forecast
    fig1.add_trace(go.Scatter(
        x=fut_dates_dt, y=fut_pred,
        name=f'{model_choice} Forecast', mode='lines',
        line=dict(color='#34D399', width=2.5, dash='dash'),
        hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra>Forecast</extra>'
    ))

    # Prophet confidence band
    if fut_lo is not None and fut_hi is not None:
        fig1.add_trace(go.Scatter(
            x=list(fut_dates_dt) + list(fut_dates_dt)[::-1],
            y=list(fut_hi) + list(fut_lo)[::-1],
            fill='toself', fillcolor='rgba(52,211,153,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='80% CI', hoverinfo='skip'
        ))

    # TODAY vertical line — FIX: use shape instead of add_vline
    today_x = str(df_filtered['Date'].max().date())
    fig1.add_shape(
        type="line",
        x0=today_x, x1=today_x, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#A5B4FC", width=1, dash="dot")
    )
    fig1.add_annotation(
        x=today_x, y=1, xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color="#A5B4FC", size=11),
        yanchor="bottom"
    )

    fig1.update_layout(height=420, **PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    # Forecast Table
    st.markdown('<div class="section-header">Prediction Table</div>', unsafe_allow_html=True)
    df_tbl = pd.DataFrame({
        'Date'           : fut_dates_dt.strftime('%d %b %Y'),
        'Day'            : fut_dates_dt.strftime('%A'),
        'Predicted Sales': [f"₹{v:,.0f}" for v in fut_pred],
        'vs Daily Avg'   : [f"{'▲' if v > avg_daily else '▼'} {abs(v-avg_daily)/avg_daily*100:.1f}%"
                            for v in fut_pred],
    })
    st.dataframe(df_tbl, use_container_width=True, height=300, hide_index=True)

    # Summary metrics
    st.markdown('<div class="section-header">Forecast Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predicted",  format_inr(float(np.sum(fut_pred))))
    m2.metric("Peak Day",         format_inr(float(np.max(fut_pred))))
    m3.metric("Lowest Day",       format_inr(float(np.min(fut_pred))))
    m4.metric("Daily Average",    format_inr(float(np.mean(fut_pred))))

# ─── TAB 2: ACTUAL vs PREDICTED ──────────
with tab2:
    st.markdown('<div class="section-header">Model Validation — Last 60 Days</div>', unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    r2 = float(1 - np.var(test_actual - test_pred) / np.var(test_actual))
    v1.metric("MAPE",     f"{model_mape}%")
    v2.metric("RMSE",     f"{calc_rmse(test_actual, test_pred):,.0f}")
    v3.metric("R² Score", f"{r2:.3f}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_dates_dt, y=test_actual,
        name='Actual Sales', mode='lines',
        line=dict(color='#F0F2FF', width=2.5),
        hovertemplate='%{x|%d %b}<br>Actual: ₹%{y:,.0f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=test_dates_dt, y=test_pred,
        name=f'{model_choice} Prediction', mode='lines',
        line=dict(color='#F87171', width=2, dash='dash'),
        hovertemplate='%{x|%d %b}<br>Predicted: ₹%{y:,.0f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=list(test_dates_dt) + list(test_dates_dt)[::-1],
        y=list(test_actual) + list(test_pred)[::-1],
        fill='toself', fillcolor='rgba(248,113,113,0.05)',
        line=dict(color='rgba(0,0,0,0)'), name='Error', hoverinfo='skip'
    ))
    fig2.update_layout(height=380, **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 3: HISTORICAL TRENDS ────────────
with tab3:
    st.markdown('<div class="section-header">Monthly Sales Trend</div>', unsafe_allow_html=True)

    monthly_df = df_filtered.copy()
    monthly_df['Month'] = monthly_df['Date'].dt.to_period('M').astype(str)
    m_agg = monthly_df.groupby('Month')['Sales'].sum().reset_index()

    fig3 = go.Figure(go.Bar(
        x=m_agg['Month'], y=m_agg['Sales'],
        marker=dict(
            color=m_agg['Sales'],
            colorscale=[[0,'#1E2130'],[0.5,'#4F46E5'],[1,'#A5B4FC']],
            showscale=False
        ),
        hovertemplate='%{x}<br>₹%{y:,.0f}<extra></extra>'
    ))
    # Rolling avg
    ds = df_filtered.sort_values('Date')
    fig3.add_trace(go.Scatter(
        x=ds['Date'], y=ds['Sales'].rolling(30).mean(),
        name='30d Rolling Avg', mode='lines',
        line=dict(color='#FBBF24', width=2),
        hovertemplate='%{x|%d %b %Y}<br>Avg: ₹%{y:,.0f}<extra></extra>'
    ))
    fig3.update_layout(height=380, bargap=0.2, **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Weekly Sales Pattern</div>', unsafe_allow_html=True)
    dow_df = df_filtered.copy()
    dow_df['DOW'] = dow_df['Date'].dt.day_name()
    order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    d_agg  = dow_df.groupby('DOW')['Sales'].mean().reindex(order).reset_index()

    fig4 = go.Figure(go.Bar(
        x=d_agg['DOW'], y=d_agg['Sales'],
        marker=dict(color=['#4F46E5']*5 + ['#A5B4FC','#6366F1']),
        hovertemplate='%{x}<br>Avg: ₹%{y:,.0f}<extra></extra>'
    ))
    fig4.update_layout(height=260, showlegend=False, bargap=0.3, **PLOT_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)