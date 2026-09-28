# pages/forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import base64
from pathlib import Path

from sklearn.linear_model import LinearRegression

from data_loader import load_data
from styles import apply_global_styles, render_sidebar, YELLOW


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Visitor Forecast | Penang Ferry Museum",
    page_icon=None,
    layout="wide",
)

apply_global_styles()
render_sidebar(current_page="forecast")


# ============================================================
# LOAD DATA
# ============================================================

df = load_data().copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.rename(columns={
    "reviewer": "name",
    "author": "name",
    "user": "name",
    "reviewer_name": "name",
    "review_url": "reviewUrl",
    "url": "reviewUrl",
})


# ============================================================
# HEADER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "images (1).jpg"

if IMAGE_PATH.exists():
    with open(IMAGE_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 0; margin-bottom: -10px;">
            <img src="data:image/jpeg;base64,{img_b64}"
                width="180"
                style="display: inline-block; vertical-align: middle;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("""
<h1 style="
    text-align: center;
    color: #292929;
    font-family: 'Poppins', Arial, sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    line-height: 1.2;
">
    Visitor Trend Forecast
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="
    text-align: center;
    color: #666666;
    font-family: 'Lato', Arial, sans-serif;
    font-size: 18px;
    font-weight: 500;
    margin: 2px 0 0 0;
    padding: 0;
">
    Predicting future visitor activity from past review patterns
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
    text-align: center;
    color: #F2B705;
    font-size: 16px;
    margin: 4px 0 12px 0;
    padding: 0;
">
    ━━━━━
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    Reviews are a strong signal of visitor activity. By studying
    <b>when</b> reviews are left, we can forecast <b>when</b> the next wave
    of visitors is likely to arrive, helping with staffing, promotion,
    and event planning.
</p>
""", unsafe_allow_html=True)

# >>> CHANGED: "How does this work?" expander dibuang sepenuhnya
# >>> END CHANGED


# ============================================================
# PREPARE DATA
# ============================================================

attractions = sorted(
    df["attraction_name"].dropna().astype(str).unique().tolist()
)

st.markdown("---")
st.markdown('<p class="pfm-section-title">1) Choose an Attraction</p>',
            unsafe_allow_html=True)

default_idx = 0
if "Penang Ferry Museum" in attractions:
    default_idx = attractions.index("Penang Ferry Museum")

selected_attraction = st.selectbox(
    "Which attraction would you like to forecast?",
    attractions,
    index=default_idx,
    key="forecast_attraction",
    help="Taip untuk cari attraction",
)

attraction_df = df[df["attraction_name"] == selected_attraction].copy()
attraction_df = attraction_df.dropna(subset=["date"])

if len(attraction_df) < 20:
    st.warning(
        f"Not enough dated reviews for {selected_attraction} to build a "
        "reliable forecast. Please choose another attraction."
    )
    st.stop()


# ============================================================
# 2) HISTORICAL TREND
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">2) Historical Visitor Rhythm</p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    Each bar is the number of reviews received in that month. Taller bars
    mean more visitors. Gaps may indicate closures or missing data.
</p>
""", unsafe_allow_html=True)

monthly = (
    attraction_df
    .assign(month=attraction_df["date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")
    .size()
    .reset_index(name="Reviews")
    .sort_values("month")
)

if not monthly.empty:
    full_range = pd.date_range(
        monthly["month"].min(), monthly["month"].max(), freq="MS"
    )
    monthly = (
        monthly.set_index("month")
        .reindex(full_range, fill_value=0)
        .rename_axis("month")
        .reset_index()
    )

monthly["Label"] = monthly["Reviews"].astype(str)

hist_chart = (
    alt.Chart(monthly)
    .mark_bar(color="#2F6B7C", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("month:T", title="Month",
                axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        y=alt.Y("Reviews:Q", title="Reviews per month"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("Reviews:Q", title="Reviews"),
        ],
    )
    .properties(height=320)
)

st.altair_chart(hist_chart, use_container_width=True)


# ============================================================
# 3) SEASONALITY
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">3) Which Months Bring the Most Visitors?</p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    By looking at all years together, we can see which months consistently
    receive the most reviews, a strong signal for peak visitor seasons.
</p>
""", unsafe_allow_html=True)

attraction_df["month_name"] = attraction_df["date"].dt.strftime("%b")
attraction_df["month_num"] = attraction_df["date"].dt.month

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

seasonal = (
    attraction_df
    .groupby(["month_num", "month_name"])
    .size()
    .reset_index(name="Reviews")
    .sort_values("month_num")
)

seasonal["Label"] = seasonal["Reviews"].astype(str)

peak_row = seasonal.loc[seasonal["Reviews"].idxmax()]
peak_month = peak_row["month_name"]

st.markdown(
    f"""
    <div style="
        background-color: #FFF9E6;
        border-left: 4px solid {YELLOW};
        padding: 12px 18px;
        border-radius: 6px;
        margin: 6px 0 16px 0;
        font-size: 15px;
        color: #292929;
    ">
        Peak month historically: <b>{peak_month}</b>
        with <b>{int(peak_row['Reviews'])}</b> reviews.
        Plan staffing and promotions accordingly.
    </div>
    """,
    unsafe_allow_html=True,
)

seasonal["Highlight"] = seasonal["month_name"].apply(
    lambda m: "Peak" if m == peak_month else "Normal"
)

season_chart = (
    alt.Chart(seasonal)
    .mark_bar(cursor="pointer", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("month_name:N", title="",
                sort=month_order,
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reviews:Q", title="Total reviews across all years"),
        color=alt.Color(
            "Highlight:N",
            scale=alt.Scale(
                domain=["Peak", "Normal"],
                range=[YELLOW, "#2F6B7C"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("Reviews:Q", title="Total Reviews"),
        ],
    )
    .properties(height=300)
)

season_labels = (
    alt.Chart(seasonal)
    .mark_text(dy=-8, fontWeight="bold", color="#555", fontSize=12)
    .encode(x="month_name:N", y="Reviews:Q", text="Label:N")
)

st.altair_chart(season_chart + season_labels, use_container_width=True)


# ============================================================
# 4) DAY-OF-WEEK PATTERN
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">4) Which Days Do Visitors Leave Reviews?</p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    Most visitors post their review within one or two days of visiting.
    So the day a review was posted is a good proxy for the day they visited.
</p>
""", unsafe_allow_html=True)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

attraction_df["day_of_week"] = attraction_df["date"].dt.day_name()

dow = (
    attraction_df
    .groupby("day_of_week")
    .size()
    .reindex(day_order, fill_value=0)
    .reset_index(name="Reviews")
)

dow["Label"] = dow["Reviews"].astype(str)

dow["Highlight"] = dow["day_of_week"].apply(
    lambda d: "Weekend" if d in ["Saturday", "Sunday"] else "Weekday"
)

dow_chart = (
    alt.Chart(dow)
    .mark_bar(cursor="pointer", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("day_of_week:N", title="", sort=day_order,
                axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("Reviews:Q", title="Total reviews"),
        color=alt.Color(
            "Highlight:N",
            scale=alt.Scale(
                domain=["Weekend", "Weekday"],
                range=[YELLOW, "#2F6B7C"],
            ),
            legend=alt.Legend(title="Type"),
        ),
        tooltip=[
            alt.Tooltip("day_of_week:N", title="Day"),
            alt.Tooltip("Reviews:Q", title="Reviews"),
        ],
    )
    .properties(height=300)
)

dow_labels = (
    alt.Chart(dow)
    .mark_text(dy=-8, fontWeight="bold", color="#555", fontSize=12)
    .encode(x="day_of_week:N", y="Reviews:Q", text="Label:N")
)

st.altair_chart(dow_chart + dow_labels, use_container_width=True)


# ============================================================
# 5) FORECAST
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">5) Forecast - Linear Regression </p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    The AI learns two things:
    Trend: is visitor volume going up, down, or flat over time?
    Season: which months naturally attract more visitors?
    It combines both to project the next few months.
</p>
""", unsafe_allow_html=True)

months_ahead = st.slider(
    "How many months ahead should we forecast?",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
    key="forecast_months",
)


if len(monthly) >= 6:

    hist = monthly.copy()
    hist["t"] = np.arange(len(hist))

    X = hist[["t"]].values
    y = hist["Reviews"].values
    trend_model = LinearRegression().fit(X, y)

    hist["trend_pred"] = trend_model.predict(X)
    hist["residual"] = hist["Reviews"] - hist["trend_pred"]

    hist["month_num"] = hist["month"].dt.month
    seasonal_factor = hist.groupby("month_num")["residual"].mean().to_dict()

    residual_std = hist["residual"].std()

    last_date = hist["month"].max()
    future_dates = pd.date_range(
        last_date + pd.DateOffset(months=1),
        periods=months_ahead,
        freq="MS",
    )

    future_t = np.arange(len(hist), len(hist) + months_ahead).reshape(-1, 1)
    future_trend = trend_model.predict(future_t)

    future_seasonal = np.array([
        seasonal_factor.get(d.month, 0) for d in future_dates
    ])

    future_pred = future_trend + future_seasonal
    future_pred = np.maximum(future_pred, 0)

    forecast_df = pd.DataFrame({
        "month": future_dates,
        "Forecast": future_pred.round(1),
        "Lower": np.maximum(future_pred - residual_std, 0).round(1),
        "Upper": (future_pred + residual_std).round(1),
        "Type": "Forecast",
    })

    hist_simple = hist[["month", "Reviews"]].rename(
        columns={"Reviews": "Forecast"}
    ).copy()
    hist_simple["Lower"] = hist_simple["Forecast"]
    hist_simple["Upper"] = hist_simple["Forecast"]
    hist_simple["Type"] = "Historical"

    combined = pd.concat(
        [hist_simple, forecast_df],
        ignore_index=True,
    )

    # Build pure trend line dataframe (past + future)
    hist_trend_df = pd.DataFrame({
        "month": hist["month"],
        "Trend": hist["trend_pred"],
    })
    future_trend_df = pd.DataFrame({
        "month": future_dates,
        "Trend": future_trend,
    })
    trend_line_df = pd.concat(
        [hist_trend_df, future_trend_df],
        ignore_index=True,
    )

    total_forecast = int(future_pred.sum())
    avg_forecast = round(float(future_pred.mean()), 1)
    peak_forecast_row = forecast_df.loc[forecast_df["Forecast"].idxmax()]
    peak_forecast_month = peak_forecast_row["month"].strftime("%b %Y")
    peak_forecast_value = int(peak_forecast_row["Forecast"])

    recent_same_len = hist["Reviews"].iloc[-months_ahead:].sum()
    if recent_same_len > 0:
        change = (total_forecast - recent_same_len) / recent_same_len * 100
    else:
        change = 0

    # Auto warning kalau data tak stabil atau tak cukup
    if residual_std > avg_forecast * 0.5:
        st.warning(
            "⚠️ The historical data is highly volatile (inconsistent ups and downs). "
            "Treat this forecast as a **direction only**, not a precise number. "
            "Use it for rough planning, not final decisions."
        )

    if len(monthly) < 12:
        st.info(
            f"ℹ️ Historical data covers only **{len(monthly)} months**. "
            "Seasonal patterns may not yet be fully reliable. "
            "Ideally, at least 12 months of data is needed."
        )

    # Slider + toggle dalam baris yang sama
    col_slider, col_toggle = st.columns([3, 1])
    with col_toggle:
        show_trend = st.checkbox("Show pure trend line", value=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            f"Predicted reviews (next {months_ahead} months)",
            f"{total_forecast:,}",
        )
    with c2:
        st.metric("Average per month", f"{avg_forecast:.1f}")
    with c3:
        st.metric("Expected peak", peak_forecast_month)
    with c4:
        st.metric(
            "Change vs last period",
            f"{change:+.1f}%",
        )

    hist_line = (
        alt.Chart(combined[combined["Type"] == "Historical"])
        .mark_line(color="#2F6B7C", strokeWidth=3, point=alt.OverlayMarkDef(size=50))
        .encode(
            x=alt.X("month:T", title="Month",
                    axis=alt.Axis(format="%b %Y", labelAngle=-45)),
            y=alt.Y("Forecast:Q", title="Reviews per month"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("Forecast:Q", title="Reviews", format=".0f"),
            ],
        )
    )

    forecast_line = (
        alt.Chart(combined[combined["Type"] == "Forecast"])
        .mark_line(
            color=YELLOW, strokeWidth=3, strokeDash=[6, 4],
            point=alt.OverlayMarkDef(size=70, color=YELLOW),
        )
        .encode(
            x="month:T",
            y="Forecast:Q",
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("Forecast:Q", title="Predicted", format=".1f"),
            ],
        )
    )

    trend_line = (
        alt.Chart(trend_line_df)
        .mark_line(
            color="#888888",
            strokeWidth=2,
            strokeDash=[4, 4],
            opacity=0.6,
        )
        .encode(
            x="month:T",
            y=alt.Y("Trend:Q", title="Reviews per month"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("Trend:Q", title="Pure Trend", format=".1f"),
            ],
        )
    )

    # Chart berubah ikut toggle
    if show_trend:
        forecast_chart = (
            (trend_line + hist_line + forecast_line)
            .properties(height=380)
        )
    else:
        forecast_chart = (
            (hist_line + forecast_line)
            .properties(height=380)
        )

    st.altair_chart(forecast_chart, use_container_width=True)

    st.caption(
        "Solid blue line: past reviews. "
        "Grey dashed line: pure linear trend. "
        "Dashed yellow line: AI forecast (trend + seasonality)."
    )

    # >>> CHANGED: Ringkasan naratif ditukar ke English
    st.markdown(f"""
    <div style="
        background-color: #F0F7FF;
        border-left: 4px solid #2F6B7C;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 14px 0;
        font-size: 15px;
        line-height: 1.8;
        color: #292929;
    ">
        📌 <b>Summary:</b> Based on data from 
        <b>{hist['month'].min().strftime('%b %Y')}</b> to 
        <b>{hist['month'].max().strftime('%b %Y')}</b>, 
        the forecast shows a change of <b>{change:+.1f}%</b> 
        compared to the previous period. The peak is expected in 
        <b>{peak_forecast_month}</b> with around 
        <b>{peak_forecast_value} reviews</b>. 
        On average, approximately <b>{avg_forecast:.0f} reviews per month</b> 
        are expected during this forecast period.
    </div>
    """, unsafe_allow_html=True)
    # >>> END CHANGED

    st.markdown("#### Forecast details")

    table_df = forecast_df.copy()
    table_df["Month"] = table_df["month"].dt.strftime("%b %Y")
    table_df = table_df[["Month", "Forecast", "Lower", "Upper"]]
    table_df = table_df.rename(columns={
        "Forecast": "Predicted Reviews",
        "Lower": "Low Estimate",
        "Upper": "High Estimate",
    })

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Download CSV button
    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download forecast as CSV",
        data=csv,
        file_name=f"forecast_{selected_attraction.replace(' ', '_')}_{months_ahead}m.csv",
        mime="text/csv",
    )


# ============================================================
# 6) WHAT THE AI LEARNED
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">6) What the AI Learned</p>',
            unsafe_allow_html=True)

if len(monthly) >= 6:

    if len(monthly) >= 3:
        recent_avg = hist["Reviews"].tail(3).mean()
        older_avg = hist["Reviews"].head(3).mean()

        if recent_avg > older_avg * 1.1:
            trend_msg = "Growing. Reviews have been increasing."
            trend_color = "#2E7D32"
        elif recent_avg < older_avg * 0.9:
            trend_msg = "Declining. Reviews have been decreasing."
            trend_color = "#C62828"
        else:
            trend_msg = "Stable. Review volume has been consistent."
            trend_color = "#F2B705"

    season_msg = (
        f"Peak season: visitors are most active in {peak_month}."
    )

    slow_row = seasonal.loc[seasonal["Reviews"].idxmin()]
    slow_msg = (
        f"Slow season: {slow_row['month_name']} is usually the quietest month."
    )

    best_day = dow.loc[dow["Reviews"].idxmax(), "day_of_week"]
    day_msg = f"Busiest day of week: {best_day}."

    st.markdown(
        f"""
        <div style="
            background-color: #FAFAFA;
            border-radius: 12px;
            padding: 22px 26px;
            border: 1px solid #EEE;
            font-size: 16px;
            line-height: 1.9;
            color: #292929;
        ">
            <div style="color: {trend_color}; font-weight: 600;">{trend_msg}</div>
            <div>{season_msg}</div>
            <div>{slow_msg}</div>
            <div>{day_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7) SUGGESTED ACTIONS
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">7) Suggested Actions</p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    Based on the forecast, here are some practical next steps.
</p>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown(f"### Prepare for {peak_month}")
        st.markdown(
            f"- Schedule more staff during {peak_month}.\n"
            f"- Run promotions before the peak.\n"
            f"- Ensure ticket systems can handle higher volume."
        )

    with st.container(border=True):
        st.markdown("### Promote during slow months")
        st.markdown(
            f"- {slow_row['month_name']} is usually quiet. Ideal for events or discounts.\n"
            "- Target locals during off-peak season."
        )

with c2:
    with st.container(border=True):
        st.markdown("### Optimise for weekend peak")
        st.markdown(
            "- Most reviews arrive on weekends.\n"
            "- Ensure weekend operations run smoothly."
        )

    with st.container(border=True):
        st.markdown("### Monitor monthly")
        st.markdown(
            "Revisit this forecast monthly. As new reviews come in, "
            "the AI updates its prediction automatically."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Method: linear trend + monthly seasonality learned from your own review dates. "
    "Forecast is indicative, not a guarantee."
)