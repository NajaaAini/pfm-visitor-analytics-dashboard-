# pages/detail_review.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
import base64
import re
from pathlib import Path
from streamlit_folium import st_folium
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from data_loader import load_data
from styles import apply_global_styles, render_sidebar


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Detail Review | Penang Ferry Museum",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()


# ============================================================
# CUSTOM CSS — Warna Search Box
# ============================================================

st.markdown("""
<style>
div[data-baseweb="select"] input {
    background-color: #FFF9E6 !important;
    color: #292929 !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
div[data-baseweb="select"] input:focus {
    border: 2px solid #F2B705 !important;
    box-shadow: 0 0 0 2px rgba(242, 183, 5, 0.2) !important;
    outline: none !important;
}
div[data-baseweb="select"] > div {
    border-color: #F2B705 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] input::placeholder {
    color: #A08000 !important;
    font-style: italic !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #FFF3CC !important;
}

/* Keyword search box styling */
div[data-testid="stTextInput"] input {
    background-color: #FFF9E6 !important;
    color: #292929 !important;
    border: 1px solid #F2B705 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 2px solid #F2B705 !important;
    box-shadow: 0 0 0 2px rgba(242, 183, 5, 0.2) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #A08000 !important;
    font-style: italic !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data().copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.rename(columns={
    "reviewer": "name",
    "author": "name",
    "user": "name",
    "user_name": "name",
    "reviewer_name": "name",
    "review_url": "reviewUrl",
    "url": "reviewUrl",
    "link": "reviewUrl",
    "review_link": "reviewUrl",
    "google_url": "reviewUrl",
    "owner_response": "responseFromOwnerText",
    "response": "responseFromOwnerText",
    "owner_response_text": "responseFromOwnerText",
    "response_from_owner_text": "responseFromOwnerText",
})

if "reviewUrl" not in df.columns:
    df["reviewUrl"] = ""

if "sentiment" not in df.columns:
    df["sentiment"] = df["rating"].apply(
        lambda x: "Positive" if pd.notna(x) and x >= 4
        else ("Neutral" if pd.notna(x) and x == 3
        else ("Negative" if pd.notna(x) else "Unknown"))
    )


# ============================================================
# SIDEBAR
# ============================================================

# ----- 1) Brand + navigation -----
render_sidebar(
    current_page="detail_review",
    show_attraction_selector=False,
    show_extra_filters=False,
)


# ----- 2) Attraction Selection -----
attraction_options = sorted(df["attraction_name"].dropna().unique())

with st.sidebar:

    st.markdown(
        """
        <p style="
            font-family:Poppins,Arial,sans-serif;
            font-size:14px;
            font-weight:700;
            color:#292929;
            margin-bottom:5px;
        ">
            Attraction Selection
        </p>
        """,
        unsafe_allow_html=True,
    )

    selected_attraction = st.selectbox(
        "Select Attraction:",
        options=attraction_options,
        label_visibility="collapsed",
        key="sidebar_attraction",
    )


# ----- 3) TIMELINE FILTER -----
valid_dates = df["date"].dropna()

timeline_start = None
timeline_end = None

if not valid_dates.empty:
    global_min = valid_dates.min().date()
    global_max = valid_dates.max().date()

    with st.sidebar:

        st.markdown(
            """
            <p style="
                font-family:Poppins,Arial,sans-serif;
                font-size:14px;
                font-weight:700;
                color:#292929;
                margin-bottom:5px;
            ">
                 Timeline Filter
            </p>
            """,
            unsafe_allow_html=True,
        )

        preset = st.selectbox(
            "Quick range:",
            ["All time", "Past Week", "Past Month", "Past 3 Months",
             "Past 6 Months", "Past Year", "Past 2 Years", "Custom"],
            index=0,
            key="detail_date_preset",
        )

        today = pd.Timestamp.today().normalize()

        if preset == "All time":
            start_date, end_date = global_min, global_max
        elif preset == "Past Week":
            start_date = (today - pd.Timedelta(days=7)).date()
            end_date = today.date()
        elif preset == "Past Month":
            start_date = (today - pd.Timedelta(days=30)).date()
            end_date = today.date()
        elif preset == "Past 3 Months":
            start_date = (today - pd.Timedelta(days=90)).date()
            end_date = today.date()
        elif preset == "Past 6 Months":
            start_date = (today - pd.Timedelta(days=180)).date()
            end_date = today.date()
        elif preset == "Past Year":
            start_date = (today - pd.Timedelta(days=365)).date()
            end_date = today.date()
        elif preset == "Past 2 Years":
            start_date = (today - pd.Timedelta(days=730)).date()
            end_date = today.date()
        else:  # Custom
            picked = st.date_input(
                "Custom range:",
                value=(global_min, global_max),
                min_value=global_min,
                max_value=global_max,
                key="detail_custom_date_range",
            )
            if isinstance(picked, tuple) and len(picked) == 2:
                start_date, end_date = picked
            else:
                start_date, end_date = global_min, global_max

        if start_date < global_min:
            start_date = global_min
        if end_date > global_max:
            end_date = global_max

        st.caption(f"Range: **{start_date}** → **{end_date}**")

    timeline_start = start_date
    timeline_end = end_date


# ----- 4) Review & Text Filters -----
with st.sidebar:

    st.markdown("---")
    st.markdown(
        """
        <p style="
            font-family:Poppins,Arial,sans-serif;
            font-size:14px;
            font-weight:700;
            color:#292929;
            margin-bottom:5px;
        ">
            Review &amp; Text Filters
        </p>
        """,
        unsafe_allow_html=True,
    )

    rating_filter = st.selectbox(
        "Filter by Star Rating:",
        options=["All Ratings", "5 ⭐", "4 ⭐", "3 ⭐", "2 ⭐", "1 ⭐"],
        key="sidebar_rating_filter",
    )

    sentiment_filter = st.selectbox(
        "Filter by Sentiment:",
        options=["All Sentiments", "Positive", "Neutral", "Negative"],
        key="sidebar_sentiment_filter",
    )

    max_words_limit = st.slider(
        "Max Words Shown:", 20, 200, 100, 10, key="sidebar_max"
    )

    min_words_limit = 4  # default

    st.caption(
        "Use the filters above to explore attraction reviews "
        "and visitor feedback."
    )


# ============================================================
# APPLY TIMELINE FILTER
# ============================================================

if timeline_start is not None and timeline_end is not None:
    mask = ((df["date"].dt.date >= timeline_start)
            & (df["date"].dt.date <= timeline_end))
    df = df[mask].copy()


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
    Detailed Review
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
    Deep-dive into visitor reviews &amp; feedback
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


# ============================================================
# SUB-HEADING HELPER
# ============================================================

def sub_heading(text):
    st.markdown(
        f'<p style="font-size:15px;font-weight:700;color:#292929;'
        f'margin:14px 0 6px 0;text-transform:uppercase;letter-spacing:0.5px;">'
        f'{text}</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# KEYWORD HIGHLIGHT HELPER
# ============================================================

def highlight_keyword(text, keyword):
    """Wrap keyword occurrences with markdown highlight."""
    if not keyword or not keyword.strip():
        return text
    pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
    return pattern.sub(
        lambda m: f"**:orange[{m.group(0)}]**",
        text,
    )


# ============================================================
# 1) DATASET OVERVIEW
# ============================================================

st.markdown('<p class="pfm-section-title">1) Dataset Overview</p>',
            unsafe_allow_html=True)

total_reviews = len(df)
total_attractions = df["attraction_name"].nunique()
overall_avg = df["rating"].mean()
overall_avg_rating = round(overall_avg, 2) if pd.notna(overall_avg) else "N/A"
platforms = ", ".join(df["platform"].dropna().unique()) if "platform" in df.columns else "Google Maps"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Total Reviews",
        f"{total_reviews:,}",
        help=f"Total number of reviews: {total_reviews:,}",
    )
with k2:
    st.metric(
        "Total Attractions",
        total_attractions,
        help=f"{total_attractions} attractions with reviews",
    )
with k3:
    st.metric(
        "Average Rating",
        f"{overall_avg_rating} / 5.0",
        help=f"Average rating across {total_reviews:,} reviews",
    )
with k4:
    st.metric(
        "Platform",
        platforms,
        help="Source of review data",
    )


# ============================================================
# 2) ATTRACTION SUMMARY
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">2) Attraction Summary</p>',
            unsafe_allow_html=True)

summary = (
    df.groupby("attraction_name")
    .agg(
        Reviews=("rating", "size"),
        Average_Rating=("rating", "mean")
    )
    .reset_index()
)

summary["Average_Rating"] = summary["Average_Rating"].round(2)
summary = summary.sort_values("Reviews", ascending=False)

st.dataframe(summary, use_container_width=True, hide_index=True)


# ============================================================
# 3) DETAILED ATTRACTION ANALYSIS
# ============================================================

st.markdown("---")

st.markdown("""
<p class="pfm-page-title">Detailed Attraction Analysis</p>
<p class="pfm-page-subtitle">Review insights and visitor feedback</p>
<p class="pfm-yellow-line">━━━━━</p>
""", unsafe_allow_html=True)

st.markdown(f'<p class="pfm-section-title">{selected_attraction}</p>',
            unsafe_allow_html=True)

filtered_df = df[df["attraction_name"] == selected_attraction].copy()
analysis_df = filtered_df.copy()

if rating_filter != "All Ratings":
    star_val = int(rating_filter[0])
    analysis_df = analysis_df[analysis_df["rating"] == star_val]

analysis_df["sentiment"] = analysis_df["rating"].apply(
    lambda x: "Positive" if pd.notna(x) and x >= 4
    else ("Neutral" if pd.notna(x) and x == 3
    else ("Negative" if pd.notna(x) else "Unknown"))
)

if sentiment_filter != "All Sentiments":
    analysis_df = analysis_df[analysis_df["sentiment"] == sentiment_filter]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Filtered Reviews", len(analysis_df))
with c2:
    avg_filtered = analysis_df["rating"].mean()
    st.metric("Average Rating",
              round(avg_filtered, 2) if pd.notna(avg_filtered) else "N/A")
with c3:
    platform_str = (", ".join(analysis_df["platform"].dropna().unique())
                    if "platform" in analysis_df.columns and not analysis_df.empty
                    else "Google Maps")
    st.metric("Platform", platform_str)


# ============================================================
# 4) REVIEWS SUMMARY — table + download + keyword search
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">3) Reviews Summary</p>',
            unsafe_allow_html=True)

# ---------- KEYWORD SEARCH (Summary table) ----------
summary_keyword = st.text_input(
    "🔍 Filter reviews in this table",
    placeholder="Type a keyword to filter the table (e.g., 'ferry', 'parking', 'staff')...",
    key="summary_keyword_search",
)

table_df = analysis_df[
    analysis_df["text"].astype(str).str.strip() != ""
].copy()

if summary_keyword and summary_keyword.strip():
    table_df = table_df[
        table_df["text"].str.contains(
            summary_keyword.strip(), case=False, na=False, regex=False
        )
    ].copy()
    st.caption(
        f"🔎 **{len(table_df)}** review(s) match **'{summary_keyword.strip()}'**"
    )

if not table_df.empty:
    display_df = table_df[["platform", "rating", "sentiment",
                           "date", "text"]].copy()
    display_df["date"] = pd.to_datetime(
        display_df["date"], errors="coerce"
    ).dt.strftime("%d %b %Y")
    display_df["date"] = display_df["date"].fillna("—")

    display_df = display_df.rename(columns={
        "platform": "Platform", "rating": "Rating",
        "sentiment": "Sentiment", "date": "Date", "text": "Review"
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered reviews",
        data=csv,
        file_name=f"{selected_attraction}_reviews.csv",
        mime="text/csv",
    )
else:
    if summary_keyword and summary_keyword.strip():
        st.warning(f"No reviews match **'{summary_keyword.strip()}'**.")
    else:
        st.info("No reviews with text match the selected filters.")


# ============================================================
# 5) REVIEW DETAILS — keyword search + highlight
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">4) Review Details</p>',
            unsafe_allow_html=True)

# ---------- KEYWORD SEARCH (Review Details) ----------
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    keyword = st.text_input(
        "🔍 Search reviews",
        placeholder="Type a keyword (e.g., 'parking', 'staff', 'clean')...",
        key="review_keyword_search",
        label_visibility="collapsed",
    )

with search_col2:
    show_all = st.checkbox(
        "Show all matches",
        value=False,
        key="review_show_all",
        help="By default shows the latest 10 matching reviews.",
    )

# ---------- APPLY SEARCH ----------
review_df = analysis_df[
    analysis_df["text"].astype(str).str.strip() != ""
].copy()

if keyword and keyword.strip():
    review_df = review_df[
        review_df["text"].str.contains(
            keyword.strip(), case=False, na=False, regex=False
        )
    ].copy()

review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
review_df = review_df.sort_values("date", ascending=False)

total_matches = len(review_df)

if not show_all:
    review_df = review_df.head(10)

# ---------- RESULT COUNT ----------
if keyword and keyword.strip():
    if total_matches == 0:
        st.warning(f"No reviews match **'{keyword.strip()}'**.")
    else:
        shown = len(review_df)
        st.caption(
            f"🔎 Found **{total_matches}** review(s) matching "
            f"**'{keyword.strip()}'** · showing **{shown}**"
        )
else:
    st.caption(
        f"Showing latest **{len(review_df)}** reviews with full details."
    )

# ---------- RENDER REVIEWS ----------
if not review_df.empty:
    for _, row in review_df.iterrows():
        rating_val = row["rating"] if pd.notna(row["rating"]) else 0
        name_val = row.get("name", "Anonymous")
        if pd.isna(name_val) or str(name_val).strip() == "":
            name_val = "Anonymous"

        text_val = str(row.get("text", "")).strip()

        # Truncate long text (but only when not searching — full text is useful in search mode)
        if not (keyword and keyword.strip()):
            if len(text_val) > 400:
                text_val = text_val[:400] + "..."

        # Highlight keyword if searching
        if keyword and keyword.strip():
            display_text = highlight_keyword(text_val, keyword)
        else:
            display_text = text_val

        review_url = row.get("reviewUrl", "")
        owner_response = row.get("responseFromOwnerText", "")
        date_val = row["date"]
        date_str = date_val.strftime("%d %b %Y") if pd.notna(date_val) else "—"

        sentiment_val = row.get("sentiment", "")

        with st.container(border=True):
            st.markdown(
                f"**{name_val}** · **{rating_val:.0f}★** · *{date_str}* · {sentiment_val}"
            )
            st.markdown(display_text)

            if pd.notna(owner_response) and str(owner_response).strip():
                st.success(f"**Owner:** {owner_response}")

            if pd.notna(review_url) and str(review_url).strip():
                st.markdown(f"[Read on Google]({review_url})")
else:
    if not (keyword and keyword.strip()):
        st.info("No reviews available for the selected filters.")


# ============================================================
# 6) REVIEW TEXT ANALYSIS
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">5) Review Text Analysis</p>',
            unsafe_allow_html=True)

text_series = analysis_df["text"].astype(str)
text_series = text_series[text_series.str.strip() != ""]

corpus = " ".join(text_series.tolist())

if corpus.strip():
    wc1, wc2 = st.columns(2)

    with wc1:
        st.markdown('<p class="pfm-card-title">Unigram Word Cloud</p>',
                    unsafe_allow_html=True)
        wc = WordCloud(
            width=500, height=350, background_color="white",
            colormap="Wistia",
            min_word_length=min_words_limit,
            max_words=max_words_limit,
        ).generate(corpus)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with wc2:
        st.markdown('<p class="pfm-card-title">Bigram Word Cloud</p>',
                    unsafe_allow_html=True)
        try:
            vec = CountVectorizer(ngram_range=(2, 2), stop_words="english")
            X = vec.fit_transform(text_series.tolist())
            freqs = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1))
        except ValueError:
            freqs = {}

        if freqs:
            wc = WordCloud(
                width=500, height=350, background_color="white",
                colormap="YlOrBr",
                max_words=max_words_limit,
            ).generate_from_frequencies(freqs)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Not enough text for bigram word cloud.")
else:
    st.info("No text available for the selected filters.")


# ============================================================
# 7) ATTRACTION LOCATIONS
# ============================================================

st.markdown("---")
st.markdown('<p class="pfm-section-title">6) Attraction Locations</p>',
            unsafe_allow_html=True)

st.markdown("""
<p class="pfm-description">
    Explore the locations of Penang Ferry Museum and its competitors across Penang.
</p>
""", unsafe_allow_html=True)


# ============================================================
# CUSTOM LOCATIONS (HARDCODED)
# ============================================================
# Semua marker guna icon ferry (ship):
#   dataset → merah (red)
#   custom  → biru  (blue)
# ============================================================

CUSTOM_LOCATIONS = [
    {
        "name": "Kek Lok Si Temple",
        "lat": 5.3993,
        "lng": 100.2736,
    },
    {
        "name": "Chew Jetty",
        "lat": 5.413576782644611,
        "lng": 100.34017462330446,
    },
    {
        "name": "Hin Bus Depot",
        "lat": 5.412464110636859,
        "lng": 100.32808540109448,
    },
    {
        "name": "Gurney Bay Park",
        "lat": 5.433383267917253,
        "lng": 100.32331398624294,
    },
    {
        "name": "Akuarium Tunku Abdul Rahman (AkuaTAR)",
        "lat": 5.28710975671712,
        "lng": 100.28840845142153,
    },
]


# ---------- BASE COORDS FROM DATAFRAME ----------
coords_per_attraction = (
    df.dropna(subset=["lat", "lng"])
    .groupby("attraction_name")[["lat", "lng"]]
    .first()
    .reset_index()
)

coords_per_attraction["source"] = "dataset"

# ---------- GABUNG CUSTOM + DATASET ----------
custom_df = pd.DataFrame(CUSTOM_LOCATIONS)
if not custom_df.empty:
    custom_df = custom_df.rename(columns={"name": "attraction_name"})
    custom_df["source"] = "custom"

    map_coords = pd.concat(
        [coords_per_attraction, custom_df],
        ignore_index=True,
    )
else:
    map_coords = coords_per_attraction


# ---------- SELECTOR ----------
map_selected_attraction = st.selectbox(
    "Select an attraction to view on the map",
    ["All"] + sorted(map_coords["attraction_name"].tolist()),
    key="map_attraction_selector",
)


# ---------- VIEW: SELECTED ATTRACTION ----------
if map_selected_attraction != "All":

    row = map_coords[
        map_coords["attraction_name"] == map_selected_attraction
    ].iloc[0]

    lat = row["lat"]
    lng = row["lng"]
    is_custom = row["source"] == "custom"

    marker_color = "blue" if is_custom else "red"

    # Data review hanya wujud kalau attraction tu ada dalam df
    specific_df = df[df["attraction_name"] == map_selected_attraction].copy()

    col_info, col_map = st.columns([1, 1])

    with col_info:

        st.markdown("""
        <p class="pfm-card-title">
            Attraction Overview
        </p>
        """, unsafe_allow_html=True)

        if specific_df.empty:
            # Custom location — takda data review
            st.markdown(
                f"""
                <div class="pfm-card">
                <p class="pfm-card-title">{map_selected_attraction}</p>
                <p class="pfm-description">
                    This is a manually added location — no review data
                    available in the dataset.
                </p>
                <b>Latitude:</b> {lat:.5f}<br>
                <b>Longitude:</b> {lng:.5f}<br>
                <b>Source:</b> Manual entry
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            total_revs = len(specific_df)

            avg_rating = specific_df["rating"].mean()
            avg_rating = round(avg_rating, 2) if pd.notna(avg_rating) else "N/A"

            specific_df["sentiment"] = specific_df["rating"].apply(
                lambda x: "Positive" if pd.notna(x) and x >= 4
                else ("Neutral" if pd.notna(x) and x == 3
                else ("Negative" if pd.notna(x) else "Unknown"))
            )

            positive_count = specific_df["sentiment"].eq("Positive").sum()
            negative_count = specific_df["sentiment"].eq("Negative").sum()

            positive_rate = (positive_count / total_revs * 100) if total_revs > 0 else 0
            negative_rate = (negative_count / total_revs * 100) if total_revs > 0 else 0

            specific_df["date"] = pd.to_datetime(
                specific_df["date"], errors="coerce"
            )

            latest_date = specific_df["date"].max()

            if pd.notna(latest_date):
                latest_date_text = latest_date.strftime("%d %b %Y")
            else:
                latest_date_text = "N/A"

            st.markdown(
                f"""
                <div class="pfm-card">

                <p class="pfm-card-title">
                    {map_selected_attraction}
                </p>

                <p class="pfm-description">
                    Performance summary based on the available
                    Google Maps reviews.
                </p>

                <b>Reviews Tracked:</b>
                {total_revs:,}<br>

                <b>Average Rating:</b>
                {avg_rating} / 5.0<br>

                <b>Positive Reviews:</b>
                {positive_rate:.1f}%<br>

                <b>Negative Reviews:</b>
                {negative_rate:.1f}%<br>

                <b>Latest Review:</b>
                {latest_date_text}<br>

                <b>Platform:</b>
                Google Maps

                </div>
                """,
                unsafe_allow_html=True
            )

    with col_map:

        st.markdown("""
        <p class="pfm-card-title">
            Location
        </p>
        """, unsafe_allow_html=True)

        m = folium.Map(
            location=[lat, lng],
            zoom_start=15,
            control_scale=True,
        )

        folium.Marker(
            [lat, lng],
            popup=map_selected_attraction,
            tooltip=map_selected_attraction,
            icon=folium.Icon(
                color=marker_color,
                icon="ship",
                prefix="fa",
            ),
        ).add_to(m)

        st_folium(
            m,
            height=350,
            use_container_width=True,
        )

        # Legend
        st.markdown("""
        <div style="
            display:flex;
            gap:24px;
            margin-top:8px;
            font-family:'Lato',Arial,sans-serif;
            font-size:13px;
            color:#444;
        ">
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="
                    display:inline-block;
                    width:12px;height:12px;
                    border-radius:50%;
                    background:#E74C3C;
                "></span>
                <span><b>Dataset</b> — attractions with review data</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="
                    display:inline-block;
                    width:12px;height:12px;
                    border-radius:50%;
                    background:#3498DB;
                "></span>
                <span><b>Custom</b> — manually added locations (no review data)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------- VIEW: ALL ATTRACTIONS ----------
else:

    m = folium.Map(
        location=[5.4164, 100.3400],
        zoom_start=13,
        control_scale=True,
    )

    # Dataset markers — merah, icon ship
    for _, row in coords_per_attraction.iterrows():

        folium.Marker(
            [row["lat"], row["lng"]],
            popup=row["attraction_name"],
            tooltip=row["attraction_name"],
            icon=folium.Icon(
                color="red",
                icon="ship",
                prefix="fa",
            ),
        ).add_to(m)

    # Custom markers — biru, icon ship
    for loc in CUSTOM_LOCATIONS:
        folium.Marker(
            [loc["lat"], loc["lng"]],
            popup=f"{loc['name']}",
            tooltip=f"{loc['name']}",
            icon=folium.Icon(
                color="blue",
                icon="ship",
                prefix="fa",
            ),
        ).add_to(m)

    st_folium(
        m,
        height=500,
        use_container_width=True,
    )

    # Legend
    st.markdown("""
    <div style="
        display:flex;
        gap:24px;
        margin-top:8px;
        font-family:'Lato',Arial,sans-serif;
        font-size:13px;
        color:#444;
    ">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="
                display:inline-block;
                width:12px;height:12px;
                border-radius:50%;
                background:#E74C3C;
            "></span>
            <span><b>Dataset</b> — attractions with review data</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="
                display:inline-block;
                width:12px;height:12px;
                border-radius:50%;
                background:#3498DB;
            "></span>
            <span><b>Custom</b> — manually added locations (no review data)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
