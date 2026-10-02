#  Penang Ferry Museum — Competitor Analysis Dashboard 


An interactive **Streamlit analytics dashboard** that turns Google Maps reviews into actionable business intelligence for **Penang Ferry Museum (PFM)** and benchmarks its performance against competing attractions across Penang, Malaysia. - (https://pfm-visitor-analytics.streamlit.app/)


![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1) Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dashboard Pages](#-dashboard-pages)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Requirements](#-data-requirements)
- [Usage](#-usage)
- [Machine Learning Components](#-machine-learning-components)
- [Design System](#-design-system)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 2) Overview

Penang Ferry Museum receives visitor feedback through Google Maps reviews — but raw reviews alone don't tell the full story. This dashboard:

- **Analyses** PFM's reviews for sentiment, keywords, and owner responsiveness
- **Compares** PFM against competitor attractions in Penang
- **Discovers** visitor topics automatically using LDA topic modelling
- **Forecasts** future visitor activity using linear regression + seasonality
- **Recommends** concrete actions for management based on data gaps

All insights are delivered through a clean, branded, multi-page Streamlit interface.

---

## 3) Features

### 3.1) Analytics
- **Sentiment analysis** derived from star ratings (4–5★ = Positive, 3★ = Neutral, 1–2★ = Negative)
- **Owner response tracking** — response rate, response to negative reviews, sample responses
- **Visit context breakdown** — Weekday / Weekend / Public Holiday
- **Keyword extraction** — top positive & negative terms per attraction
- **Trend analysis** — monthly volume, monthly average rating

### 3.2) Machine Learning
- **LDA topic modelling** with human-readable topic labels
- **Rate-per-review gap analysis** to fairly compare attractions of different sizes
- **Linear regression forecasting** combined with monthly seasonality

### 3.3) Visualisation
- Interactive **Altair** charts (bars, lines, forecasts with confidence bands)
- **WordCloud** visualisations (unigram & bigram)
- **Folium** interactive maps for attraction locations

### 3.4) Interactivity
- Timeline filters (presets + custom range)
- Multi-select competitor picker
- Rating & sentiment filters
- Search reviews by keyword
- **Download CSV** for filtered reviews and forecasts
- Expandable sample reviews per topic / owner response

---

## 4) Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| **1** | **PFM Analysis** (Home) | Deep-dive into Penang Ferry Museum's own reviews — KPIs, satisfaction, owner response, visit context, trends, keywords, representative reviews, management insights |
| **2** | **Executive Overview** | Cross-attraction KPIs, top 3 attractions, review volume, visit context distribution, monthly trends |
| **3** | **Competitor Comparison** | Side-by-side benchmarking + LDA topic modelling to find opportunity gaps ranked by rate difference |
| **4** | **Detailed Review** | Filterable review explorer, word clouds, interactive location map |
| **5** | **Visitor Forecast** | Linear regression + seasonality to predict future review volume with confidence bands and suggested actions |


