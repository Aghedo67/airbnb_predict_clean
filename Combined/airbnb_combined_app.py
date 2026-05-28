import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Dublin Airbnb Intelligence Suite",
    layout="wide",
    page_icon="🍀"
)

# -------------------- CUSTOM STYLE --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }
    .main { background-color: #f5f3ef; }

    /* Airbnb-inspired accent */
    .stMetric label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { color: #FF385C; font-family: 'DM Serif Display', serif; font-size: 1.6rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: #ede9e3; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 18px; font-weight: 500; color: #555; }
    .stTabs [aria-selected="true"] { background: #fff; color: #FF385C; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF385C;
        color: white;
        border: none;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        transition: background 0.2s;
    }
    .stButton>button:hover { background-color: #e02d50; color: white; }

    /* Section cards */
    .card-section {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        border: 1px solid #e8e3dc;
    }

    /* Metric card */
    .stMetric {
        background: #ffffff;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid #e8e3dc;
    }

    /* Success / info boxes */
    .stSuccess, .stInfo { border-radius: 10px; }

    /* Selectbox, slider labels */
    label { color: #444 !important; font-size: 0.88rem !important; font-weight: 500 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1a1a1a;
    }
    [data-testid="stSidebar"] * { color: #f0ece4 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #aaa !important; }
    </style>
""", unsafe_allow_html=True)


# ==================== DATA & MODEL LOADING ====================

@st.cache_data
def load_aggregated():
    return pd.read_csv("dublin_aggregated_df(1).csv")

@st.cache_data
def load_raw():
    df = pd.read_csv('dublin_merged_df(1).csv.gz', compression='gzip')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    return df

@st.cache_resource
def load_price_model():
    return joblib.load('xgb_model_log(1).pkl')

@st.cache_resource
def load_recommendation_model():
    return joblib.load('Recommendation_system/baseline_model.pkl')

df = load_aggregated()
raw_df = load_raw()
price_model = load_price_model()
rec_model = load_recommendation_model()

# ==================== ENCODING HELPER ====================

@st.cache_data
def get_freq_encoding(neighbourhood, property_type, df):
    n_freq = df[df['neighbourhood'] == neighbourhood]['neighbourhood_freq'].mean()
    p_freq = df[df['property_type'] == property_type]['property_type_freq'].mean()
    if pd.isna(n_freq):
        n_freq = df['neighbourhood_freq'].mean()
    if pd.isna(p_freq):
        p_freq = df['property_type_freq'].mean()
    return n_freq, p_freq


# ==================== RECOMMENDATION HELPER ====================

import requests
from PIL import Image
from io import BytesIO

def get_image_safe(url):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(BytesIO(r.content))
    except Exception:
        return None

def show_recommendations(user_id, listings_df, model):
    try:
        all_ids = listings_df['id_x'].unique()
        seen_ids = listings_df[listings_df['reviewer_id'] == user_id]['id_x'].unique()
        to_predict = list(set(all_ids) - set(seen_ids))
        pairs = [(user_id, lid, 0) for lid in to_predict]
        preds = model.test(pairs)
        top5 = sorted(preds, key=lambda x: x.est, reverse=True)[:5]

        st.markdown(f"#### Top picks for guest `{user_id}`")
        for i, rec in enumerate(top5, 1):
            row = listings_df[listings_df['id_x'] == rec.iid]
            if row.empty:
                continue
            listing = row.iloc[0]
            with st.container():
                c1, c2 = st.columns([1, 2])
                with c1:
                    img = get_image_safe(listing.get('picture_url', ''))
                    if img:
                        st.image(img, use_column_width=True)
                    else:
                        st.markdown(
                            "<div style='height:120px;background:#f0ece4;border-radius:10px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:#aaa;font-size:0.8rem;'>No image</div>",
                            unsafe_allow_html=True
                        )
                with c2:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Nightly Rate", f"€{listing.get('price','–')}")
                    m2.metric("Guests", f"{int(listing.get('accommodates', 0))}")
                    m3.metric("Bedrooms", f"{int(listing.get('bedrooms', 0))}")
                    with st.expander("Property details"):
                        st.write(listing.get('description', 'No description available.'))
                        st.markdown(f"**Neighbourhood:** {listing.get('neighbourhood','–')}")
                    listing_url = listing.get('listing_url', '')
                    if listing_url:
                        st.link_button("View on Airbnb →", listing_url)
            st.divider()
    except Exception as e:
        st.error(f"Could not generate recommendations: {e}")


# ==================== HEADER ====================

st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:0.25rem;'>
        <span style='font-size:2.4rem;'>🍀</span>
        <div>
            <h1 style='margin:0;font-size:2rem;color:#1a1a1a;'>Dublin Airbnb Intelligence Suite</h1>
            <p style='margin:0;color:#888;font-size:0.95rem;'>Price prediction · Personalised recommendations · Market insights</p>
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #e8e3dc;margin:1rem 0;'>
""", unsafe_allow_html=True)

# ==================== KPI ROW ====================

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Nightly Price", f"€{round(df['price'].mean(), 0):.0f}")
k2.metric("Avg Rating", f"{round(df['review_scores_rating'].mean(), 2)}")
k3.metric("Total Listings", f"{len(df):,}")
k4.metric("Neighbourhoods", f"{df['neighbourhood'].nunique()}")

st.markdown("<br>", unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "📈 Visual Insights",
    "🧠 Model Insights",
    "📉 ANOVA Analysis",
    "💰 Price Predictor",
    "🔍 Recommendations"
])


# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df, use_container_width=True)


# ==================== TAB 2: VISUAL INSIGHTS ====================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Price vs Rating")
        st.scatter_chart(df, x='price', y='review_scores_rating')

        top_types = raw_df['property_type'].value_counts().nlargest(10).index
        pt_data = (
            raw_df[raw_df['property_type'].isin(top_types)]
            .groupby('property_type')['price'].mean()
            .reset_index()
        )
        st.subheader("Avg Price by Property Type (Top 10)")
        st.bar_chart(pt_data, x='property_type', y='price')

    with c2:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['price'].dropna(), bins=30, color='#FF385C', edgecolor='white', alpha=0.85)
        ax.set_xlabel("Price (€)")
        ax.set_ylabel("Count")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        monthly = raw_df.groupby('month')['price'].mean().reset_index()
        monthly['month'] = pd.to_datetime(monthly['month'], format='%m').dt.strftime('%B')
        st.subheader("Monthly Average Price")
        st.line_chart(monthly, x='month', y='price')

    st.subheader("Geographical Distribution")
    st.map(raw_df.sample(min(1000, len(raw_df))), latitude='latitude', longitude='longitude', width='stretch')


# ==================== TAB 3: MODEL INSIGHTS ====================
with tab3:
    st.subheader("Model Performance & Reliability")

    st.markdown("**Test-set metrics**")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", "0.6925", help="Explains 69% of price variance.")
    m2.metric("RMSE (log)", "0.3910")
    m3.metric("MAE (log)", "0.2665")

    st.markdown("**Cross-validated metrics**")
    m4, m5, m6 = st.columns(3)
    m4.metric("CV R²", "0.6918")
    m5.metric("CV RMSE (log)", "0.3784")
    m6.metric("CV MAE (log)", "0.2721")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Feature Importance (Top 20 — XGBoost)")

        # Top 20 features from the chart in the uploaded image
        feature_names_top20 = [
            'accommodates', 'bedrooms', 'room_type_Entire home/apt',
            'room_type_Shared room', 'bathrooms', 'minimum_nights',
            'distance_to_center', 'compound_scores', 'review_scores_location',
            'reviews_per_month', 'instant_bookable', 'host_response_rate',
            'host_identity_verified', 'room_type_Hotel room',
            'review_scores_cleanliness', 'estimated_occupancy_l365d',
            'number_of_reviews', 'host_response_time',
            'review_scores_accuracy', 'host_listings_count'
        ]
        importance_vals = [
            0.205, 0.157, 0.075, 0.062, 0.038, 0.030,
            0.029, 0.026, 0.024, 0.022, 0.022, 0.021,
            0.021, 0.020, 0.020, 0.020, 0.019, 0.019,
            0.018, 0.018
        ]

        fi_df = pd.DataFrame({"Feature": feature_names_top20, "Importance": importance_vals})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(8, 7))
        bars = ax2.barh(fi_df["Feature"], fi_df["Importance"], color='#FF385C', edgecolor='none', height=0.65)
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Top 20 Drivers of Listing Price", fontsize=12, pad=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(left=False)
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        fig2.tight_layout()
        st.pyplot(fig2)

    with col_right:
        st.subheader("Model Limitations")
        st.info("""
**Unexplained Variance (31%)**
The model cannot capture intangibles like interior design, photography quality, or specific landmark proximity.

**Mild overfitting**
Train/test RMSE gap of ~0.11 suggests some memorisation of training noise.

**Log-scale effect**
Errors on log-transformed prices may understate issues with extreme-priced listings.
        """)


# ==================== TAB 4: ANOVA ====================
with tab4:
    try:
        import plotly.express as px

        st.markdown("## ANOVA Feature Significance")

        ca, cb, cc = st.columns(3)
        ca.success("🏆 **Top drivers**\n\nAccommodates · Bedrooms · Property type · Location scores")
        cb.warning("⚖️ **Moderate impact**\n\nRoom type · Instant booking · Season · Cleanliness")
        cc.error("⚠️ **Low / insignificant**\n\nBathrooms · Number of reviews")

        anova_data = pd.DataFrame({
            "Feature": [
                "Accommodates", "Bedrooms", "Property Type",
                "Location Score", "Room Type", "Instant Bookable",
                "Season", "Bathrooms"
            ],
            "F_value": [48323, 6501, 1261, 7849, 209, 562, 14, 0.5],
            "p_value": [0.0, 0.0, 0.0, 0.0, 6.08e-136, 2.56e-124, 3.23e-9, 0.476]
        })
        anova_data["Significance"] = anova_data["p_value"].apply(
            lambda x: "Significant (p < 0.05)" if x < 0.05 else "Not Significant"
        )

        fig_a = px.bar(
            anova_data, x="F_value", y="Feature",
            color="Significance", orientation="h",
            hover_data={"F_value": True, "p_value": ':.2e', "Feature": False},
            title="ANOVA Feature Significance",
            color_discrete_map={
                "Significant (p < 0.05)": "#FF385C",
                "Not Significant": "#aaa"
            }
        )
        fig_a.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_a, use_container_width=True)
        st.info("Hover over each bar to see exact F-values and p-values. Features with p < 0.05 are statistically significant drivers of price.")

    except ImportError:
        st.warning("Plotly not installed. Run `pip install plotly` to enable this chart.")


# ==================== TAB 5: PRICE PREDICTOR ====================
with tab5:
    st.subheader("Predict Your Listing Price")
    st.markdown("All inputs correspond to the **top 20 XGBoost features** identified from model analysis.")

    with st.expander("ℹ️ About the features used", expanded=False):
        st.write(
            "This predictor uses the exact 20 most important features from the XGBoost model: "
            "accommodates, bedrooms, room type, bathrooms, minimum nights, distance to centre, "
            "review sentiment (compound scores), review location score, reviews per month, "
            "instant bookable, host response rate, host identity verified, review cleanliness score, "
            "estimated occupancy, number of reviews, host response time, review accuracy score, "
            "and host listings count."
        )

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Property details**")
        accommodates     = st.slider("Accommodates", 1, 20, 2)
        bedrooms         = st.slider("Bedrooms", 0, 10, 1)
        bathrooms        = st.slider("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
        room_type        = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        property_type    = st.selectbox("Property type", df['property_type'].dropna().unique())
        neighbourhood    = st.selectbox("Neighbourhood", df['neighbourhood'].dropna().unique())

    with col_b:
        st.markdown("**Host details**")
        host_response_rate      = st.slider("Host response rate", 0.0, 1.0, 0.95)
        host_response_time_val  = st.selectbox("Host response time", ["within an hour", "within a few hours", "within a day", "a few days or more"])
        host_identity_verified  = st.selectbox("Host identity verified", ["Yes", "No"])
        host_listings_count     = st.number_input("Host listings count", 1, 500, 1)
        instant_bookable        = st.selectbox("Instant bookable", ["Yes", "No"])
        minimum_nights          = st.slider("Minimum nights", 1, 365, 2)

    with col_c:
        st.markdown("**Review & performance**")
        review_scores_location   = st.slider("Location score", 0.0, 5.0, 4.7)
        review_scores_cleanliness= st.slider("Cleanliness score", 0.0, 5.0, 4.6)
        review_scores_accuracy   = st.slider("Accuracy score", 0.0, 5.0, 4.7)
        number_of_reviews        = st.number_input("Number of reviews", 0, 500_000, 20)
        reviews_per_month        = st.number_input("Reviews per month", 0.0, 100.0, 1.5, step=0.1)
        estimated_occupancy      = st.slider("Estimated occupancy (days/yr)", 0, 365, 180)
        compound_scores          = st.slider("Sentiment score (compound)", -1.0, 1.0, 0.85, step=0.01)
        distance_to_center       = st.number_input("Distance to city centre (km)", 0.0, 50.0, 3.0, step=0.1)

    st.markdown("")
    predict_btn = st.button("🚀 Predict Price")

    if predict_btn:
        # ---- encode categorical features ----
        room_entire   = 1 if room_type == "Entire home/apt" else 0
        room_shared   = 1 if room_type == "Shared room" else 0
        room_hotel    = 1 if room_type == "Hotel room" else 0
        # private room is the reference class (all zeros)

        host_time_map = {
            "within an hour": 0,
            "within a few hours": 1,
            "within a day": 2,
            "a few days or more": 3
        }
        host_response_time_enc = host_time_map[host_response_time_val]

        n_freq, p_freq = get_freq_encoding(neighbourhood, property_type, df)

        input_dict = {
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'room_type_Entire home/apt': room_entire,
            'room_type_Shared room': room_shared,
            'bathrooms': bathrooms,
            'minimum_nights': minimum_nights,
            'distance_to_center': distance_to_center,
            'compound_scores': compound_scores,
            'review_scores_location': review_scores_location,
            'reviews_per_month': reviews_per_month,
            'instant_bookable': 1 if instant_bookable == "Yes" else 0,
            'host_response_rate': host_response_rate,
            'host_identity_verified': 1 if host_identity_verified == "Yes" else 0,
            'room_type_Hotel room': room_hotel,
            'review_scores_cleanliness': review_scores_cleanliness,
            'estimated_occupancy_l365d': estimated_occupancy,
            'number_of_reviews': number_of_reviews,
            'host_response_time': host_response_time_enc,
            'review_scores_accuracy': review_scores_accuracy,
            'host_listings_count': host_listings_count,
            # keep neighbourhood/property freq as fallback columns if model uses them
            'neighbourhood_freq': n_freq,
            'property_type_freq': p_freq,
        }

        input_df = pd.DataFrame([input_dict])

        try:
            input_df = input_df[price_model.feature_names_in_]
        except Exception:
            pass

        pred_log = price_model.predict(input_df)
        price_est = np.exp(pred_log[0])

        st.success(f"### 💰 Estimated nightly price: €{round(price_est, 2)}")

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Point estimate", f"€{round(price_est, 2)}")
        col_res2.metric("Conservative (−15%)", f"€{round(price_est * 0.85, 2)}")
        col_res3.metric("Optimistic (+15%)", f"€{round(price_est * 1.15, 2)}")

        st.caption(
            "Range reflects model uncertainty (~±15%). "
            "Actual prices may vary with seasonality, photography quality, and listing description."
        )


# ==================== TAB 6: RECOMMENDATIONS ====================
with tab6:
    st.subheader("Personalised Stay Recommendations")
    st.markdown("Discover Dublin listings tailored to a guest's past review history.")

    with st.sidebar:
        st.title("🍀 DublinStay")
        st.markdown("---")
        top_ids = raw_df['reviewer_id'].dropna().value_counts().head(10).index.tolist()
        sidebar_id = st.selectbox("Quick test profiles:", top_ids)
        st.info("Select a reviewer ID to pre-fill the recommendation tool.")

    user_input = st.number_input(
        "Enter Reviewer / Guest ID:",
        value=int(sidebar_id),
        min_value=1,
        step=1
    )

    col_rec1, col_rec2, col_rec3 = st.columns(3)
    col_rec1.metric("Training samples", "276,126")
    col_rec2.metric("RMSE", "0.4238")
    col_rec3.metric("MAE", "0.3101")

    st.markdown("---")

    if st.button("🔍 Find Recommendations"):
        with st.spinner("Analysing Dublin rentals for this guest…"):
            show_recommendations(user_input, raw_df, rec_model)

    with st.expander("Model notes & limitations"):
        st.write("""
**Positivity bias** — Review sentiment is overwhelmingly positive (0.8–1.0 range), making it harder to distinguish excellent from merely good listings.

**Cold start** — A guest with no review history cannot receive personalised recommendations. The system needs existing review data to work.

**Temporal drift** — Recommendations reflect historical preferences and may not account for a listing's recent changes in quality or price.
        """)
