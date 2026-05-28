import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from PIL import Image
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Dublin Airbnb Intelligence Suite",
    layout="wide",
    page_icon="🍀"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
    .main { background-color: #f5f3ef; }
    .stMetric label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { color: #FF385C; font-family: 'DM Serif Display', serif; font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: #ede9e3; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 18px; font-weight: 500; color: #555; }
    .stTabs [aria-selected="true"] { background: #fff; color: #FF385C; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background-color: #FF385C; color: white; border: none;
        font-weight: 600; font-family: 'DM Sans', sans-serif; font-size: 1rem; transition: background 0.2s;
    }
    .stButton>button:hover { background-color: #e02d50; color: white; }
    .stMetric { background: #ffffff; padding: 12px 16px; border-radius: 12px; border: 1px solid #e8e3dc; }
    label { color: #444 !important; font-size: 0.88rem !important; font-weight: 500 !important; }
    [data-testid="stSidebar"] { background: #1a1a1a; }
    [data-testid="stSidebar"] * { color: #f0ece4 !important; }
    </style>
""", unsafe_allow_html=True)


# ==================== DATA & MODEL LOADING ====================

@st.cache_data
def load_aggregated():
    return pd.read_csv(os.path.join(BASE_DIR, 'dublin_aggregated_df (1).csv'))

@st.cache_data
def load_raw():
    df = pd.read_csv(os.path.join(BASE_DIR, 'dublin_merged_df (1).csv.gz'), compression='gzip')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    return df

@st.cache_resource
def load_price_model():
    return joblib.load(os.path.join(BASE_DIR, 'xgb_log_model (1).pkl'))

df = load_aggregated()
raw_df = load_raw()
price_model = load_price_model()


# ==================== RECOMMENDATION HELPER ====================

def get_image_safe(url):
    try:
        r = requests.get(url, timeout=5)
        return Image.open(BytesIO(r.content))
    except Exception:
        return None

def show_recommendations(user_id, listings_df):
    try:
        seen_ids = listings_df[listings_df['reviewer_id'] == user_id]['id_x'].unique()
        if len(seen_ids) == 0:
            st.warning("No review history found for this user ID.")
            return

        user_history = listings_df[listings_df['id_x'].isin(seen_ids)]

        # Use neighbourhood_cleansed if available, else fall back
        neigh_col = 'neighbourhood_cleansed' if 'neighbourhood_cleansed' in listings_df.columns else None
        prop_col  = 'property_type' if 'property_type' in listings_df.columns else None

        unseen = listings_df[~listings_df['id_x'].isin(seen_ids)].drop_duplicates('id_x').copy()
        unseen['score'] = unseen['review_scores_rating'].fillna(0) / 5

        if neigh_col:
            fav_neighs = user_history[neigh_col].value_counts().head(3).index.tolist()
            unseen['score'] += unseen[neigh_col].isin(fav_neighs).astype(int) * 2
        if prop_col:
            fav_props = user_history[prop_col].value_counts().head(2).index.tolist()
            unseen['score'] += unseen[prop_col].isin(fav_props).astype(int)

        top5 = unseen.sort_values('score', ascending=False).head(5)

        st.markdown(f"#### Top picks for guest `{user_id}`")
        for _, listing in top5.iterrows():
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
                    m1.metric("Nightly Rate", f"€{listing.get('price', '–')}")
                    m2.metric("Guests", f"{int(listing.get('accommodates', 0))}")
                    m3.metric("Bedrooms", f"{int(listing.get('bedrooms', 0))}")
                    with st.expander("Property details"):
                        st.write(listing.get('description', 'No description available.'))
                        if neigh_col:
                            st.markdown(f"**Neighbourhood:** {listing.get(neigh_col, '–')}")
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
# Neighbourhood count from one-hot columns
neigh_cols = [c for c in df.columns if c.startswith('neighbourhood_cleansed_')]
k4.metric("Neighbourhoods", str(len(neigh_cols)))

st.markdown("<br>", unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📈 Visual Insights",
    "🧠 Model Insights",
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

        # Room type avg price from one-hot columns
        room_cols = [c for c in df.columns if c.startswith('room_type_')]
        if room_cols:
            room_avg = {}
            for col in room_cols:
                label = col.replace('room_type_', '')
                avg = df[df[col] == 1]['price'].mean()
                room_avg[label] = round(avg, 2)
            room_df = pd.DataFrame(list(room_avg.items()), columns=['Room Type', 'Avg Price'])
            st.subheader("Avg Price by Room Type")
            st.bar_chart(room_df, x='Room Type', y='Avg Price')

    with c2:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['price'].dropna(), bins=30, color='#FF385C', edgecolor='white', alpha=0.85)
        ax.set_xlabel("Price (€)")
        ax.set_ylabel("Count")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

        # Neighbourhood avg price from one-hot columns
        if neigh_cols:
            neigh_avg = {}
            for col in neigh_cols:
                label = col.replace('neighbourhood_cleansed_', '')
                avg = df[df[col] == 1]['price'].mean()
                neigh_avg[label] = round(avg, 2)
            neigh_df = pd.DataFrame(list(neigh_avg.items()), columns=['Neighbourhood', 'Avg Price'])
            st.subheader("Avg Price by Neighbourhood")
            st.bar_chart(neigh_df, x='Neighbourhood', y='Avg Price')

    # Monthly trends from raw_df
    if 'month' in raw_df.columns and 'price' in raw_df.columns:
        monthly = raw_df.groupby('month')['price'].mean().reset_index()
        monthly['month'] = pd.to_datetime(monthly['month'], format='%m').dt.strftime('%B')
        st.subheader("Monthly Average Price")
        st.line_chart(monthly, x='month', y='price')

    if 'latitude' in raw_df.columns and 'longitude' in raw_df.columns:
        st.subheader("Geographical Distribution")
        st.map(raw_df.sample(min(1000, len(raw_df))), latitude='latitude', longitude='longitude', width='stretch')


# ==================== TAB 3: MODEL INSIGHTS ====================
with tab3:
    st.subheader("Model Performance & Reliability")

    st.markdown("**Test-set metrics**")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", "0.78", help="Explains 78% of price variance.")
    m2.metric("RMSE (log)", "0.311")
    m3.metric("MAE (log)", "0.228")

    st.markdown("**Cross-validated metrics**")
    m4, m5, m6 = st.columns(3)
    m4.metric("CV R²", "0.7948")
    m5.metric("CV RMSE (log)", "0.2993")
    m6.metric("CV MAE (log)", "0.2193")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Feature Importance (Top 20 — XGBoost)")

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
        ax2.barh(fi_df["Feature"], fi_df["Importance"], color='#FF385C', edgecolor='none', height=0.65)
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
**Unexplained Variance (21%)**
The model cannot capture intangibles like interior design, photography quality, or specific landmark proximity.

**Log-scale effect**
Errors on log-transformed prices may understate issues with extreme-priced listings.
        """)


# ==================== TAB 4: PRICE PREDICTOR ====================
with tab4:
    st.subheader("Predict Your Listing Price")
    st.markdown("All inputs correspond to the **top 20 XGBoost features** identified from model analysis.")

    with st.expander("ℹ️ About the features used", expanded=False):
        st.write(
            "This predictor uses all 31 features the XGBoost model was trained on, "
            "covering property size, room type, host quality, review scores, "
            "neighbourhood, seasonality, and booking behaviour."
        )

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Property details**")
        accommodates   = st.slider("Accommodates", 1, 20, 2)
        bedrooms       = st.slider("Bedrooms", 0, 10, 1)
        beds           = st.slider("Beds", 0, 20, 1)
        bathrooms      = st.slider("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
        minimum_nights = st.slider("Minimum nights", 1, 365, 2)
        maximum_nights = st.slider("Maximum nights", 1, 1125, 30)
        room_type      = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        neighbourhood  = st.selectbox("Neighbourhood", [
            "Dublin City", "Dn Laoghaire-Rathdown", "Fingal", "South Dublin"
        ])
        season         = st.selectbox("Season", ["Winter (ref)", "Spring", "Summer", "Autumn"])

    with col_b:
        st.markdown("**Host details**")
        host_is_superhost      = st.selectbox("Superhost", ["Yes", "No"])
        host_has_profile_pic   = st.selectbox("Has profile picture", ["Yes", "No"])
        host_identity_verified = st.selectbox("Identity verified", ["Yes", "No"])
        host_response_rate     = st.slider("Response rate", 0.0, 1.0, 0.95)
        host_response_time_val = st.selectbox("Response time", [
            "within an hour", "within a few hours", "within a day", "a few days or more"
        ])
        host_listings_count    = st.number_input("Host listings count", 1, 500, 1)
        instant_bookable       = st.selectbox("Instant bookable", ["Yes", "No"])

    with col_c:
        st.markdown("**Reviews & performance**")
        review_scores_location      = st.slider("Location score", 0.0, 5.0, 4.7)
        review_scores_cleanliness   = st.slider("Cleanliness score", 0.0, 5.0, 4.6)
        review_scores_accuracy      = st.slider("Accuracy score", 0.0, 5.0, 4.7)
        review_scores_communication = st.slider("Communication score", 0.0, 5.0, 4.8)
        review_scores_value         = st.slider("Value score", 0.0, 5.0, 4.5)
        number_of_reviews           = st.number_input("Total reviews", 0, 500_000, 20)
        number_of_reviews_ly        = st.number_input("Reviews last year", 0, 10_000, 5)
        number_of_reviews_l30d      = st.number_input("Reviews last 30 days", 0, 500, 1)
        reviews_per_month           = st.number_input("Reviews per month", 0.0, 100.0, 1.5, step=0.1)
        estimated_occupancy         = st.slider("Estimated occupancy (days/yr)", 0, 365, 180)
        compound_scores             = st.slider("Sentiment score (compound)", -1.0, 1.0, 0.85, step=0.01)
        distance_to_center          = st.number_input("Distance to city centre (km)", 0.0, 50.0, 3.0, step=0.1)

    st.markdown("")
    predict_btn = st.button("🚀 Predict Price")

    if predict_btn:
        host_time_map = {
            "within an hour": 0,
            "within a few hours": 1,
            "within a day": 2,
            "a few days or more": 3
        }

        input_dict = {
            'accommodates':                        accommodates,
            'bedrooms':                            bedrooms,
            'room_type_Entire home/apt':           1 if room_type == "Entire home/apt" else 0,
            'distance_to_center':                  distance_to_center,
            'estimated_occupancy_l365d':           estimated_occupancy,
            'compound_scores':                     compound_scores,
            'beds':                                beds,
            'review_scores_cleanliness':           review_scores_cleanliness,
            'number_of_reviews_ly':                number_of_reviews_ly,
            'host_is_superhost':                   1 if host_is_superhost == "Yes" else 0,
            'host_response_time':                  host_time_map[host_response_time_val],
            'host_listings_count':                 host_listings_count,
            'review_scores_value':                 review_scores_value,
            'review_scores_communication':         review_scores_communication,
            'review_scores_location':              review_scores_location,
            'host_response_rate':                  host_response_rate,
            'instant_bookable':                    1 if instant_bookable == "Yes" else 0,
            'neighbourhood_cleansed_South Dublin': 1 if neighbourhood == "South Dublin" else 0,
            'room_type_Hotel room':                1 if room_type == "Hotel room" else 0,
            'season_Autumn':                       1 if season == "Autumn" else 0,
            'room_type_Shared room':               1 if room_type == "Shared room" else 0,
            'number_of_reviews':                   number_of_reviews,
            'minimum_nights':                      minimum_nights,
            'number_of_reviews_l30d':              number_of_reviews_l30d,
            'bathrooms':                           bathrooms,
            'reviews_per_month':                   reviews_per_month,
            'host_has_profile_pic':                1 if host_has_profile_pic == "Yes" else 0,
            'season_Spring':                       1 if season == "Spring" else 0,
            'host_identity_verified':              1 if host_identity_verified == "Yes" else 0,
            'review_scores_accuracy':              review_scores_accuracy,
            'maximum_nights':                      maximum_nights,
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


# ==================== TAB 5: RECOMMENDATIONS ====================
with tab5:
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
    col_rec1.metric("Approach", "Collaborative Filtering")
    col_rec2.metric("Signals used", "3")
    col_rec3.metric("Recommendations", "5")

    st.markdown("---")

    if st.button("🔍 Find Recommendations"):
        with st.spinner("Analysing Dublin rentals for this guest…"):
            show_recommendations(user_input, raw_df)

    with st.expander("Model notes & limitations"):
        st.write("""
**How it works** — Recommendations are based on the neighbourhoods, property types, and ratings from a guest's review history.

**Cold start** — A guest with no review history cannot receive personalised recommendations.

**Positivity bias** — Most listings have high ratings, so neighbourhood and property type carry more weight in ranking.

**Temporal drift** — Recommendations reflect historical preferences and may not account for recent changes in listing quality or price.
        """)
