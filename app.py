# 1. Importing libraries and downloading files
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 2. Loading Models and preprocessor
@st.cache_resource
def load_artifacts():
    tabular_model = joblib.load('models/tabular_baseline_model.pkl')
    preprocessor = joblib.load('models/tabular_baseline_preprocessor.pkl')
    nlp_pipeline = joblib.load('models/nlp_baseline_pipeline.pkl')
    meta_learner = joblib.load('models/stacked_meta_learner.pkl')
    return tabular_model, preprocessor, nlp_pipeline, meta_learner

tabular_model, preprocessor, nlp_pipeline, meta_learner = load_artifacts()
# 3. Cleaning and Lemmatizing Synopsis Text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_synopsis(text):
    
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(w) for w in review if w not in stop_words]
    return ' '.join(review)

# 4. Standardizing Numerical, Categorical, Multi-Label
def get_tabular_features(row_df, genres, studios, themes):
    # 1. Standard numeric/categorical features via the saved ColumnTransformer
    std_arr = preprocessor.transform(row_df)
    std_cols = preprocessor.get_feature_names_out()
    std_df = pd.DataFrame(std_arr, columns=std_cols)

    # 2. Multi-label columns — genres/themes/studios
    trained_cols = tabular_model.feature_names_in_ if hasattr(tabular_model, "feature_names_in_") else None
    mlb_row = pd.DataFrame(0, index=[0], columns=[
        c for c in (trained_cols if trained_cols is not None else [])
        if c.startswith(('studios_', 'genres_', 'themes_'))
    ])
    for g in genres:
        col = f"genres_{g}"
        if col in mlb_row.columns:
            mlb_row[col] = 1
    for s in studios:
        col = f"studios_{s}"
        if col in mlb_row.columns:
            mlb_row[col] = 1
    for t in themes:
        col = f"themes_{t}"
        if col in mlb_row.columns:
            mlb_row[col] = 1

    final_row = pd.concat([std_df, mlb_row], axis=1)
    if trained_cols is not None:
        final_row = final_row.reindex(columns=trained_cols, fill_value=0)

    return final_row
# 5. UI Streamlit 

## Title and Headline
st.title("Anime Rating Predictor")
st.write("Predicts whether an anime is likely to be rated 7.0+ on MyAnimeList, using metadata and synopsis.")

## Ingesting the New Anime Data
st.header("Anime Details")
episodes = st.number_input("Episodes", min_value=1, value=12)
duration_mins = st.number_input("Duration (mins)", min_value=1, value=24)
type_ = st.selectbox("Type", ["TV", "Movie", "Special"])
source = st.selectbox("Source", ["Original", "Manga", "Novel", "Game", "Other", "Mixed Media", "Audio"])
rating = st.selectbox("Rating", [
    "G - All Ages", "PG - Children", "PG-13 - Teens 13 or older",
    "R - 17+ (violence & profanity)", "R+ - Mild Nudity", "Rx - Hentai"
])

st.header("Genres, Studios, Themes")

@st.cache_data
def get_mlb_options(_model, prefix):
    trained_cols = _model.feature_names_in_
    options = [c[len(prefix):] for c in trained_cols if c.startswith(prefix)]
    return sorted(options)

genre_options = get_mlb_options(tabular_model, "genres_")
studio_options = get_mlb_options(tabular_model, "studios_")
theme_options = get_mlb_options(tabular_model, "themes_")

genres = st.multiselect("Genres", genre_options, default=["Action", "Adventure"] if "Action" in genre_options else [])
studios = st.multiselect("Studio(s)", studio_options)
themes = st.multiselect("Theme(s)", theme_options)

st.header("Synopsis")
synopsis = st.text_area("Paste or write a synopsis", height=150)

if st.button("Predict"):
    if not synopsis.strip():
        st.error("Please enter a synopsis before predicting.")
    else:
        row = pd.DataFrame([{
            "episodes": episodes,
            "duration_mins": duration_mins,
            "type": type_,
            "source": source,
            "rating": rating,
        }])

        try:
            tab_features = get_tabular_features(row, genres, studios, themes)
            tab_proba = tabular_model.predict_proba(tab_features.values)[:, 1][0]

            cleaned = clean_synopsis(synopsis)
            nlp_proba = nlp_pipeline.predict_proba([cleaned])[:, 1][0]

            final_proba = meta_learner.predict_proba([[tab_proba, nlp_proba]])[:, 1][0]

            st.subheader("Result")
            col1, col2 = st.columns(2)
            col1.metric("Tabular model", f"{tab_proba:.2f}")
            col2.metric("NLP model", f"{nlp_proba:.2f}")

            if final_proba >= 0.5:
                st.success(f"Predicted: **High-rated** (confidence: {final_proba:.2f})")
            else:
                st.info(f"Predicted: **Not high-rated** (confidence: {1 - final_proba:.2f})")

        except Exception as e:
            st.error(f"Something went wrong processing this input: {e}")