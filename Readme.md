# Anime Rating Predictor
Predicting the **high-rated** status of an anime — whether the anime crosses the threshold (MyAnimeList score >=7.0). Predicted using structured metadata (anime genres, release season, episode, studio etc.) and synopsis text, combined through a stacking ensemble.

## Links
[Live Web App](https://anime-rating-predictor-sfhulhtj2gtzsgihsywffa.streamlit.app/)

## Objective
At the initial stage of an anime release, it does not accumulate enough community ratings. This project predicts, using only information available at release, whether an anime will have high-rated status or not.

## Tech Stack & Tools
- **Python Language**: `python 3.14`
- **Data Preparation & Model Training**: `numpy`, `pandas`, `scikit-learn`, `nltk`
- **Deployment**: `streamlit` for running and deploying the app.

## Repository Structure

```
anime-rating-predictor/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_tabular_baseline.ipynb
│   ├── 03_nlp_baseline.ipynb
│   └── 04_combined_model.ipynb
├── models/
├── data/
├── app.py
├── requirements.txt
└── README.md
```
## Pipeline

### Data Preparation: 
- **Notebook**: `01_data_preparation.ipynb`
- **Dataset Used**: MyAnimeList anime dataset (30k records & 28 features). Contains metadata and synopsis text as major features.
- **Data Processing**: 
  - Removed non-essential, redundant features. 
  - Processed data for both tabular and text.
  - Handled missing values, categorical and multi-label features.
  - Performed EDA, both univariate and bivariate.
  - **Target Feature**: Threshold (`score`) selected for determining the status of `high-rated` anime.
  
### Model Training:
- **Notebook**: `02_tabular_baseline.ipynb` | `03_nlp_baseline.ipynb` | `04_combined_model.ipynb`
- Dataset trained in three separate paths to determine the baseline score.
  1. **Tabular**: `Random Forest` trained on structured metadata.
  2. **NLP**: `TF-IDF` and `Logistic Regression` for synopsis text data.
- Ensemble stacking model combining both of them.

#### Results & Observation

| Model | Test Macro F1 | Test ROC-AUC |
|---|---|---|
| Logistic Regression (tabular) | 0.736 | 0.847 |
| Random Forest (tabular, tuned) | 0.784 | 0.882 |
| Logistic Regression (NLP, synopsis) | 0.701 | 0.803 |
| **Stacking Ensemble (final)** | **0.791** | **0.889** |

- Stacking Ensemble outperforms both standalone models.
- The meta-learner weights the tabular model's prediction (`w=9.57`) more heavily than the NLP model's (`w=3.56`). This makes metadata the primary driver for rating, with synopsis acting as a meaningful secondary signal.

### Deployment
- Streamlit is used to deploy `app.py`. It takes anime metadata and synopsis and returns a prediction from the full stacking ensemble, determining whether the anime's status is high-rated or not.

## Key Decision & Findings
- **High-Rated Threshold** is `7.0`, chosen because roughly `70%` of anime scores in this dataset fall below this, making it a natural point to split.
- Data split is consistent across all three models.
- Found and corrected a data-loading bug where studios/genres/themes were read back from CSV as strings instead of lists, causing them to be encoded incorrectly. Fixing this had minimal impact on final metrics, showing these features contributed less signal than the primary metadata.
- For the **Tabular** baseline, `RandomForest` is chosen over `LogisticRegression` because of better metrics despite overfitting — reaching `0.882` test ROC-AUC vs `0.847`. The overfitting is addressed through hyperparameter tuning, reducing the train/test Macro F1 gap from `0.192` to `0.134`.
- For the **NLP** baseline, `LogisticRegression` is chosen over `MultinomialNB` despite comparable Macro F1, because of substantially higher recall (`0.719` vs. `0.528`), better fitting the project objective — missing a high-rated anime is worse than flagging a mediocre one.
- **Ensemble Model** uses stacking over simple averaging, so the standalone models' weights are learned, not guessed.

## Launching
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitation & Future Improvements
- The dataset we trained on is not validated against genuinely new anime.
- Synopsis-based signal is weaker than metadata — a transformer-based embedding (vs. TF-IDF) could close some of that gap.