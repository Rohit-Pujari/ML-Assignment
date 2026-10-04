# Aspect-Based Sentiment Analysis on Hotel Reviews

UE24CS352A Machine Learning, Mini-Project (Problem Statement #1)

**Team:** [Rohit Pujari, PES1UG25CS836], [Sujan Anjankumar, PES1UG24CS632]  
**Section:** [Section]

## Problem
Given the text of a hotel review, predict the user's sentiment for each aspect of the stay (Overall, Value, Rooms, Location, Cleanliness, Service), not just one overall score. Based on the Stanford CS229 (2016) report "Aspect-based Sentiment Analysis on Hotel Reviews".

## Dataset
TripAdvisor hotel reviews (Wang, Lu and Zhai, KDD 2010/2011), downloaded from the Series 00040 "Trip Advisor Data" release.

- 246,399 reviews of 1,850 hotels (173,420 after cleaning, see below)
- Per-review ratings (1-5) for 8 aspects: Overall, Value, Rooms, Location, Cleanliness, Front Desk, Service, Business Service
- `-1` means the user did not rate that aspect
- Review text is split across 3 files; commas in the text were replaced with semicolons

Files (not included in this repo because of size, place them in the project root):

| File | Contents |
|---|---|
| `00040-00000001.dat` | ratings |
| `00040-00000002.dat` | review text, part 1 |
| `00040-00000003.dat` | review text, part 2 |
| `00040-00000004.dat` | review text, part 3 |

The original download names the text files with suffixes like `_3`, `_5`. Rename them to the names above, or edit the file names in `main.ipynb`.

## Approach
1. **Load:** the ratings file is parsed by hand (price values like `1,438` contain a comma and break `pd.read_csv`). The ratings and text files are row-aligned, so they are joined side by side.
2. **Clean:** drop rows with Overall rating 0 and reviews shorter than 20 words.
3. **Aspects used:** Overall, Value, Rooms, Location, Cleanliness, Service. Front Desk and Business Service are dropped because they are missing in roughly half or more of the reviews.
4. **Split:** random sample of 60,000 reviews, 80% train and 20% test.
5. **Features:** `CountVectorizer` with word unigrams and bigrams, `min_df=5`, up to 200,000 features.
6. **Models:** one model per aspect (rows where that aspect is `-1` are excluded), using Linear SVM (class-weight balanced) and Multinomial Naive Bayes.
7. **Tasks:** exact star rating (5 classes) and polarity (1-2 stars negative, 3 neutral, 4-5 positive).
8. **Metrics:** accuracy and macro-F1 (macro-F1 matters because the classes are imbalanced).

## Results (polarity task)

| Aspect | SVM acc | SVM macro-F1 | NB acc | NB macro-F1 |
|---|---|---|---|---|
| Overall | 0.858 | 0.706 | 0.854 | 0.719 |
| Value | 0.782 | 0.638 | 0.804 | 0.665 |
| Rooms | 0.773 | 0.624 | 0.779 | 0.627 |
| Location | 0.804 | 0.481 | 0.812 | 0.486 |
| Cleanliness | 0.824 | 0.613 | 0.818 | 0.606 |
| Service | 0.783 | 0.613 | 0.805 | 0.650 |

Exact star prediction is harder (accuracy about 0.48 to 0.64). The full table for both tasks is saved to `baseline_results.csv` when you run the notebook.

## Setup
Python 3.12 is recommended.

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
1. Put the four `.dat` files in the project root (see Dataset).
2. Open `main.ipynb` and run all cells from top to bottom. This loads and cleans the data, trains the models, prints the results table, and saves the trained vectorizer and models to `absa_model.joblib`.
3. Demo: after step 2 has created `absa_model.joblib`, run

```bash
streamlit run app.py
```

and paste a hotel review into the page to see the predicted sentiment for each aspect.

## Project structure
```
main.ipynb            data loading, training, evaluation, saves the model
app.py                Streamlit demo
requirements.txt      dependencies
baseline_results.csv  results table (created by the notebook)
absa_model.joblib     trained model (created by the notebook, not committed)
```

## Limitations
- Models predict all six aspects for every review, even for aspects the review does not mention.
- Rare classes (1-star, neutral) are predicted much worse than common ones, as shown by the gap between accuracy and macro-F1.
- Only a 60,000-review sample was used for training and testing.