# Aspect Based Sentiment Analysis on Hotel Reviews

University ML mini project based on Yangyang Yu, [“Aspect based Sentiment Analysis on Hotel Reviews”](https://cs229.stanford.edu/proj2016spr/report/032.pdf), Stanford CS229 (2016). The notebook `main.ipynb` remains the original working baseline. The added pipeline compares its bag of n-grams models with one shared, multi-task DistilBERT model.

## Data

Place the ratings file and three row-aligned text files in the project root:

| File | Contents |
| --- | --- |
| `00040-00000001.dat` | Ratings |
| `00040-00000002.dat` | Review text, part 1 |
| `00040-00000003.dat` | Review text, part 2 |
| `00040-00000004.dat` | Review text, part 3 |

Edit `RATINGS_FILE` and `TEXT_FILES` at the top of `data.py` if the local filenames differ. The shared loader reproduces the notebook parsing (including ratings from the last eight comma-split fields), joins rows by position, retains Overall > 0 and reviews with at least 20 words, samples 60,000 rows with seed 42, and uses the fixed 48,000/12,000 split. Labels map ratings 1–2 to negative, 3 to neutral, and 4–5 to positive. Unrated aspects are omitted from training and evaluation.

## Setup

Use Python 3.12. For the project GPU environment, activate the existing `~/ml312` virtual environment, then install dependencies with `uv`:

```bash
source ~/ml312/bin/activate
uv pip install -r requirements.txt
```

PyTorch uses CUDA when available and falls back to CPU.

## Train and run

Train the bag of n-grams baseline:

```bash
python train_baseline.py
```

This fits a `CountVectorizer` with unigrams/bigrams and six LinearSVC plus six MultinomialNB classifiers. It writes `models/baseline.joblib` and `results/baseline_results.csv`.

Check the DistilBERT pipeline on 2,000 training reviews for one epoch:

```bash
python train_bert.py --smoke
```

Run full training (48,000 training rows, two epochs):

```bash
python train_bert.py
```

The first run downloads `distilbert-base-uncased`. The model and tokenizer are saved under `models/bert/`, and metrics go to `results/bert_results.csv`.

Launch the live demo:

```bash
streamlit run app.py
```

The app can show predictions and a random held-out review with its true aspect labels, baseline feature contributions, keyword based sentence highlights, and available result tables/charts. Train the corresponding model(s) first to enable their predictions.

## Comparison

Both scripts evaluate on the same 12,000 test reviews and report per-aspect accuracy and macro-F1 on rated rows. Macro-F1 is informative because aspect ratings are imbalanced, so common classes can yield good accuracy while rare classes remain difficult. Results shown by the app are read from the CSV files produced by actual runs.

Training artifacts and the raw data are ignored by Git (`*.dat`, model files, `models/`, `out/`).
