"""Train one balanced LinearSVC and MultinomialNB per aspect."""
from pathlib import Path

import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

from data import ASPECTS, load_data, polarity


def main():
    train, test = load_data()
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.2, max_features=200000)
    x_train = vec.fit_transform(train["text"])
    x_test = vec.transform(test["text"])
    def fit_aspect(aspect):
        # SciPy sparse indexing requires a NumPy array; a pandas Series mask
        # does not implement the .nonzero() protocol expected by SciPy.
        tr = (train[aspect] != -1).to_numpy()
        te = (test[aspect] != -1).to_numpy()
        y_train = train.loc[tr, aspect].map(polarity)
        y_test = test.loc[te, aspect].map(polarity)
        result = {}
        for name, model in (
            ("SVM", LinearSVC(class_weight="balanced", max_iter=5000)),
            ("Naive Bayes", MultinomialNB()),
        ):
            model.fit(x_train[tr], y_train)
            pred = model.predict(x_test[te])
            result[name] = (model, {
                "aspect": aspect, "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
            })
        return aspect, result

    # Each aspect is independent. Threading keeps the shared sparse matrices
    # in memory once instead of copying them into worker processes.
    fitted = Parallel(n_jobs=min(len(ASPECTS), os.cpu_count() or 1), prefer="threads")(
        delayed(fit_aspect)(aspect) for aspect in ASPECTS
    )
    svm_models, nb_models, records = {}, {}, []
    for aspect, result in fitted:
        svm_models[aspect] = result["SVM"][0]
        nb_models[aspect] = result["Naive Bayes"][0]
        records.extend(result[name][1] for name in ("SVM", "Naive Bayes"))

    root = Path(__file__).resolve().parent
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    joblib.dump((vec, svm_models, nb_models), root / "models" / "baseline.joblib")
    results = pd.DataFrame(records)
    results.to_csv(root / "results" / "baseline_results.csv", index=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
