"""Interactive demo for the baseline and multi-task DistilBERT models."""
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from data import ASPECTS, load_data, polarity
from train_bert import LABELS, MultiAspectDistilBert, MAX_LENGTH

ROOT = Path(__file__).resolve().parent
COLORS = {"negative": "#dc2626", "neutral": "#6b7280", "positive": "#16a34a"}
PALE_COLORS = {"negative": "#fee2e2", "neutral": "#e5e7eb", "positive": "#dcfce7"}


@st.cache_resource
def load_models():
    baseline_path = ROOT / "models" / "baseline.joblib"
    baseline = joblib.load(baseline_path) if baseline_path.exists() else None
    bert_dir = ROOT / "models" / "bert"
    bert, tokenizer = None, None
    if (bert_dir / "model.pt").exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_dir = bert_dir if (bert_dir / "config.json").exists() else "distilbert-base-uncased"
        bert = MultiAspectDistilBert(
            model_name=str(config_dir), pretrained=config_dir != bert_dir
        ).to(device)
        state = torch.load(bert_dir / "model.pt", map_location=device, weights_only=True)
        bert.load_state_dict(state)
        bert.eval()
        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    return baseline, bert, tokenizer


def predict_all(text, resources):
    baseline, bert, tokenizer = resources
    text = text.replace(",", ";")
    outputs = {a: {} for a in ASPECTS}
    if baseline:
        vec, svms, nbs = baseline
        x = vec.transform([text])
        for aspect in ASPECTS:
            outputs[aspect]["SVM"] = (int(svms[aspect].predict(x)[0]), None)
            outputs[aspect]["Naive Bayes"] = (int(nbs[aspect].predict(x)[0]), None)
    if bert:
        device = next(bert.parameters()).device
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        # DistilBERT does not consume token_type_ids; omit them when supplied by
        # a tokenizer configuration that includes the field.
        batch = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            probs = torch.softmax(bert(**batch), dim=-1)[0].cpu()
        for i, aspect in enumerate(ASPECTS):
            outputs[aspect]["DistilBERT"] = (int(probs[i].argmax()), probs[i].tolist())
    return outputs


def focus_map(text):
    lexicon = {
        "Cleanliness": {"clean", "dirty", "spotless", "stain", "smell", "dust"},
        "Location": {"location", "near", "walk", "station", "central", "noisy", "street"},
        "Value": {"price", "expensive", "cheap", "worth", "value"},
        "Rooms": {"room", "rooms", "bed", "bathroom", "view"},
        "Service": {"staff", "service", "reception", "helpful", "rude"},
        "Overall": set(),
    }
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [(sentence, [a for a, words in lexicon.items()
                        if words and set(re.findall(r"[a-z]+", sentence.lower())) & words])
            for sentence in sentences if sentence]


def show_highlight(text):
    from html import escape
    colors = {"Cleanliness": ("#166534", "#dcfce7"), "Location": ("#1e40af", "#dbeafe"),
              "Value": ("#854d0e", "#fef3c7"), "Rooms": ("#6b21a8", "#f3e8ff"),
              "Service": ("#9f1239", "#ffe4e6")}
    parts = []
    for sentence, aspects in focus_map(text):
        badges = " ".join(
            f'<span style="display:inline-block;color:{colors[a][0]};background:{colors[a][1]};'
            f'border:1px solid {colors[a][0]};border-radius:10px;padding:1px 7px;'
            f'font-size:0.78em;font-weight:600">{escape(a)}</span>'
            for a in aspects
        )
        if not badges:
            badges = '<span style="color:#374151;background:#f3f4f6;border:1px solid #9ca3af;border-radius:10px;padding:1px 7px;font-size:0.78em">No aspect keyword match</span>'
        parts.append(
            '<div style="margin:8px 0;padding:10px 12px;background:#ffffff;color:#111827;'
            'border:1px solid #d1d5db;border-radius:6px;line-height:1.55">'
            f'{escape(sentence)}<div style="margin-top:6px">{badges}</div></div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def explain(text, aspect, resources):
    baseline = resources[0]
    if not baseline:
        st.info("Train the baseline first to view feature contributions.")
        return
    vec, svms, nbs = baseline
    x = vec.transform([text.replace(",", ";")])
    terms = vec.get_feature_names_out()
    svm = svms[aspect]
    svm_class = int(svm.predict(x)[0])
    svm_scores = svm.coef_[svm_class] * x.toarray()[0]
    nb = nbs[aspect]
    nb_class = int(nb.predict(x)[0])
    nb_scores = nb.feature_log_prob_[nb_class] * x.toarray()[0]
    cols = st.columns(2)
    for col, title, scores in ((cols[0], "SVM: strongest supporting n-grams", svm_scores),
                               (cols[1], "Naive Bayes: log-probability contributors", nb_scores)):
        indices = x.indices
        ranked = sorted(indices, key=lambda j: scores[j], reverse=True)[:10]
        if ranked:
            frame = pd.DataFrame({"term": terms[ranked], "score": scores[ranked]}).sort_values("score")
            with col:
                st.caption(title)
                st.bar_chart(frame.set_index("term"))
        else:
            col.caption("No vocabulary terms matched.")


@st.cache_data
def get_test_reviews():
    return load_data()[1]


def main():
    st.set_page_config(page_title="Hotel Review Aspect Sentiment", layout="wide")
    st.title("Hotel review aspect sentiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.sidebar:
        st.subheader("Runtime")
        st.write(f"Device in use: **{device}**")
        st.write(f"GPU: **{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable'}**")
    with st.spinner("Loading saved models…"):
        resources = load_models()
    baseline, bert, _ = resources
    if not baseline:
        st.warning("Baseline artifacts missing. Run `python train_baseline.py`.")
    if not bert:
        st.info("DistilBERT artifacts missing. Run `python train_bert.py --smoke` or train it fully.")

    predict_tab, decide_tab, results_tab = st.tabs(["Predict", "How the models decide", "Results"])
    with predict_tab:
        if "review_text" not in st.session_state:
            st.session_state.review_text = ""
        if st.button("Random test review"):
            test = get_test_reviews()
            row = test.sample(n=1, random_state=None).iloc[0]
            st.session_state.review_text = row["text"]
            st.session_state.random_review_text = row["text"]
            st.session_state.true_ratings = {a: int(row[a]) for a in ASPECTS}
        st.caption("Paste your own review below, or load a random held-out review to compare with its true ratings.")
        text = st.text_area("Enter a review", key="review_text", height=160,
                            placeholder="Paste a hotel review here…")
        if text.strip():
            output = predict_all(text, resources)
            truth = (st.session_state.get("true_ratings", {})
                     if text == st.session_state.get("random_review_text") else {})
            rows = []
            for aspect in ASPECTS:
                row = {"Aspect": aspect}
                for model in ("SVM", "Naive Bayes", "DistilBERT"):
                    value = output[aspect].get(model)
                    row[model] = LABELS[value[0]] if value else "Unavailable"
                rating = truth.get(aspect, -1)
                row["True label"] = LABELS[polarity(rating)] if rating != -1 else "Not rated"
                rows.append(row)
            prediction_table = pd.DataFrame(rows)

            def prediction_style(row):
                styles = [""] * len(row)
                truth_label = row["True label"]
                for model_name in ("SVM", "Naive Bayes", "DistilBERT"):
                    label = row[model_name]
                    style = (f"background-color: {PALE_COLORS[label]}; color: #111827; font-weight: 600"
                             if label in PALE_COLORS else "color: #111827")
                    if truth_label in LABELS and label in LABELS and label != truth_label:
                        style += "; font-weight: bold; border: 2px solid #111827"
                    styles[row.index.get_loc(model_name)] = style
                return styles

            st.dataframe(prediction_table.style.apply(prediction_style, axis=1),
                         use_container_width=True, hide_index=True)
            if truth:
                st.caption("Prediction cells with a dark outline disagree with the true label.")
            for aspect in ASPECTS:
                info = output[aspect].get("DistilBERT")
                if info and info[1]:
                    st.caption(f"{aspect}: DistilBERT confidence")
                    st.progress(info[1][info[0]], text=f"{LABELS[info[0]]}: {info[1][info[0]]:.1%}")
            if truth:
                st.caption("Rows with a true label show the random test review's rated aspect labels.")
    with decide_tab:
        text = st.session_state.get("review_text", "").strip()
        if not text:
            st.info("Enter a review in the Predict tab first.")
        else:
            aspect = st.selectbox("Aspect to explain", ASPECTS)
            explain(text, aspect, resources)
            st.subheader("Sentence to aspect highlights")
            show_highlight(text)
    with results_tab:
        paths = [ROOT / "results" / "baseline_results.csv", ROOT / "results" / "bert_results.csv"]
        frames = [pd.read_csv(p) for p in paths if p.exists()]
        if frames:
            results = pd.concat(frames, ignore_index=True)
            st.dataframe(results, use_container_width=True, hide_index=True)
            chart_data = results.copy()
            chart_data["aspect_model"] = chart_data["aspect"] + " / " + chart_data["model"]
            chart = chart_data.set_index("aspect_model")[["accuracy", "macro_f1"]]
            st.bar_chart(chart)
            st.caption("Class imbalance can leave rare negative and neutral labels with much lower macro-F1 than accuracy.")
        else:
            st.info("No results yet. Run the training scripts to create results/*.csv.")


if __name__ == "__main__":
    main()
