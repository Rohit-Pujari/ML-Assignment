"""Shared TripAdvisor data loading and deterministic train/test split."""
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

# Edit these names here when the downloaded archive uses different suffixes.
RATINGS_FILE = "00040-00000001.dat"
TEXT_FILES = [
    "00040-00000002.dat",
    "00040-00000003.dat",
    "00040-00000004.dat",
]
ASPECTS = ["Overall", "Value", "Rooms", "Location", "Cleanliness", "Service"]
ALL_ASPECTS = [
    "Overall", "Value", "Rooms", "Location", "Cleanliness", "Front Desk",
    "Service", "Business Service",
]
TEXT_RE = re.compile(r"^(\d+),(.*?),(\d\d/\d\d/\d{4}),(.*)$")


def polarity(rating):
    """Map a star rating to negative=0, neutral=1, positive=2; preserve -1."""
    if rating == -1:
        return -1
    return 0 if rating <= 2 else (1 if rating == 3 else 2)


def load_data(root=None):
    """Return the fixed 48k/12k split after the notebook's exact cleaning."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    rows = []
    with (root / RATINGS_FILE).open(encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            rows.append([p[0], p[1]] + [int(x) for x in p[-8:]])
    ratings = pd.DataFrame(rows, columns=["Hotel", "User"] + ALL_ASPECTS)

    texts = []
    for filename in TEXT_FILES:
        with (root / filename).open(encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                match = TEXT_RE.match(line.rstrip("\n"))
                if match is None:
                    raise ValueError(f"Could not parse a row in {filename}: {line[:120]!r}")
                texts.append(match.groups())
    text = pd.DataFrame(texts, columns=["hotel", "user", "date", "text"])

    df = pd.concat([ratings, text[["date", "text"]]], axis=1)
    df = df[df["Overall"] > 0]
    df = df[df["text"].str.split().str.len() >= 20]
    # The supplied dataset replaces these already; keep train/demo behavior aligned.
    df["text"] = df["text"].str.replace(",", ";", regex=False)
    small = df.sample(60000, random_state=42)
    train, test = train_test_split(small, test_size=0.2, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)
