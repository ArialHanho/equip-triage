"""Synthetic memo routing. This experiment never controls equipment."""
import argparse
import csv
import json
import platform
import re
import time
import unicodedata
from pathlib import Path
from statistics import median

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
LABELS = ("data", "equipment", "measurement", "setup")
ABSTAIN = "unknown"
KEYWORDS = {
    "measurement": ("치구", "접촉", "저항", "측정", "기준편", "프로브", "반복 시험"),
    "data": ("일부만", "로그", "호스트", "통신", "수집", "이벤트", "시각"),
    "setup": ("레시피", "설정", "조건", "허용", "상한", "하한"),
    "equipment": ("모터", "소음", "진동", "이송", "냉각", "팬", "발열", "구동축"),
}


def load():
    with (ROOT / "memo_corpus.tsv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    expected_splits = ("train", "validation", "test")
    data = {s: [] for s in expected_splits}
    groups = {}
    normalized = set()
    for row in rows:
        s, group, label, memo = (row[k] for k in ("split", "group", "label", "text"))
        if s not in data or label not in (*LABELS, ABSTAIN) or not group or not memo:
            raise ValueError("invalid row")
        if groups.setdefault(group, s) != s:
            raise ValueError(f"group across splits: {group}")
        key = re.sub(r"\W+", "", unicodedata.normalize("NFKC", memo)).casefold()
        if not key or key in normalized:
            raise ValueError(f"duplicate or empty memo: {memo}")
        normalized.add(key)
        data[s].append(row)
    for s, group in data.items():
        wanted = set(LABELS if s == "train" else (*LABELS, ABSTAIN))
        if {r["label"] for r in group} != wanted:
            raise ValueError(f"unexpected label coverage in {s}")
    return data


def texts(rows):
    return [r["text"] for r in rows]


def rule(memos):
    return np.array([[sum(word in memo for word in KEYWORDS[label])
                      for label in LABELS] for memo in memos], dtype=float)


def classify(scores, cutoff):
    winners = scores.argmax(axis=1)
    strong = scores.max(axis=1) >= cutoff
    unique = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) == 1
    return [LABELS[i] if ok and single else ABSTAIN
            for i, ok, single in zip(winners, strong, unique, strict=True)]


def summary(rows, pred):
    truth = [r["label"] for r in rows]
    known = [i for i, y in enumerate(truth) if y != ABSTAIN]
    unknown = [i for i, y in enumerate(truth) if y == ABSTAIN]
    accepted = [i for i, p in enumerate(pred) if p != ABSTAIN]
    return {
        "macro_f1_known_including_abstention": round(float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "known_recall": dict(zip(LABELS, [round(float(x), 4) for x in recall_score(truth, pred, labels=LABELS, average=None, zero_division=0)])),
        "known_routed": [sum(i in accepted for i in known), len(known)],
        "unknown_flagged": [sum(pred[i] == ABSTAIN for i in unknown), len(unknown)],
        "accepted_errors": [sum(pred[i] != truth[i] for i in accepted), len(accepted)],
    }


def evaluate(predict, data, candidates, latency):
    val = data["validation"]
    val_scores = predict(texts(val))
    ranks = []
    for cutoff in candidates:
        stats = summary(val, classify(val_scores, cutoff))
        # Break utility ties by macro-F1, then the more conservative cutoff.
        n_wrong = stats["accepted_errors"][0]
        n_correct = stats["accepted_errors"][1] - n_wrong
        # Pilot decision cost: an incorrect action weighs twice a correct routing.
        ranks.append((n_correct - 2 * n_wrong, stats["macro_f1_known_including_abstention"], cutoff))
    _, _, cutoff = max(ranks)
    test = data["test"]
    predictions = classify(predict(texts(test)), cutoff)
    val_result = summary(val, classify(val_scores, cutoff))
    result = {
        "validation_cutoff": cutoff,
        "validation": val_result,
        "test": summary(test, predictions),
        "test_errors_or_abstentions": [
            {"group": r["group"], "text": r["text"], "expected": r["label"], "predicted": y}
            for r, y in zip(test, predictions) if r["label"] != y or y == ABSTAIN],
    }
    if latency:
        predict(texts(test[:1]))
        samples = []
        for _ in range(3):
            for row in test:
                start = time.perf_counter()
                predict([row["text"]])
                samples.append((time.perf_counter() - start) * 1000)
        result["cpu_single_memo_latency_ms"] = {
            "n": len(samples), "p50": round(median(samples), 4), "p95": round(float(np.percentile(samples, 95)), 4)
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = load()
    tr = data["train"]
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
                          LogisticRegression(C=5.0, random_state=args.seed, max_iter=1000))
    model.fit(texts(tr), [r["label"] for r in tr])
    assert list(model.classes_) == list(LABELS)
    result = {
        "data": "synthetic, hand-authored; not actual line data",
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "splits": {s: {"rows": len(rs), "groups": sorted({r["group"] for r in rs})} for s, rs in data.items()},
        "rule": evaluate(rule, data, [1.0, 2.0, 3.0, 4.0], args.latency),
        "tfidf_logistic": evaluate(model.predict_proba, data, [i / 100 for i in range(25, 101, 5)], args.latency),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
