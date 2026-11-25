"""Linear controls sharing the same normalized observations and candidate labels."""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data import ACTIONS
from routing import (
    CONTINUES,
    HOLD,
    UNKNOWN,
    Output,
    decode_states,
    direct_output,
    state_output,
)


def vectorizer():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)


def global_text(row):
    head = f"registry {row.domains[0]} then {row.domains[1]} "
    return head + " || ".join(
        f"binding {s}: {line}" for s, line in zip(row.binding, row.lines)
    )


def fit_lr(features, labels):
    return LogisticRegression(C=4, max_iter=500).fit(features, labels)


def probs(model, features, num):
    scores = np.zeros((features.shape[0], num))
    scores[:, model.classes_] = model.predict_proba(features)
    return scores


def packet_models(rows, examples):
    vec = vectorizer()
    x = vec.fit_transform([global_text(r) for r in rows])
    direct = fit_lr(x, [ACTIONS.index(e.gold) for e in examples])
    targets = np.array([[k if k >= 0 else 2 for k in r.binding] for r in rows])
    statuses = np.array(
        [
            [e.states[k] if k >= 0 else UNKNOWN for k in r.binding]
            for e, r in zip(examples, rows)
        ]
    )
    heads = [fit_lr(x, labels) for labels in (*targets.T, *statuses.T)]

    def score(batch, method):
        values = vec.transform([global_text(r) for r in batch])
        if method.startswith("char_direct"):
            return direct_output(
                batch, probs(direct, values, 5), guarded=method != "char_direct_raw"
            )
        pred = [probs(h, values, 3) for h in heads]
        guesses = np.stack([p.argmax(1) for p in pred], axis=1)
        use_binding = method == "char_mentions_bound"
        states = np.full((len(batch), 2), UNKNOWN)
        actions = np.full(len(batch), HOLD)
        for i, r in enumerate(batch):
            slots = list(r.binding) if use_binding else guesses[i, :2].tolist()
            if r.complete and sorted(slots) == [0, 1]:
                for j, slot in enumerate(slots):
                    states[i, slot] = guesses[i, 2 + j]
                actions[i] = decode_states(r.domains, states[i], [0, 0])
        confidence = np.stack(
            [p.max(1) for p in (pred[2:] if use_binding else pred)]
        ).min(0)
        return Output(actions, confidence, states)

    return score


def candidate_models(rows, examples):
    vec = vectorizer().fit([line for r in rows for line in r.lines if line])

    def channels(batch, include_other=True):
        own = []
        other = []
        header = []
        for r in batch:
            for slot, domain in enumerate(r.domains):
                own.append(" ".join(s for s, k in zip(r.lines, r.binding) if k == slot))
                other.append(
                    " ".join(s for s, k in zip(r.lines, r.binding) if k != slot)
                )
                header.append(
                    [int(j == domain) for j in range(5)]
                    + [int(slot == 0), int(slot == 1)]
                )
        base = hstack([vec.transform(own), csr_matrix(header)], format="csr")
        return base, hstack(
            [base, vec.transform(other)], format="csr"
        ) if include_other else None

    labels = np.array([e.states for e in examples]).ravel()
    x, y = channels(rows)
    bound, global_model = fit_lr(x, labels), fit_lr(y, labels)

    def score(batch, method):
        own, all_features = channels(batch, method == "char_states_global")
        if method == "char_states_global":
            scores = probs(global_model, all_features, 3)
        else:
            scores = probs(bound, own, 3)
        return state_output(
            batch, scores.reshape(-1, 2, 3), exact=not method.endswith("wta")
        )

    return score
