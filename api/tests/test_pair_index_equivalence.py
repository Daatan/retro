"""retro#1056: the indexed dedupe/clustering must be bit-identical to the all-pairs loops.

The references below are verbatim copies of the quadratic implementations that
shipped before retro#1056. Cluster ids feed the settlement independent-cluster
count (retro#372) and the stats feed retro#355 tuning, so "close" is not good
enough: every output, including the tie-break choice of representative and every
histogram band, must match exactly.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from forecast_api.clustering import (
    SIMILARITY_BANDS,
    ClusterStats,
    cluster_texts_with_stats,
    jaccard,
    shingles,
)
from forecast_api.dedup import _MIN_CLUSTER_TOKENS, _jaccard, _title_tokens, dedupe_syndicated


def _ref_dedupe(results, *, title_of, url_of, priority_of, threshold=0.8):
    reps, rep_tokens, url_to_cluster = [], [], {}
    for item in results:
        url = (url_of(item) or "").strip()
        idx = url_to_cluster.get(url) if url else None
        if idx is None:
            tokens = _title_tokens(title_of(item))
            if threshold > 0.0 and len(tokens) >= _MIN_CLUSTER_TOKENS:
                best = threshold
                for i, rt in enumerate(rep_tokens):
                    if len(rt) < _MIN_CLUSTER_TOKENS:
                        continue
                    sim = _jaccard(tokens, rt)
                    if sim >= best:
                        best, idx = sim, i
            if idx is None:
                idx = len(reps)
                reps.append(item)
                rep_tokens.append(tokens)
                if url:
                    url_to_cluster[url] = idx
                continue
        if priority_of(item) > priority_of(reps[idx]):
            reps[idx] = item
        if url:
            url_to_cluster.setdefault(url, idx)
    return reps


def _ref_cluster(texts, *, threshold, shingle_size=3):
    n = len(texts)
    sets = [shingles(t, shingle_size) if t else frozenset() for t in texts]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_j, pairs, hist = 0.0, 0, [0] * SIMILARITY_BANDS
    for i in range(n):
        if not sets[i]:
            continue
        for j in range(i + 1, n):
            if not sets[j]:
                continue
            s = jaccard(sets[i], sets[j])
            pairs += 1
            max_j = max(max_j, s)
            hist[min(int(s * SIMILARITY_BANDS), SIMILARITY_BANDS - 1)] += 1
            if s >= threshold:
                rx, ry = find(i), find(j)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    ids, out = {}, []
    for i in range(n):
        out.append(ids.setdefault(find(i), len(ids)))
    return tuple(out), ClusterStats(
        rows=n, textful=sum(1 for s in sets if s), pairs=pairs, max_jaccard=max_j,
        histogram=tuple(hist),
    )


# A small vocabulary with a skewed frequency so pools share common words (the case
# where a naive inverted index degrades) as well as rare ones.
_VOCAB = [f"w{i}" for i in range(60)] + ["iran", "us", "nuclear", "talks", "deal", "the", "of"]


def _text(rnd: random.Random) -> str:
    k = rnd.choice([0, 1, 2, 3, 4, 5, 8, 12, 20, 30])
    return " ".join(rnd.choices(_VOCAB, weights=[1] * 60 + [9] * 7, k=k))


def _pool(seed: int, n: int):
    rnd = random.Random(seed)
    rows: list[SimpleNamespace] = []
    for i in range(n):
        if rows and rnd.random() < 0.25:  # near-copy of an earlier row
            src = rnd.choice(rows).title.split()
            if src and rnd.random() < 0.5:
                src[rnd.randrange(len(src))] = rnd.choice(_VOCAB)
            title = " ".join(src)
        else:
            title = _text(rnd)
        url = rnd.choice(["", f"u{i}", f"u{rnd.randrange(max(i, 1))}"])
        rows.append(SimpleNamespace(title=title, url=url, cred=rnd.choice([0.5, 1.0, 1.0, 1.5])))
    return rows


_KW = dict(title_of=lambda r: r.title, url_of=lambda r: r.url, priority_of=lambda r: r.cred)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.8, 0.95, 1.0])
def test_dedupe_matches_quadratic_reference(seed, threshold):
    rows = _pool(seed, 120)
    got = dedupe_syndicated(rows, threshold=threshold, **_KW)
    want = _ref_dedupe(rows, threshold=threshold, **_KW)
    assert [id(r) for r in got] == [id(r) for r in want]


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.2, 0.4, 0.7, 1.0])
@pytest.mark.parametrize("shingle_size", [1, 3])
def test_clustering_matches_quadratic_reference(seed, threshold, shingle_size):
    rnd = random.Random(seed)
    texts = [r.title if rnd.random() > 0.1 else rnd.choice([None, ""]) for r in _pool(seed, 90)]
    got = cluster_texts_with_stats(texts, threshold=threshold, shingle_size=shingle_size)
    want = _ref_cluster(texts, threshold=threshold, shingle_size=shingle_size)
    assert got == want


def test_degenerate_inputs():
    for texts in ([], [None], ["a b c"], ["", "", ""], ["x y z", "x y z"]):
        assert cluster_texts_with_stats(texts, threshold=0.4) == _ref_cluster(texts, threshold=0.4)
    assert dedupe_syndicated([], **_KW) == []
