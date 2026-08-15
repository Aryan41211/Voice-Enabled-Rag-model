import numpy as np

from benchmarks.metrics import mrr, percentile, recall_at_k


def test_percentile_matches_numpy():
    data = [float(x) for x in range(100)]
    for p in (0, 25, 50, 70, 100):
        assert percentile(data, p) == float(np.percentile(data, p))


def test_percentile_empty_raises():
    try:
        percentile([], 50)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_recall_at_k_partial_and_full():
    gold = {"a", "b"}
    assert recall_at_k(["a", "x", "y"], gold, 3) == 0.5
    assert recall_at_k(["a", "b", "x"], gold, 3) == 1.0
    assert recall_at_k(["x", "y", "z"], gold, 3) == 0.0


def test_recall_at_k_empty_gold():
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_mrr_first_and_late():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["c", "a", "b"], {"b"}) == 1.0 / 3
    assert mrr(["x", "y"], {"a"}) == 0.0
