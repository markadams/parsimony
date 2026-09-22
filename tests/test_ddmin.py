"""ddmin against a synthetic oracle with a known answer.

Testing the search independently of any floating-point emulation is the whole
point of reimplementing it: with an oracle we know the minimal set exactly, so a
regression in the search shows up here rather than as a mysteriously worse
precision assignment three layers up.
"""

from __future__ import annotations

import string

import pytest

from parsimony import ddmin


def oracle(required):
    """Predicate that holds exactly when every element of `required` is present."""
    req = set(required)
    calls = []

    def holds(subset):
        calls.append(len(subset))
        return req.issubset(subset)

    holds.calls = calls
    return holds


def test_finds_exact_minimal_set():
    items = list(string.ascii_lowercase[:16])
    required = {"c", "f", "i"}
    holds = oracle(required)
    assert set(ddmin(items, holds)) == required


def test_result_is_one_minimal():
    items = list(string.ascii_lowercase[:12])
    holds = oracle({"b", "j"})
    found = ddmin(items, holds)
    for x in found:
        assert not holds([y for y in found if y != x])


def test_empty_when_nothing_required():
    holds = oracle(set())
    assert ddmin(list("abcdef"), holds) == []


def test_everything_required():
    items = list("abcde")
    holds = oracle(set(items))
    assert set(ddmin(items, holds)) == set(items)


def test_handles_empty_input():
    assert ddmin([], lambda s: False) == []


@pytest.mark.parametrize("n,k", [(16, 1), (32, 3), (64, 2), (64, 8)])
def test_cheaper_than_exhaustive(n, k):
    """ddmin must stay near its O(n^2) bound, not degenerate to brute force."""
    items = [f"v{i}" for i in range(n)]
    required = {items[i] for i in range(0, n, max(n // k, 1))[:k]}
    holds = oracle(required)
    assert set(ddmin(items, holds)) == required
    assert len(holds.calls) < n * n


def test_predicate_sees_only_subsets():
    items = list("abcdefgh")
    seen = []

    def holds(subset):
        seen.append(set(subset))
        return "d" in subset

    ddmin(items, holds)
    assert all(s <= set(items) for s in seen)
