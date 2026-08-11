"""Multi-column input rules: the column budget, hard-to-soft matching, and per-column splits."""

from __future__ import annotations

import numpy as np
import pytest

from olinda.data.reference import MAX_COLUMNS, check_column_budget, match_hard_columns
from olinda.data.split import split_reference_to_indices


# ── column budget ────────────────────────────────────────────────────────────


def test_budget_allows_up_to_the_cap():
  check_column_budget([f"c{i}" for i in range(MAX_COLUMNS)])  # must not raise


def test_budget_rejects_more_than_the_cap():
  with pytest.raises(ValueError, match=rf"{MAX_COLUMNS + 1} value columns"):
    check_column_budget([f"c{i}" for i in range(MAX_COLUMNS + 1)])


# ── hard → soft column matching ──────────────────────────────────────────────


def test_exact_match():
  assert match_hard_columns(["tox", "sol"], ["tox"]) == {"tox": "tox"}


def test_suffix_match_is_the_real_world_case():
  """Teacher outputs carry a suffix the experimental file does not."""
  soft = ["abaumannii_inhibition_probability", "ecoli_inhibition_probability"]
  assert match_hard_columns(soft, ["abaumannii_inhibition"]) == {
    "abaumannii_inhibition": "abaumannii_inhibition_probability"
  }


def test_hyphen_separator_also_matches():
  assert match_hard_columns(["tox-score"], ["tox"]) == {"tox": "tox-score"}


def test_exact_match_wins_over_suffix():
  mapping = match_hard_columns(["tox", "tox_probability"], ["tox"])
  assert mapping == {"tox": "tox"}


def test_ambiguous_prefix_raises_and_names_candidates():
  soft = ["tb_active_probability", "tb_inactive_probability"]
  with pytest.raises(ValueError) as e:
    match_hard_columns(soft, ["tb"])
  assert "ambiguous" in str(e.value)
  assert "tb_active_probability" in str(e.value)


def test_separator_prevents_false_prefix_match():
  """'tox' must NOT silently match 'toxicity_probability'."""
  with pytest.raises(ValueError, match="matches no soft column"):
    match_hard_columns(["toxicity_probability"], ["tox"])


def test_unmatched_hard_column_raises():
  with pytest.raises(ValueError, match="matches no soft column"):
    match_hard_columns(["a_probability"], ["zzz"])


def test_two_hard_columns_claiming_one_soft_column_raises():
  """Nested prefixes can both resolve to the same target — that must not silently pick one."""
  with pytest.raises(ValueError, match="claimed by both"):
    match_hard_columns(["a_b_probability"], ["a", "a_b"])


def test_soft_only_columns_are_simply_absent():
  """A soft column with no hard counterpart is not an error — it stays soft-only."""
  mapping = match_hard_columns(["a_probability", "b_probability"], ["a"])
  assert mapping == {"a": "a_probability"}
  assert "b_probability" not in mapping.values()


# ── per-column value-stratified split ────────────────────────────────────────


def test_split_is_per_column_and_respects_each_column_nan_mask():
  """Two columns with different missing rows must get different, self-consistent partitions."""
  n = 2000
  rng = np.random.default_rng(0)
  a = rng.random(n).astype(np.float32)
  b = rng.random(n).astype(np.float32)
  a[:200] = np.nan  # different holes per column
  b[-300:] = np.nan

  ta, va, ia = split_reference_to_indices(a, val_frac=0.1)
  tb, vb, ib = split_reference_to_indices(b, val_frac=0.1)

  for idx_t, idx_v, y, info in ((ta, va, a, ia), (tb, vb, b, ib)):
    both = np.concatenate([idx_t, idx_v])
    assert np.isfinite(y[both]).all()  # never trains on a missing value
    assert set(both.tolist()) == set(np.where(np.isfinite(y))[0].tolist())
    assert set(idx_t.tolist()).isdisjoint(idx_v.tolist())
    assert info["n_dropped"] == int((~np.isfinite(y)).sum())

  assert set(ta.tolist()) != set(tb.tolist())  # genuinely independent splits


def test_split_validation_spans_the_value_range():
  """Value-stratified, not random: validation must cover low and high values alike."""
  n = 5000
  y = np.linspace(0.0, 1.0, n).astype(np.float32)
  _, val_idx, _ = split_reference_to_indices(y, val_frac=0.1)
  vals = y[val_idx]
  assert vals.min() < 0.02 and vals.max() > 0.98
  # evenly spread across deciles, which a random split would not guarantee
  counts, _ = np.histogram(vals, bins=10, range=(0.0, 1.0))
  assert counts.min() > 0


def test_split_is_deterministic():
  y = np.random.default_rng(3).random(1000).astype(np.float32)
  t1, v1, _ = split_reference_to_indices(y, seed=42)
  t2, v2, _ = split_reference_to_indices(y, seed=42)
  assert np.array_equal(t1, t2) and np.array_equal(v1, v2)
