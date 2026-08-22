# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import pytest
import torch

from serving.cache import SlotCache


def make(capacity=3, max_seq_len=8, n_layers=2, n_heads=2, head_dim=4):
    return SlotCache(n_layers, capacity, n_heads, max_seq_len, head_dim)


def kv(cache, width, fill):
    shape = (cache.capacity, 2, width, 4)
    return torch.full(shape, fill), torch.full(shape, -fill)


def test_a_fresh_cache_holds_nothing():
    cache = make()
    assert cache.lengths.tolist() == [0, 0, 0]
    assert not cache.key_mask().any()


def test_reserve_then_append_makes_the_new_tokens_visible():
    cache = make()
    cache.reserve(3)
    k, v = kv(cache, 3, 1.0)
    keys, values = cache.append(0, k, v)
    assert cache.lengths.tolist() == [3, 3, 3]
    assert cache.key_mask()[0].tolist() == [True] * 3 + [False] * 5
    assert torch.equal(keys[:, :, :3], k)
    assert torch.equal(values[:, :, :3], v)


def test_writes_land_after_what_is_already_stored():
    cache = make()
    cache.reserve(2)
    cache.append(0, *kv(cache, 2, 1.0))
    cache.reserve(1)
    keys, _ = cache.append(0, *kv(cache, 1, 5.0))
    assert cache.lengths.tolist() == [3, 3, 3]
    assert keys[0, 0, :3, 0].tolist() == [1.0, 1.0, 5.0]


def test_inactive_rows_neither_grow_nor_become_visible():
    """An idle slot is carried through the arithmetic but stays masked out."""
    cache = make()
    active = torch.tensor([True, False, True])
    cache.reserve(2, active)
    cache.append(0, *kv(cache, 2, 1.0))
    assert cache.lengths.tolist() == [2, 0, 2]
    assert not cache.key_mask()[1].any()


def test_an_idle_rows_stale_write_stays_out_of_reach():
    cache = make()
    cache.reserve(1, torch.tensor([True, False, False]))
    cache.append(0, *kv(cache, 1, 9.0))
    assert cache.lengths[1].item() == 0
    assert not cache.key_mask()[1].any()


def test_layers_do_not_share_storage():
    cache = make(n_layers=2)
    cache.reserve(1)
    cache.append(0, *kv(cache, 1, 1.0))
    cache.append(1, *kv(cache, 1, 2.0))
    assert cache.keys[0][0, 0, 0, 0].item() == 1.0
    assert cache.keys[1][0, 0, 0, 0].item() == 2.0


def test_every_layer_writes_at_the_same_position():
    cache = make(n_layers=2)
    cache.reserve(2)
    cache.append(0, *kv(cache, 2, 1.0))
    cache.append(1, *kv(cache, 2, 2.0))
    assert cache.lengths.tolist() == [2, 2, 2]


def test_reset_frees_a_slot_without_touching_its_neighbours():
    cache = make()
    cache.reserve(4)
    cache.append(0, *kv(cache, 4, 1.0))
    cache.reset(1)
    assert cache.lengths.tolist() == [4, 0, 4]
    assert cache.key_mask()[0].sum().item() == 4
    assert cache.key_mask()[1].sum().item() == 0


def test_a_reused_slot_starts_from_zero():
    cache = make()
    cache.reserve(4)
    cache.append(0, *kv(cache, 4, 1.0))
    cache.reset(0)
    cache.reserve(1, torch.tensor([True, False, False]))
    keys, _ = cache.append(0, *kv(cache, 1, 7.0))
    assert cache.lengths[0].item() == 1
    assert keys[0, 0, 0, 0].item() == 7.0


def test_running_past_the_window_is_refused():
    cache = make(max_seq_len=4)
    cache.reserve(4)
    cache.append(0, *kv(cache, 4, 1.0))
    with pytest.raises(ValueError, match="past the 4 token window"):
        cache.reserve(1)


def test_a_full_row_does_not_block_a_shorter_one():
    cache = make(max_seq_len=4)
    cache.reserve(4, torch.tensor([True, False, False]))
    cache.append(0, *kv(cache, 4, 1.0))
    cache.reserve(1, torch.tensor([False, True, True]))
    assert cache.lengths.tolist() == [4, 1, 1]


def test_each_row_writes_at_its_own_offset():
    """Rows sit at different lengths, so one shared write index would be wrong."""
    cache = make(capacity=2, max_seq_len=8)
    cache.reserve(3, torch.tensor([True, False]))
    cache.append(0, *kv(cache, 3, 1.0))
    assert cache.lengths.tolist() == [3, 0]
    cache.reserve(1)
    keys, _ = cache.append(0, *kv(cache, 1, 4.0))
    assert cache.lengths.tolist() == [4, 1]
    assert keys[0, 0, :4, 0].tolist() == [1.0, 1.0, 1.0, 4.0]
    assert keys[1, 0, 0, 0].item() == 4.0
    # Position 1 still holds what the inactive pass wrote, and the mask
    # is what keeps it out of the arithmetic.
    assert keys[1, 0, 1, 0].item() == 1.0
    assert cache.key_mask()[1].tolist() == [True] + [False] * 7


def test_appending_before_reserving_is_refused():
    cache = make()
    with pytest.raises(RuntimeError, match="reserve"):
        cache.append(0, *kv(cache, 1, 1.0))


def test_appending_a_different_width_than_reserved_is_refused():
    cache = make()
    cache.reserve(2)
    with pytest.raises(ValueError, match="reserved 2"):
        cache.append(0, *kv(cache, 3, 1.0))


def test_reserve_needs_a_positive_width():
    cache = make()
    with pytest.raises(ValueError, match="at least 1"):
        cache.reserve(0)
