# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import pytest

from serving.scheduler import Request, Scheduler


def make(capacity=2, max_seq_len=16):
    return Scheduler(model=None, capacity=capacity, max_seq_len=max_seq_len)


def test_admission_is_bounded_by_capacity():
    sched = make(capacity=2)
    for _ in range(5):
        sched.submit(Request(prompt=[1, 2]))
    assert len(sched.admit()) == 2
    assert len(sched.running) == 2
    assert len(sched.waiting) == 3


def test_a_retired_sequence_frees_its_slot_for_the_queue():
    """The point of continuous batching: a slot refills mid-flight."""
    sched = make(capacity=1)
    sched.submit(Request(prompt=[1]))
    sched.submit(Request(prompt=[2]))
    first = sched.admit()[0]
    assert not sched.admit()
    sched.retire(first)
    second = sched.admit()
    assert len(second) == 1
    assert second[0].slot == first.slot
    assert second[0].request.prompt == [2]


def test_slots_are_never_double_booked():
    sched = make(capacity=3)
    for _ in range(3):
        sched.submit(Request(prompt=[1]))
    slots = [s.slot for s in sched.admit()]
    assert sorted(slots) == [0, 1, 2]


def test_generated_excludes_the_prompt():
    sched = make()
    sched.submit(Request(prompt=[7, 8, 9]))
    seq = sched.admit()[0]
    assert seq.generated == []
    seq.tokens.append(42)
    assert seq.generated == [42]
    assert seq.position == 3


def test_finish_on_token_budget():
    sched = make()
    sched.submit(Request(prompt=[1], max_new_tokens=2))
    seq = sched.admit()[0]
    seq.tokens.append(5)
    assert not sched._is_finished(seq)
    seq.tokens.append(6)
    assert sched._is_finished(seq)


def test_finish_on_stop_token():
    sched = make()
    sched.submit(Request(prompt=[1], max_new_tokens=99, stop_token=0))
    seq = sched.admit()[0]
    seq.tokens.append(4)
    assert not sched._is_finished(seq)
    seq.tokens.append(0)
    assert sched._is_finished(seq)


def test_finish_on_context_limit():
    sched = make(max_seq_len=8)
    sched.submit(Request(prompt=[1], max_new_tokens=99))
    seq = sched.admit()[0]
    seq.position = 8
    assert sched._is_finished(seq)


def test_a_prompt_that_cannot_fit_is_refused_at_submit():
    sched = make(max_seq_len=4)
    with pytest.raises(ValueError, match="does not fit"):
        sched.submit(Request(prompt=[1, 2, 3, 4]))


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        Scheduler(model=None, capacity=0, max_seq_len=8)


def test_pending_counts_both_queued_and_running():
    sched = make(capacity=1)
    sched.submit(Request(prompt=[1]))
    sched.submit(Request(prompt=[2]))
    assert sched.pending == 2
    sched.admit()
    assert sched.pending == 2
    sched.retire(sched.running[0])
    assert sched.pending == 1


def test_request_ids_are_unique():
    assert Request(prompt=[1]).id != Request(prompt=[1]).id
