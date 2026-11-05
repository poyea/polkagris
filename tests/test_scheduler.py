# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

import pytest
import torch

from capstone.model import ModelConfig, Transformer
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
    assert seq.position == 3
    seq.tokens.append(42)
    assert seq.generated == [42]
    assert seq.position == 4


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
    seq.tokens.extend(range(6))
    assert not sched._is_finished(seq)
    seq.tokens.append(9)
    assert sched._is_finished(seq)


def test_a_prompt_that_cannot_fit_is_refused_at_submit():
    sched = make(max_seq_len=4)
    with pytest.raises(ValueError, match="does not fit"):
        sched.submit(Request(prompt=[1, 2, 3, 4]))


def test_an_empty_prompt_is_refused_at_submit():
    sched = make()
    with pytest.raises(ValueError, match="empty"):
        sched.submit(Request(prompt=[]))


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


def test_retiring_twice_is_refused():
    """A second retire would hand the same slot to two sequences at once."""
    sched = make(capacity=1)
    sched.submit(Request(prompt=[1]))
    seq = sched.admit()[0]
    sched.retire(seq)
    with pytest.raises(ValueError, match="not running"):
        sched.retire(seq)
    assert sched._free == [0]


def test_retiring_something_never_admitted_is_refused():
    from serving.scheduler import Sequence

    sched = make(capacity=2)
    stray = Sequence(request=Request(prompt=[1]), slot=99, tokens=[1])
    with pytest.raises(ValueError, match="not running"):
        sched.retire(stray)
    assert sched._free == [0, 1]


def test_a_stale_copy_cannot_evict_the_live_sequence():
    from serving.scheduler import Sequence

    sched = make(capacity=1)
    sched.submit(Request(prompt=[1]))
    live = sched.admit()[0]
    stale = Sequence(request=live.request, slot=live.slot, tokens=[1])
    with pytest.raises(ValueError, match="not running"):
        sched.retire(stale)
    assert sched.running[live.slot] is live


def test_a_request_that_generates_nothing_is_refused():
    sched = make()
    with pytest.raises(ValueError, match="max_new_tokens"):
        sched.submit(Request(prompt=[1], max_new_tokens=0))


def tiny_model(seed=1859, vocab=64, seq_len=32):
    """A model whose logits depend on history.

    At the default init the head copies its last input token, so every
    sequence would continue identically and a cache that leaked between slots
    would still pass. Widening the weights makes attention decide the token.
    """
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, dim=32, n_layers=2, n_heads=4, seq_len=seq_len)
    model = Transformer(cfg).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.6)
    return model


def alone(model, prompt, budget):
    """The same continuation, one sequence at a time and no scheduler."""
    tokens = list(prompt)
    for _ in range(budget):
        with torch.no_grad():
            logits = model(torch.tensor([tokens]))
        tokens.append(int(logits[0, -1].argmax()))
    return tokens[len(prompt):]


PROMPTS = [[3, 9, 1], [7], [5, 2, 8, 4], [11, 12], [20, 21, 22, 23, 24]]


@pytest.mark.parametrize("capacity", [1, 2, 3, 5])
def test_batching_does_not_change_what_a_sequence_generates(capacity):
    """Capacity is a throughput knob, so it must not be visible in the output.

    Below 5 the slots refill mid-flight, which is where a stale key or a
    misaligned position would show up as one sequence reading another's cache.
    """
    model = tiny_model()
    budget = 7
    expected = [alone(model, prompt, budget) for prompt in PROMPTS]

    sched = Scheduler(model=model, capacity=capacity, max_seq_len=32)
    for prompt in PROMPTS:
        sched.submit(Request(prompt=prompt, max_new_tokens=budget))
    finished = sched.run()

    assert len(finished) == len(PROMPTS)
    by_id = {sequence.request.id: sequence.generated for sequence in finished}
    assert [by_id[key] for key in sorted(by_id)] == expected


def test_the_continuations_actually_differ():
    """Guards the test above: identical output would make it prove nothing."""
    model = tiny_model()
    generated = [tuple(alone(model, prompt, 7)) for prompt in PROMPTS]
    assert len(set(generated)) == len(PROMPTS)
    assert any(row[0] != prompt[-1] for prompt, row in zip(PROMPTS, generated))


def test_a_drained_scheduler_hands_every_slot_and_cache_row_back():
    model = tiny_model()
    sched = Scheduler(model=model, capacity=2, max_seq_len=32)
    for prompt in PROMPTS:
        sched.submit(Request(prompt=prompt, max_new_tokens=4))
    sched.run()
    assert sched.pending == 0
    assert sorted(sched._free) == [0, 1]
    # A reused slot must not see the previous tenant's keys.
    assert sched.cache.lengths.tolist() == [0, 0]
    assert not sched.cache.key_mask().any()


def test_step_returns_what_it_retired():
    model = tiny_model()
    sched = Scheduler(model=model, capacity=2, max_seq_len=32)
    sched.submit(Request(prompt=[3, 9], max_new_tokens=1))
    sched.submit(Request(prompt=[4, 1], max_new_tokens=3))
    sched.admit()
    # Prefill yields the first token, so a one-token budget finishes at once.
    retired = sched.step()
    assert [s.request.max_new_tokens for s in retired] == [1]
    assert retired[0].done and len(retired[0].generated) == 1
    assert len(sched.running) == 1


def test_prefill_advances_a_fresh_slot_by_exactly_one_token():
    model = tiny_model()
    sched = Scheduler(model=model, capacity=1, max_seq_len=32)
    sched.submit(Request(prompt=[5, 2, 8], max_new_tokens=9))
    sequence = sched.admit()[0]
    sched.step()
    assert sequence.prefilled
    assert len(sequence.generated) == 1
    # The cache holds the prompt; the token just sampled is written next pass.
    assert sched.cache.lengths[sequence.slot].item() == 3
    sched.step()
    assert len(sequence.generated) == 2
    assert sched.cache.lengths[sequence.slot].item() == 4


def test_generated_tokens_carry_no_graph():
    """Serving never backpropagates, and a retained graph would hold every
    activation for the life of the sequence."""
    model = tiny_model()
    sched = Scheduler(model=model, capacity=1, max_seq_len=32)
    sched.submit(Request(prompt=[3, 9], max_new_tokens=3))
    sched.run()
    assert all(isinstance(token, int) for token in sched.finished[0].tokens)


def test_run_restores_training_mode():
    model = tiny_model()
    model.train()
    sched = Scheduler(model=model, capacity=1, max_seq_len=32)
    sched.submit(Request(prompt=[3, 9], max_new_tokens=2))
    sched.run()
    assert model.training


def test_the_cache_follows_the_weights():
    model = tiny_model()
    sched = Scheduler(model=model, capacity=2, max_seq_len=32)
    sched.submit(Request(prompt=[3, 9], max_new_tokens=2))
    sched.run()
    assert sched.cache.keys[0].device == next(model.parameters()).device


def test_a_stop_token_retires_mid_budget():
    model = tiny_model()
    budget = 7
    expected = alone(model, [3, 9, 1], budget)
    sched = Scheduler(model=model, capacity=1, max_seq_len=32)
    sched.submit(Request(prompt=[3, 9, 1], max_new_tokens=budget, stop_token=expected[2]))
    finished = sched.run()
    assert finished[0].generated == expected[:3]


def test_temperature_zero_is_reproducible_and_sampling_is_not_pinned_to_it():
    model = tiny_model()

    def generate(temperature, seed):
        torch.manual_seed(seed)
        sched = Scheduler(model=model, capacity=1, max_seq_len=32)
        sched.submit(Request(prompt=[3, 9, 1], max_new_tokens=8, temperature=temperature))
        return sched.run()[0].generated

    assert generate(0.0, 1) == generate(0.0, 2)
    assert any(generate(2.0, seed) != generate(0.0, 1) for seed in (1, 2, 3))
