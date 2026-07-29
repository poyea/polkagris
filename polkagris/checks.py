# Copyright (c) 2026 John Law
# SPDX-License-Identifier: MIT

"""Runner for the per-component `checks.py` scripts.

Each check is a no-argument function that measures something and returns a
one-line result. Raise `Pending` for a property not established yet, `Skip`
when the platform cannot measure it. An `AssertionError` is a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


class Pending(Exception):
    """Not established yet."""


class Skip(Exception):
    """This machine cannot measure it."""


def run(title: str, checks: Sequence[Callable[[], str]]) -> int:
    print(title)
    failed = 0
    for check in checks:
        label = check.__name__.replace("_", " ")
        try:
            note = check()
        except Pending as exc:
            print(f"  pending  {label}: {exc}")
        except Skip as exc:
            print(f"  skip     {label}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL     {label}: {exc}")
        else:
            print(f"  ok       {label}: {note}")
    return 1 if failed else 0
