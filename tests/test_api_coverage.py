"""This wrapper must cover the binding, or say why not.

``fastfields.numpy`` wraps ``fastfields.dlpack``, which in turn binds every
public entry point of the C++ hub (enforced by
``fastfields-dlpack/tests/test_api_coverage.py``). So "which hub operations
can a numpy user reach?" reduces to "which ``fastfields.dlpack`` names does
this package wrap?" -- a question answerable here, without the C++ tree, by
comparing the two ``__all__`` lists.

An operation may be absent on purpose. Those live in :data:`DELIBERATE` with
the reason, and the package docstring repeats the list for users. The point of
the test is the *other* direction: an op that goes missing by accident -- a new
binding nobody wrapped, or one dropped in a refactor -- fails here instead of
being found by a user.

Spelling: the hub's accumulate primitives exist only in place
(``field_adddiag_``), and this wrapper adds an out-of-place spelling
(``field_adddiag``) that calls the same binding; conversely ``dt_euclidean``
exists only out of place and the wrapper adds ``dt_euclidean_``. Either
spelling counts as covering the binding -- except where the binding itself has
both (``sym_solve``/``sym_solve_``, ``{field,flow}_precond[_]``), which are
genuinely different operations and are matched exactly.
"""

from __future__ import annotations

import fastfields.dlpack as ffb
import pytest

import fastfields.numpy as ffn

#: Bindings this wrapper deliberately does not expose -> why.
DELIBERATE = {
    op: (
        "adjoint of a pushpull op, for differentiating through `grid`; there "
        "is no autograd on numpy, so it stays raw-layer only "
        "(fastfields.dlpack.%s)" % op
    )
    for op in (
        "pull_backward",
        "push_backward",
        "count_backward",
        "grad_backward",
    )
}

#: Names in ``fastfields.dlpack.__all__`` that are not operations.
NON_OPS = {"__version__", "backend"}


def _binding_ops() -> set[str]:
    return {n for n in ffb.__all__ if n not in NON_OPS}


def _covered(op: str) -> bool:
    """Does this wrapper expose ``op``, under either spelling?"""
    names = set(ffn.__all__)
    if op in names:
        return True
    # Alias only where the binding does not itself declare both spellings.
    alias = op[:-1] if op.endswith("_") else op + "_"
    if alias in _binding_ops():
        return False
    return alias in names


def test_no_accidental_gap():
    missing = sorted(
        op
        for op in _binding_ops()
        if not _covered(op) and op not in DELIBERATE
    )
    assert not missing, (
        "fastfields.dlpack operations this wrapper neither exposes nor "
        f"documents as deliberately absent: {missing}. Wrap them, or add "
        "them to DELIBERATE in this file (and to the 'What this wrapper "
        "deliberately does not expose' section of the package docstring) "
        "with the reason."
    )


def test_deliberate_gaps_are_really_absent():
    """A documented gap that got filled should be removed from the list."""
    stale = sorted(op for op in DELIBERATE if _covered(op))
    assert not stale, (
        f"listed as deliberately absent but now exposed: {stale}; drop them "
        "from DELIBERATE and from the package docstring."
    )


def test_deliberate_gaps_still_exist_in_the_binding():
    """Guard against a typo / a renamed binding silently emptying the list."""
    unknown = sorted(op for op in DELIBERATE if op not in _binding_ops())
    assert not unknown, (
        f"DELIBERATE names no longer in fastfields.dlpack.__all__: {unknown}"
    )


def test_docstring_lists_every_deliberate_gap():
    doc = ffn.__doc__ or ""
    for op in DELIBERATE:
        assert op in doc, (
            f"{op} is deliberately absent but the package docstring does not "
            "mention it; a gap the user cannot read about is just a gap"
        )


@pytest.mark.parametrize("name", sorted(ffn.__all__))
def test_dunder_all_resolves(name):
    assert hasattr(ffn, name), f"__all__ names {name!r}, which does not exist"
