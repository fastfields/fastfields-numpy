"""Spline-interpolation gather / scatter (pushpull) — numpy.

Channel-last, x-first coordinate convention. For a spatial rank ``D`` (1, 2 or
3), with ``B`` leading batch dims:

* ``inp``  : ``(*B, *inshape,  C)``  — the volume (spline coefficients)
* ``grid`` : ``(*B, *outshape, D)``  — sampling coordinates, in voxels
* ``out``  : ``(*B, *outshape, C)``  — the pulled samples (``pull``)

``inp`` and ``grid`` must share the same rank ``B + D + 1``; ``D`` is taken
from ``grid``'s trailing axis. ``push`` is the numerical adjoint of ``pull``.
"""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array, _as_spline

__all__ = ["pull", "push", "count", "grad"]


def _spatial_shape(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Normalise a spatial-size argument to a length-``ndim`` tuple."""
    if np.isscalar(shape):
        return (int(shape),) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


def pull(
    inp: np.ndarray,
    grid: np.ndarray,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Sample (pull) ``inp`` at the coordinates in ``grid``.

    Returns ``(*B, *outshape, C)`` with ``outshape`` and ``C`` taken from
    ``grid`` and ``inp``. ``extrapolate``: 1 always, 0 not past voxel centres,
    -1 not past edges.
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    out = np.zeros(grid.shape[:-1] + (inp.shape[-1],), dtype=inp.dtype)
    _ff.pull(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def push(
    inp: np.ndarray,
    grid: np.ndarray,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Splat (push) ``inp`` into a volume of spatial size ``shape``.

    The adjoint of :func:`pull`. ``inp``/``grid`` are
    ``(*B, *outshape, {C,D})``; the result is ``(*B, *shape, C)`` (``shape``
    gives the ``D`` spatial sizes of the target volume). Values are accumulated
    (the buffer is pre-zeroed here).
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial_shape(shape, ndim)
    out = np.zeros(
        grid.shape[:nbatch] + spatial + (inp.shape[-1],), dtype=inp.dtype
    )
    _ff.push(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def count(
    grid: np.ndarray,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> np.ndarray:
    """Splat ones into a volume of spatial size ``shape`` (push of all-ones).

    Returns ``(*B, *shape, 1)``.
    """
    grid = _as_float_array(grid, "grid")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial_shape(shape, ndim)
    out = np.zeros(grid.shape[:nbatch] + spatial + (1,), dtype=grid.dtype)
    _ff.count(
        out,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
    )
    return out


def grad(
    inp: np.ndarray,
    grid: np.ndarray,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> np.ndarray:
    """Sample the spatial gradients of ``inp`` at ``grid``.

    Returns ``(*B, *outshape, C, D)`` — one gradient vector of length ``D`` per
    sampled channel value. ``abs=True`` takes the absolute value per component.
    """
    inp = _as_float_array(inp, "inp")
    grid = _as_float_array(grid, "grid", dtype=inp.dtype)
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    out = np.zeros(grid.shape[:-1] + (inp.shape[-1], ndim), dtype=inp.dtype)
    _ff.grad(
        out,
        inp,
        grid,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        extrapolate=int(extrapolate),
        abs=bool(abs),
    )
    return out
