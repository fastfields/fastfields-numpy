"""Shared helpers: argument normalisation, validation and batch-broadcast.

These helpers are used across the distance-transform, symmetric-matrix and
resampling wrappers. numpy is a hard dependency of this package, so (unlike the
cupy/torch backends) nothing is imported lazily here.
"""

from __future__ import annotations

from typing import Any

# `order`/`bound` normalisation is centralised in fastfields.helpers so
# every backend shares one implementation; re-export under the private
# names the numpy modules (_dt, _resample) already import.
from fastfields.helpers import as_bound as _as_bound  # noqa: F401
from fastfields.helpers import as_spline as _as_spline  # noqa: F401

import numpy as np

# The bindings only implement float32/float64 kernels.
_FLOAT_DTYPES = (np.float32, np.float64)


def _as_float_array(
    x: Any,
    name: str,
    *,
    dtype: Any = None,
    copy: bool = False,
) -> np.ndarray:
    """Return a float32/float64 numpy array for ``x``.

    Read-only *inputs* are **not** forced contiguous: the underlying C++
    library is fully stride-aware (it consumes DLPack strides directly), so a
    non-contiguous input is passed zero-copy with its native strides. Only two
    situations trigger a copy: ``copy=True`` (a functional in-place op needs a
    private, C-contiguous writable buffer) and a read-only source (numpy
    cannot export a read-only array through DLPack).

    Parameters
    ----------
    x : array_like
        Input to convert (numpy array, python scalar/sequence, ...).
    name : str
        Argument name used in error messages.
    dtype : numpy dtype, optional
        If given, cast ``x`` to this dtype (must be float32/float64). Otherwise
        integers / python floats are promoted to float64.
    copy : bool, default=False
        Force a fresh, C-contiguous writable buffer (for in-place bindings).

    Returns
    -------
    numpy.ndarray
        A float32/float64 array (a zero-copy view when possible).

    Raises
    ------
    TypeError
        If the resulting dtype is not float32/float64.
    """
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype not in _FLOAT_DTYPES:
        # Promote integers / python floats to float64 by default.
        arr = arr.astype(np.float64, copy=False)
    if arr.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"{name} must be float32 or float64, got dtype {arr.dtype}"
        )
    if copy:
        # Functional in-place ops write through this buffer, so it must be a
        # private, C-contiguous copy that does not alias the caller's input.
        return np.array(arr, dtype=arr.dtype, copy=True, order="C")
    if not arr.flags.writeable:
        # numpy refuses to export a read-only array through DLPack; copy those.
        arr = arr.copy()
    return arr


def _validate_inplace(x: Any, name: str = "x") -> np.ndarray:
    """Validate ``x`` for an in-place op and return it unchanged.

    The underlying library is fully stride-aware (it receives the array's
    DLPack strides and indexes accordingly), so an in-place call writes
    directly into ``x`` regardless of its memory layout -- no contiguous copy
    is made. We therefore accept *any* float32/float64 numpy array here, which
    keeps in-place ops zero-copy even for non-contiguous views (a core
    memory-efficiency feature of the library). We only reject non-arrays and
    wrong dtypes, where an in-place write could not land in the caller's buffer
    or would need a lossy cast.

    Parameters
    ----------
    x : numpy.ndarray
        Array to be written in place.
    name : str, optional
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        ``x`` unchanged.

    Raises
    ------
    TypeError
        If ``x`` is not a float32/float64 numpy array.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"inplace=True requires a numpy ndarray for {name}")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"inplace=True requires a float32/float64 array for {name}"
        )
    return x


# --------------------------------------------------------------------------- #
# zero-copy batch-dim broadcasting                                            #
# --------------------------------------------------------------------------- #
#
# The raw bindings require every input tensor of an op to share the same batch
# (leading) dimensions -- they do not broadcast. We normalise inputs to a
# common broadcast batch shape *without copying*: each input is re-strided to
# the target shape, using its real stride on axes that already match and a
# 0-stride on axes that are broadcast (size 1 -> N). The result shares memory
# with the source, so the big inputs are never duplicated; the stride-aware
# C++ library handles the 0-strides natively.
#
# We use ``as_strided`` on the (writable) source, not ``np.broadcast_to``,
# because ``broadcast_to`` returns a *read-only* view, which numpy then refuses
# to export through DLPack ("cannot export readonly array").


def _bcast_view(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return a zero-copy, DLPack-exportable broadcast of ``a`` to ``shape``.

    Parameters
    ----------
    a : numpy.ndarray
        Array to broadcast.
    shape : tuple of int
        Target shape (right-aligned against ``a``'s axes).

    Returns
    -------
    numpy.ndarray
        A 0-stride ``as_strided`` view sharing memory with ``a``.

    Raises
    ------
    ValueError
        If ``a`` cannot be broadcast to ``shape``.
    """
    shape = tuple(shape)
    if a.shape == shape:
        return a
    strides = [0] * len(shape)
    # right-align a's axes against the target shape
    for i in range(1, a.ndim + 1):
        dim = a.shape[-i]
        if dim == shape[-i]:
            strides[-i] = a.strides[-i]
        elif dim == 1:
            strides[-i] = 0
        else:
            raise ValueError(
                f"cannot broadcast array of shape {a.shape} to {shape}"
            )
    return np.lib.stride_tricks.as_strided(a, shape, tuple(strides))


def _broadcast_batch(
    specs: list[tuple[np.ndarray, int]],
) -> tuple[tuple[int, ...], list[np.ndarray]]:
    """Broadcast the batch dims of several arrays to a common shape.

    Parameters
    ----------
    specs : list of (numpy.ndarray, int)
        ``(array, n_core)`` pairs, where ``n_core`` is the number of trailing
        (core) axes that must be left untouched.

    Returns
    -------
    batch_shape : tuple of int
        The common broadcast batch shape.
    views : list of numpy.ndarray
        Each input broadcast (zero-copy) to ``batch_shape + its core dims``.
    """
    batch = np.broadcast_shapes(*[a.shape[: a.ndim - nc] for a, nc in specs])
    views = [_bcast_view(a, batch + a.shape[a.ndim - nc :]) for a, nc in specs]
    return batch, views
