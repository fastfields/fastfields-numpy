"""Compact-symmetric linear-algebra wrappers (numpy).

A compact-symmetric matrix of channel size ``C`` is stored along the last axis
as ``C*(C+1)//2`` values: the diagonal first, then the rows of the upper
triangle. ``mat`` therefore has trailing dim ``C*(C+1)//2`` and vectors have
trailing dim ``C``.
"""

from __future__ import annotations

import math

import fastfields.dlpack as _ff

import numpy as np

from ._util import (
    _as_float_array,
    _bcast_view,
    _broadcast_batch,
    _validate_inplace,
)

__all__ = [
    "sym_channels_from_packed",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_invert",
]


def sym_channels_from_packed(packed_len: int) -> int:
    """Return the channel count ``C`` with ``C*(C+1)/2 == packed_len``.

    Parameters
    ----------
    packed_len : int
        Length of the packed (compact-symmetric) trailing dimension.

    Returns
    -------
    int
        The number of channels ``C``.

    Raises
    ------
    ValueError
        If ``packed_len`` is not a triangular number ``C*(C+1)/2``.
    """
    c = int((math.isqrt(8 * packed_len + 1) - 1) // 2)
    if c * (c + 1) // 2 != packed_len:
        raise ValueError(
            f"packed length {packed_len} is not a triangular number "
            "(expected C*(C+1)/2 for some integer C)"
        )
    return c


def _check_sym(
    mat: np.ndarray,
    vec: np.ndarray,
    matname: str = "mat",
    vecname: str = "vec",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate dtypes/channels of a packed matrix + vector (no batch check).

    Batch dims are broadcast later, so we only enforce the channel relation
    ``vec.shape[-1] == C``.

    Parameters
    ----------
    mat : numpy.ndarray
        Packed compact-symmetric matrix, trailing dim ``C*(C+1)//2``.
    vec : numpy.ndarray
        Vector, trailing dim ``C``.
    matname, vecname : str, optional
        Argument names used in error messages.

    Returns
    -------
    mat : numpy.ndarray
        Float matrix (native strides preserved).
    vec : numpy.ndarray
        Float vector cast to ``mat``'s dtype.
    c : int
        The channel count ``C``.

    Raises
    ------
    ValueError
        If ``vec``'s channel count does not match the matrix packing.
    """
    mat = _as_float_array(mat, matname)
    vec = _as_float_array(vec, vecname, dtype=mat.dtype)
    c = sym_channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"{vecname} has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    return mat, vec, c


def sym_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Compute ``H @ vec`` for a compact-symmetric packed matrix ``H``.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrix, trailing dim ``C*(C+1)/2`` (diagonal first,
        then upper rows).
    vec : numpy.ndarray
        Vector, trailing dim ``C``.

    Returns
    -------
    numpy.ndarray
        The product, shaped ``broadcast_batch + (C,)``. The batch (leading)
        dims of ``mat`` and ``vec`` are broadcast together.
    """
    mat, vec, c = _check_sym(mat, vec)
    batch, (mat_b, vec_b) = _broadcast_batch([(mat, 1), (vec, 1)])
    out = np.empty(batch + (c,), dtype=mat.dtype)
    _ff.sym_matvec(out, mat_b, vec_b)
    return out


def _addsub_matvec_(
    out0: np.ndarray,
    mat: np.ndarray,
    vec: np.ndarray,
    binding,
) -> np.ndarray:
    """Shared in-place ``out0 (+|-)= H @ vec`` implementation.

    ``out0`` is mutated in place and returned. It fixes the batch (leading)
    shape; ``mat`` and ``vec`` are broadcast (zero-copy) to that batch shape.
    The trailing-``_`` contract requires the caller's ``out0`` buffer to be
    updated, so we write through it via DLPack and never materialise a private
    copy (matching the cupy backend).
    """
    # out0 fixes the output buffer: validate it in place (no copy) so the
    # binding writes through the caller's array.
    out0 = _validate_inplace(out0, "out0")
    # mat/vec must match out0's dtype (its buffer cannot be re-typed in place).
    mat = _as_float_array(mat, "mat", dtype=out0.dtype)
    vec = _as_float_array(vec, "vec", dtype=out0.dtype)
    c = sym_channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"vec has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    if out0.shape[-1] != c:
        raise ValueError("out0 must have the same channel count as vec")
    # out0 fixes the batch; broadcast mat/vec onto it (zero-copy views).
    batch = out0.shape[:-1]
    mat_b = _bcast_view(mat, batch + (mat.shape[-1],))
    vec_b = _bcast_view(vec, batch + (c,))
    binding(out0, mat_b, vec_b)
    return out0


def sym_addmatvec_(
    out0: np.ndarray, mat: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """Accumulate ``out0 += H @ vec`` **in place**; returns ``out0``.

    The trailing ``_`` denotes an in-place op: the caller's ``out0`` array is
    mutated directly (written through its DLPack buffer) and also returned for
    convenience.

    Parameters
    ----------
    out0 : numpy.ndarray
        Accumulator, trailing dim ``C``, mutated in place. Must be a
        float32/float64 array; it fixes the batch (leading) shape.
    mat : numpy.ndarray
        Compact-symmetric matrix, trailing dim ``C*(C+1)/2``. Broadcast to
        ``out0``'s batch shape.
    vec : numpy.ndarray
        Vector, trailing dim ``C``. Broadcast to ``out0``'s batch shape.

    Returns
    -------
    numpy.ndarray
        ``out0`` (the same array object), now holding ``out0 + H @ vec``.

    Raises
    ------
    TypeError
        If ``out0`` is not a float32/float64 numpy array.
    ValueError
        If ``out0``'s channel count does not match ``vec``, or ``mat``/``vec``
        cannot be broadcast to ``out0``'s batch shape.
    """
    return _addsub_matvec_(out0, mat, vec, _ff.sym_addmatvec_)


def sym_submatvec_(
    out0: np.ndarray, mat: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """Accumulate ``out0 -= H @ vec`` **in place**; returns ``out0``.

    The trailing ``_`` denotes an in-place op: the caller's ``out0`` array is
    mutated directly (written through its DLPack buffer) and also returned for
    convenience.

    Parameters
    ----------
    out0 : numpy.ndarray
        Accumulator, trailing dim ``C``, mutated in place. Must be a
        float32/float64 array; it fixes the batch (leading) shape.
    mat : numpy.ndarray
        Compact-symmetric matrix, trailing dim ``C*(C+1)/2``. Broadcast to
        ``out0``'s batch shape.
    vec : numpy.ndarray
        Vector, trailing dim ``C``. Broadcast to ``out0``'s batch shape.

    Returns
    -------
    numpy.ndarray
        ``out0`` (the same array object), now holding ``out0 - H @ vec``.

    Raises
    ------
    TypeError
        If ``out0`` is not a float32/float64 numpy array.
    ValueError
        If ``out0``'s channel count does not match ``vec``, or ``mat``/``vec``
        cannot be broadcast to ``out0``'s batch shape.
    """
    return _addsub_matvec_(out0, mat, vec, _ff.sym_submatvec_)


def sym_matvec_backward(grad: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Backward of :func:`sym_matvec`: gradient w.r.t. the packed matrix.

    Parameters
    ----------
    grad : numpy.ndarray
        Upstream gradient, trailing dim ``C``.
    vec : numpy.ndarray
        The vector from the forward pass, trailing dim ``C``.

    Returns
    -------
    numpy.ndarray
        Gradient in packed form, trailing dim ``C*(C+1)/2``. Batch dims of
        ``grad`` and ``vec`` are broadcast together.

    Raises
    ------
    ValueError
        If ``grad`` and ``vec`` have different channel counts.
    """
    grad = _as_float_array(grad, "grad")
    vec = _as_float_array(vec, "vec", dtype=grad.dtype)
    if grad.shape[-1] != vec.shape[-1]:
        raise ValueError("grad and vec must have the same channel count")
    c = grad.shape[-1]
    packed = c * (c + 1) // 2
    batch, (grad_b, vec_b) = _broadcast_batch([(grad, 1), (vec, 1)])
    out = np.empty(batch + (packed,), dtype=grad.dtype)
    _ff.sym_matvec_backward(out, grad_b, vec_b)
    return out


def sym_solve(
    mat: np.ndarray, vec: np.ndarray, weight: np.ndarray | None = None
) -> np.ndarray:
    """Solve ``(H + diag(weight)) @ x = vec`` for ``x``.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrix, trailing dim ``C*(C+1)/2``.
    vec : numpy.ndarray
        Right-hand side, trailing dim ``C``.
    weight : numpy.ndarray, optional
        Diagonal regulariser added to ``H``, trailing dim ``C``.

    Returns
    -------
    numpy.ndarray
        The solution ``x``, shaped ``broadcast_batch + (C,)``. Batch dims of
        ``mat``/``vec``/``weight`` are broadcast together.

    Raises
    ------
    ValueError
        If ``weight``'s channel count does not match ``vec``.
    """
    mat, vec, c = _check_sym(mat, vec)
    if weight is None:
        batch, (mat_b, vec_b) = _broadcast_batch([(mat, 1), (vec, 1)])
        out = np.empty(batch + (c,), dtype=mat.dtype)
        _ff.sym_solve(out, mat_b, vec_b)
    else:
        w = _as_float_array(weight, "weight", dtype=mat.dtype)
        if w.shape[-1] != c:
            raise ValueError("weight must have the same channel count as vec")
        batch, (mat_b, vec_b, w_b) = _broadcast_batch(
            [(mat, 1), (vec, 1), (w, 1)]
        )
        out = np.empty(batch + (c,), dtype=mat.dtype)
        _ff.sym_solve(out, mat_b, vec_b, w_b)
    return out


def sym_invert(mat: np.ndarray) -> np.ndarray:
    """Invert a compact-symmetric packed matrix; result is also packed.

    Parameters
    ----------
    mat : numpy.ndarray
        Compact-symmetric matrix, trailing dim ``C*(C+1)/2``.

    Returns
    -------
    numpy.ndarray
        The packed inverse, same shape as ``mat``.
    """
    mat = _as_float_array(mat, "mat")
    # Output must be a real C-contiguous buffer even if ``mat`` is strided.
    out = np.empty(mat.shape, dtype=mat.dtype)
    _ff.sym_invert(out, mat)
    return out
