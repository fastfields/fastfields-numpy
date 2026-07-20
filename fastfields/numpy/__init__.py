"""fastfields.numpy: a friendly numpy interface over the ``fastfields.dlpack`` bindings.

The underlying bindings operate *in place* / write through pre-allocated output
arrays.  This package wraps them so every public function takes numpy arrays and
**returns** freshly allocated numpy arrays (never clobbering the caller's input
unless ``inplace=True`` is explicitly requested).

Batch (leading) dimensions are **broadcast**: the raw bindings require every
input tensor of an op to share the same batch dims (they do not broadcast), so
these wrappers normalise inputs to a common broadcast batch shape and allocate
outputs with that shape.  The broadcast is done **zero-copy**: each input is
re-strided to the target batch shape with 0-strides on the broadcast axes (a
view that shares memory with the original), which the stride-aware C++ library
consumes directly without any copy.  Returned outputs therefore carry the
broadcast batch shape.

Spline order and boundary condition arguments accept an ``int``, one of the
re-exported :class:`Spline` / :class:`Bound` enums, or a friendly string
(e.g. ``"cubic"``, ``"dct2"``).
"""

from __future__ import annotations

import math

import numpy as np

import fastfields.dlpack as _ff
from fastfields.dlpack import Bound, Spline

__all__ = [
    "Spline",
    "Bound",
    "euclidean_distance_transform",
    "l1_distance_transform",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec",
    "sym_submatvec",
    "sym_solve",
    "sym_invert",
    "resample",
    "restriction",
    "spline_coeff",
    "spline_distance_table",
    "spline_distance_brent",
    "spline_distance_gaussnewton",
    "mesh_distance",
    "sym_channels_from_packed",
]

_FLOAT_DTYPES = (np.float32, np.float64)

# --------------------------------------------------------------------------- #
# argument normalisation helpers                                              #
# --------------------------------------------------------------------------- #

_SPLINE_ALIASES = {
    "nearest": Spline.Nearest,
    "constant": Spline.Nearest,
    "linear": Spline.Linear,
    "quadratic": Spline.Quadratic,
    "cubic": Spline.Cubic,
    "fourth": Spline.FourthOrder,
    "fifth": Spline.FifthOrder,
    "sixth": Spline.SixthOrder,
    "seventh": Spline.SeventhOrder,
}

_BOUND_ALIASES = {
    "zero": Bound.Zero,
    "zeros": Bound.Zero,
    "replicate": Bound.Replicate,
    "nearest": Bound.Replicate,
    "dct1": Bound.DCT1,
    "dct2": Bound.DCT2,
    "neumann": Bound.DCT2,
    "reflect": Bound.DCT2,
    "dst1": Bound.DST1,
    "dst2": Bound.DST2,
    "dirichlet": Bound.DST2,
    "dft": Bound.DFT,
    "wrap": Bound.DFT,
    "circular": Bound.DFT,
    "nocheck": Bound.NoCheck,
}


def _as_spline(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _SPLINE_ALIASES:
            raise ValueError(
                f"unknown spline order {value!r}; "
                f"expected an int 0..7 or one of {sorted(_SPLINE_ALIASES)}"
            )
        return int(_SPLINE_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"spline order must be in 0..7, got {ivalue}")
    return ivalue


def _as_bound(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _BOUND_ALIASES:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"expected an int 0..7 or one of {sorted(_BOUND_ALIASES)}"
            )
        return int(_BOUND_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"boundary condition must be in 0..7, got {ivalue}")
    return ivalue


def _as_float_array(x, name, *, dtype=None, copy=False):
    """Return a C-contiguous CPU float32/float64 numpy array."""
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
    # ascontiguousarray copies when needed; force a copy on request so an
    # in-place binding call does not touch the caller's buffer.
    out = np.ascontiguousarray(arr)
    if copy and out is arr:
        out = arr.copy()
    return out


def _validate_inplace(x, name="x"):
    """Validate ``x`` for an in-place op and return it unchanged.

    The underlying library is fully stride-aware (it receives the array's
    DLPack strides and indexes accordingly), so an in-place call writes
    directly into ``x`` regardless of its memory layout -- no contiguous copy
    is made. We therefore accept *any* float32/float64 numpy array here, which
    keeps in-place ops zero-copy even for non-contiguous views (a core
    memory-efficiency feature of the library). We only reject non-arrays and
    wrong dtypes, where an in-place write could not land in the caller's
    buffer or would need a lossy cast.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"inplace=True requires a numpy ndarray for {name}")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"inplace=True requires a float32/float64 array for {name}")
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
# We use ``as_strided`` on the (writable) source rather than ``np.broadcast_to``
# because ``broadcast_to`` returns a *read-only* view, which numpy then refuses
# to export through DLPack ("cannot export readonly array").


def _bcast_view(a, shape):
    """Return a zero-copy, DLPack-exportable broadcast of ``a`` to ``shape``."""
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


def _broadcast_batch(specs):
    """Broadcast the batch dims of several arrays to a common shape.

    ``specs`` is a list of ``(array, n_core)`` pairs, where ``n_core`` is the
    number of trailing (core) axes that must be left untouched. Returns
    ``(batch_shape, [views...])`` where each view is broadcast to
    ``batch_shape + that array's core dims`` (zero-copy).
    """
    batch = np.broadcast_shapes(*[a.shape[: a.ndim - nc] for a, nc in specs])
    views = [_bcast_view(a, batch + a.shape[a.ndim - nc:]) for a, nc in specs]
    return batch, views


# --------------------------------------------------------------------------- #
# distance transforms                                                         #
# --------------------------------------------------------------------------- #


def euclidean_distance_transform(x, voxel_spacing=1.0, *, inplace=False):
    """Squared Euclidean distance transform along the **last** axis.

    ``x`` must hold ``0`` at feature locations and ``+inf`` elsewhere.
    Returns a new array (unless ``inplace=True``, in which case ``x`` is
    modified and returned -- it must then already be a float32/float64 array;
    any memory layout is fine, the write is zero-copy via DLPack strides).
    """
    if inplace:
        _validate_inplace(x)
        _ff.dt_euclidean(x, float(voxel_spacing))
        return x
    out = _as_float_array(x, "x", copy=True)
    _ff.dt_euclidean(out, float(voxel_spacing))
    return out


def l1_distance_transform(x, voxel_spacing=1.0, *, inplace=False):
    """L1 distance transform along the **last** axis (see
    :func:`euclidean_distance_transform`)."""
    if inplace:
        _validate_inplace(x)
        _ff.dt_l1(x, float(voxel_spacing))
        return x
    out = _as_float_array(x, "x", copy=True)
    _ff.dt_l1(out, float(voxel_spacing))
    return out


# --------------------------------------------------------------------------- #
# compact-symmetric linear algebra                                            #
# --------------------------------------------------------------------------- #


def sym_channels_from_packed(packed_len: int) -> int:
    """Number of channels ``C`` such that ``C*(C+1)/2 == packed_len``."""
    c = int((math.isqrt(8 * packed_len + 1) - 1) // 2)
    if c * (c + 1) // 2 != packed_len:
        raise ValueError(
            f"packed length {packed_len} is not a triangular number "
            "(expected C*(C+1)/2 for some integer C)"
        )
    return c


def _check_sym(mat, vec, matname="mat", vecname="vec"):
    """Validate dtypes/channels of a packed matrix + vector (no batch check).

    Batch dims are broadcast later, so we only enforce the channel relation
    ``vec.shape[-1] == C`` and return the contiguous float arrays plus ``C``.
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


def sym_matvec(mat, vec):
    """``H @ vec`` where ``H`` is a compact-symmetric packed matrix.

    ``mat`` has trailing dim ``C*(C+1)/2`` (diagonal first, then upper rows),
    ``vec`` has trailing dim ``C``.  The batch (leading) dims of ``mat`` and
    ``vec`` are broadcast; the result has the broadcast batch shape + ``(C,)``.
    """
    mat, vec, c = _check_sym(mat, vec)
    batch, (mat_b, vec_b) = _broadcast_batch([(mat, 1), (vec, 1)])
    out = np.empty(batch + (c,), dtype=mat.dtype)
    _ff.sym_matvec(out, mat_b, vec_b)
    return out


def sym_addmatvec(out0, mat, vec):
    """``out0 + H @ vec`` (returns a new array; ``out0`` is not modified).

    Batch dims of ``out0``, ``mat`` and ``vec`` are broadcast together.
    """
    mat, vec, c = _check_sym(mat, vec)
    out0 = _as_float_array(out0, "out0", dtype=mat.dtype)
    if out0.shape[-1] != c:
        raise ValueError("out0 must have the same channel count as vec")
    batch, (out_b, mat_b, vec_b) = _broadcast_batch(
        [(out0, 1), (mat, 1), (vec, 1)]
    )
    out = np.array(out_b, dtype=mat.dtype)  # materialise a contiguous buffer
    _ff.sym_addmatvec_(out, mat_b, vec_b)
    return out


def sym_submatvec(out0, mat, vec):
    """``out0 - H @ vec`` (returns a new array; ``out0`` is not modified)."""
    mat, vec, c = _check_sym(mat, vec)
    out0 = _as_float_array(out0, "out0", dtype=mat.dtype)
    if out0.shape[-1] != c:
        raise ValueError("out0 must have the same channel count as vec")
    batch, (out_b, mat_b, vec_b) = _broadcast_batch(
        [(out0, 1), (mat, 1), (vec, 1)]
    )
    out = np.array(out_b, dtype=mat.dtype)  # materialise a contiguous buffer
    _ff.sym_submatvec_(out, mat_b, vec_b)
    return out


def sym_matvec_backward(grad, vec):
    """Backward of :func:`sym_matvec`: gradient w.r.t. the packed matrix.

    ``grad`` and ``vec`` both have trailing dim ``C`` (batch dims broadcast);
    the result has the packed trailing dim ``C*(C+1)/2``.
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


def sym_solve(mat, vec, weight=None):
    """Solve ``(H + diag(weight)) @ x = vec`` for ``x``.

    ``weight`` (optional) has trailing dim ``C`` matching ``vec``.  Batch dims
    of ``mat``, ``vec`` (and ``weight``) are broadcast together.
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


def sym_invert(mat):
    """Invert a compact-symmetric packed matrix; result is also packed."""
    mat = _as_float_array(mat, "mat")
    out = np.empty_like(mat)
    _ff.sym_invert(out, mat)
    return out


# --------------------------------------------------------------------------- #
# spline coefficients / resampling                                            #
# --------------------------------------------------------------------------- #


def spline_coeff(x, order=3, bound="dct2", *, inplace=False):
    """Spline-coefficient prefilter along the **last** axis.

    Orders 0 and 1 are no-ops.  Returns a new array unless ``inplace=True``.
    """
    spline = _as_spline(order)
    bnd = _as_bound(bound)
    if inplace:
        _validate_inplace(x)
        _ff.spline_coeff(x, spline, bnd)
        return x
    out = _as_float_array(x, "x", copy=True)
    _ff.spline_coeff(out, spline, bnd)
    return out


def _resize_shape_scale(in_shape, ndim, factor, shape):
    """Return (out_shape, scale-list) for resample/restriction.

    ``scale[d]`` is the input-index step per output-index step
    (align-corners convention: ``(in-1)/(out-1)``).
    """
    batch = tuple(in_shape[:-ndim]) if ndim < len(in_shape) else ()
    spatial_in = tuple(in_shape[-ndim:])

    if shape is not None:
        if np.isscalar(shape):
            out_spatial = (int(shape),) * ndim
        else:
            out_spatial = tuple(int(s) for s in shape)
        if len(out_spatial) != ndim:
            raise ValueError(f"shape must have length ndim={ndim}")
    elif factor is not None:
        if np.isscalar(factor):
            factors = (float(factor),) * ndim
        else:
            factors = tuple(float(f) for f in factor)
        if len(factors) != ndim:
            raise ValueError(f"factor must have length ndim={ndim}")
        out_spatial = tuple(
            max(1, int(round(n * f))) for n, f in zip(spatial_in, factors)
        )
    else:
        out_spatial = spatial_in  # identity

    scale = []
    for n_in, n_out in zip(spatial_in, out_spatial):
        if n_out > 1 and n_in > 1:
            scale.append((n_in - 1) / (n_out - 1))
        else:
            scale.append(1.0)
    return batch + out_spatial, scale


def _infer_ndim(ndim, factor, shape, x_ndim):
    if ndim is not None:
        return int(ndim)
    if shape is not None and not np.isscalar(shape):
        return len(shape)
    if factor is not None and not np.isscalar(factor):
        return len(factor)
    return 1


def resample(x, factor=None, shape=None, *, order=2, bound="dct2",
             ndim=None, shift=0.0):
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Provide either ``factor`` (per-axis multiplier, scalar or sequence) or
    ``shape`` (explicit output spatial size).  With neither, this is the
    identity.  Returns a new array.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    out_shape, scale = _resize_shape_scale(x.shape, ndim, factor, shape)
    out = np.zeros(out_shape, dtype=x.dtype)
    _ff.resample(out, x, spline=_as_spline(order), bound=_as_bound(bound),
                 shift=float(shift), scale=scale, ndim=ndim)
    return out


def restriction(x, factor=None, shape=None, *, order=2, bound="dct2",
                ndim=None, shift=0.0):
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The output buffer is zeroed before the (accumulating) binding call.
    Returns a new array.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    out_shape, scale = _resize_shape_scale(x.shape, ndim, factor, shape)
    out = np.zeros(out_shape, dtype=x.dtype)  # must be pre-zeroed (accumulated)
    _ff.restriction(out, x, spline=_as_spline(order), bound=_as_bound(bound),
                    shift=float(shift), scale=scale, ndim=ndim)
    return out


# --------------------------------------------------------------------------- #
# point-to-spline / point-to-mesh distance                                    #
# --------------------------------------------------------------------------- #
#
# NOTE: the underlying fastfields-cpu-lib shape checks for these ops are
# inconsistent with the kernels (see this package's report / README notes),
# so they are exposed as thin pass-throughs following the documented
# distance.h contract but are not otherwise validated here.


def spline_distance_table(loc, coeff, times, *, order=3, bound="dct2"):
    """Point-to-spline (squared) distance via a dictionary of candidate times.

    ``loc`` is ``(*B, D)``, ``coeff`` ``(*B, N, D)``, ``times`` ``(*B, K)``.
    Batch dims of ``loc``/``coeff``/``times`` (core dims ``(D,)``, ``(N, D)``,
    ``(K,)`` respectively) are broadcast; ``time``/``dist`` are allocated with
    the broadcast batch shape.  Returns ``(dist, time)`` each shaped ``(*B,)``.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    times = _as_float_array(times, "times", dtype=loc.dtype)
    batch, (loc_b, coeff_b, times_b) = _broadcast_batch(
        [(loc, 1), (coeff, 2), (times, 1)]
    )
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_table(time, dist, loc_b, coeff_b, times_b,
                        _as_spline(order), _as_bound(bound))
    return dist, time


def spline_distance_brent(loc, coeff, *, max_iter=64, tol=1e-6, step=0.1,
                          order=3, bound="dct2"):
    """Point-to-spline (squared) distance via Brent's method.

    ``loc`` is ``(*B, D)``, ``coeff`` ``(*B, N, D)``; batch dims broadcast.
    Returns ``(dist, time)``."""
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_brent(time, dist, loc_b, coeff_b, int(max_iter), float(tol),
                        float(step), _as_spline(order), _as_bound(bound))
    return dist, time


def spline_distance_gaussnewton(loc, coeff, *, max_iter=64, tol=1e-6,
                                order=3, bound="dct2"):
    """Point-to-spline (squared) distance via Gauss-Newton.

    ``loc`` is ``(*B, D)``, ``coeff`` ``(*B, N, D)``; batch dims broadcast.
    Returns ``(dist, time)``."""
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_gaussnewton(time, dist, loc_b, coeff_b, int(max_iter),
                              float(tol), _as_spline(order), _as_bound(bound))
    return dist, time


def mesh_distance(loc, vertices, faces, *, signed=True, naive=False,
                  return_nearest=False):
    """Point-to-triangular-mesh (squared) distance.

    ``loc`` is ``(*B, D)``, ``vertices`` ``(*B, N, D)``, ``faces`` ``(*B, M, D)``
    (vertex indices).  Batch dims (core dims ``(D,)``, ``(N, D)``, ``(M, D)``)
    are broadcast; ``dist`` (and ``nearest`` if requested) are allocated with
    the broadcast batch shape.  Returns ``dist`` of shape ``(*B,)`` (and the
    nearest-vertex index array when ``return_nearest=True``).
    """
    loc = _as_float_array(loc, "loc")
    vertices = _as_float_array(vertices, "vertices", dtype=loc.dtype)
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    batch, (loc_b, vert_b, faces_b) = _broadcast_batch(
        [(loc, 1), (vertices, 2), (faces, 2)]
    )
    dist = np.empty(batch, dtype=loc.dtype)
    nearest = None
    if return_nearest:
        nearest = np.empty(batch, dtype=np.int64)
    _ff.dt_mesh(dist, nearest, loc_b, vert_b, faces_b,
                bool(signed), bool(naive))
    if return_nearest:
        return dist, nearest
    return dist
