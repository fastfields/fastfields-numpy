"""fastfields_numpy: a friendly numpy interface over the ``fastfields_bind`` bindings.

The underlying bindings operate *in place* / write through pre-allocated output
arrays.  This package wraps them so every public function takes numpy arrays and
**returns** freshly allocated numpy arrays (never clobbering the caller's input
unless ``inplace=True`` is explicitly requested).

Spline order and boundary condition arguments accept an ``int``, one of the
re-exported :class:`Spline` / :class:`Bound` enums, or a friendly string
(e.g. ``"cubic"``, ``"dct2"``).
"""

from __future__ import annotations

import math

import numpy as np

import fastfields_bind as _ff
from fastfields_bind import Bound, Spline

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
    mat = _as_float_array(mat, matname)
    vec = _as_float_array(vec, vecname, dtype=mat.dtype)
    if mat.dtype != vec.dtype:
        vec = vec.astype(mat.dtype)
        vec = np.ascontiguousarray(vec)
    c = sym_channels_from_packed(mat.shape[-1])
    if vec.shape[-1] != c:
        raise ValueError(
            f"{vecname} has {vec.shape[-1]} channels but the packed matrix "
            f"encodes {c} channels"
        )
    if mat.shape[:-1] != vec.shape[:-1]:
        raise ValueError(
            f"batch shapes differ: {matname}{mat.shape[:-1]} vs "
            f"{vecname}{vec.shape[:-1]}"
        )
    return mat, vec, c


def sym_matvec(mat, vec):
    """``H @ vec`` where ``H`` is a compact-symmetric packed matrix.

    ``mat`` has trailing dim ``C*(C+1)/2`` (diagonal first, then upper rows),
    ``vec`` has trailing dim ``C``.  Returns an array shaped like ``vec``.
    """
    mat, vec, _ = _check_sym(mat, vec)
    out = np.empty_like(vec)
    _ff.sym_matvec(out, mat, vec)
    return out


def sym_addmatvec(out0, mat, vec):
    """``out0 + H @ vec`` (returns a new array; ``out0`` is not modified)."""
    mat, vec, _ = _check_sym(mat, vec)
    out = _as_float_array(out0, "out0", dtype=mat.dtype, copy=True)
    if out.shape != vec.shape:
        raise ValueError("out0 must have the same shape as vec")
    _ff.sym_addmatvec_(out, mat, vec)
    return out


def sym_submatvec(out0, mat, vec):
    """``out0 - H @ vec`` (returns a new array; ``out0`` is not modified)."""
    mat, vec, _ = _check_sym(mat, vec)
    out = _as_float_array(out0, "out0", dtype=mat.dtype, copy=True)
    if out.shape != vec.shape:
        raise ValueError("out0 must have the same shape as vec")
    _ff.sym_submatvec_(out, mat, vec)
    return out


def sym_matvec_backward(grad, vec):
    """Backward of :func:`sym_matvec`: gradient w.r.t. the packed matrix.

    ``grad`` and ``vec`` both have trailing dim ``C``; the result has the packed
    trailing dim ``C*(C+1)/2``.
    """
    grad = _as_float_array(grad, "grad")
    vec = _as_float_array(vec, "vec", dtype=grad.dtype)
    if grad.shape != vec.shape:
        raise ValueError("grad and vec must have the same shape")
    c = grad.shape[-1]
    packed = c * (c + 1) // 2
    out = np.empty(grad.shape[:-1] + (packed,), dtype=grad.dtype)
    _ff.sym_matvec_backward(out, grad, vec)
    return out


def sym_solve(mat, vec, weight=None):
    """Solve ``(H + diag(weight)) @ x = vec`` for ``x``.

    ``weight`` (optional) has trailing dim ``C`` matching ``vec``.
    """
    mat, vec, c = _check_sym(mat, vec)
    out = np.empty_like(vec)
    if weight is None:
        _ff.sym_solve(out, mat, vec)
    else:
        w = _as_float_array(weight, "weight", dtype=mat.dtype)
        if w.shape[-1] != c or w.shape[:-1] != vec.shape[:-1]:
            raise ValueError("weight must be broadcastable-shaped like vec")
        _ff.sym_solve(out, mat, vec, w)
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


def _spline_dist_outputs(loc, coeff):
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch = loc.shape[:-1]
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    return loc, coeff, dist, time


def spline_distance_table(loc, coeff, times, *, order=3, bound="dct2"):
    """Point-to-spline (squared) distance via a dictionary of candidate times.

    ``loc`` is ``(*B, D)``, ``coeff`` ``(N, D)``, ``times`` ``(K,)``.
    Returns ``(dist, time)`` each shaped ``(*B,)``.
    """
    loc, coeff, dist, time = _spline_dist_outputs(loc, coeff)
    times = _as_float_array(times, "times", dtype=loc.dtype)
    _ff.dt_spline_table(time, dist, loc, coeff, times,
                        _as_spline(order), _as_bound(bound))
    return dist, time


def spline_distance_brent(loc, coeff, *, max_iter=64, tol=1e-6, step=0.1,
                          order=3, bound="dct2"):
    """Point-to-spline (squared) distance via Brent's method.
    Returns ``(dist, time)``."""
    loc, coeff, dist, time = _spline_dist_outputs(loc, coeff)
    _ff.dt_spline_brent(time, dist, loc, coeff, int(max_iter), float(tol),
                        float(step), _as_spline(order), _as_bound(bound))
    return dist, time


def spline_distance_gaussnewton(loc, coeff, *, max_iter=64, tol=1e-6,
                                order=3, bound="dct2"):
    """Point-to-spline (squared) distance via Gauss-Newton.
    Returns ``(dist, time)``."""
    loc, coeff, dist, time = _spline_dist_outputs(loc, coeff)
    _ff.dt_spline_gaussnewton(time, dist, loc, coeff, int(max_iter),
                              float(tol), _as_spline(order), _as_bound(bound))
    return dist, time


def mesh_distance(loc, vertices, faces, *, signed=True, naive=False,
                  return_nearest=False):
    """Point-to-triangular-mesh (squared) distance.

    ``loc`` is ``(*B, D)``, ``vertices`` ``(N, D)``, ``faces`` ``(M, D)`` (vertex
    indices).  Returns ``dist`` of shape ``(*B,)`` (and the nearest-vertex index
    array when ``return_nearest=True``).
    """
    loc = _as_float_array(loc, "loc")
    vertices = _as_float_array(vertices, "vertices", dtype=loc.dtype)
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    dist = np.empty(loc.shape[:-1], dtype=loc.dtype)
    nearest = None
    if return_nearest:
        nearest = np.empty(loc.shape[:-1], dtype=np.int64)
    _ff.dt_mesh(dist, nearest, loc, vertices, faces,
                bool(signed), bool(naive))
    if return_nearest:
        return dist, nearest
    return dist
