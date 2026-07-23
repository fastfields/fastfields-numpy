"""Distance-transform wrappers (numpy).

Euclidean / L1 distance transforms along the last axis, plus point-to-spline
and point-to-mesh distances.
"""

from __future__ import annotations

import fastfields.dlpack as _ff

import numpy as np

from ._util import (
    _as_bound,
    _as_float_array,
    _as_spline,
    _broadcast_batch,
    _validate_inplace,
)

__all__ = [
    "dt_euclidean",
    "dt_l1",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
]


# --------------------------------------------------------------------------- #
# distance transforms                                                         #
# --------------------------------------------------------------------------- #


def dt_euclidean(
    x: np.ndarray, voxel_spacing: float = 1.0, *, inplace: bool = False
) -> np.ndarray:
    """Squared Euclidean distance transform along the **last** axis.

    Parameters
    ----------
    x : numpy.ndarray
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.
    inplace : bool, default=False
        If ``True``, modify ``x`` in place and return it (``x`` must then
        already be a float32/float64 array; any memory layout is fine, the
        write is zero-copy via DLPack strides).

    Returns
    -------
    numpy.ndarray
        The distance transform (a new array unless ``inplace=True``).
    """
    if inplace:
        _validate_inplace(x)
        _ff.dt_euclidean(x, float(voxel_spacing))
        return x
    out = _as_float_array(x, "x", copy=True)
    _ff.dt_euclidean(out, float(voxel_spacing))
    return out


def dt_l1(
    x: np.ndarray, voxel_spacing: float = 1.0, *, inplace: bool = False
) -> np.ndarray:
    """L1 distance transform along the **last** axis.

    See :func:`dt_euclidean` for the input convention and the
    meaning of the parameters.

    Parameters
    ----------
    x : numpy.ndarray
        Input holding ``0`` at feature locations and ``+inf`` elsewhere.
    voxel_spacing : float, default=1.0
        Physical spacing between samples along the last axis.
    inplace : bool, default=False
        Modify ``x`` in place and return it (see above).

    Returns
    -------
    numpy.ndarray
        The distance transform (a new array unless ``inplace=True``).
    """
    if inplace:
        _validate_inplace(x)
        _ff.dt_l1(x, float(voxel_spacing))
        return x
    out = _as_float_array(x, "x", copy=True)
    _ff.dt_l1(out, float(voxel_spacing))
    return out


# --------------------------------------------------------------------------- #
# point-to-spline / point-to-mesh distance                                    #
# --------------------------------------------------------------------------- #
#
# NOTE: the underlying fastfields-cpu-lib shape checks for these ops are
# inconsistent with the kernels (see this package's report / README notes),
# so they are exposed as thin pass-throughs following the documented
# distance.h contract but are not otherwise validated here.


def dt_spline_table(
    loc: np.ndarray,
    coeff: np.ndarray,
    times: np.ndarray,
    *,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-spline (squared) distance via a dictionary of candidate times.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape ``(*B, D)``.
    coeff : numpy.ndarray
        Spline coefficients, shape ``(*B, N, D)``.
    times : numpy.ndarray
        Candidate times, shape ``(*B, K)``.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : numpy.ndarray
        Squared distance per query point, shape ``(*B,)``.
    time : numpy.ndarray
        Best time per query point, shape ``(*B,)``.

    Notes
    -----
    Batch dims of ``loc``/``coeff``/``times`` (core dims ``(D,)``, ``(N, D)``,
    ``(K,)``) are broadcast; outputs use the broadcast batch shape.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    times = _as_float_array(times, "times", dtype=loc.dtype)
    batch, (loc_b, coeff_b, times_b) = _broadcast_batch(
        [(loc, 1), (coeff, 2), (times, 1)]
    )
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_table(
        time,
        dist,
        loc_b,
        coeff_b,
        times_b,
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_spline_brent(
    loc: np.ndarray,
    coeff: np.ndarray,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    step: float = 0.1,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-spline (squared) distance via Brent's method.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape ``(*B, D)``.
    coeff : numpy.ndarray
        Spline coefficients, shape ``(*B, N, D)``.
    max_iter : int, default=64
        Maximum number of Brent iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    step : float, default=0.1
        Initial bracketing step.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : numpy.ndarray
        Squared distance per query point, shape ``(*B,)``.
    time : numpy.ndarray
        Best time per query point, shape ``(*B,)``.

    Notes
    -----
    Batch dims of ``loc`` (core ``(D,)``) and ``coeff`` (core ``(N, D)``) are
    broadcast together.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_brent(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        float(step),
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_spline_gaussnewton(
    loc: np.ndarray,
    coeff: np.ndarray,
    *,
    max_iter: int = 64,
    tol: float = 1e-6,
    order: int | str = 3,
    bound: int | str = "dct2",
) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-spline (squared) distance via Gauss-Newton.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape ``(*B, D)``.
    coeff : numpy.ndarray
        Spline coefficients, shape ``(*B, N, D)``.
    max_iter : int, default=64
        Maximum number of Gauss-Newton iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    order : int or str, default=3
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.

    Returns
    -------
    dist : numpy.ndarray
        Squared distance per query point, shape ``(*B,)``.
    time : numpy.ndarray
        Best time per query point, shape ``(*B,)``.

    Notes
    -----
    Batch dims of ``loc`` (core ``(D,)``) and ``coeff`` (core ``(N, D)``) are
    broadcast together.
    """
    loc = _as_float_array(loc, "loc")
    coeff = _as_float_array(coeff, "coeff", dtype=loc.dtype)
    batch, (loc_b, coeff_b) = _broadcast_batch([(loc, 1), (coeff, 2)])
    dist = np.empty(batch, dtype=loc.dtype)
    time = np.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_gaussnewton(
        time,
        dist,
        loc_b,
        coeff_b,
        int(max_iter),
        float(tol),
        _as_spline(order),
        _as_bound(bound),
    )
    return dist, time


def dt_mesh(
    loc: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Point-to-triangular-mesh (squared) distance.

    Parameters
    ----------
    loc : numpy.ndarray
        Query points, shape ``(*B, D)``.
    vertices : numpy.ndarray
        Mesh vertices, shape ``(*B, N, D)``.
    faces : numpy.ndarray
        Triangle vertex indices, shape ``(*B, M, D)`` (cast to int64).
    signed : bool, default=True
        Return signed distances.
    naive : bool, default=False
        Use the naive (brute-force) algorithm.
    return_nearest : bool, default=False
        Also return the nearest-vertex index per query point.

    Returns
    -------
    dist : numpy.ndarray
        (Squared) distance per query point, shape ``(*B,)``.
    nearest : numpy.ndarray, optional
        Nearest-vertex index per query point, only if ``return_nearest`` is
        ``True``.

    Notes
    -----
    Batch dims (core dims ``(D,)``, ``(N, D)``, ``(M, D)``) are broadcast; the
    outputs use the broadcast batch shape.
    """
    loc = _as_float_array(loc, "loc")
    vertices = _as_float_array(vertices, "vertices", dtype=loc.dtype)
    # faces is an integer index array; keep native strides (stride-aware
    # kernel) but ensure it is writable so numpy can export it via DLPack.
    faces = np.asarray(faces, dtype=np.int64)
    if not faces.flags.writeable:
        faces = faces.copy()
    batch, (loc_b, vert_b, faces_b) = _broadcast_batch(
        [(loc, 1), (vertices, 2), (faces, 2)]
    )
    dist = np.empty(batch, dtype=loc.dtype)
    nearest = None
    if return_nearest:
        nearest = np.empty(batch, dtype=np.int64)
    _ff.dt_mesh(
        dist, nearest, loc_b, vert_b, faces_b, bool(signed), bool(naive)
    )
    if return_nearest:
        return dist, nearest
    return dist
