"""Spline-coefficient prefilter and resampling/restriction (numpy)."""

from __future__ import annotations

from typing import Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array, _as_spline, _validate_inplace

__all__ = [
    "spline_coeff",
    "resample",
    "restriction",
]


def spline_coeff(
    x: np.ndarray,
    order: int | str = 3,
    bound: int | str = "dct2",
    *,
    inplace: bool = False,
) -> np.ndarray:
    """Spline-coefficient prefilter along the **last** axis.

    Parameters
    ----------
    x : numpy.ndarray
        Input samples.
    order : int or str, default=3
        Spline order (orders 0 and 1 are no-ops).
    bound : int or str, default="dct2"
        Boundary condition.
    inplace : bool, default=False
        If ``True``, modify ``x`` in place and return it.

    Returns
    -------
    numpy.ndarray
        Spline coefficients (a new array unless ``inplace=True``).
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


def _resize_shape_scale(
    in_shape: tuple[int, ...],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[tuple[int, ...], list[float]]:
    """Return the output shape and per-dim scale for resample/restriction.

    Parameters
    ----------
    in_shape : tuple of int
        Shape of the input array.
    ndim : int
        Number of trailing spatial dimensions to resize.
    factor : float or sequence of float, optional
        Per-axis multiplier (mutually exclusive with ``shape``).
    shape : int or sequence of int, optional
        Explicit output spatial size (mutually exclusive with ``factor``).

    Returns
    -------
    out_shape : tuple of int
        The full output shape (batch dims + spatial dims).
    scale : list of float
        Per-dim input-index step per output-index step (align-corners
        convention ``(in-1)/(out-1)``).

    Raises
    ------
    ValueError
        If ``factor``/``shape`` do not have length ``ndim``.
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


def _infer_ndim(
    ndim: int | None,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
    x_ndim: int,
) -> int:
    """Infer the number of spatial dimensions to resize.

    Parameters
    ----------
    ndim : int, optional
        Explicit spatial-dimension count (used verbatim if given).
    factor : float or sequence of float, optional
        Per-axis factor; a sequence implies ``len(factor)`` dims.
    shape : int or sequence of int, optional
        Output shape; a sequence implies ``len(shape)`` dims.
    x_ndim : int
        Rank of the input (currently unused; kept for signature stability).

    Returns
    -------
    int
        The inferred spatial-dimension count (defaults to 1).
    """
    if ndim is not None:
        return int(ndim)
    if shape is not None and not np.isscalar(shape):
        return len(shape)
    if factor is not None and not np.isscalar(factor):
        return len(factor)
    return 1


def resample(
    x: np.ndarray,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: int | None = None,
    shift: float = 0.0,
) -> np.ndarray:
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    factor : float or sequence of float, optional
        Per-axis multiplier (scalar or sequence). Mutually exclusive with
        ``shape``; with neither, this is the identity.
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    shift : float, default=0.0
        Sampling shift.

    Returns
    -------
    numpy.ndarray
        The resampled array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..x.ndim``.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    out_shape, scale = _resize_shape_scale(x.shape, ndim, factor, shape)
    out = np.zeros(out_shape, dtype=x.dtype)
    _ff.resample(
        out,
        x,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=float(shift),
        scale=scale,
        ndim=ndim,
    )
    return out


def restriction(
    x: np.ndarray,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: int | None = None,
    shift: float = 0.0,
) -> np.ndarray:
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The output buffer is zeroed before the (accumulating) binding call.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    factor : float or sequence of float, optional
        Per-axis multiplier (scalar or sequence). Mutually exclusive with
        ``shape``.
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order.
    bound : int or str, default="dct2"
        Boundary condition.
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    shift : float, default=0.0
        Sampling shift.

    Returns
    -------
    numpy.ndarray
        The restricted array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..x.ndim``.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    out_shape, scale = _resize_shape_scale(x.shape, ndim, factor, shape)
    out = np.zeros(out_shape, dtype=x.dtype)  # pre-zeroed (accumulated into)
    _ff.restriction(
        out,
        x,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=float(shift),
        scale=scale,
        ndim=ndim,
    )
    return out
