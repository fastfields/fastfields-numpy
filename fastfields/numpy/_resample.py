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


# torch-interpol anchor conventions (see ``interpol.resize``). Each anchor is
# identified by its first (lower-cased) letter, so both the full name
# (``"centers"``) and the abbreviation (``"c"``) are accepted.
_ANCHORS = ("c", "e", "f", "l")
# Uniform (dim-independent) sampling shift for the non-centers anchors; the
# resize kernel offsets the sampling grid by ``shift * (scale[d] - 1)``.
_ANCHOR_SHIFT = {"e": 0.5, "f": 0.0, "l": 1.0}


def _anchor_scale_shift(
    anchor: str,
    spatial_in: Sequence[int],
    spatial_out: Sequence[int],
) -> tuple[list[float], float]:
    """Map a torch-interpol ``anchor`` to a per-dim scale and scalar shift.

    The fastfields resize kernel samples input coordinate
    ``scale[d] * loc[d] + shift * (scale[d] - 1)`` for output index ``loc``.
    The four anchors of ``interpol.resize`` map onto ``(scale, shift)`` as:

    ==========  =====================  =======
    anchor      scale[d]               shift
    ==========  =====================  =======
    ``centers`` ``(in-1)/(out-1)``     ``0.0``
    ``edges``   ``in/out``             ``0.5``
    ``first``   ``in/out``             ``0.0``
    ``last``    ``in/out``             ``1.0``
    ==========  =====================  =======

    Parameters
    ----------
    anchor : str
        Anchor name or abbreviation (``centers``/``edges``/``first``/``last``
        or ``c``/``e``/``f``/``l``); matched case-insensitively on the first
        letter, mirroring ``interpol.resize``.
    spatial_in, spatial_out : sequence of int
        Input and output spatial sizes (length ``ndim``).

    Returns
    -------
    scale : list of float
        Per-dim input-index step per output-index step.
    shift : float
        Scalar sampling shift shared across dimensions.

    Raises
    ------
    ValueError
        If ``anchor`` is empty or its first letter is not one of ``c/e/f/l``.
    """
    key = str(anchor)[:1].lower()
    if key not in _ANCHORS:
        raise ValueError(
            f"anchor must be one of centers/edges/first/last, got {anchor!r}"
        )
    if key == "c":
        scale = [
            ((n_in - 1) / (n_out - 1)) if (n_in > 1 and n_out > 1) else 1.0
            for n_in, n_out in zip(spatial_in, spatial_out)
        ]
        return scale, 0.0
    scale = [n_in / n_out for n_in, n_out in zip(spatial_in, spatial_out)]
    return scale, _ANCHOR_SHIFT[key]


def _resize_shapes(
    in_shape: tuple[int, ...],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the batch, input-spatial, and output-spatial shapes.

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
    batch : tuple of int
        Leading (non-resized) dimensions.
    spatial_in : tuple of int
        Input spatial shape (length ``ndim``).
    out_spatial : tuple of int
        Output spatial shape (length ``ndim``).

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

    return batch, spatial_in, out_spatial


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
    anchor: str = "centers",
    shift: float | None = None,
    scale: Sequence[float] | None = None,
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
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention, matching ``interpol.resize``. Determines the
        per-dim scale and the default ``shift``:

        * ``centers`` -- align first/last samples (``(in-1)/(out-1)`` step);
        * ``edges`` -- align the outer voxel *edges* (half-voxel shift);
        * ``first`` -- anchor the first voxel (``in/out`` step, no shift);
        * ``last`` -- anchor the last voxel.

        Abbreviations (``"c"``/``"e"``/``"f"``/``"l"``) are accepted.
    shift : float, optional
        Sampling shift override. When omitted the shift implied by ``anchor``
        is used; pass a value to override it (advanced use).
    scale : sequence of float, optional
        Per-dim scale override (default: derived from ``anchor`` and the
        shapes), length ``ndim``.

    Returns
    -------
    numpy.ndarray
        The resampled array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..x.ndim`` or ``anchor`` is unknown.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    batch, spatial_in, out_spatial = _resize_shapes(
        x.shape, ndim, factor, shape
    )
    eff_scale, anchor_shift = _anchor_scale_shift(
        anchor, spatial_in, out_spatial
    )
    if scale is not None:
        eff_scale = [float(s) for s in scale]
        if len(eff_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    out = np.zeros(batch + out_spatial, dtype=x.dtype)
    _ff.resample(
        out,
        x,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=anchor_shift if shift is None else float(shift),
        scale=eff_scale,
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
    anchor: str = "centers",
    shift: float | None = None,
    scale: Sequence[float] | None = None,
) -> np.ndarray:
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The output buffer is zeroed before the (accumulating) binding call. The
    ``anchor`` convention matches :func:`resample`; because the scale is
    derived from this call's own (input, output) shapes, a ``resample`` and a
    matching ``restriction`` use reciprocal scales and the same shift -- the
    adjoint relationship the binding expects.

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
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention (see :func:`resample`).
    shift : float, optional
        Sampling shift override (see :func:`resample`).
    scale : sequence of float, optional
        Per-dim scale override (see :func:`resample`).

    Returns
    -------
    numpy.ndarray
        The restricted array.

    Raises
    ------
    ValueError
        If ``ndim`` is outside ``1..x.ndim`` or ``anchor`` is unknown.
    """
    x = _as_float_array(x, "x")
    ndim = _infer_ndim(ndim, factor, shape, x.ndim)
    if ndim < 1 or ndim > x.ndim:
        raise ValueError(f"ndim must be in 1..{x.ndim}, got {ndim}")
    batch, spatial_in, out_spatial = _resize_shapes(
        x.shape, ndim, factor, shape
    )
    eff_scale, anchor_shift = _anchor_scale_shift(
        anchor, spatial_in, out_spatial
    )
    if scale is not None:
        eff_scale = [float(s) for s in scale]
        if len(eff_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    # pre-zeroed (the binding accumulates into `out`)
    out = np.zeros(batch + out_spatial, dtype=x.dtype)
    _ff.restriction(
        out,
        x,
        spline=_as_spline(order),
        bound=_as_bound(bound),
        shift=anchor_shift if shift is None else float(shift),
        scale=eff_scale,
        ndim=ndim,
    )
    return out
