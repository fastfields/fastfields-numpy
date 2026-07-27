"""Spatial regularisers — numpy.

Two operator families over the last ``ndim`` spatial axes:

* **field** — a multi-channel field ``(*batch, *spatial, C)``. The
  ``absolute`` / ``membrane`` / ``bending`` penalties are **per-channel**
  (a scalar broadcasts to all ``C`` channels, or pass a length-``C`` sequence).
* **flow** — a vector flow field. The penalties are **scalars**.

Each family offers the operator (``*_matvec``, apply the regulariser) and its
diagonal (``*_diag``, a preconditioner). With ``voxel_size=None`` all voxels
are unit size; with only ``absolute`` the operator is a per-channel scaling.
"""

from __future__ import annotations

from typing import Optional, Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array

__all__ = [
    "field_matvec",
    "field_diag",
    "flow_matvec",
    "flow_diag",
    "flow_relax",
]


def _per_channel(
    value: float | Sequence[float] | None, channels: int, name: str
) -> Optional[list]:
    """Normalise a per-channel penalty to a length-``channels`` list/None."""
    if value is None:
        return None
    if np.isscalar(value):
        return [float(value)] * channels
    out = [float(v) for v in value]
    if len(out) != channels:
        raise ValueError(
            f"{name} must be a scalar or a length-C={channels} sequence, "
            f"got {value!r}"
        )
    return out


def _voxel_size(
    value: float | Sequence[float] | None, ndim: int
) -> Optional[list]:
    if value is None:
        return None
    if np.isscalar(value):
        return [float(value)] * ndim
    out = [float(v) for v in value]
    if len(out) != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or a length-ndim={ndim} sequence, "
            f"got {value!r}"
        )
    return out


def field_matvec(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the field regulariser ``out = L @ inp`` (shape of ``inp``)."""
    inp = _as_float_array(inp, "inp")
    channels = inp.shape[-1]
    out = np.zeros_like(inp)
    _ff.field_matvec(
        out,
        inp,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_diag(
    shape: Sequence[int],
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal (preconditioner) of the field regulariser, shaped ``shape``.

    ``shape`` is the full field shape ``(*batch, *spatial, C)``.
    """
    out = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    channels = out.shape[-1]
    _ff.field_diag(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_matvec(
    inp: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the flow regulariser; same shape as ``inp``.

    ``shears`` (Lamé mu) and ``div`` (Lamé lambda) add the linear-elastic
    penalty, which couples the flow channels; a non-zero value selects the
    full combined stencil.
    """
    inp = _as_float_array(inp, "inp")
    out = np.zeros_like(inp)
    _ff.flow_matvec(
        out,
        inp,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_diag(
    shape: Sequence[int],
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal (preconditioner) of the flow regulariser, shaped ``shape``."""
    out = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
    _ff.flow_diag(
        out,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def flow_relax(
    flow: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Refine ``flow`` in place with ``nb_iter`` relaxation sweeps.

    Solves ``(H + L) x = g`` where ``H`` is the per-voxel symmetric Hessian
    ``hes`` (packed ``ndim*(ndim+1)/2`` last axis), ``L`` the flow regulariser
    (same penalties as :func:`flow_matvec`), and ``g`` the gradient ``grd``.
    ``flow`` is the warm start, mutated in place and returned.
    """
    flow = _as_float_array(flow, "flow")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    _ff.flow_relax(
        flow,
        hes,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return flow
