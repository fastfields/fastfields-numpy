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

from typing import NamedTuple, Optional, Sequence

import fastfields.dlpack as _ff

import numpy as np

from ._util import _as_bound, _as_float_array, _broadcast_batch

__all__ = [
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_addkernel",
    "field_addkernel_",
    "field_subkernel",
    "field_subkernel_",
    "field_relax",
    "field_cg",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
    "field_precond_",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
    "flow_addkernel",
    "flow_addkernel_",
    "flow_subkernel",
    "flow_subkernel_",
    "flow_relax",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_precond",
    "flow_precond_",
    "flow_forward",
    "CGInfo",
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


# ---------------------------------------------------------------------------
# In-place accumulate ops (`out (+/-)= ...`)
# ---------------------------------------------------------------------------
#
# There is exactly ONE kernel behind each of these: the in-place-only C
# primitive `ff::{field,flow}_{matvec,diag}_{add,sub}_`, restored from
# jitfields' `op='+'` / `op='-'` entry points. The out-of-place spelling simply
# copies the caller's array first and then runs the same primitive on the copy
# -- it is not a second code path, and it never materialises a separate
# temporary for the regulariser result.


def _field_matvec_acc(
    inp,
    field,
    absolute,
    membrane,
    bending,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = _as_float_array(inp, "inp")
    field = _as_float_array(field, "field")
    acc = inp if inplace else inp.copy()
    channels = field.shape[-1]
    fn = _ff.field_submatvec_ if sub else _ff.field_addmatvec_
    fn(
        acc,
        field,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def _field_diag_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subdiag_ if sub else _ff.field_adddiag_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


# ---------------------------------------------------------------------------
# The regularised system (H + L)
# ---------------------------------------------------------------------------
#
# `*_forward` applies (H + L), `*_precond[_]` applies the Jacobi approximation
# of its inverse, and `*_relax` / `field_cg` solve it -- where H is the
# per-voxel data-term Hessian (`mat`, compact-symmetric) and L the
# regulariser. Each of these is ONE binding, `fastfields.dlpack.{field,flow}_
# {forward,precond,precond_}`.
#
# They used to be assembled here in Python out of `*_matvec` + `sym_matvec`
# (forward) and `*_diag` + `sym_solve` (preconditioner), because the hub
# entry points existed in C++ but had never been bound. That composition
# computes the same values; it just materialises a full-size temporary for the
# regulariser result and walks the field twice. The hub entry points are now
# bound, so these call them.
#
# Batch (leading) dims of `mat` and the vector are still broadcast, as they
# were through `sym_solve`/`sym_matvec`. The *spatial* axes and the channel
# axis are not: the regulariser's stencil is spatial, so a mismatch there is
# an error rather than something to broadcast (same rule as `field_relax`).
# `ndim + 1` trailing axes are therefore the core shape.


def _reg_broadcast(mat, vec, ndim, mat_name="mat", vec_name="vec"):
    """Broadcast the batch dims of a (Hessian, vector) pair for a reg op."""
    mat = _as_float_array(mat, mat_name)
    vec = _as_float_array(vec, vec_name)
    core = int(ndim) + 1
    if mat.ndim < core or vec.ndim < core:
        raise ValueError(
            f"{mat_name} and {vec_name} must have at least ndim+1={core} "
            f"axes (*spatial, C); got {mat.shape} and {vec.shape}"
        )
    batch, (mat_b, vec_b) = _broadcast_batch([(mat, core), (vec, core)])
    return batch, mat_b, vec_b


def _field_kernel_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subkernel_ if sub else _ff.field_addkernel_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def _flow_matvec_acc(
    inp,
    flow,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = _as_float_array(inp, "inp")
    flow = _as_float_array(flow, "flow")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_submatvec_ if sub else _ff.flow_addmatvec_
    fn(
        acc,
        flow,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def _flow_diag_acc(
    inp,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subdiag_ if sub else _ff.flow_adddiag_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


def _flow_kernel_acc(
    inp,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = _as_float_array(inp, "inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subkernel_ if sub else _ff.flow_addkernel_
    fn(
        acc,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return acc


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


def _field_channels(
    channels: int | None,
    *penalties: float | Sequence[float] | None,
) -> int:
    """Infer the channel count ``C`` from an explicit value or penalty lengths.

    A per-channel penalty passed as a sequence fixes ``C`` (its length); scalar
    / None penalties broadcast. Defaults to 1 if nothing pins it down.
    """
    if channels is not None:
        return int(channels)
    for p in penalties:
        if p is not None and not np.isscalar(p):
            return len(p)
    return 1


def field_kernel(
    ndim: int,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    channels: int | None = None,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Materialise the field regulariser's per-channel Toeplitz stencil.

    Returns the small centred kernel that, convolved with a field, reproduces
    :func:`field_matvec`. The shape is ``(*k, C)`` (channels are independent —
    no cross-channel matrix), where ``k`` is the stencil width per spatial dim:
    1 (absolute only), 3 (membrane) or 5 (bending). The channel count ``C`` is
    taken from ``channels`` if given, else inferred from the per-channel
    penalty lengths (default 1).
    """
    ndim = int(ndim)
    channels = _field_channels(channels, absolute, membrane, bending)
    if bending is not None:
        width = 5
    elif membrane is not None:
        width = 3
    else:
        width = 1
    out = np.zeros([width] * ndim + [channels], dtype=dtype)
    _ff.field_kernel(
        out,
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


def field_relax(
    field: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """Refine ``field`` in place with ``nb_iter`` relaxation sweeps.

    Solves ``(H + L) x = g`` where ``H`` is the per-voxel compact-symmetric
    Hessian ``hes`` (packed ``C*(C+1)/2`` last axis), ``L`` the field
    regulariser (same per-channel penalties as :func:`field_matvec`), and
    ``g`` the gradient ``grd``. ``field`` is the warm start, mutated in place
    and returned.
    """
    field = _as_float_array(field, "field")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    channels = field.shape[-1]
    _ff.field_relax(
        field,
        hes,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return field


def field_matvec_rls(
    inp: np.ndarray,
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """RLS/JRLS-weighted variant of :func:`field_matvec`.

    ``wgt`` has shape ``(*batch, *spatial, 1)`` for a single weight shared
    across all channels (JRLS), or ``(*batch, *spatial, C)`` for a genuine
    per-channel weight (RLS, ``C`` matching ``inp``'s channel count) --
    the trailing dimension of ``wgt`` selects which mode is used. Same
    shape/penalty conventions as :func:`field_matvec` otherwise.
    """
    inp = _as_float_array(inp, "inp")
    wgt = _as_float_array(wgt, "wgt")
    channels = inp.shape[-1]
    out = np.zeros_like(inp)
    _ff.field_matvec_rls(
        out,
        inp,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_diag_rls(
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    channels: int | None = None,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Diagonal (preconditioner) of :func:`field_matvec_rls`.

    ``wgt`` selects JRLS (trailing dim 1) vs RLS (trailing dim ``C``), same
    as :func:`field_matvec_rls`. The output channel count ``C`` is taken
    from ``channels`` if given, else from ``wgt``'s trailing dimension when
    it is not 1 (RLS), else inferred from the per-channel penalty lengths
    (default 1) exactly as :func:`field_kernel` does.
    """
    wgt = _as_float_array(wgt, "wgt")
    if channels is None:
        channels = (
            wgt.shape[-1]
            if wgt.shape[-1] != 1
            else _field_channels(None, absolute, membrane, bending)
        )
    out = np.zeros(wgt.shape[:-1] + (channels,), dtype=dtype)
    _ff.field_diag_rls(
        out,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_relax_rls(
    field: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    wgt: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> np.ndarray:
    """RLS/JRLS-weighted variant of :func:`field_relax`.

    Refines ``field`` in place with ``nb_iter`` relaxation sweeps of
    ``(H + L(w)) x = g``, where ``L(w)`` is the weighted field regulariser
    (same ``wgt`` conventions as :func:`field_matvec_rls`). ``field`` is
    the warm start, mutated in place and returned.
    """
    field = _as_float_array(field, "field")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    wgt = _as_float_array(wgt, "wgt")
    channels = field.shape[-1]
    _ff.field_relax_rls(
        field,
        hes,
        grd,
        wgt,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
    )
    return field


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


def flow_kernel(
    ndim: int,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Materialise the flow regulariser's Toeplitz convolution stencil.

    Returns the small centred kernel that, convolved with a flow field,
    reproduces :func:`flow_matvec`. The shape is ``(*k, ndim)`` for the
    per-channel vector stencil, or ``(*k, ndim, ndim)`` when ``shears``/``div``
    select the cross-channel (Lamé) matrix stencil, where ``k`` is the stencil
    width per spatial dim: 1 (absolute only), 3 (membrane/Lamé) or 5 (bending).
    """
    ndim = int(ndim)
    is_matrix = shears != 0.0 or div != 0.0
    if shears == div == membrane == bending == 0.0:
        width = 1
    elif bending == 0.0:
        width = 3
    else:
        width = 5
    shape = [width] * ndim + [ndim]
    if is_matrix:
        shape += [ndim]
    out = np.zeros(shape, dtype=dtype)
    _ff.flow_kernel(
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


def flow_matvec_rls(
    inp: np.ndarray,
    wgt: np.ndarray,
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
    """RLS/JRLS-weighted variant of :func:`flow_matvec`.

    ``wgt`` has shape ``(*batch, *spatial, 1)``: unlike the field family, the
    flow weighting is always *joint* -- the trailing axis of a flow field
    holds the components of one displacement vector, so a single weight is
    shared across them. Same shape/penalty conventions as :func:`flow_matvec`
    otherwise.

    ``bending`` is **not supported** with weighting (no weighted bending
    kernel exists, as in ``jitfields``); a non-zero value raises.
    """
    inp = _as_float_array(inp, "inp")
    wgt = _as_float_array(wgt, "wgt")
    out = np.zeros_like(inp)
    _ff.flow_matvec_rls(
        out,
        inp,
        wgt,
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


def flow_diag_rls(
    wgt: np.ndarray,
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
    """Diagonal (preconditioner) of :func:`flow_matvec_rls`.

    Unlike plain :func:`flow_diag`, which builds from a *shape*, this one is
    driven by the weight map: the output is ``wgt.shape[:-1] + (ndim,)``,
    since a flow field's channel count is ``ndim``. Same ``wgt`` conventions
    as :func:`flow_matvec_rls`.
    """
    wgt = _as_float_array(wgt, "wgt")
    out = np.zeros(wgt.shape[:-1] + (ndim,), dtype=dtype)
    _ff.flow_diag_rls(
        out,
        wgt,
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


def flow_relax_rls(
    flow: np.ndarray,
    hes: np.ndarray,
    grd: np.ndarray,
    wgt: np.ndarray,
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
    """RLS/JRLS-weighted variant of :func:`flow_relax`.

    Refines ``flow`` in place with ``nb_iter`` relaxation sweeps of
    ``(H + L(w)) x = g``, where ``L(w)`` is the weighted flow regulariser
    (same ``wgt`` conventions as :func:`flow_matvec_rls`). ``flow`` is the
    warm start, mutated in place and returned.
    """
    flow = _as_float_array(flow, "flow")
    hes = _as_float_array(hes, "hes")
    grd = _as_float_array(grd, "grd")
    wgt = _as_float_array(wgt, "wgt")
    _ff.flow_relax_rls(
        flow,
        hes,
        grd,
        wgt,
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


def flow_precond(
    mat: np.ndarray,
    vec: np.ndarray,
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
    """Apply the preconditioner ``(M + diag(R)) \\ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat``; ``diag(R)`` is the
    diagonal of the flow regulariser (same penalties as :func:`flow_matvec`).
    One kernel (``fastfields.dlpack.flow_precond``). Leading batch dims of
    ``mat``/``vec`` broadcast; the ``ndim`` spatial axes and the component axis
    must match.

    See :func:`flow_precond_` for the in-place form.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.flow_precond(
        out,
        mat,
        vec,
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


def flow_forward(
    mat: np.ndarray,
    vec: np.ndarray,
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
    """Apply the forward matrix-vector product ``(M + R) @ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat`` and ``R`` the flow
    regulariser operator. One kernel
    (``fastfields.dlpack.flow_forward``). Leading batch dims of ``mat``/``vec``
    broadcast; the ``ndim`` spatial axes and the component axis must match.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.flow_forward(
        out,
        mat,
        vec,
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


# --- accumulate variants -------------------------------------------------
#
# jitfields exposes ``_add`` / ``_sub`` (write a fresh array) and trailing-
# underscore in-place (``_add_`` / ``_sub_``) forms of the flow operator and
# its diagonal. They are thin compositions over :func:`flow_matvec` /
# :func:`flow_diag` — ``out = inp ± op(...)`` — so no new kernel is needed.


def flow_addmatvec(
    inp: np.ndarray,
    flow: np.ndarray,
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
    """Return ``inp + L @ flow`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy --
    one kernel, no separate temporary for the regulariser result.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_submatvec(
    inp: np.ndarray,
    flow: np.ndarray,
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
    """Return ``inp - L @ flow`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy --
    one kernel, no separate temporary for the regulariser result.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_addmatvec_(
    inp: np.ndarray,
    flow: np.ndarray,
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
    """In place ``inp += L @ flow`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_submatvec_(
    inp: np.ndarray,
    flow: np.ndarray,
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
    """In place ``inp -= L @ flow`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def flow_adddiag(
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
    """Return ``inp + diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_subdiag(
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
    """Return ``inp - diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_adddiag_(
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
    """In place ``inp += diag(L)`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_subdiag_(
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
    """In place ``inp -= diag(L)`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_precond(
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the preconditioner ``(M + diag(R)) \\ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat``; ``diag(R)`` is the
    (per-channel) diagonal of the field regulariser. One kernel
    (``fastfields.dlpack.field_precond``). Leading batch dims of
    ``mat``/``vec`` broadcast; the ``ndim`` spatial axes and the channel axis
    must match.

    See :func:`field_precond_` for the in-place form.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.field_precond(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_forward(
    mat: np.ndarray,
    vec: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Apply the forward matrix-vector product ``(M + R) @ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat`` and ``R`` the field
    regulariser operator. One kernel
    (``fastfields.dlpack.field_forward``). Leading batch dims of
    ``mat``/``vec`` broadcast; the ``ndim`` spatial axes and the channel axis
    must match.
    """
    batch, mat, vec = _reg_broadcast(mat, vec, ndim)
    channels = vec.shape[-1]
    out = np.empty(batch + vec.shape[vec.ndim - int(ndim) - 1 :], vec.dtype)
    _ff.field_forward(
        out,
        mat,
        vec,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return out


def field_addmatvec(
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp + L @ field`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy --
    one kernel, no separate temporary for the regulariser result.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_submatvec(
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp - L @ field`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy --
    one kernel, no separate temporary for the regulariser result.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_addmatvec_(
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place ``inp += L @ field`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_submatvec_(
    inp: np.ndarray,
    field: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place ``inp -= L @ field`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_adddiag(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp + diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_subdiag(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp - diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_adddiag_(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place ``inp += diag(L)`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_subdiag_(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place ``inp -= diag(L)`` (``inp`` a float array); returns it.

    Calls the fused in-place C primitive directly.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


# ---------------------------------------------------------------------------
# In-place preconditioners and the CG solver
# ---------------------------------------------------------------------------


def field_precond_(
    sol: np.ndarray,
    mat: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In-place :func:`field_precond`: ``sol = (M + diag(R)) \\ sol``.

    ``sol`` holds the gradient on entry and the preconditioned solution on
    exit; it is mutated and returned. Unlike :func:`field_precond` nothing is
    broadcast -- an in-place op writes through the caller's buffer, so its
    shape is the answer's shape.
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    channels = sol.shape[-1]
    _ff.field_precond_(
        sol,
        mat,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return sol


def flow_precond_(
    sol: np.ndarray,
    mat: np.ndarray,
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
    """In-place :func:`flow_precond`: ``sol = (M + diag(R)) \\ sol``.

    ``sol`` holds the gradient on entry and the preconditioned solution on
    exit; it is mutated and returned.
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    _ff.flow_precond_(
        sol,
        mat,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=float(absolute),
        membrane=float(membrane),
        bending=float(bending),
        shears=float(shears),
        div=float(div),
        bound=_as_bound(bound),
        ndim=ndim,
    )
    return sol


class CGInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`field_cg`."""

    #: Number of CG iterations actually run (<= the requested ``nb_iter``).
    iterations: int
    #: Final relative residual ``||g - (H + L) x|| / ||g||``.
    residual: float


def field_cg(
    sol: np.ndarray,
    mat: np.ndarray,
    grd: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 32,
    tol: float = 1e-8,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, CGInfo]:
    """Solve ``(M + R) x = grd`` by preconditioned conjugate gradients.

    The Krylov alternative to :func:`field_relax`'s Gauss-Seidel sweeps on the
    same system, Jacobi-preconditioned by :func:`field_precond`'s
    ``(M + diag(R))^-1``.

    ``sol`` is both the warm start and the output: it is **mutated in place**
    and returned, exactly like :func:`field_relax`. Pass a zero-filled array
    for a cold start, or a previous estimate to continue from it.

    Iteration stops after ``nb_iter`` steps, or as soon as the relative
    residual drops below ``tol`` (``tol <= 0`` runs every iteration), or early
    if the operator turns out not to be positive definite along the current
    search direction. Pass ``return_info=True`` to find out which happened.

    Notes
    -----
    This operation is **CPU-only** in ``fastfields-lib`` (its dot products need
    a device-side reduction the CUDA backend does not implement yet,
    fastfields-lib#34). That costs this wrapper nothing -- numpy arrays are
    host memory -- but it is why ``fastfields.cupy`` has no ``field_cg``.

    Parameters
    ----------
    sol : numpy.ndarray
        Warm start in, solution out ``(*batch, *spatial, C)``.
    mat : numpy.ndarray
        Per-voxel compact-symmetric Hessian ``(*batch, *spatial, C*(C+1)/2)``.
    grd : numpy.ndarray
        Right-hand side / gradient ``(*batch, *spatial, C)``.
    nb_iter : int, default=32
        Maximum number of CG iterations.
    tol : float, default=1e-8
        Relative residual tolerance; ``<= 0`` disables the early exit.
    return_info : bool, default=False
        Also return a :class:`CGInfo` with the iteration count and residual.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, CGInfo)
        ``sol``, mutated in place -- plus the diagnostics if asked for.
    """
    sol = _as_float_array(sol, "sol")
    mat = _as_float_array(mat, "mat")
    grd = _as_float_array(grd, "grd")
    channels = sol.shape[-1]
    iterations, residual = _ff.field_cg(
        sol,
        mat,
        grd,
        voxel_size=_voxel_size(voxel_size, ndim),
        absolute=_per_channel(absolute, channels, "absolute"),
        membrane=_per_channel(membrane, channels, "membrane"),
        bending=_per_channel(bending, channels, "bending"),
        bound=_as_bound(bound),
        ndim=ndim,
        nb_iter=int(nb_iter),
        tol=float(tol),
    )
    if return_info:
        return sol, CGInfo(int(iterations), float(residual))
    return sol


# ---------------------------------------------------------------------------
# Stencil (kernel) accumulate variants
# ---------------------------------------------------------------------------
#
# The same `inp ± op(...)` pattern as the *_adddiag / *_subdiag family, for
# the materialised Toeplitz stencil rather than the diagonal. `inp` is shaped
# like the output of :func:`field_kernel` / :func:`flow_kernel`.


def field_addkernel(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp + K`` as a **new** array, shaped like ``inp``.

    ``K`` is the field regulariser's stencil (see :func:`field_kernel`, whose
    output shape ``inp`` must match). Copies ``inp`` and runs the in-place
    accumulate primitive on the copy.
    """
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_subkernel(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """Return ``inp - K`` as a **new** array, shaped like ``inp``."""
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_addkernel_(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place: ``inp += K``. Mutates and returns ``inp``."""
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_subkernel_(
    inp: np.ndarray,
    absolute: float | Sequence[float] | None = None,
    membrane: float | Sequence[float] | None = None,
    bending: float | Sequence[float] | None = None,
    *,
    voxel_size: float | Sequence[float] | None = None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> np.ndarray:
    """In place: ``inp -= K``. Mutates and returns ``inp``."""
    return _field_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def flow_addkernel(
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
    """Return ``inp + K`` as a **new** array, shaped like ``inp``.

    ``K`` is the flow regulariser's stencil (see :func:`flow_kernel`, whose
    output shape ``inp`` must match).
    """
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_subkernel(
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
    """Return ``inp - K`` as a **new** array, shaped like ``inp``."""
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_addkernel_(
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
    """In place: ``inp += K``. Mutates and returns ``inp``."""
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_subkernel_(
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
    """In place: ``inp -= K``. Mutates and returns ``inp``."""
    return _flow_kernel_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )
