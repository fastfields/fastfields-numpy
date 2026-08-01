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

from ._sym import sym_matvec, sym_solve
from ._util import _as_bound, _as_float_array

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
    "field_precond",
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
    "flow_relax",
    "flow_precond",
    "flow_forward",
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
    A composition of :func:`flow_diag` and ``sym_solve`` — no new kernel.
    """
    vec = _as_float_array(vec, "vec")
    diag = flow_diag(
        vec.shape,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
    )
    return sym_solve(mat, vec, diag)


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
    regulariser operator. A composition of ``sym_matvec`` and
    :func:`flow_matvec` — no new kernel.
    """
    vec = _as_float_array(vec, "vec")
    out = sym_matvec(mat, vec)
    out = out + flow_matvec(
        vec,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
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
    (per-channel) diagonal of the field regulariser. A composition of
    :func:`field_diag` and ``sym_solve`` — no new kernel.
    """
    vec = _as_float_array(vec, "vec")
    diag = field_diag(
        vec.shape,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
    )
    return sym_solve(mat, vec, diag)


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
    regulariser operator. A composition of ``sym_matvec`` and
    :func:`field_matvec` — no new kernel.
    """
    vec = _as_float_array(vec, "vec")
    out = sym_matvec(mat, vec)
    out = out + field_matvec(
        vec,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
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
