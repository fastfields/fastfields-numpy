"""The regularised system ``(H + L)`` and the stencil accumulate family.

``field_forward`` / ``field_precond`` / ``flow_forward`` / ``flow_precond``
used to be assembled here in Python, out of ``*_matvec`` + ``sym_matvec`` and
``*_diag`` + ``sym_solve``, because the hub entry points existed in C++ but
had never been bound. They now call one fused binding each. The composition is
therefore the right reference: it is a different path through the library, so
agreement to the bit says the re-pointing preserved the semantics (including
the batch broadcasting the composition got from ``sym_solve``/``sym_matvec``),
and it is also what a user upgrading across this change is entitled to.

``field_precond_`` / ``flow_precond_`` (in place), ``field_cg``, and the
``*_addkernel`` / ``*_subkernel`` family are newly exposed and are checked
against their out-of-place / non-accumulating counterparts.
"""

import numpy as np
import pytest

import fastfields.numpy as ff

NDIM = 2
SPATIAL = (5, 6)
VS = [1.0, 1.2]
FIELD_PEN = {
    "absolute": [0.1, 0.2],
    "membrane": [0.5, 0.3],
    "bending": [0.05, 0.02],
}
FLOW_PEN = {
    "absolute": 0.1,
    "membrane": 0.5,
    "bending": 0.0,
    "shears": 0.2,
    "div": 0.1,
}
KW = {"voxel_size": VS, "bound": "dct2", "ndim": NDIM}


def _operands(channels, seed=0):
    """A positive-definite compact-symmetric Hessian and a vector field."""
    rng = np.random.default_rng(seed)
    npack = channels * (channels + 1) // 2
    vec = rng.standard_normal(SPATIAL + (channels,))
    hes = np.zeros(SPATIAL + (npack,))
    for c in range(channels):
        hes[..., c] = 3.0 + c + rng.random(SPATIAL)
    hes[..., channels:] = 0.1 * rng.standard_normal(
        SPATIAL + (npack - channels,)
    )
    return hes, vec


# --------------------------------------------------------------------------- #
# forward / preconditioner vs. the composition they replaced                  #
# --------------------------------------------------------------------------- #


def test_field_forward_matches_sym_matvec_plus_field_matvec():
    hes, vec = _operands(2)
    got = ff.field_forward(hes, vec, **FIELD_PEN, **KW)
    ref = ff.sym_matvec(hes, vec) + ff.field_matvec(vec, **FIELD_PEN, **KW)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_flow_forward_matches_sym_matvec_plus_flow_matvec():
    hes, vec = _operands(NDIM)
    got = ff.flow_forward(hes, vec, **FLOW_PEN, **KW)
    ref = ff.sym_matvec(hes, vec) + ff.flow_matvec(vec, **FLOW_PEN, **KW)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_field_precond_matches_sym_solve_with_field_diag():
    hes, vec = _operands(2)
    got = ff.field_precond(hes, vec, **FIELD_PEN, **KW)
    diag = ff.field_diag(vec.shape, **FIELD_PEN, **KW)
    np.testing.assert_allclose(
        got, ff.sym_solve(hes, vec, diag), rtol=1e-12, atol=1e-12
    )


def test_flow_precond_matches_sym_solve_with_flow_diag():
    hes, vec = _operands(NDIM)
    got = ff.flow_precond(hes, vec, **FLOW_PEN, **KW)
    diag = ff.flow_diag(vec.shape, **FLOW_PEN, **KW)
    np.testing.assert_allclose(
        got, ff.sym_solve(hes, vec, diag), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("op", ["forward", "precond"])
def test_leading_batch_dims_still_broadcast(op):
    """The composition broadcast batch dims via sym_*; the kernel must too."""
    hes, vec = _operands(2)
    single = getattr(ff, f"field_{op}")(hes, vec, **FIELD_PEN, **KW)

    batched = getattr(ff, f"field_{op}")(
        hes[None],
        np.broadcast_to(vec, (3,) + vec.shape).copy(),
        **FIELD_PEN,
        **KW,
    )
    assert batched.shape == (3,) + single.shape
    for i in range(3):
        np.testing.assert_allclose(batched[i], single, rtol=1e-12, atol=1e-12)


def test_spatial_rank_below_ndim_is_rejected():
    hes, vec = _operands(2)
    with pytest.raises(ValueError, match="at least ndim"):
        ff.field_forward(hes[0, 0], vec[0, 0], **FIELD_PEN, ndim=NDIM)


# --------------------------------------------------------------------------- #
# in-place preconditioners                                                    #
# --------------------------------------------------------------------------- #


def test_field_precond_inplace_agrees_and_mutates():
    hes, vec = _operands(2)
    expected = ff.field_precond(hes, vec, **FIELD_PEN, **KW)
    sol = vec.copy()
    returned = ff.field_precond_(sol, hes, **FIELD_PEN, **KW)
    assert returned is sol
    np.testing.assert_allclose(sol, expected, rtol=1e-12, atol=1e-12)


def test_flow_precond_inplace_agrees_and_mutates():
    hes, vec = _operands(NDIM)
    expected = ff.flow_precond(hes, vec, **FLOW_PEN, **KW)
    sol = vec.copy()
    returned = ff.flow_precond_(sol, hes, **FLOW_PEN, **KW)
    assert returned is sol
    np.testing.assert_allclose(sol, expected, rtol=1e-12, atol=1e-12)


def test_out_of_place_precond_leaves_its_input_alone():
    hes, vec = _operands(2)
    before = vec.copy()
    ff.field_precond(hes, vec, **FIELD_PEN, **KW)
    np.testing.assert_array_equal(vec, before)


# --------------------------------------------------------------------------- #
# field_cg                                                                    #
# --------------------------------------------------------------------------- #


def test_field_cg_solves_what_field_forward_applies():
    hes, grd = _operands(2, seed=3)
    sol = np.zeros_like(grd)
    returned = ff.field_cg(
        sol, hes, grd, **FIELD_PEN, **KW, nb_iter=200, tol=1e-12
    )
    assert returned is sol  # in place, like field_relax
    applied = ff.field_forward(hes, sol, **FIELD_PEN, **KW)
    np.testing.assert_allclose(applied, grd, rtol=1e-8, atol=1e-10)


def test_field_cg_return_info():
    hes, grd = _operands(2, seed=3)
    sol, info = ff.field_cg(
        np.zeros_like(grd),
        hes,
        grd,
        **FIELD_PEN,
        **KW,
        nb_iter=200,
        tol=1e-12,
        return_info=True,
    )
    assert isinstance(info.iterations, int)
    assert 0 < info.iterations <= 200
    assert info.residual < 1e-10
    # The reported residual is the real one.
    applied = ff.field_forward(hes, sol, **FIELD_PEN, **KW)
    rel = np.linalg.norm(applied - grd) / np.linalg.norm(grd)
    np.testing.assert_allclose(info.residual, rel, rtol=1e-3, atol=1e-12)


def test_field_cg_honours_the_iteration_cap():
    hes, grd = _operands(2, seed=5)
    _, info = ff.field_cg(
        np.zeros_like(grd),
        hes,
        grd,
        **FIELD_PEN,
        **KW,
        nb_iter=3,
        tol=0.0,
        return_info=True,
    )
    assert info.iterations <= 3


def test_field_cg_warm_start():
    hes, grd = _operands(2, seed=4)
    sol = np.zeros_like(grd)
    ff.field_cg(sol, hes, grd, **FIELD_PEN, **KW, nb_iter=200, tol=1e-12)
    _, info = ff.field_cg(
        sol.copy(),
        hes,
        grd,
        **FIELD_PEN,
        **KW,
        nb_iter=200,
        tol=1e-8,
        return_info=True,
    )
    assert info.iterations <= 2


# --------------------------------------------------------------------------- #
# stencil accumulate family                                                   #
# --------------------------------------------------------------------------- #


def _field_kernel():
    return ff.field_kernel(
        NDIM, **FIELD_PEN, channels=2, voxel_size=VS, bound="dct2"
    )


def _flow_kernel():
    return ff.flow_kernel(NDIM, **FLOW_PEN, voxel_size=VS, bound="dct2")


def test_field_addkernel_on_zeros_is_field_kernel():
    k = _field_kernel()
    np.testing.assert_allclose(
        ff.field_addkernel(np.zeros_like(k), **FIELD_PEN, **KW),
        k,
        rtol=1e-12,
        atol=1e-12,
    )


def test_field_subkernel_on_zeros_is_minus_field_kernel():
    k = _field_kernel()
    np.testing.assert_allclose(
        ff.field_subkernel(np.zeros_like(k), **FIELD_PEN, **KW),
        -k,
        rtol=1e-12,
        atol=1e-12,
    )


def test_flow_addkernel_on_zeros_is_flow_kernel():
    k = _flow_kernel()
    np.testing.assert_allclose(
        ff.flow_addkernel(np.zeros_like(k), **FLOW_PEN, **KW),
        k,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "family,kernel,pen",
    [("field", _field_kernel, FIELD_PEN), ("flow", _flow_kernel, FLOW_PEN)],
)
def test_add_then_sub_kernel_round_trips(family, kernel, pen):
    start = kernel() * 0.5
    acc = getattr(ff, f"{family}_addkernel")(start, **pen, **KW)
    back = getattr(ff, f"{family}_subkernel")(acc, **pen, **KW)
    np.testing.assert_allclose(back, start, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
@pytest.mark.parametrize("sign", ["add", "sub"])
def test_inplace_kernel_mutates_and_matches_functional(family, pen, sign):
    kernel = _field_kernel if family == "field" else _flow_kernel
    start = kernel() * 0.25
    expected = getattr(ff, f"{family}_{sign}kernel")(start, **pen, **KW)

    buf = start.copy()
    returned = getattr(ff, f"{family}_{sign}kernel_")(buf, **pen, **KW)
    assert returned is buf
    np.testing.assert_allclose(buf, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "family,pen", [("field", FIELD_PEN), ("flow", FLOW_PEN)]
)
def test_functional_kernel_accumulate_leaves_input_alone(family, pen):
    kernel = _field_kernel if family == "field" else _flow_kernel
    start = kernel() * 0.25
    before = start.copy()
    getattr(ff, f"{family}_addkernel")(start, **pen, **KW)
    np.testing.assert_array_equal(start, before)
