"""Tests for fastfields_numpy (the friendly numpy wrappers)."""

import numpy as np
import pytest

import fastfields.numpy as ff

# --------------------------------------------------------------------------- #
# distance transforms                                                         #
# --------------------------------------------------------------------------- #


def _dt_reference(inp, voxel_spacing, cost):
    """Brute force: out[..., i] = min_j inp[..., j] + cost(spacing*(i-j))."""
    n = inp.shape[-1]
    flat = inp.reshape(-1, n)
    out = np.full_like(flat, np.inf)
    for r in range(flat.shape[0]):
        for i in range(n):
            best = np.inf
            for j in range(n):
                best = min(best, flat[r, j] + cost(voxel_spacing * (i - j)))
            out[r, i] = best
    return out.reshape(inp.shape)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_matches_bruteforce(dtype):
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=dtype,
    )
    ref = _dt_reference(inp, 1.5, lambda d: d * d)
    out = ff.dt_euclidean(inp, voxel_spacing=1.5)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    # input must be untouched (returns a new array)
    assert np.isinf(inp[0, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_l1_matches_bruteforce(dtype):
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=dtype,
    )
    ref = _dt_reference(inp, 2.0, lambda d: abs(d))
    out = ff.dt_l1(inp, voxel_spacing=2.0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_dt_handles_noncontiguous_input():
    # Build a strided / transposed (non-C-contiguous) view; the wrapper must
    # copy it into a contiguous buffer and still produce the right answer.
    base = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=np.float64,
    )
    view = np.asfortranarray(base)  # F-contiguous, not C-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    ref = _dt_reference(base, 1.0, lambda d: d * d)
    out = ff.dt_euclidean(view)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

    # A transposed view of a 3d array (last axis is the strided one).
    vol = np.stack([base, base + 0.0], axis=0)  # (2,2,7)
    tview = np.swapaxes(vol, 1, 2)  # (2,7,2), strided last
    assert not tview.flags["C_CONTIGUOUS"]
    ref_t = _dt_reference(tview.astype(np.float64), 1.0, lambda d: d * d)
    out_t = ff.dt_euclidean(tview)
    np.testing.assert_allclose(out_t, ref_t, rtol=1e-6, atol=1e-6)


def test_dt_inplace():
    inp = np.array([[0, np.inf, np.inf, 0, np.inf]], dtype=np.float32)
    ref = _dt_reference(inp, 1.0, lambda d: d * d)
    ret = ff.dt_euclidean(inp, inplace=True)
    assert ret is inp
    np.testing.assert_allclose(inp, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# compact-symmetric linear algebra                                            #
# --------------------------------------------------------------------------- #


def _pack_symmetric(mats):
    """Dense (B,C,C) symmetric -> compact (B, C*(C+1)/2) diagonal-then-rows."""
    B, C, _ = mats.shape
    packed = np.zeros((B, C * (C + 1) // 2), dtype=mats.dtype)
    for b in range(B):
        idx = 0
        for k in range(C):
            packed[b, idx] = mats[b, k, k]
            idx += 1
        for i in range(C):
            for j in range(i + 1, C):
                packed[b, idx] = mats[b, i, j]
                idx += 1
    return packed


def _random_symmetric(B, C, seed, posdef=False):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((B, C, C))
    m = m + np.transpose(m, (0, 2, 1))
    if posdef:
        m = np.einsum("bij,bkj->bik", m, m) + C * np.eye(C)[None]
    return m


@pytest.mark.parametrize("C", [2, 3])
def test_sym_matvec_matches_dense(C):
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    packed = _pack_symmetric(mats)

    out = ff.sym_matvec(packed, vec)
    ref = np.einsum("bij,bj->bi", mats, vec)
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)
    assert out.shape == vec.shape


@pytest.mark.parametrize("C", [2, 3])
def test_sym_solve_inverts_matvec(C):
    B = 6
    mats = _random_symmetric(B, C, seed=C, posdef=True)
    packed = _pack_symmetric(mats)
    x = np.random.default_rng(7 + C).standard_normal((B, C))

    b = ff.sym_matvec(packed, x)
    x_rec = ff.sym_solve(packed, b)
    # the posdef solve accumulates in reduced precision, so loosen tolerance
    np.testing.assert_allclose(x_rec, x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_invert_then_matvec_is_identity(C):
    B = 4
    mats = _random_symmetric(B, C, seed=20 + C, posdef=True)
    packed = _pack_symmetric(mats)

    inv_packed = ff.sym_invert(packed)
    # (H^-1) applied to each column of H should give the identity.
    vecs = np.eye(C)[None].repeat(B, axis=0)  # (B, C, C): columns are e_k
    for k in range(C):
        col = np.ascontiguousarray(mats[:, :, k])  # H[:,k]
        rec = ff.sym_matvec(inv_packed, col)  # H^-1 @ H[:,k]
        np.testing.assert_allclose(rec, vecs[:, :, k], rtol=1e-5, atol=1e-5)


def test_sym_matvec_broadcasts_batch_dims():
    # hessian batch (1,) vs vec batch (5,): the wrapper must broadcast the
    # batch dims and produce the manually-broadcast dense result.
    C, B = 3, 5
    mats = _random_symmetric(1, C, seed=42)  # batch (1,)
    packed = _pack_symmetric(mats)  # (1, 6)
    vec = np.random.default_rng(7).standard_normal((B, C))  # batch (5,)

    out = ff.sym_matvec(packed, vec)
    assert out.shape == (B, C)
    dense = np.broadcast_to(mats, (B, C, C))
    ref = np.einsum("bij,bj->bi", dense, vec)
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


def test_sym_matvec_broadcast_is_zero_copy():
    # The big input must be broadcast as a 0-stride view (no copy). We probe
    # the wrapper's internal broadcast helper directly to assert zero-copy.
    from fastfields.numpy._util import _bcast_view

    big = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((1, 6))
    )
    view = _bcast_view(big, (100000, 6))
    assert view.shape == (100000, 6)
    assert view.strides[0] == 0  # broadcast axis has 0 stride
    assert np.shares_memory(view, big)  # no copy of the big input
    # end-to-end: mismatched batches still zero-copy on the larger operand
    packed = np.ascontiguousarray(
        _pack_symmetric(_random_symmetric(1, 2, seed=1))
    )
    vec = np.random.default_rng(2).standard_normal((256, 2))
    out = ff.sym_matvec(packed, vec)
    assert out.shape == (256, 2)


def test_sym_solve_broadcasts_weight():
    C, B = 3, 4
    mats = _random_symmetric(1, C, seed=99, posdef=True)  # batch (1,)
    packed = _pack_symmetric(mats)
    w = (
        np.abs(np.random.default_rng(1).standard_normal((B, C))) + 0.5
    )  # (4, C)
    x = np.random.default_rng(2).standard_normal((B, C))

    dense = np.broadcast_to(mats, (B, C, C))
    b = np.einsum("bij,bj->bi", dense, x) + w * x
    x_rec = ff.sym_solve(packed, b, weight=w)
    # the solve accumulates in reduced precision, so loosen tolerance
    np.testing.assert_allclose(x_rec, x, rtol=1e-4, atol=1e-4)


def test_sym_solve_with_weight():
    C, B = 3, 4
    mats = _random_symmetric(B, C, seed=99, posdef=True)
    packed = _pack_symmetric(mats)
    w = np.abs(np.random.default_rng(1).standard_normal((B, C))) + 0.5
    x = np.random.default_rng(2).standard_normal((B, C))

    # (H + diag(w)) x = b
    b = ff.sym_matvec(packed, x) + w * x
    x_rec = ff.sym_solve(packed, b, weight=w)
    np.testing.assert_allclose(x_rec, x, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# in-place add/sub matvec (C3: trailing-`_` must mutate the caller's array)   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("C", [2, 3])
def test_sym_addmatvec_mutates_caller_inplace(C):
    # Regression for fastfields-lib#17 (C3): the trailing-`_` variant must
    # write through the caller's accumulator, not return a private copy.
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    packed = _pack_symmetric(mats)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    out0 = np.random.default_rng(200 + C).standard_normal((B, C))

    expected = out0 + np.einsum("bij,bj->bi", mats, vec)
    ret = ff.sym_addmatvec_(out0, packed, vec)
    # returned object IS the caller's array, and it was updated in place.
    assert ret is out0
    np.testing.assert_allclose(out0, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_submatvec_mutates_caller_inplace(C):
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    packed = _pack_symmetric(mats)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    out0 = np.random.default_rng(200 + C).standard_normal((B, C))

    expected = out0 - np.einsum("bij,bj->bi", mats, vec)
    ret = ff.sym_submatvec_(out0, packed, vec)
    assert ret is out0
    np.testing.assert_allclose(out0, expected, rtol=1e-8, atol=1e-8)


def test_sym_addmatvec_broadcasts_onto_accumulator():
    # out0 fixes the batch shape; mat/vec (batch (1,)) broadcast onto it.
    C, B = 3, 4
    mats = _random_symmetric(1, C, seed=11)  # batch (1,)
    packed = _pack_symmetric(mats)  # (1, 6)
    vec = np.random.default_rng(3).standard_normal((1, C))  # batch (1,)
    out0 = np.zeros((B, C))

    ff.sym_addmatvec_(out0, packed, vec)
    dense = np.broadcast_to(mats, (B, C, C))
    ref = np.broadcast_to(np.einsum("bij,bj->bi", dense, vec), (B, C))
    np.testing.assert_allclose(out0, ref, rtol=1e-8, atol=1e-8)


def test_sym_addmatvec_channel_mismatch_raises():
    packed = np.zeros(3)  # encodes C=2
    vec = np.zeros(3)  # C=3
    out0 = np.zeros(3)
    with pytest.raises(ValueError):
        ff.sym_addmatvec_(out0, packed, vec)


# --------------------------------------------------------------------------- #
# spline coeff / resample                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_factor_one_identity(dtype):
    x = np.array([1, 2, 3, 4, 5], dtype=dtype)
    out = ff.resample(x, factor=1, order="linear", bound="dct2")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_resample_no_args_identity_2d_last_axis():
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = ff.resample(x, order="linear", ndim=1)  # identity on last axis
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_resample_upsample_shape_and_endpoints():
    x = np.arange(5, dtype=np.float64)
    out = ff.resample(x, factor=2, order="linear")
    # factor 2 -> 10 samples; align-corners keeps the endpoints and reproduces
    # the linear ramp over the same [0, 4] extent.
    assert out.shape == (10,)
    np.testing.assert_allclose(
        out, np.linspace(0, 4, 10), rtol=1e-6, atol=1e-6
    )


def test_resample_shape_arg():
    x = np.arange(4, dtype=np.float64)
    out = ff.resample(x, shape=7, order="linear")
    assert out.shape == (7,)
    np.testing.assert_allclose(out, np.linspace(0, 3, 7), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spline_coeff_shape_dtype_preserved(dtype):
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=dtype)
    out = ff.spline_coeff(x, order=3, bound="dct2")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # input not modified
    np.testing.assert_allclose(x, [[1, 2, 3, 4, 5]])


def test_spline_coeff_order1_noop():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out = ff.spline_coeff(x, order=1)
    np.testing.assert_allclose(out, x, rtol=1e-12, atol=1e-12)


def test_restriction_runs_and_shapes():
    x = np.arange(9, dtype=np.float64)
    out = ff.restriction(x, shape=5, order="linear")
    assert out.shape == (5,)
    assert out.dtype == x.dtype


# --------------------------------------------------------------------------- #
# argument normalisation                                                      #
# --------------------------------------------------------------------------- #


def test_enum_and_string_args_equivalent():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    a = ff.spline_coeff(x, order=ff.Spline.Cubic, bound=ff.Bound.DCT2)
    b = ff.spline_coeff(x, order="cubic", bound="dct2")
    c = ff.spline_coeff(x, order=3, bound=3)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, c)


def test_bad_spline_and_bound_raise():
    x = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        ff.spline_coeff(x, order="nonsense")
    with pytest.raises(ValueError):
        ff.spline_coeff(x, bound="nonsense")


def test_channels_from_packed():
    assert ff.sym_channels_from_packed(3) == 2
    assert ff.sym_channels_from_packed(6) == 3
    with pytest.raises(ValueError):
        ff.sym_channels_from_packed(4)


# ------------------------------------------------------------------------- #
# in-place on non-contiguous arrays (library is stride-aware -> zero copy)   #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inplace_noncontiguous(dtype):
    # A strided (non-contiguous) view of the last axis must be writable in
    # place: the write lands in the caller's buffer, no contiguous copy.
    base = np.full((3, 16), np.inf, dtype=dtype)
    base[:, 0] = 0.0
    base[:, 8] = 0.0
    x = base[:, ::2]  # shape (3, 8), stride 2 (non-contig)
    assert not x.flags["C_CONTIGUOUS"]
    ref = np.ascontiguousarray(x)
    ff.dt_euclidean(ref, inplace=True)
    ret = ff.dt_euclidean(x, inplace=True)
    assert ret is x  # returned the same object
    assert np.allclose(x, ref)  # correct result on the strided view
    assert np.allclose(base[:, ::2], ref)  # write landed in the parent buffer


def test_inplace_rejects_non_array_and_bad_dtype():
    with pytest.raises(TypeError):
        ff.dt_euclidean([0.0, 1.0], inplace=True)  # not ndarray
    with pytest.raises(TypeError):
        ff.dt_euclidean(
            np.zeros(4, dtype=np.int32), inplace=True
        )  # wrong dtype
