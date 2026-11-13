"""nki_scstat.py: switch 9s (R2_REFRESH_STATS, lane THR10, work/p2/thr10/NOTES.md): the fp8 scale refresh's weight
statistics in one NKI pass a unit. Under switches 9c and 9d every optimizer step ends with two compiled torch graphs
that read every MLP and attention weight (and both value-embedding tables) to form each layer's scale row: their
amax, the largest row norm of W_fc and W_v, the largest column norm of W_proj (mlp8_scales_into 6.32 ms and
attn8_scales_into 4.72 ms a step on PF6's merged line, 1.33 GB read: about 120 GB/s). Here each unit's statistics
come from one kernel call that streams each weight once by natural row DMAs (the two programs of an LNC 2 rank take
one weight each, or half a table each), and torch forms the same scale row from them with the scale module's own
functions (mlp_row, attn_row: forward_scales' tail verbatim).

Every statistics output is [nprg x 128, W] fp32: program k assembles its columns in one [128, W] SBUF tile (zeroed
first) and stores it once into rows k x 128 to k x 128 + 127 (switch 9i's resmix form: one store a program, no
4-byte column stores). nprg is 2 under kernel[2] (LNC 2) and 1 under a plain call; programs() says which, and the
torch functions read the program rows they need (_prog).
  _mlp_stats_kernel(w_fc, w_proj) -> st [nprg x 128, 8] fp32, cs [128, F] fp32
      W_fc [F, d] bf16 by its F / 128 row tiles, in program 0's rows: column 0 the max and column 1 the min of each
      partition's elements, column 2 the largest of each partition's row sums of squares (each row on the free axis:
      activation_reduce, square then add, in fp32). W_proj [d, F] by its d / 128 row tiles, in the last program's
      rows: column 3 max, column 4 min; cs[p, j] the sum over the partition's rows of W_proj[., j]^2 (fp32), torch
      adding the 128 partitions for the column norms.
  _attn_stats_kernel(wq, wk, wv, wp) -> st [nprg x 128, 16] fp32: max and min of each of the four [d, d] weights
      (program 0's rows: columns 0-1 q, 2-3 k; the last program's rows: 4-5 v, 7-8 p) and column 6 (the last
      program's rows) the largest of each partition's row sums of squares of W_v.
  _amax_stats_kernel(t) -> st [nprg x 128, 2] fp32: each program's max (column 0) and min (column 1) over its share
      of a [V, d] table's row tiles (half each under kernel[2]).

What is exact: every max and min (an element's value), so amax = max(max, -min) is bit for bit forward_scales' abs
amax, and every power-of-two scale that depends on amax alone (s_w, i_w and the descales); vb (|lambda| x the table's
amax). What is not: the row and column sums of squares are fp32 sums in another order than torch's (the squares
themselves are exact: a bf16 value squared fits fp32), so rho2, kappa2 and rho_v2 may differ from forward_scales' in
their last bits, and the power-of-two scales derived from them (s_h, the backward's s_dp, s_y) differ only when a
value sits within those bits of a power-of-two boundary. sim_st.py measures both on random and trained-scale weights.

Written for the device-side NKI parser frontend under switch 9c's rules: every loop `for name in range(...)` with one
plain target; `if` on trace-time integers only (program_id included); no list, comprehension, `while`, conditional
expression or chained assignment in a traced body; every slice bound a plain name or a literal; no PSUM. Shapes (the
torch callers in train.py check them): every row count a multiple of 128 and at least 256.
"""

import os

import torch

_P = 128
_LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
ST_MLP = 8
ST_ATTN = 16
ST_AMAX = 2

_HAS_NKI = False
_NKI_ERROR = None
try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    _VEC = nisa.engine.vector

    def _shard():
        if nl.program_ndim() > 0:
            return nl.program_id(0), nl.num_programs(0)
        return 0, 1

    def _span(n, pid, nprg):
        per = n // nprg
        rem = n - per * nprg
        i0 = pid * per
        if pid < rem:
            i0 = i0 + pid
        else:
            i0 = i0 + rem
        cnt = per
        if pid < rem:
            cnt = cnt + 1
        return i0, i0 + cnt

    def _rows_stats(w, R, K, mx, mn, rs, t0, t1, dt):
        """Row tiles t0 .. t1 - 1 of w [R, K]: mx, mn [P, 1] fp32 the running max and min of each partition's
        elements; rs [P, 1] fp32 (when given) the running max of each partition's row sums of squares."""
        f32 = nl.float32
        for i in range(t1 - t0):
            r0 = (t0 + i) * _P
            r1 = r0 + _P
            x = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=w[r0:r1, 0:K])
            tmx = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            tmn = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tmx, op=nl.maximum, data=x, axis=1)
            nisa.tensor_reduce(dst=tmn, op=nl.minimum, data=x, axis=1)
            if rs is not None:
                sq = nl.ndarray((_P, K), dtype=f32, buffer=nl.sbuf)
                trs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=trs)
            if i == 0:
                nisa.tensor_copy(dst=mx, src=tmx, engine=_VEC)
                nisa.tensor_copy(dst=mn, src=tmn, engine=_VEC)
                if rs is not None:
                    nisa.tensor_copy(dst=rs, src=trs, engine=_VEC)
            else:
                nisa.tensor_tensor(dst=mx, data1=mx, data2=tmx, op=nl.maximum)
                nisa.tensor_tensor(dst=mn, data1=mn, data2=tmn, op=nl.minimum)
                if rs is not None:
                    nisa.tensor_tensor(dst=rs, data1=rs, data2=trs, op=nl.maximum)

    def _col_stats(w, R, K, mx, mn, cs_acc, dt):
        """Every row tile of w [R, K]: mx, mn as _rows_stats; cs_acc [P, K] fp32 the sum over each partition's rows
        of the squares (exact squares, fp32 adds in row-tile order)."""
        f32 = nl.float32
        NR = R // _P
        for i in range(NR):
            r0 = i * _P
            r1 = r0 + _P
            x = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=w[r0:r1, 0:K])
            tmx = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            tmn = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tmx, op=nl.maximum, data=x, axis=1)
            nisa.tensor_reduce(dst=tmn, op=nl.minimum, data=x, axis=1)
            sq = nl.ndarray((_P, K), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sq, op=nl.square, data=x)
            if i == 0:
                nisa.tensor_copy(dst=mx, src=tmx, engine=_VEC)
                nisa.tensor_copy(dst=mn, src=tmn, engine=_VEC)
                nisa.tensor_copy(dst=cs_acc, src=sq, engine=_VEC)
            else:
                nisa.tensor_tensor(dst=mx, data1=mx, data2=tmx, op=nl.maximum)
                nisa.tensor_tensor(dst=mn, data1=mn, data2=tmn, op=nl.minimum)
                nisa.tensor_tensor(dst=cs_acc, data1=cs_acc, data2=sq, op=nl.add)

    def _put_col(tile, c, src):
        """tile[:, c] = src [P, 1] (one VectorE copy into the program's assembled tile)."""
        c1 = c + 1
        nisa.tensor_copy(dst=tile[0:_P, c:c1], src=src, engine=_VEC)

    def _store_rows(st, pid, tile, W):
        """st[pid x 128 : pid x 128 + 128, 0:W] = tile [P, W] (one DMA: the program's own rows)."""
        p0 = pid * _P
        p1 = p0 + _P
        nisa.dma_copy(dst=st[p0:p1, 0:W], src=tile)

    @nki.jit
    def _mlp_stats_kernel(w_fc, w_proj):
        """Switch 9s: one MLP unit's statistics (the module docstring). Program 0 reads W_fc, the last program
        W_proj (one program reads both under a plain call)."""
        F = w_fc.shape[0]
        D = w_fc.shape[1]
        dt = w_fc.dtype
        f32 = nl.float32
        pid, nprg = _shard()
        st = nl.ndarray((nprg * _P, ST_MLP), dtype=f32, buffer=nl.shared_hbm)
        cs = nl.ndarray((_P, F), dtype=f32, buffer=nl.shared_hbm)
        tile = nl.ndarray((_P, ST_MLP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(tile, value=0.0)
        if pid == 0:
            mx = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            mn = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            rs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            _rows_stats(w_fc, F, D, mx, mn, rs, 0, F // _P, dt)
            _put_col(tile, 0, mx)
            _put_col(tile, 1, mn)
            _put_col(tile, 2, rs)
        if pid == nprg - 1:
            mx2 = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            mn2 = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            acc = nl.ndarray((_P, F), dtype=f32, buffer=nl.sbuf)
            _col_stats(w_proj, D, F, mx2, mn2, acc, dt)
            _put_col(tile, 3, mx2)
            _put_col(tile, 4, mn2)
            nisa.dma_copy(dst=cs[0:_P, 0:F], src=acc)
        _store_rows(st, pid, tile, ST_MLP)
        return st, cs

    @nki.jit
    def _attn_stats_kernel(wq, wk, wv, wp):
        """Switch 9s: one attention layer's statistics (the module docstring). Program 0 reads W_q and W_k, the last
        program W_v and W_p (one program reads all four under a plain call)."""
        D = wq.shape[1]
        dt = wq.dtype
        f32 = nl.float32
        NR = D // _P
        pid, nprg = _shard()
        st = nl.ndarray((nprg * _P, ST_ATTN), dtype=f32, buffer=nl.shared_hbm)
        tile = nl.ndarray((_P, ST_ATTN), dtype=f32, buffer=nl.sbuf)
        nisa.memset(tile, value=0.0)
        if pid == 0:
            a = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            b = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            _rows_stats(wq, D, D, a, b, None, 0, NR, dt)
            _put_col(tile, 0, a)
            _put_col(tile, 1, b)
            c = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            e = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            _rows_stats(wk, D, D, c, e, None, 0, NR, dt)
            _put_col(tile, 2, c)
            _put_col(tile, 3, e)
        if pid == nprg - 1:
            g = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            h = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            rs = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            _rows_stats(wv, D, D, g, h, rs, 0, NR, dt)
            _put_col(tile, 4, g)
            _put_col(tile, 5, h)
            _put_col(tile, 6, rs)
            u = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            v = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
            _rows_stats(wp, D, D, u, v, None, 0, NR, dt)
            _put_col(tile, 7, u)
            _put_col(tile, 8, v)
        _store_rows(st, pid, tile, ST_ATTN)
        return st

    @nki.jit
    def _amax_stats_kernel(t):
        """Switch 9s: each program's max and min over its share of a [V, d] table's row tiles (half each under
        kernel[2]; all of them under a plain call), in its own rows."""
        V = t.shape[0]
        D = t.shape[1]
        dt = t.dtype
        f32 = nl.float32
        pid, nprg = _shard()
        st = nl.ndarray((nprg * _P, ST_AMAX), dtype=f32, buffer=nl.shared_hbm)
        t0, t1 = _span(V // _P, pid, nprg)
        mx = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
        mn = nl.ndarray((_P, 1), dtype=f32, buffer=nl.sbuf)
        _rows_stats(t, V, D, mx, mn, None, t0, t1, dt)
        tile = nl.ndarray((_P, ST_AMAX), dtype=f32, buffer=nl.sbuf)
        _put_col(tile, 0, mx)
        _put_col(tile, 1, mn)
        _store_rows(st, pid, tile, ST_AMAX)
        return st

    _HAS_NKI = True
except Exception as exc:  # nki unavailable (CPU dev box): the references below still work
    _NKI_ERROR = f"{type(exc).__name__}: {exc}"


def _launch(kernel):
    """kernel[2] under an LNC=2 runtime (one weight a physical core), plain call otherwise."""
    return kernel[_LNC] if _LNC == 2 else kernel


def programs() -> int:
    """The programs a launch runs (_launch's): a statistics output holds programs() x 128 rows."""
    return 2 if _LNC == 2 else 1


# ----------------------------------------------------------------------------- torch: the rows from the statistics

def _prog(st: torch.Tensor, k: int) -> torch.Tensor:
    """Program k's 128 rows of a statistics output [nprg x 128, W] (k = -1: the last program's)."""
    n = st.shape[0] // _P
    k = k % n
    return st[k * _P:(k + 1) * _P]


def _amax(st, c_max, c_min):
    return torch.maximum(st[:, c_max].amax(), -st[:, c_min].amin())


def mlp_row(M, st: torch.Tensor, cs: torch.Tensor, d: int) -> torch.Tensor:
    """Switch 9s: switch 9c's scale row [NSC] fp32 from _mlp_stats_kernel's statistics, by M's (the 9c module's) own
    functions and constants: nki_mlp_fused8.forward_scales' tail verbatim, with amax = max(max, -min) (exact),
    rho2 = the largest row sum of squares, kappa2 = the largest column sum (the partitions' partial sums added)."""
    s0, s1 = _prog(st, 0), _prog(st, -1)
    amax_fc = _amax(s0, 0, 1)
    amax_pj = _amax(s1, 3, 4)
    s_wfc, i_wfc = M._pow2_le(amax_fc, M.E4M3_MAX, M.E_W[0], M.E_W[1])
    s_wpj, i_wpj = M._pow2_le(amax_pj, M.E4M3_MAX, M.E_W[0], M.E_W[1])
    rho2 = s0[:, 2].amax()
    kappa2 = cs.sum(dim=0).amax()
    s_h, i_h, c_h, _ic = M._pow4_le(rho2 * (M.H_BOUND_FACTOR * d), M.E4M3_MAX, M.E_H[0], M.E_H[1])
    zero = torch.zeros_like(amax_fc)
    return torch.stack([s_wfc, i_wfc * (1.0 / M.X_SCALE), c_h, s_wpj, i_h * i_wpj, zero, zero, zero, zero, rho2, kappa2,
                        i_wfc, i_wpj, s_h, amax_fc, amax_pj])


def table_amax(st: torch.Tensor) -> torch.Tensor:
    """Switch 9s: a table's exact abs amax from _amax_stats_kernel's statistics (every program's rows)."""
    a = _amax(_prog(st, 0), 0, 1)
    for k in range(1, st.shape[0] // _P):
        a = torch.maximum(a, _amax(_prog(st, k), 0, 1))
    return a


def attn_row(A, st: torch.Tensor, vb: torch.Tensor, d: int) -> torch.Tensor:
    """Switch 9s: switch 9d's scale row [NSC] fp32 from _attn_stats_kernel's statistics and vb, by A's (the 9d
    module's) own functions and constants: forward_scales' tail verbatim (amax exact, rho_v2 the largest row sum)."""
    s0, s1 = _prog(st, 0), _prog(st, -1)
    amax_qkv = torch.stack([_amax(s0, 0, 1), _amax(s0, 2, 3), _amax(s1, 4, 5)]).amax()
    amax_p = _amax(s1, 7, 8)
    s_wqkv, i_wqkv = A._pow2_le(amax_qkv, A.E4M3_MAX, A.E_W[0], A.E_W[1])
    s_wp, i_wp = A._pow2_le(amax_p, A.E4M3_MAX, A.E_W[0], A.E_W[1])
    rho_v2 = s1[:, 6].amax()
    vbf = vb.detach().float().reshape(())
    b_y = A.y_bound(rho_v2, d, vbf)
    s_y, i_y = A._pow2_le(b_y, A.E4M3_MAX, A.E_W[0], A.E_W[1])
    zero = torch.zeros_like(amax_qkv)
    s_x = torch.full_like(amax_qkv, A.X_SCALE)
    return torch.stack([s_x, s_wqkv, i_wqkv * (1.0 / A.X_SCALE), s_y, s_wp, i_y * i_wp, i_wqkv, i_wp, rho_v2, amax_qkv,
                        amax_p, b_y, vbf, zero, zero, zero])


# ----------------------------------------------------------------------------- references (CPU, proofs)

def _part(w: torch.Tensor) -> torch.Tensor:
    """w [R, K] as [R / 128, 128, K] fp32: partition p holds rows p, p + 128, ... (the kernels' row tiles)."""
    return w.float().reshape(w.shape[0] // _P, _P, w.shape[1])


def mlp_stats_ref(w_fc: torch.Tensor, w_proj: torch.Tensor, nprg: int = 0):
    """What _mlp_stats_kernel computes, in torch on the same bf16 values (the sums in torch's order), in its layout:
    st [nprg x 128, 8] (program 0's rows columns 0-2, the last program's columns 3-4, the rest 0), cs [128, F]."""
    nprg = nprg or programs()
    a, b = _part(w_fc), _part(w_proj)
    st = torch.zeros(nprg * _P, ST_MLP)
    q = (nprg - 1) * _P
    st[0:_P, 0] = a.amax(dim=(0, 2))
    st[0:_P, 1] = a.amin(dim=(0, 2))
    st[0:_P, 2] = (a * a).sum(dim=2).amax(dim=0)
    st[q:q + _P, 3] = b.amax(dim=(0, 2))
    st[q:q + _P, 4] = b.amin(dim=(0, 2))
    cs = (b * b).sum(dim=0)
    return st, cs


def attn_stats_ref(wq, wk, wv, wp, nprg: int = 0):
    """What _attn_stats_kernel computes, in torch, in its layout: st [nprg x 128, 16] (program 0's rows columns 0-3,
    the last program's columns 4-8, the rest 0)."""
    nprg = nprg or programs()
    st = torch.zeros(nprg * _P, ST_ATTN)
    q = (nprg - 1) * _P
    for c, w, r0 in ((0, wq, 0), (2, wk, 0), (4, wv, q), (7, wp, q)):
        x = _part(w)
        st[r0:r0 + _P, c] = x.amax(dim=(0, 2))
        st[r0:r0 + _P, c + 1] = x.amin(dim=(0, 2))
    v = _part(wv)
    st[q:q + _P, 6] = (v * v).sum(dim=2).amax(dim=0)
    return st


def amax_stats_ref(t: torch.Tensor, nprg: int = 0):
    """What _amax_stats_kernel computes, in torch, in its layout: st [nprg x 128, 2], program k's rows the max and
    min over its share of the row tiles (the kernel's split: the first V / 128 mod nprg programs one tile more)."""
    nprg = nprg or programs()
    x = _part(t)
    n = x.shape[0]
    per, rem = n // nprg, n % nprg
    st = torch.zeros(nprg * _P, ST_AMAX)
    i0 = 0
    for k in range(nprg):
        cnt = per + (1 if k < rem else 0)
        st[k * _P:(k + 1) * _P, 0] = x[i0:i0 + cnt].amax(dim=(0, 2))
        st[k * _P:(k + 1) * _P, 1] = x[i0:i0 + cnt].amin(dim=(0, 2))
        i0 += cnt
    return st


def cc_cases(z, bf16, f32):
    """cc_b2.py's device-compile cases for this module at the training shapes (numpy zeros)."""
    D = 1536
    return {"st_mlp_6144": (_mlp_stats_kernel, dict(w_fc=z((6144, D), dtype=bf16), w_proj=z((D, 6144), dtype=bf16))),
            "st_mlp_3072": (_mlp_stats_kernel, dict(w_fc=z((3072, D), dtype=bf16), w_proj=z((D, 3072), dtype=bf16))),
            "st_attn": (_attn_stats_kernel, dict(wq=z((D, D), dtype=bf16), wk=z((D, D), dtype=bf16),
                                                 wv=z((D, D), dtype=bf16), wp=z((D, D), dtype=bf16))),
            "st_ve": (_amax_stats_kernel, dict(t=z((32768, D), dtype=f32)))}
