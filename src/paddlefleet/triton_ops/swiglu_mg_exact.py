# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""与 MG（torch.compile 出来的 Triton kernel）**逐位相同**的 SwiGLU。

## 为什么非得用 Triton 复刻

MG 的 `swiglu` / `weighted_swiglu`（`megatron/core/fusions/fused_bias_swiglu.py:15,69`）
带 `@jit_fuser`，而 `jit_fuser = torch.compile`（`megatron/core/jit.py:8,20`），
所以基线其实是 Inductor 生成的一个 Triton kernel。抓出它的源码（`TORCH_LOGS=output_code`）：

    tmp0  = load(x1).to(fp32)
    tmp8  = load(x2).to(fp32)
    tmp11 = load(w)                      # 每行一个
    tmp3  = libdevice.exp(-tmp0)
    tmp5  = tmp3 + 1.0
    tmp6  = tmp0 / tmp5                  # ← **Triton 的普通除法**
    tmp9  = tmp6 * tmp8
    tmp12 = tmp9 * tmp11
    store(bf16(tmp12))                   # 全程 fp32，只在存的时候 round 一次

关键在 `tmp6 = tmp0 / tmp5`：**Triton 的 `/` 不是 round-to-nearest**
（本目录 `situ_glu.py:34` 已经踩过这个坑，那里专门用 `libdevice.div_rn` 换回 RN）。
实测 torch 侧「编译版 silu」与「eager 版 silu」在 fp32 上差
**1922/112000 个元素、各差 1 个 ULP**；而 Paddle 的 `F.silu`、
`x/(1+exp(-x))`、`x*sigmoid(x)` 都与 **eager** 版一致，因此都对不上编译版。

这 1 个 fp32 ULP 平时被最后的 bf16 舍入吸收掉，但只要乘积正好落在两个 bf16 的
**中点**上就会翻位 —— 实测 64 个专家里有 2 个元素这样翻掉（P-11）。
所以只能用同样的 Triton `/`（即普通 `/`，**不要** `div_rn`）把这条链复刻一遍。

## 判据

`tests/single_card_tests/models/run_moe_experts_cmp.py` 里 64 个专家的
`fc2 入(激活后)` 与 `专家出口` 全部逐位相同。
"""

from __future__ import annotations

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra.cuda import libdevice  # noqa: E402


@enable_compat_on_triton_kernel
@triton.jit
def _swiglu_fwd_kernel(  # pragma: no cover - triton kernel body compiles to PTX
    x_ptr,
    w_ptr,
    out_ptr,
    n_cols,
    xnumel,
    HAS_W: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """逐元素照抄 Inductor 生成的那条链，顺序与舍入点一个不改。"""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # 长上下文下 row * stride 会超 int32，偏移一律用 int64（抄 situ_glu.py 的做法）
    idx = xindex.to(tl.int64)
    col = idx % n_cols
    row = idx // n_cols
    base = row * (2 * n_cols) + col

    gate = tl.load(x_ptr + base, mask=xmask).to(tl.float32)
    up = tl.load(x_ptr + base + n_cols, mask=xmask).to(tl.float32)

    # 这里必须是普通 `/`，与 Inductor 生成的 `tmp0 / tmp5` 一致；
    # 换成 libdevice.div_rn 会得到「更正确」但与基线不同的结果。
    act = gate / (libdevice.exp(-gate) + 1.0)
    out = act * up
    if HAS_W:
        out = out * tl.load(w_ptr + row, mask=xmask, eviction_policy="evict_last").to(
            tl.float32
        )
    tl.store(out_ptr + idx, out, mask=xmask)


def swiglu_forward_triton(
    x: paddle.Tensor, weights: paddle.Tensor | None = None
) -> paddle.Tensor:
    """``SwiGLU(x)``（可选再乘 per-token 权重），与 MG 的编译版逐位相同。

    Args:
        x: ``[..., 2 * n_cols]``，前半是 gate、后半是 up（与 MG 的
            ``torch.chunk(y, 2, -1)`` 一致）。
        weights: ``[..., 1]`` 或 ``[...]`` 的 per-token 权重；``None`` 表示不乘。

    Returns:
        ``[..., n_cols]``，dtype 与 ``x`` 相同。
    """
    ori_shape = list(x.shape)
    n_cols = ori_shape[-1] // 2
    x2d = x.reshape([-1, ori_shape[-1]])
    rows = x2d.shape[0]
    out = paddle.empty([rows, n_cols], dtype=x.dtype)
    if rows == 0 or n_cols == 0:
        return out.reshape(ori_shape[:-1] + [n_cols])

    w = weights
    if w is not None:
        w = w.reshape([-1]).astype(paddle.float32)
        assert w.shape[0] == rows, (
            f"weights 的行数 {w.shape[0]} 与 x 的行数 {rows} 不一致"
        )

    xnumel = rows * n_cols
    XBLOCK = 1024
    grid = (triton.cdiv(xnumel, XBLOCK),)
    _swiglu_fwd_kernel[grid](
        x2d,
        w if w is not None else x2d,  # 不用时给个可读指针占位
        out,
        n_cols,
        xnumel,
        HAS_W=w is not None,
        XBLOCK=XBLOCK,
        num_warps=4,
    )
    return out.reshape(ori_shape[:-1] + [n_cols])


# ---------------------------------------------------------------------------
# 反向：同样照抄 Inductor 生成的两个 kernel
# ---------------------------------------------------------------------------
# MG 的 `swiglu_back` / `weighted_swiglu_back` 也带 @jit_fuser，抓出来是两个 kernel：
#
# ① input_grad（pointwise，输出 [rows, 2*n_cols]）
#      前半（对 y_1 求导）用的是 **tl.sigmoid**：
#          out = ((gp * s) * ((y1 * (1 - s)) + 1)) * y2      s = tl.sigmoid(y1)
#      后半（对 y_2 求导）用的是 **exp + 普通除法**（来自 F.silu）：
#          out = gp * (y1 / (exp(-y1) + 1))
#      其中 gp = g（不带权重时）或 g * w（带权重时）。
#      注意前后两半用了**不同**的 sigmoid 实现 —— 这不是笔误，是 MG 源码里
#      `torch.sigmoid(y_1)` 与 `F.silu(y_1)` 分别被 Inductor 展开的结果。
#
# ② weights_grad（行归约，输出 [rows, 1] fp32）
#          sum_r( ((y1 / (exp(-y1) + 1)) * y2) * g )
#      归约用 `tl.sum(..., 1)`，块宽是 `next_power_of_2(n_cols)`（896 → 1024），
#      掩掉的 lane 填 0。**块宽会改变归约树的形状**，所以这里必须取同一个值。


@enable_compat_on_triton_kernel
@triton.jit
def _swiglu_bwd_input_kernel(  # pragma: no cover - triton kernel body compiles to PTX
    g_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    n_cols,
    xnumel,
    HAS_W: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    idx = xindex.to(tl.int64)
    col = idx % (2 * n_cols)
    row = idx // (2 * n_cols)
    is_gate = col < n_cols
    half_col = tl.where(is_gate, col, col - n_cols)

    g = tl.load(g_ptr + row * n_cols + half_col, mask=xmask, other=0.0).to(tl.float32)
    if HAS_W:
        g = g * tl.load(w_ptr + row, mask=xmask, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + row * (2 * n_cols) + half_col, mask=xmask, other=0.0).to(
        tl.float32
    )

    # 前半：tl.sigmoid（对应 MG 的 torch.sigmoid(y_1)）
    s = tl.sigmoid(gate)
    up = tl.load(
        x_ptr + row * (2 * n_cols) + n_cols + half_col, mask=xmask, other=0.0
    ).to(tl.float32)
    d_gate = ((g * s) * ((gate * (1.0 - s)) + 1.0)) * up
    # 后半：exp + 普通除法（对应 MG 的 F.silu(y_1)），普通 `/` 不能换 div_rn
    d_up = g * (gate / (libdevice.exp(-gate) + 1.0))

    tl.store(out_ptr + idx, tl.where(is_gate, d_gate, d_up), mask=xmask)


# Inductor 对 weights_grad 那个归约用的是 `persistent_reduction`，
# `size_hints={'x':128,'r0_':1024}`。XBLOCK 与 num_warps 会改变归约树的形状
# ⇒ 直接影响 fp32 的最后 1 位。下面两个值是实测扫出来能与基线逐位相同的组合
# （判据脚本：tests/single_card_tests/models/test_swiglu_bwd_parity.py）。
_WGRAD_XBLOCK = 1
# ⚠️ `num_warps` **不能是常量** —— 它必须随 `R0_BLOCK` 变。
# Inductor 的 persistent_reduction 是按归约块大小挑 warp 数的，实测（判据：
# `megatron_ref/probe_swiglu_bwd_k.py`，稠密输入、两侧比 `weights_grad`）：
#   n_cols=896（TP=1）→ R0_BLOCK=1024 → **num_warps=8** 逐位相同
#   n_cols=448（TP=2）→ R0_BLOCK= 512 → **num_warps=4** 逐位相同
#                                        （8 的话差 88/126，≤1.5 fp32 ULP）
# 两个点都落在 `R0_BLOCK / 128` 上。
# 原来写死 8 是因为当初只在 896 这一个形状上扫过 —— TP>1 把 `moe_ffn_hidden`
# 减半之后就落到了没验过的形状上，这正是 Gate 8 TP 档 `router.weight` 那
# 69/81920 个 bf16 差异的源头（doc/06 P-20 五十八/五十九）。
_WGRAD_NUM_WARPS = 8  # 仅作为 R0_BLOCK=1024 时的取值 / 显式传参时的默认


def _wgrad_num_warps_for(r0_block: int) -> int:
    """按归约块大小推 `num_warps`，照 Inductor 的 heuristic。

    实测锚点：R0_BLOCK=1024 → 8、R0_BLOCK=512 → 4。
    夹在 [1, 8] 之间（再大 Inductor 也不会超过 8）。
    """
    return max(1, min(8, r0_block // 128))


@enable_compat_on_triton_kernel
@triton.jit
def _swiglu_bwd_wgrad_kernel(  # pragma: no cover - triton kernel body compiles to PTX
    x_ptr,
    g_ptr,
    out_ptr,
    xnumel,
    r0_numel,
    n_cols,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """行归约，**二维 block 布局照抄 Inductor 的 persistent_reduction**。

    归约树的形状由 `[XBLOCK, R0_BLOCK]` 的形状和 `num_warps` 共同决定，
    写成一维 `tl.arange(R0_BLOCK)` + `tl.sum(..., 0)` 得到的结果会差 1 个 fp32 ULP
    （实测 e44 有 12/39 行不同）。所以这里必须保持二维 + `tl.sum(..., 1)`。
    """
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    row = xindex.to(tl.int64)
    r = r0_index.to(tl.int64)

    gate = tl.load(x_ptr + row * (2 * n_cols) + r, r0_mask & xmask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        x_ptr + row * (2 * n_cols) + n_cols + r, r0_mask & xmask, other=0.0
    ).to(tl.float32)
    g = tl.load(g_ptr + row * n_cols + r, r0_mask & xmask, other=0.0).to(tl.float32)

    silu = gate / (libdevice.exp(-gate) + 1.0)
    prod = (silu * up) * g
    acc = tl.broadcast_to(prod, [XBLOCK, R0_BLOCK])
    acc = tl.where(r0_mask & xmask, acc, 0)
    tl.store(out_ptr + xindex, tl.sum(acc, 1)[:, None], xmask)


def swiglu_backward_triton(
    grad_out: paddle.Tensor,
    x: paddle.Tensor,
    weights: paddle.Tensor | None = None,
    *,
    wgrad_xblock: int = _WGRAD_XBLOCK,
    wgrad_num_warps: int = _WGRAD_NUM_WARPS,
):
    """``SwiGLU`` 的反向，与 MG 编译版逐位相同。

    Returns:
        ``(input_grad, weights_grad)``；``weights is None`` 时 ``weights_grad`` 为 ``None``。
    """
    ori_shape = list(x.shape)
    n_cols = ori_shape[-1] // 2
    x2d = x.reshape([-1, ori_shape[-1]])
    g2d = grad_out.reshape([-1, n_cols])
    rows = x2d.shape[0]
    input_grad = paddle.empty([rows, ori_shape[-1]], dtype=x.dtype)

    w = None
    if weights is not None:
        w = weights.reshape([-1]).astype(paddle.float32)

    if rows and n_cols:
        xnumel = rows * ori_shape[-1]
        XBLOCK = 1024
        _swiglu_bwd_input_kernel[(triton.cdiv(xnumel, XBLOCK),)](
            g2d,
            w if w is not None else g2d,
            x2d,
            input_grad,
            n_cols,
            xnumel,
            HAS_W=w is not None,
            XBLOCK=XBLOCK,
            num_warps=4,
        )

    weights_grad = None
    if w is not None:
        weights_grad = paddle.empty([rows, 1], dtype=paddle.float32)
        if rows and n_cols:
            R0_BLOCK = triton.next_power_of_2(n_cols)
            # 调用方没显式指定时，按 R0_BLOCK 推 —— 不要用那个写死的 8
            if wgrad_num_warps is _WGRAD_NUM_WARPS:
                wgrad_num_warps = _wgrad_num_warps_for(R0_BLOCK)
            _swiglu_bwd_wgrad_kernel[(triton.cdiv(rows, wgrad_xblock),)](
                x2d,
                g2d,
                weights_grad,
                rows,
                n_cols,
                n_cols,
                XBLOCK=wgrad_xblock,
                R0_BLOCK=R0_BLOCK,
                num_warps=wgrad_num_warps,
            )
        weights_grad = weights_grad.reshape(weights.shape)

    return input_grad.reshape(ori_shape), weights_grad


def wgrad_reduction_block_is_single_stage(n_cols: int) -> bool:
    """`tl.sum` 单块归约能覆盖 ``n_cols`` 吗（决定能否与 Inductor 的归约树对齐）。

    Inductor 对 896 列用的是 ``R0_BLOCK=1024`` 的单块归约。列数再大它会换成
    多阶段归约，归约树形状随之改变，本文件的单块 kernel 就**不再等价** ——
    那种情况下调用方应该退回 eager 实现，而不是假装对得上。
    """
    return triton.next_power_of_2(n_cols) <= 2048

