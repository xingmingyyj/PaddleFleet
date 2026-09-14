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

"""与 TransformerEngine `FusedAdam`（precision-aware + param remainder）**逐位相同**的 AdamW。

## 为什么非得用 Triton 复刻

HyperBody 的优化器配置（`recipes/hyperbody/hyperbody.py:200-205`）是

    use_precision_aware_optimizer = True
    main_params_dtype = fp32   exp_avg_dtype = bf16   exp_avg_sq_dtype = bf16

再叠上 `OptimizerConfig.store_param_remainders` 默认 `True`，MCore 传给 TE 的是
`master_weights=True / use_decoupled_grad=True / master_weight_dtype=fp32`
（`megatron/core/optimizer/__init__.py:562-583`），实际落到
`multi_tensor_adam_param_remainder` 这个 kernel 上。它有两处「照抄不可」的语义：

**(1) fp32 master 用「bf16 权重 + int16 余数」存，而且是带符号的位偏移。**

    master_bits = (uint32(p_bf16_bits) << 16) + int32(remainder)     # 带符号加
    p_bf16      = round_to_nearest_even(master)                      # 不是截断！

若按直觉理解成「截断 + 无符号低 16 位」，约**一半**元素的 bf16 权重会差 1 个 bf16 ULP
（实测 2033/4096）。

**(2) `libtransformer_engine.so` 是 `--use_fast_math` 编出来的。**

    $ strings libtransformer_engine.so | grep use_fast_math
    --use_fast_math

⇒ kernel 里的 `/` 是 `div.approx.f32`、`sqrtf` 是 `sqrt.approx.f32`，**都不是 IEEE
就近舍入**。numpy / torch / paddle 的除法与开方全是 RN，所以在
`q = m̂ / (√v̂ + eps)` 上必然差最后 1 个 bit —— 实测 2456/4096 个元素各差 1 ULP，
无论怎么调 FMA 位置、eps 位置、bias-correction 折叠方式都消不掉。

而 **Triton 的裸 `/` 和 `tl.sqrt` 正好落到 approx 版本上**（本目录 `situ_glu.py:34`
早就踩过「Triton 的 `/` 不是 RN」这个坑，那里反过来用 `libdevice.div_rn` 换回 RN）。
用 Triton 写这个 kernel，4096 个跨 8 个量级的元素上 **q 逐位命中**。

判据脚本：`megatron_ref/dump_optimizer_step.py`（参照侧产锚点 + 自校验）
与 `tests/single_card_tests/models/run_optimizer_step_cmp.py`（迁移侧）。

⚠️ 本文件**只依赖 triton**，不 import paddle —— 参照侧的探针会按路径直接加载它，
保证两侧用的是同一份 kernel 源码（不是两份复制品）。
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _adamw_pr_kernel(
    p_ptr,  # bfloat16：模型权重（同时是 fp32 master 的高 16 位）
    rem_ptr,  # int16 ：fp32 master 的带符号余数
    m_ptr,  # bfloat16：exp_avg
    v_ptr,  # bfloat16：exp_avg_sq
    g_ptr,  # float32 ：梯度（main_grads_dtype=fp32）
    n,
    lr,
    beta1,
    beta2,
    c1,  # 1.0f - beta1，在**host 上按 fp32**算好（见下面的 note）
    c2,  # 1.0f - beta2
    bc1,  # 1 - beta1**step
    bc2,  # 1 - beta2**step
    eps,
    wd,
    BLOCK: tl.constexpr,
):
    """一步 AdamW，逐位对齐 `multi_tensor_adam_param_remainder`。

    note 1：`c1/c2/bc1/bc2` 必须在 host 上先落到 fp32 再传进来。kernel 收到的
    `beta1` 是 `float`，所以设备上的 `1 - beta1` 是 `1.0f - 0.9f = 0x3DCCCCD0`；
    若在 python 里按 double 算 `1 - 0.9` 再转 fp32 得到 `0x3DCCCCCD`，差 3 个 ULP。

    note 2：**m / v 必须按真正的 bf16 张量读**（`tl.load(...).to(tl.float32)`），
    不能用「uint16 位模式左移 16 位再 bitcast」那套。两者数值完全相同，
    但后者会让 Triton 给下面的 `/` 与 `tl.sqrt` 选**另一种下降**，
    实测 q 差 1123/4096（各 1~2 个 fp32 ULP）。按 bf16 读则逐位命中。
    这一条是 Gate 4.5 里最反直觉的一处。
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = off < n

    # ---- 拼出 fp32 master：bf16 高 16 位（无符号扩展）+ int16 余数（有符号扩展）----
    pb = tl.load(p_ptr + off, mask=msk).to(tl.uint16, bitcast=True).to(tl.int32)
    rb = tl.load(rem_ptr + off, mask=msk, other=0).to(tl.int32)
    master = ((pb << 16) + rb).to(tl.float32, bitcast=True)

    m = tl.load(m_ptr + off, mask=msk, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + off, mask=msk, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=msk, other=0.0)

    # ---- 与 C 源码逐字对应（含左结合：(1-b2)*g*g == ((1-b2)*g)*g）----
    m = beta1 * m + c1 * g
    v = beta2 * v + (c2 * g) * g
    # 下面三个 `/` 和 `tl.sqrt` 刻意用裸算子 —— 它们才是 approx 版本，别换成 div_rn
    q = (m / bc1) / (tl.sqrt(v / bc2) + eps)
    # 衰减项是普通加法；最后一步是**一次舍入**的 FMA（实测：两次舍入差 669/4096）
    master = libdevice.fma_rn(-lr, q + wd * master, master)

    # ---- 拆回 (bf16, int16 带符号余数) ----
    # 用整数「加半个 ULP 再右移」而不是 `master.to(tl.bfloat16)`：
    # 两者只在**正好落在中点**时不同（`cvt.rn` 取偶，这里取远离零），
    # 实测 6 步 × 4096 里恰好有 1 个元素踩到，取远离零才对得上。
    mbits = master.to(tl.int32, bitcast=True)
    pb2 = (mbits + 32768) >> 16
    rem_new = mbits - (pb2 << 16)

    tl.store(
        p_ptr + off,
        (pb2 & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        mask=msk,
    )
    tl.store(rem_ptr + off, rem_new.to(tl.int16), mask=msk)
    tl.store(m_ptr + off, m.to(tl.bfloat16), mask=msk)
    tl.store(v_ptr + off, v.to(tl.bfloat16), mask=msk)


def adamw_precision_aware_step(
    param_bf16,
    rem_i16,
    m_bf16,
    v_bf16,
    grad_fp32,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    block: int = 1024,
    bc1: float | None = None,
    bc2: float | None = None,
) -> None:
    """原地跑一步。四个状态张量都被就地更新。

    张量只要能提供 `.data_ptr()` 与 `.numel()` 即可 —— torch 与 paddle 都满足，
    所以参照侧探针和迁移侧优化器共用这一份实现。

    Args:
        param_bf16: 模型权重（bfloat16）；同时充当 fp32 master 的高 16 位
        rem_i16: fp32 master 的带符号余数（int16）
        m_bf16 / v_bf16: exp_avg / exp_avg_sq（bfloat16）
        grad_fp32: 梯度（float32）
        step: 从 1 开始的步数，用于 bias correction
        bc1 / bc2: 覆盖 bias correction（只给探针扫描用，正常不要传）
    """
    import struct

    def f32(x: float) -> float:
        """把 python double 夹到 fp32 再变回 double，保证传进 kernel 的是 fp32 值。"""
        return struct.unpack("f", struct.pack("f", x))[0]

    b1, b2 = f32(beta1), f32(beta2)
    # bias correction：kernel 里是 `1 - std::pow(beta, step)`，`std::pow(float,int)`
    # 在 C++ 里返回 **double**，减法也在 double 里做，最后才赋给 float。
    # 所以正确的做法是「double 里算完再落 fp32」，而不是先把 pow 落到 fp32。
    if bc1 is None:
        bc1 = f32(1.0 - b1**step)
    if bc2 is None:
        bc2 = f32(1.0 - b2**step)
    n = int(param_bf16.numel())
    _adamw_pr_kernel[(triton.cdiv(n, block),)](
        param_bf16,
        rem_i16,
        m_bf16,
        v_bf16,
        grad_fp32,
        n,
        f32(lr),
        b1,
        b2,
        f32(1.0 - b1),
        f32(1.0 - b2),
        bc1,
        bc2,
        f32(eps),
        f32(weight_decay),
        BLOCK=block,
    )
