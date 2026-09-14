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
"""TF32 策略：让 fp32 GEMM 能逐 op 对齐 TransformerEngine，同时保持 conv 不用 TF32。

## 为什么需要这个模块

逐位对齐 HyperEncoder 时撞上两个**互相矛盾**的需求（doc/06_逐位对齐问题清单.md P-9）：

1. **conv 必须不用 TF32**。Paddle 的 cuDNN 卷积默认走 TF32，而 `FLAGS_cudnn_allow_tf32=0`
   只管前向，bwd-filter / bwd-data 不生效（doc/03 F-24）。唯一有效的开关是 CUDA 级的
   环境变量 ``NVIDIA_TF32_OVERRIDE=0``。

2. **router 的 fp32 GEMM 必须用 TF32**。MG 的 ``RouterGatingLinearFunction`` 前反向都走
   ``te_general_gemm``，TE 对 fp32 GEMM 一律请求 ``CUBLAS_COMPUTE_32F_FAST_TF32``。
   cuBLASLt 日志实测同一个 GEMM 两边选了不同 kernel：

       TE      : COMPUTE_32F_FAST_TF32  algoId=21 tile=64x64 stages=32x6 splitsK=13
       torch/pd: COMPUTE_32F            algoId=20 tile=64x64 stages=8x5  splitsK=13

   累加顺序不同 ⇒ router logits 差 28317/40960 个元素（max_abs 1.43e-06），一路放大到
   MoE 输出。这里请求 TF32 **不损精度**：router 的 x/w 都是 bf16 upcast 上来的（尾数 7 位），
   TF32 尾数 10 位能精确表示，乘法无舍入 —— 实测两边离 fp64 精确解都只有 ~2.7e-07。

``NVIDIA_TF32_OVERRIDE`` 是全局的，没法同时满足这两条。

## 解法：利用「各库在第一次使用时缓存 override」这个事实

实测（``/tmp/probe_warmup_order.py`` 的结论）：cuBLAS 和 cuDNN 各自在**建 handle 时**
读一次 ``NVIDIA_TF32_OVERRIDE`` 并缓存，之后再改环境变量对它们无效。所以只要控制
**两个库的初始化时机**：

    override 摘掉 → 跑一个假 fp32 matmul  → cuBLAS 记住「允许 TF32」
    override=0    → 跑一个假 conv         → cuDNN 记住「禁止 TF32」
    之后 override 一直保持 0，fp32 GEMM 用不用 TF32 由 ``FLAGS_cublas_allow_tf32``
    逐 op 决定（这个 flag 是每次调用都读的）。

于是 conv 全程不用 TF32，而 ``te_fp32_gemm_math()`` 块内的 fp32 GEMM 走 TF32。

## 使用

进程里**第一个 GPU 算子之前**调用一次 :func:`init_tf32_math_policy`，
然后在需要对齐 TE 的 fp32 GEMM 外面套 :class:`te_fp32_gemm_math`。
两者都由 ``PADDLEFLEET_ROUTER_GEMM_TE_MATH=1`` 统一开关，默认关（不影响其它模型）。
"""

from __future__ import annotations

import logging
import os

import paddle

_logger = logging.getLogger(__name__)

ENV_FLAG = "PADDLEFLEET_ROUTER_GEMM_TE_MATH"

# 第二个开关：把「反向的算子链」显式写成与 MG（eager torch autograd）同一条。
# 为什么需要单独一个开关：Paddle 与 torch 的 autograd 对同一个前向表达式会选**不同的
# 算子链**，逐位对齐时必须把链写死（不是归约顺序的问题 —— doc/06 P-12 已经证过，
# `paddle.sum` 与 `torch.sum` 在这些长度上是一致的）。
# 这类改动会影响共享代码路径上其它模型的数值，所以默认关，只在对齐腿打开。
# 目前覆盖：eager softmax 的反向（fusions/fused_softmax.py）。
ENV_MG_EXACT_BACKWARD = "PADDLEFLEET_MG_EXACT_BACKWARD"

# 第三个开关：EP=1 也走 **token dispatcher** 那条路（默认 EP=1 走
# `_forward_single_card_moe`）。
# 为什么需要它（doc/06 P-18）：Paddle 的 MoE 在 `expert_model_parallel_size > 1` 时
# 换成 dispatcher 实现，而 Gate 3/4 手工对齐的是**单卡那条**；MG 两档都走 dispatcher，
# 所以 MG 天然 EP 不变、Paddle 不是。要修 dispatcher 那条路，就得先把 **EP 这一维隔离掉**
# —— 单卡强制走 dispatcher，与现成的 MG EP=1 锚点比，变量只剩「dispatcher vs MG」。
ENV_MG_EXACT_MOE_DISPATCHER = "PADDLEFLEET_MG_EXACT_MOE_DISPATCHER"

_initialized = False
_warned = False


def enabled() -> bool:
    """是否启用「fp32 GEMM 对齐 TE」策略。"""
    return os.environ.get(ENV_FLAG, "0") == "1"


def mg_exact_backward_enabled() -> bool:
    """是否把反向的算子链显式写成与 MG 同一条（见 :data:`ENV_MG_EXACT_BACKWARD`）。"""
    return os.environ.get(ENV_MG_EXACT_BACKWARD, "0") == "1"


def mg_exact_moe_dispatcher_enabled() -> bool:
    """EP=1 是否也走 token dispatcher（见 :data:`ENV_MG_EXACT_MOE_DISPATCHER`）。"""
    return os.environ.get(ENV_MG_EXACT_MOE_DISPATCHER, "0") == "1"


def init_tf32_math_policy() -> bool:
    """按上面描述的顺序初始化 cuBLAS / cuDNN 的 TF32 许可。返回是否真的做了初始化。

    必须在进程里**任何 GPU 算子之前**调用；重复调用是安全的（第二次直接返回）。
    """
    global _initialized
    if _initialized or not enabled():
        return False

    saved_override = os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    saved_flag = paddle.get_flags(["FLAGS_cublas_allow_tf32"])[
        "FLAGS_cublas_allow_tf32"
    ]
    # ① override 摘掉 + 请求 TF32 的状态下建 cuBLAS handle
    paddle.set_flags({"FLAGS_cublas_allow_tf32": True})
    paddle.matmul(
        paddle.zeros([8, 8], dtype="float32"), paddle.zeros([8, 8], dtype="float32")
    )
    paddle.device.synchronize()

    # ② 把 override 恢复成 0（conv 靠它关 TF32），再建 cuDNN handle
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0" if saved_override is None else saved_override
    paddle.nn.Conv2D(1, 1, 1)(paddle.zeros([1, 1, 4, 4], dtype="float32"))
    paddle.device.synchronize()

    # ③ 恢复 flag：默认关，只有 te_fp32_gemm_math() 块内才打开
    paddle.set_flags({"FLAGS_cublas_allow_tf32": saved_flag})
    _initialized = True
    _logger.info(
        "[tf32_math] cuBLAS 已在 override 摘掉时初始化（允许 TF32），"
        "cuDNN 已在 NVIDIA_TF32_OVERRIDE=%s 时初始化（禁止 TF32）",
        os.environ.get("NVIDIA_TF32_OVERRIDE"),
    )
    return True


class te_fp32_gemm_math:
    """在 with 块内让 fp32 GEMM 走 TF32 compute type，对齐 TE 的 kernel 选择。

    需要 :func:`init_tf32_math_policy` 已经跑过；否则 cuBLAS 可能是在
    ``NVIDIA_TF32_OVERRIDE=0`` 下初始化的，这个 flag 就不起作用（会告警一次）。
    """

    def __enter__(self):
        global _warned
        self._saved = None
        if not enabled():
            return self
        if not _initialized and not _warned:
            _warned = True
            _logger.warning(
                "[tf32_math] %s=1 但 init_tf32_math_policy() 没跑过；"
                "若 cuBLAS 已在 NVIDIA_TF32_OVERRIDE=0 下初始化，本块不会生效。",
                ENV_FLAG,
            )
        self._saved = paddle.get_flags(["FLAGS_cublas_allow_tf32"])[
            "FLAGS_cublas_allow_tf32"
        ]
        if not self._saved:
            paddle.set_flags({"FLAGS_cublas_allow_tf32": True})
        return self

    def __exit__(self, *exc):
        if self._saved is not None and not self._saved:
            paddle.set_flags({"FLAGS_cublas_allow_tf32": self._saved})
        return False
