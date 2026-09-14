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

"""与 Megatron/TE 的 precision-aware AdamW **逐位一致**的优化器（对齐腿专用）。

## 为什么不能用现成的

- `PaddleFormers` 的 `AdamWCustom`：master weight 是**独立的 fp32 张量**
  （`paddleformers/utils/optimizer.py:242-267`），而 TE 存的是
  「bf16 权重（高 16 位）+ int16 带符号余数」；而且没有独立的
  `exp_avg_dtype` / `exp_avg_sq_dtype` 旋钮（`use_lowprecision_moment` 只能让
  moment 跟随参数 dtype）。
- `PaddleFormers` 的 `adamw_triton`：公式形状就不同 —— 它是
  「先 `param *= 1-lr*coeff` 再 `param += (m/denom)*(-lr/(1-b1pow))`」，
  实测这一支与 TE 差 1361/4096（[F-34](doc/03_事实核查.md)）。

所以这里新写一个，只干一件事：**逐参数调用逐位对齐过的那个 Triton kernel**
（`triton_ops/adamw_precision_aware.py`，规格与实测见 doc/06 P-15）。

## 刻意不做的事

- **不做梯度裁剪**。对齐腿按 [D-10](doc/01_决策记录.md) 关掉裁剪：
  `total_norm` 来自 TE 的 `multi_tensor_l2norm`（分块 + block 树归约，藏在 `.so` 里），
  只逆到 ~75~87% 逐位（doc/06 P-16）。留着一个逆不准的标量会污染所有参数，
  不如显式关掉、把判据说清楚。`clip_grad` 参数刻意**不提供**。
- 不做学习率调度。`lr` 每步由调用方写进 `self.lr`（MG 的 warmup+cosine 值要外部对齐）。
- 不做 state_dict / 分布式 / offload。对齐腿是单卡固定输入的单测。

## 状态张量

| | dtype | 说明 |
|---|---|---|
| `param` | bfloat16 | 就是模型权重本身，同时充当 fp32 master 的高 16 位 |
| `remainder` | int16 | fp32 master 的带符号余数 |
| `exp_avg` / `exp_avg_sq` | bfloat16 | 与 MG 的 `exp_avg_dtype=bf16` 一致 |

初始 `remainder = 0` ⇒ master 恰好等于 bf16 权重，与 TE 的
`initialize_state(store_param_remainders=True)` 一致。
"""

from __future__ import annotations

import paddle

from ..triton_ops.utils import is_torch_compat_available

if is_torch_compat_available():
    # kernel 文件刻意不 import paddle，所以 compat 在这里打开
    paddle.enable_compat(scope={"triton"})

from ..triton_ops.adamw_precision_aware import (  # noqa: E402
    adamw_precision_aware_step,
)


class PrecisionAwareAdamW:
    """逐位对齐 TE `FusedAdam(precision-aware + param remainder)` 的 AdamW。

    Args:
        parameters: 要更新的参数（必须是 bfloat16）。
        lr: 学习率。每步可以直接改 `opt.lr`。
        beta1 / beta2 / epsilon / weight_decay: 与 MG 的 `OptimizerConfig` 对齐
            （HyperBody 是 0.9 / 0.95 / 1e-8 / 0.1，见 doc/03 F-32）。
        offload: 状态常驻 CPU，逐参数搬上 GPU 走一步再搬回来。**不影响数值**
            （kernel 逐元素、参数之间零耦合、H2D/D2H 拷贝精确），
            但能把峰值显存从「权重+梯度+状态」降到「权重+梯度+一个参数的状态」。
            encoder 2.616 B 参数时那是 29.24 GiB → 14.6 GiB（[R-10](doc/04_风险台账.md)）。
            开了之后每走完一个参数就把它的 `grad` 置 None，梯度的显存是边走边放的。
    """

    def __init__(
        self,
        parameters,
        *,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
        offload: bool = False,
    ) -> None:
        self._params = []
        for p in parameters:
            if p.stop_gradient:
                continue
            if p.dtype != paddle.bfloat16:
                raise TypeError(
                    f"{p.name}: 对齐腿要求参数是 bfloat16（MG 侧 params_dtype=bf16），"
                    f"当前是 {p.dtype}"
                )
            self._params.append(p)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.offload = offload
        self._step = 0
        # name → {"remainder", "exp_avg", "exp_avg_sq"}
        self.state: dict[str, dict[str, paddle.Tensor]] = {}

    def _state_of(self, p) -> dict:
        st = self.state.get(p.name)
        if st is None:
            st = {
                "remainder": paddle.zeros(p.shape, dtype="int16"),
                "exp_avg": paddle.zeros(p.shape, dtype="bfloat16"),
                "exp_avg_sq": paddle.zeros(p.shape, dtype="bfloat16"),
            }
            if self.offload:
                st = {k: v.cpu() for k, v in st.items()}
            self.state[p.name] = st
        return st

    @staticmethod
    def _grad_of(p):
        """取 fp32 梯度。PaddleFleet 的列/行并行层会把 fp32 累加值放在 `main_grad`。"""
        g = getattr(p, "main_grad", None)
        if g is None:
            g = p.grad
        if g is None:
            return None
        return g if g.dtype == paddle.float32 else g.astype("float32")

    @paddle.no_grad()
    def step(self) -> None:
        """一步更新。**不含梯度裁剪**（见模块 docstring 与 D-10）。"""
        self._step += 1
        for p in self._params:
            g = self._grad_of(p)
            if g is None:
                continue
            st = self._state_of(p)
            gpu = {k: v.cuda() for k, v in st.items()} if self.offload else st
            adamw_precision_aware_step(
                p,
                gpu["remainder"],
                gpu["exp_avg"],
                gpu["exp_avg_sq"],
                g.contiguous(),
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.epsilon,
                weight_decay=self.weight_decay,
                step=self._step,
            )
            if self.offload:
                self.state[p.name] = {k: v.cpu() for k, v in gpu.items()}
                del gpu
                # 逐参数放掉 fp32 梯度：2.6 B 参数时那是 9.75 GiB
                if getattr(p, "main_grad", None) is not None:
                    p.main_grad = None
                p.clear_gradient(False)

    def clear_grad(self) -> None:
        """清梯度。`main_grad` 与 `.grad` 都清。"""
        for p in self._params:
            if getattr(p, "main_grad", None) is not None:
                p.main_grad = None
            p.clear_gradient(False)
