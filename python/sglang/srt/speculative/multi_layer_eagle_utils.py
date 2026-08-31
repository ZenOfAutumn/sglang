# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

"""多层 EAGLE 投机解码 Triton 算子封装模块。

本模块从 ``triton_ops.multi_layer_eagle`` 中统一导出多层 EAGLE 算法
所需的 Triton kernel 及其 PyTorch 参考实现，供 multi_layer_eagle_worker_v2.py 调用。

导出的算子：
    - assign_hidden_states_pool_kernel:   隐藏状态池分配 kernel 入口
    - assign_hidden_states_pool_torch:     PyTorch 参考实现
    - assign_hidden_states_pool_triton:    Triton kernel 实现
    - assign_new_state_kernel:             新状态分配 kernel 入口
    - assign_new_state_triton:              Triton kernel 实现
    - rotate_input_ids_kernel:             input_ids 旋转 kernel 入口
    - rotate_input_ids_triton:              Triton kernel 实现
"""

from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
    assign_hidden_states_pool_kernel,
    assign_hidden_states_pool_torch,
    assign_hidden_states_pool_triton,
    assign_new_state_kernel,
    assign_new_state_triton,
    rotate_input_ids_kernel,
    rotate_input_ids_triton,
)

__all__ = [
    "assign_hidden_states_pool_kernel",
    "assign_hidden_states_pool_torch",
    "assign_hidden_states_pool_triton",
    "assign_new_state_kernel",
    "assign_new_state_triton",
    "rotate_input_ids_kernel",
    "rotate_input_ids_triton",
]
