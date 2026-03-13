# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch custom operator wrappers for GPU interaction kernels.

This sub-package wraps NVIDIA Warp-based interaction kernels as
``torch.library`` custom ops so they integrate cleanly with
``torch.compile`` and the autograd engine.

Modules
-------
lj
    Lennard-Jones energy and force kernels (batched, neighbor-matrix format).
"""
