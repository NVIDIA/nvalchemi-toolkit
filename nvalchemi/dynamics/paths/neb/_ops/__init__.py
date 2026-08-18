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
"""Low-level NEB tensor operations that will be moved to toolkit-ops.

Planned toolkit-ops locations
-----------------------------
``modes.py``
    ``nvalchemiops/dynamics/neb/modes.py``
``equations.py``
    ``nvalchemiops/dynamics/neb/equations.py``
``kernels.py``
    ``nvalchemiops/dynamics/neb/kernels.py``
``registry.py`` method specifications and controlled registration
    ``nvalchemiops/dynamics/neb/registry.py``
``launchers.py`` overload selection and launch
    ``nvalchemiops/dynamics/neb/launchers.py``
``torch_ops.py`` Torch validation and custom-op boundary
    ``nvalchemiops/torch/dynamics/neb.py``
``__init__.py`` exports
    ``nvalchemiops/dynamics/neb/__init__.py`` and
    ``nvalchemiops/torch/dynamics/__init__.py``

"""

from __future__ import annotations
