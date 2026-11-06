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

"""Pinned Stage 2 CPU numerical reference fixture."""

SOURCE_REFERENCE = {
    "provenance": {
        "source_commit": "a47102925287bc016fa787a581daf091291c6a6f",
        "source_checkout_head": "9756ede036ffe947bd9aa8ce2849046f225877db",
        "verified_identical_source_files": [
            "nvalchemi_csp/kernels/contact_overlap.py",
            "nvalchemi_csp/kernels/contact_kernels.py",
            "nvalchemi_csp/kernels/packer_relaxation.py",
            "nvalchemi_csp/kernels/symmetry_helpers.py",
            "nvalchemi_csp/kernels/util.py",
        ],
        "generation_command": "WARP_CACHE_DIR=/path/to/warp-cache "
        "NVALCHEMI_CSP_SOURCE=/path/to/nvalchemi_csp "
        "python "
        "test/csp/fixtures/generate_packer_stage2_cpu.py",
        "scope": "fixed-state contact outputs and one source relaxation step; "
        "CPU Warp kernels",
    },
    "tolerances": {
        "contact_atol": 0.0002,
        "contact_rtol": 0.0002,
        "step_atol": 0.0005,
        "step_rtol": 0.0005,
    },
    "multiop_skew": {
        "contacts": {
            "forces": [
                [
                    [5.181663990020752, -0.9457449316978455, 0.25645795464515686],
                    [-5.085877895355225, 0.8130142688751221, -0.3653881847858429],
                ]
            ],
            "torques": [
                [
                    [-0.02300652116537094, 0.05931607633829117, 0.7722469568252563],
                    [0.0, 0.0, 0.0],
                ]
            ],
            "virial": [
                [
                    [0.807486355304718, -0.08452748507261276, 0.029445119202136993],
                    [-0.08452748507261276, 0.05527745559811592, 0.006995706353336573],
                    [0.029445119202136993, 0.006995706353336573, 0.013658026233315468],
                ]
            ],
            "total_overlap": [5.417359352111816],
            "max_overlap": [1.735816478729248],
        },
        "one_step": {
            "cells": [
                [
                    [4.417646884918213, 0.0, 0.0],
                    [0.9902603626251221, 4.191595077514648, 0.0],
                    [0.5059409737586975, 0.40046289563179016, 4.590810775756836],
                ]
            ],
            "inverse_cells": [
                [
                    [0.2263648509979248, 0.0, 0.0],
                    [-0.05347847938537598, 0.23857265710830688, 0.0],
                    [-0.020282059907913208, -0.02081102877855301, 0.21782644093036652],
                ]
            ],
            "centers": [
                [
                    [0.17359916865825653, 0.07802092283964157, 0.12289757281541824],
                    [0.8007102012634277, 0.11168648302555084, 0.10538488626480103],
                ]
            ],
            "rotations": [
                [
                    [
                        [
                            0.8919240236282349,
                            -0.45212283730506897,
                            0.007515580393373966,
                        ],
                        [0.4521125555038452, 0.8919554948806763, 0.0031158546917140484],
                        [
                            -0.00811231229454279,
                            0.0006187825929373503,
                            0.9999669790267944,
                        ],
                    ],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ]
            ],
            "steps": [1],
        },
    },
    "p1_skew": {
        "contacts": {
            "forces": [
                [
                    [2.569981575012207, -0.44658520817756653, 0.1489882916212082],
                    [-2.569981575012207, 0.44658520817756653, -0.1489882916212082],
                ]
            ],
            "torques": [
                [
                    [-0.01488136313855648, 0.03836755454540253, 0.37170174717903137],
                    [0.0, 0.0, 0.0],
                ]
            ],
            "virial": [
                [
                    [0.7981806993484497, -0.07218092679977417, 0.03940638527274132],
                    [-0.07218092679977417, 0.03834305331110954, -0.006847640499472618],
                    [0.03940638527274132, -0.006847640499472618, 0.0022844874765723944],
                ]
            ],
            "total_overlap": [2.6592726707458496],
            "max_overlap": [1.735816478729248],
        }
    },
}
