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

# This image is for NVSkills CI skill evaluation only. It is not for
# production use or for end users, and it is not a supported way to install
# nvalchemi-toolkit. Install nvalchemi-toolkit from PyPI as the documentation
# describes.
#
# It holds a CPU-only toolkit stack so that the evaluation sandbox does not
# install it at run time. The agent runtimes come from the evaluator.

FROM python:3.12.16-slim-bookworm

# The tools the evaluator's default task image has. Sandboxes run as a
# non-root user and cannot apt-get, so the agent installers fall back to
# these when their own apt-get step fails.
RUN apt-get -o Acquire::Retries=3 update \
 && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    bash ca-certificates curl git jq procps ripgrep \
 && rm -rf /var/lib/apt/lists/*

# The venv lives outside /workspace, because the sandbox mounts an empty
# volume over /workspace. Bytecode is compiled here: the sandbox root
# filesystem is read-only, so imports could not cache it at run time.
# The CPU torch wheel comes alone from the PyTorch index, so the resolver
# never takes the CUDA wheel from PyPI. Everything else resolves on PyPI.
# antlr4-python3-runtime 4.9.3 (via physicsnemo -> hydra-core) ships only as
# a pure-Python sdist, so it is the one exemption from wheels-only.
RUN python -m venv /opt/nvalchemi-venv \
 && /opt/nvalchemi-venv/bin/pip install --no-cache-dir \
    --disable-pip-version-check --no-input --progress-bar off \
    --only-binary=:all: --no-deps \
    --index-url https://download-r2.pytorch.org/whl/cpu \
    'torch==2.13.0+cpu' \
 && /opt/nvalchemi-venv/bin/pip install --no-cache-dir \
    --disable-pip-version-check --no-input --progress-bar off \
    --only-binary=:all: --no-binary=antlr4-python3-runtime \
    'torch==2.13.0+cpu' 'nvalchemi-toolkit==0.3.0' \
    'nvalchemi-toolkit-ops==0.5.0' 'ase==3.27.0' \
 && /opt/nvalchemi-venv/bin/python -c "import nvalchemi, torch; assert nvalchemi.__version__ == '0.3.0' and not torch.cuda.is_available()"

# Sandboxes run as uid 1000 with no passwd entry. Agent installers and the
# skill upload write under $HOME, and only /tmp is writable there.
ENV HOME=/tmp/agent-home

RUN mkdir -p /workspace/skills /workspace/input /workspace/output \
    /logs/verifier /logs/agent \
 && chmod -R a+rwX /workspace /logs

WORKDIR /workspace
