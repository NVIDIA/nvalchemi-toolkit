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

"""``_await_worker_results`` orchestration logic.

The validator spawns one process per virtual rank. When a rank hits an
exception it ships an error payload and exits cleanly, but a peer blocked in a
collective never gets its partner and hangs. The wait loop must detect that
asymmetric crash and return immediately (with the crashed rank's payload)
instead of blocking the whole timeout and reporting a bare "timed out". These
tests drive the pure loop with fake processes and an injected clock — no real
multiprocessing — so the four terminal states are covered fast and on CPU.
"""

from __future__ import annotations

from nvalchemi.distributed.validate.inference import _await_worker_results


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class _FakeProc:
    """is_alive() is True until the injected clock reaches ``exit_at``."""

    def __init__(self, pid: int, exit_at: float, exitcode: int, clock: _Clock) -> None:
        self.pid = pid
        self._exit_at = exit_at
        self.exitcode = exitcode
        self._clock = clock

    def is_alive(self) -> bool:
        return self._clock() < self._exit_at


class _FakeQueue:
    def __init__(self, items=()) -> None:
        self._items = list(items)

    def empty(self) -> bool:
        return not self._items

    def get_nowait(self):
        return self._items.pop(0)


# Realistic payload shapes the workers emit.
def _success(rank: int):
    return (rank, b"<pickled-outputs>")  # 2-tuple, element 1 is bytes


def _run_error(rank: int):
    return (rank, "RUN_ERROR", "Traceback ...")  # element 1 is a str


def test_clean_success_returns_no_error():
    clk = _Clock()
    procs = [_FakeProc(p, exit_at=0.05, exitcode=0, clock=clk) for p in (1, 2)]
    q = _FakeQueue([_success(0), _success(1)])
    received, error = _await_worker_results(
        procs, q, timeout_sec=10.0, monotonic=clk, sleep=clk.sleep
    )
    assert error is None
    assert len(received) == 2
    assert clk.t < 10.0  # did not burn the timeout


def test_asymmetric_crash_returns_fast_with_payload_not_timeout():
    # rank 1 ships RUN_ERROR and exits; rank 0 stays blocked in a collective.
    clk = _Clock()
    survivor = _FakeProc(0, exit_at=float("inf"), exitcode=None, clock=clk)
    crashed = _FakeProc(1, exit_at=0.05, exitcode=0, clock=clk)
    q = _FakeQueue([_run_error(1)])
    received, error = _await_worker_results(
        [survivor, crashed], q, timeout_sec=120.0, monotonic=clk, sleep=clk.sleep
    )
    # The whole point: no bare "timed out" — the payload survives so the
    # caller's diagnostic branch can translate the real traceback.
    assert error is None
    assert any(m[1] == "RUN_ERROR" for m in received)
    assert clk.t < 120.0  # returned long before the timeout


def test_hard_crash_without_payload_reports_exit_code():
    clk = _Clock()
    survivor = _FakeProc(0, exit_at=float("inf"), exitcode=None, clock=clk)
    crashed = _FakeProc(1, exit_at=0.05, exitcode=1, clock=clk)
    received, error = _await_worker_results(
        [survivor, crashed], _FakeQueue(), timeout_sec=120.0,
        monotonic=clk, sleep=clk.sleep,
    )
    assert error is not None
    assert "exited with code 1" in error
    assert "pid=1" in error


def test_symmetric_hang_falls_through_to_timeout():
    clk = _Clock()
    procs = [_FakeProc(p, exit_at=float("inf"), exitcode=None, clock=clk) for p in (7, 8)]
    received, error = _await_worker_results(
        procs, _FakeQueue(), timeout_sec=1.0, monotonic=clk, sleep=clk.sleep
    )
    assert error is not None
    assert "timed out after 1.0s" in error
    assert clk.t >= 1.0
