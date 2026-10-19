# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""LOCAL PATCH: rollout truncation rules (repetition + wall-clock backstop).

Pure logic, no I/O and no nemo_gym imports, so it is unit-testable standalone.
"""

import json
from collections import Counter
from time import perf_counter
from typing import Any, Callable, Optional

REPETITION = "repetition"
BACKSTOP = "backstop"
MODES = ("consecutive", "anywhere")


def _normalise(value: Any) -> Any:
    """Recursively strip string whitespace; dict key order is handled by sort_keys."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def tool_call_fingerprint(tool_name: str, arguments: Any) -> str:
    """Identity of a tool call: name + arguments as canonical JSON (sorted keys, no spaces).

    Unparseable arguments fall back to the raw string, whitespace-stripped.
    """
    try:
        args = json.dumps(
            _normalise(json.loads(arguments)), sort_keys=True, separators=(",", ":")
        )
    except (json.JSONDecodeError, TypeError):
        args = (arguments or "").strip() if isinstance(arguments, str) else repr(arguments)
    return f"{tool_name}\x00{args}"


class TruncationMonitor:
    """Tracks one rollout and decides whether to abort it early.

    ``repetition_mode`` selects which repetition definition applies:
    ``consecutive`` = N identical calls back to back; ``anywhere`` = N identical
    calls at any position in the trajectory.
    """

    def __init__(
        self,
        repetition_threshold: Optional[int] = None,
        repetition_mode: str = "consecutive",
        wall_clock_seconds: Optional[float] = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if repetition_mode not in MODES:
            raise ValueError(f"repetition_mode must be one of {MODES}, got {repetition_mode!r}")
        if repetition_threshold is not None and repetition_threshold < 2:
            raise ValueError("repetition_threshold must be >= 2 or None to disable")
        self.repetition_threshold = repetition_threshold
        self.repetition_mode = repetition_mode
        self.wall_clock_seconds = wall_clock_seconds
        self._clock = clock
        self._start = clock()
        self._counts: Counter = Counter()
        self._last: Optional[str] = None
        self._run = 0
        self.call_index = 0
        self.record: Optional[dict] = None

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def observe(self, tool_name: str, arguments: Any) -> Optional[dict]:
        """Record one executed tool call; returns the abort record if repetition fired."""
        fingerprint = tool_call_fingerprint(tool_name, arguments)
        self.call_index += 1
        self._run = self._run + 1 if fingerprint == self._last else 1
        self._last = fingerprint
        self._counts[fingerprint] += 1
        if self.record is not None or self.repetition_threshold is None:
            return self.record
        seen = self._run if self.repetition_mode == "consecutive" else self._counts[fingerprint]
        if seen >= self.repetition_threshold:
            self.record = self._build(REPETITION, repeat_count=seen)
        return self.record

    def check_wall_clock(self) -> Optional[dict]:
        """Returns the abort record if the per-rollout time cap has been exceeded."""
        if self.record is not None or self.wall_clock_seconds is None:
            return self.record
        if self.elapsed >= self.wall_clock_seconds:
            self.record = self._build(BACKSTOP)
        return self.record

    def _build(self, reason: str, repeat_count: int = 0) -> dict:
        return {
            "reason": reason,
            "call_index": self.call_index,
            "elapsed_s": round(self.elapsed, 3),
            "repeat_count": repeat_count,
            "mode": self.repetition_mode if reason == REPETITION else None,
        }
