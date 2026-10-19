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
"""LOCAL PATCH tests: rollout truncation rules."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truncation import (  # noqa: E402
    BACKSTOP,
    REPETITION,
    TruncationMonitor,
    tool_call_fingerprint,
)

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def drive(calls, *, threshold=5, mode="consecutive", cap=None, clock=None, tick=0.0):
    """Replay a tool-call sequence through the monitor in app.py's observe/check order."""
    monitor = TruncationMonitor(threshold, mode, cap, clock or FakeClock())
    for name, args in calls:
        if clock is not None:
            clock.t += tick  # time passes while the tool runs
        monitor.observe(name, args)
        if monitor.record is not None:
            break
        if monitor.check_wall_clock() is not None:
            break
    return monitor


class TestFingerprint:
    def test_key_order_and_whitespace_are_normalised(self):
        assert tool_call_fingerprint(
            "python", '{"a": 1, "b": " x "}'
        ) == tool_call_fingerprint("python", '{"b":"x","a":1}')

    def test_different_tool_name_is_different(self):
        a = tool_call_fingerprint("python", "{}")
        b = tool_call_fingerprint("bash", "{}")
        assert a != b

    def test_different_values_are_different(self):
        a = tool_call_fingerprint("python", json.dumps({"code": "df.head()"}))
        b = tool_call_fingerprint("python", json.dumps({"code": "df.tail()"}))
        assert a != b

    def test_unparseable_arguments_fall_back_to_raw_string(self):
        assert tool_call_fingerprint("t", " not json ") == tool_call_fingerprint(
            "t", "not json"
        )


class TestRepetition:
    def test_loop_is_aborted(self):
        m = drive([("python", '{"code":"x"}')] * 40, threshold=5)
        assert m.record["reason"] == REPETITION
        assert m.record["call_index"] == 5
        assert m.record["repeat_count"] == 5

    def test_distinct_calls_are_not_aborted(self):
        m = drive(
            [("python", json.dumps({"code": f"step{i}"})) for i in range(200)],
            threshold=5,
        )
        assert m.record is None
        assert m.call_index == 200

    def test_interleaved_repeats_survive_consecutive_mode(self):
        calls = []
        for i in range(50):
            calls += [("python", '{"code":"probe"}'), ("python", json.dumps({"code": i}))]
        assert drive(calls, threshold=5, mode="consecutive").record is None

    def test_interleaved_repeats_caught_by_anywhere_mode(self):
        calls = []
        for i in range(50):
            calls += [("python", '{"code":"probe"}'), ("python", json.dumps({"code": i}))]
        m = drive(calls, threshold=5, mode="anywhere")
        assert m.record["reason"] == REPETITION
        assert m.record["mode"] == "anywhere"

    @pytest.mark.parametrize("threshold,expected_index", [(3, 3), (10, 10), (25, 25)])
    def test_knob_moves_the_abort_point(self, threshold, expected_index):
        m = drive([("python", '{"code":"x"}')] * 40, threshold=threshold)
        assert m.record["call_index"] == expected_index

    def test_disabled_when_threshold_is_none(self):
        assert drive([("python", '{"code":"x"}')] * 500, threshold=None).record is None

    def test_rejects_bad_mode_and_threshold(self):
        with pytest.raises(ValueError):
            TruncationMonitor(5, "sometimes", None)
        with pytest.raises(ValueError):
            TruncationMonitor(1, "consecutive", None)


class TestBackstop:
    def test_fires_on_wall_clock(self):
        clock = FakeClock()
        m = drive(
            [("python", json.dumps({"code": i})) for i in range(100)],
            threshold=None,
            cap=30.0,
            clock=clock,
            tick=4.0,
        )
        assert m.record["reason"] == BACKSTOP
        assert m.record["elapsed_s"] == pytest.approx(32.0)
        assert m.record["call_index"] == 8

    def test_does_not_fire_under_the_cap(self):
        clock = FakeClock()
        m = drive(
            [("python", json.dumps({"code": i})) for i in range(10)],
            threshold=None,
            cap=1000.0,
            clock=clock,
            tick=4.0,
        )
        assert m.record is None

    @pytest.mark.parametrize("cap,expected_calls", [(8.0, 2), (20.0, 5), (40.0, 10)])
    def test_knob_moves_the_abort_point(self, cap, expected_calls):
        clock = FakeClock()
        m = drive(
            [("python", json.dumps({"code": i})) for i in range(100)],
            threshold=None,
            cap=cap,
            clock=clock,
            tick=4.0,
        )
        assert m.record["call_index"] == expected_calls

    def test_repetition_wins_when_both_would_fire(self):
        clock = FakeClock()
        m = drive([("python", '{"code":"x"}')] * 40, threshold=2, cap=6.0, clock=clock, tick=4.0)
        assert m.record["reason"] == REPETITION


class TestAppWiring:
    """Guards the call sites in app.py; the agent itself needs nemo_gym to import."""

    def test_app_observes_and_breaks(self):
        src = open(_APP).read()
        assert "monitor.observe(" in src
        assert "monitor.check_wall_clock()" in src
        assert re.search(
            r"if monitor\.record is not None or monitor\.check_wall_clock\(\) is not None:\s*\n\s*invocation_status = \"incomplete\"\s*\n\s*break",
            src,
        )

    def test_app_propagates_the_record_to_the_run_result(self):
        src = open(_APP).read()
        assert 'result[_INTERNAL_TRUNCATION_KEY] = truncation' in src
        assert '_INTERNAL_TRUNCATION_KEY = "_ng_truncation"' in src

    def test_thresholds_are_config_fields_not_literals(self):
        src = open(_APP).read()
        for key in (
            "repetition_abort_threshold",
            "repetition_abort_mode",
            "wall_clock_abort_seconds",
        ):
            assert f"self.config.{key}" in src
