# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DABStep GRPO reward server.

Grades a rollout's final answer against a pseudo-label using the **real**
NeMo-Skills DABStep evaluator, vendored verbatim in `_dabstep_eval.py` /
`_math_grader.py` (copied from
`Skills-dabstep-eval/nemo_skills/evaluation/{evaluator/dabstep.py,math_grader.py}`).

Why not reuse `dabstep_symbolic_judge`: that server's comparison logic was measured
~7pp LENIENT versus the real evaluator (44.22% reported vs 36.89% actual, 33/450
disagreements, all lenient -- sign errors accepted, loose numeric tolerance,
over-permissive bool/string). Those lenient paths are exactly the failure modes GRPO
should be fixing, so this server uses the strict logic.

CRITICAL invariants:
  * `expected_answer` / `expected_answer_dtype` live as SIBLINGS of
    `responses_create_params`, never inside it -- the model must never see the label.
  * `expected_answer_dtype` is REQUIRED on every row. Pseudo-labels carry no dtype;
    `kaggle/preprocessing/build_dabstep_rl_dataset.py::infer_dtype` synthesises one
    (unit-tested in `kaggle/preprocessing/test_infer_dtype.py`). Falling back to "str"
    silently degrades numeric comparison to string matching -- that bug class cost
    38/450 rows once before -- so a missing dtype is counted and reported, not hidden.
  * reward is 1.0/0.0. If you ever see a CONSTANT reward across a step, the verifier is
    miswired: GRPO advantages go to zero and training learns nothing while looking green.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

from resources_servers.dabstep_rl._dabstep_eval import dabstep_equal, extract_prediction


class DABStepRLConfig(BaseResourcesServerConfig):
    name: str = "dabstep_rl"
    # Reward given when the rollout produced no assistant text at all (harness failure).
    empty_response_reward: float = 0.0
    # ADDITIVE penalty when the rollout never emits a \boxed{} answer (fragment,
    # unterminated tool call, or ran to the execution cap). Additive and applied
    # pre-advantage: a multiplicative factor is a no-op on an already-zero reward.
    no_answer_penalty: float = 0.25


class DABStepRLRunRequest(BaseRunRequest):
    model_config = {"extra": "allow"}

    expected_answer: Optional[Any] = None
    expected_answer_dtype: Optional[str] = None
    question: Optional[str] = None


class DABStepRLVerifyRequest(DABStepRLRunRequest, BaseVerifyRequest):
    pass


class DABStepRLVerifyResponse(BaseVerifyResponse):
    model_config = {"extra": "allow"}

    expected_answer_str: str
    predicted_answer: str
    expected_answer_dtype_used: Optional[str] = None
    symbolic_correct: bool
    boxed_answer_found: bool
    dtype_missing: bool


def _last_assistant_text(body: BaseVerifyRequest) -> str:
    """Last assistant message text from the Responses-API output list."""
    response = getattr(body, "response", None)
    output = getattr(response, "output", None) or []
    for o in reversed(output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [t for t in (getattr(c, "text", None) for c in content) if isinstance(t, str)]
                return "\n".join(texts).strip()
            break
    return ""


_BOXED = re.compile(r"\\boxed\s*\{")


def _coerce_expected(raw: Any) -> Any:
    """Undo the JSON-string encoding the request schema may impose on a native label."""
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if s[:1] in "[{" or (s[:1] in "\"'" and s[-1:] in "\"'"):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
    return raw


class DABStepRLResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: DABStepRLConfig

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # Every DABStep RL row shares one sandbox (mounted via ray.sub MOUNTS), so no
        # per-task session setup is needed.
        return BaseSeedSessionResponse()

    async def verify(self, body: DABStepRLVerifyRequest) -> DABStepRLVerifyResponse:
        text = _last_assistant_text(body)
        extracted = extract_prediction(text) if text else None
        predicted_answer = extracted if extracted is not None else text

        expected_answer = _coerce_expected(body.expected_answer)
        dtype = body.expected_answer_dtype
        dtype_missing = dtype is None

        correct = False
        if text:
            correct = bool(dabstep_equal(expected_answer, predicted_answer, dtype, question=body.question))

        boxed_found = bool(text and _BOXED.search(text))

        reward = 1.0 if correct else (self.config.empty_response_reward if not text else 0.0)
        # Binary reward gives no gradient separating "answered wrong at round 25" from
        # "looped to the cap and never answered" - both score 0. This makes the
        # termination failure visible to the policy gradient.
        if not boxed_found:
            reward -= self.config.no_answer_penalty

        return DABStepRLVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "expected_answer_dtype"}),
            reward=reward,
            expected_answer_str=str(expected_answer),
            predicted_answer=str(predicted_answer)[:2000],
            expected_answer_dtype_used=dtype,
            symbolic_correct=correct,
            boxed_answer_found=boxed_found,
            dtype_missing=dtype_missing,
        )


if __name__ == "__main__":
    DABStepRLResourcesServer.run_webserver()
