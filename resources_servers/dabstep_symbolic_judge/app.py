"""
DABStep symbolic-match judge resources server.

Backs `opencode_sandboxed_agent` DABStep configs. Unlike `itsla_equivalence_judge` (which grades
via an LLM judge comparing against guideline-derived free-text answers, since IT-SLA has no
independently-verified ground truth), DABStep questions have real verified ground-truth answers
(`expected_answer`/`expected_answer_dtype` in `nemo_skills/dataset/dabstep/test.jsonl`). Grading
here is therefore deterministic symbolic/numeric comparison, not an LLM call -- more reliable than
LLM judging when real ground truth exists, and avoids self-grading risk entirely.

The comparison logic (`dabstep_equal` and everything it calls) is vendored VERBATIM from
`Skills-dabstep-eval/nemo_skills/evaluation/evaluator/dabstep.py`'s `DABStepEvaluator`, including
its two external dependencies from `nemo_skills.evaluation.math_grader` -- `extract_answer`
(brace-counting `\boxed{}` extraction, handles nested braces and trailing text after the box) and
`math_equal` (symbolic/LaTeX comparison via `math_verify`/`latex2sympy2_extended`, handles
fractions, leading zeros, etc. before falling back to plain-decimal `compare_numeric`). An earlier
version of this file approximated both with a naive regex and a dropped pre-check respectively --
replaced with exact ports (see `requirements.txt`: `math-verify`, `latex2sympy2-extended`) so
grading here matches the real `ns eval` DABStep pipeline's `symbolic_correct` metric exactly, not
an approximation of it.

CRITICAL: `expected_answer`/`expected_answer_dtype` must never be read from `responses_create_params`
(what the agent sees) -- only from the sibling `expected_answer`/`metadata` fields on the verify
request body, which the agent's own prompt construction never includes. See
`environments/dabstep_smoke/data/*.jsonl` and its generation script for the enforced separation.
"""

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
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Optional

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.equivalence_llm_judge.app import (
    LLMJudgeRunRequest,
    _extract_expected_answer,
    _extract_last_assistant_text,
)

LIST_ITEM_DTYPE = {
    "list_int": "int",
    "list_float": "float",
    "list_str": "str",
    "list_str_float_pair": "str_float_pair",
}

ANSWER_LINE = re.compile(
    r"\b(?P<label>answer|output|result)\s*\d*\s*(?:\bis\b\s*[^\w\s]*|[^\w\s]+)\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)
HELPER_CALL = re.compile(r"(?:^|[.\s])(?:final_answer|print|answer)\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
PAIR_SPLIT = re.compile(r"\s*[,;]\s*(?=[^,;:\[\]{}]+:)")
FALLBACK_SPLIT = re.compile(r":|\b(?:is|are)\b", re.IGNORECASE)
NUMBER = re.compile(
    r"\s*[$€]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?\s*%?\s*(?:EUR|USD|GBP)?\s*", re.IGNORECASE
)
NUMBER_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
BOOL_TOKEN = re.compile(r"(?<!\w)(yes|no|true|false|1|0)(?!\w)", re.IGNORECASE)
CHOICE = re.compile(r"\s*([A-Z])(?:(?:\s|[^\w\s])+(.+?))?\s*", re.IGNORECASE)
NA_TEXT = re.compile(r"\s*(?:n/?a|not applicable|not available)\.?\s*", re.IGNORECASE)
EMPTY_TEXT = re.compile(r"\s*(?:none|null|nothing|empty|no answer|no results?)\.?\s*", re.IGNORECASE)
DELTA_RELATIVE_FEE = re.compile(r"\bdelta\b.*\brelative\s+fee\b.*\bchanged\s+to\b", re.IGNORECASE)


def dabstep_equal(
    expected_answer: Any, predicted_answer: Any, expected_answer_dtype: str | None = None, question: str | None = None
) -> bool:
    expected = normalize_answer(expected_answer)
    predicted = normalize_answer(predicted_answer)
    if match_answer(expected, predicted, expected_answer_dtype, question, raw_prediction=predicted_answer):
        return True
    return isinstance(predicted_answer, str) and match_fallback_lines(
        expected, predicted_answer, expected_answer_dtype, question
    )


def extract_prediction(value: Any) -> Any | None:
    if not isinstance(value, str):
        return value
    text = clean_text(value)
    if text[:1] + text[-1:] in {"[]", "{}"}:
        return None
    boxed = search_boxed(text)
    if boxed is not None:
        return clean_text(boxed)
    matches = list(ANSWER_LINE.finditer(text))
    if not matches:
        return None
    answer = next((match for match in reversed(matches) if match.group("label").casefold() == "answer"), None)
    return clean_text((answer or matches[-1]).group("value"))


def search_boxed(string: str) -> Optional[str]:
    """Exact port of nemo_skills.evaluation.math_grader.search_boxed -- brace-counting
    extraction (handles nested braces and trailing text after the closing brace), not a
    positional regex."""
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


def clean_text(value: str) -> str:
    text = value.strip(" \t\r\n.,!?#").strip("*").strip(" \t\r\n.,!?#")
    if text.startswith("```") and text.endswith("```"):
        lines = text[3:-3].strip().splitlines()
        text = "\n".join(lines[1:] if lines and re.fullmatch(r"[A-Za-z0-9_+-]+", lines[0].strip()) else lines)
    for old, new in (("$", ""), ("€", ""), ("<ctrl100>", ""), (r"\_", "_"), (r"\"", '"'), (r"\'", "'")):
        text = text.replace(old, new)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", re.sub(r"\\\s+", " ", text)).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"`":
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] == "{" and text[-1] == "}" and ":" not in text and "," not in text:
        text = text[1:-1].strip()
    call = HELPER_CALL.search(text)
    return clean_text(call.group(1)) if call else text


def normalize_answer(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_answer(item) for item in value]
    if isinstance(value, dict):
        return {
            clean_text(str(key)) if isinstance(key, str) else key: normalize_answer(item)
            for key, item in value.items()
        }
    if not isinstance(value, str):
        return value

    text = clean_text(value)
    if text and text[0] in "[{\"'":
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                return normalize_answer(parsed)
            except Exception:
                pass

    is_list, is_dict = text.startswith("[") and text.endswith("]"), text.startswith("{") and text.endswith("}")
    body = text[1:-1].strip() if is_list or is_dict else text
    if is_list and body == "":
        return []
    if is_dict and body == "":
        return {}

    if ":" in body and "\n" not in body:
        parts = [part.strip() for part in PAIR_SPLIT.split(body) if part.strip()]
        if all(":" in part for part in parts):
            pairs = [
                (
                    key.strip().strip("'\""),
                    clean_text(val.strip().strip("'\"")) if ":" in val else normalize_answer(val.strip().strip("'\"")),
                )
                for key, val in (part.split(":", 1) for part in parts)
            ]
            if is_dict:
                return {key: val for key, val in pairs}
            return [{key: val} for key, val in pairs] if is_list or len(pairs) > 1 else {pairs[0][0]: pairs[0][1]}

    if not is_dict and not NUMBER.fullmatch(body) and ("," in body or ";" in body):
        return [
            normalize_answer(part.strip().strip("'\""))
            for part in re.split(r"[,;]", body)
            if part.strip().strip("'\"")
        ]
    return [normalize_answer(body)] if is_list else text


def match_answer(
    expected: Any, predicted: Any, dtype: str | None = None, question: str | None = None, raw_prediction: Any = None
) -> bool:
    item_dtype = dtype if dtype in {"choice", "percent"} else LIST_ITEM_DTYPE.get(dtype)
    sign_flip = dtype in {"float", "int"} and isinstance(question, str) and DELTA_RELATIVE_FEE.search(question)

    if isinstance(expected, list):
        if not expected:
            return is_empty(predicted) or is_na(predicted)
        candidates = predicted if isinstance(predicted, list) else [predicted]
        if len(expected) != len(candidates):
            return False
        used: set[int] = set()
        for expected_item in expected:
            index = next(
                (
                    i
                    for i, item in enumerate(candidates)
                    if i not in used and match_answer(expected_item, item, item_dtype, question, raw_prediction=item)
                ),
                None,
            )
            if index is None:
                return False
            used.add(index)
        return True

    if isinstance(expected, dict):
        if isinstance(predicted, list) and len(predicted) == 1:
            return match_answer(expected, predicted[0], dtype, question, raw_prediction=raw_prediction)
        if len(expected) == 1:
            expected_key, expected_value = next(iter(expected.items()))
            if isinstance(predicted, dict) and len(predicted) == 1:
                predicted_key, predicted_value = next(iter(predicted.items()))
                return text_matches(expected_key, predicted_key) and match_answer(
                    expected_value, predicted_value, question=question, raw_prediction=predicted_value
                )
            return text_matches(expected_key, predicted) or match_answer(
                expected_value, predicted, question=question, raw_prediction=predicted
            )
        if not isinstance(predicted, dict) or len(expected) != len(predicted):
            return False
        used_keys: set[Any] = set()
        for expected_key, expected_value in expected.items():
            key = next(
                (
                    key
                    for key in predicted
                    if key not in used_keys
                    and text_matches(expected_key, key)
                    and match_answer(expected_value, predicted[key], question=question, raw_prediction=predicted[key])
                ),
                None,
            )
            if key is None:
                return False
            used_keys.add(key)
        return True

    if isinstance(predicted, list) and len(predicted) == 1:
        return match_answer(expected, predicted[0], dtype, question, raw_prediction=raw_prediction)
    if isinstance(predicted, dict) and len(predicted) == 1:
        predicted_key, predicted_value = next(iter(predicted.items()))
        return match_answer(expected, predicted_value, dtype, question, raw_prediction=predicted_value) or (
            isinstance(expected, str) and text_matches(expected, predicted_key)
        )

    if dtype == "percent":
        predicted_num = parse_number(predicted)
        return numbers_match(expected, predicted) or (
            predicted_num is not None
            and "%" not in str(raw_prediction)
            and numbers_match(expected, predicted_num * 100)
        )
    if dtype == "choice":
        choice = CHOICE.fullmatch(clean_text(str(expected)))
        if not choice:
            return False
        letter, label = choice.group(1).casefold(), choice.group(2)
        predicted_text = " ".join(clean_text(str(predicted)).split()).casefold()
        predicted_compact = "".join(ch for ch in predicted_text if ch.isalnum() or ch == "_")
        label_compact = (
            "".join(ch for ch in clean_text(str(label)).casefold() if ch.isalnum() or ch == "_") if label else ""
        )
        return (
            predicted_text == letter
            or text_matches(expected, predicted)
            or bool(label and (text_matches(label, predicted) or predicted_compact == letter + label_compact))
        )
    if is_na(expected):
        return is_na(predicted) or predicted is False
    if isinstance(expected, bool):
        return parse_bool(predicted) is expected or parse_bool(raw_prediction) is expected
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        predicted_num = parse_number(predicted)
        exact_int = dtype == "int" or isinstance(expected, int)
        return numbers_match(expected, predicted, exact_int) or bool(
            sign_flip and predicted_num is not None and numbers_match(abs(expected), abs(predicted_num), exact_int)
        )
    if isinstance(expected, str):
        return (
            is_empty(predicted) if is_empty(expected) else not is_na(predicted) and text_matches(expected, predicted)
        )
    return expected == predicted


def match_fallback_lines(expected: Any, prediction: str, dtype: str | None, question: str | None) -> bool:
    lines = [clean_text(line) for line in prediction.splitlines() if clean_text(line)]
    if not lines:
        return False
    fallback_lines = lines[:1] + ([] if lines[-1] == lines[0] else lines[-1:])
    if "\n" not in prediction and not any(
        FALLBACK_SPLIT.search(line) or ANSWER_LINE.search(line) or r"\boxed" in line for line in fallback_lines
    ):
        return False
    for line in fallback_lines:
        marker = ANSWER_LINE.search(line)
        suffix = marker.group("value") if marker else FALLBACK_SPLIT.split(line)[-1]
        for candidate in (line, suffix):
            extracted = extract_prediction(candidate)
            predicted = normalize_answer(extracted if extracted is not None else candidate)
            if candidate and match_answer(expected, predicted, dtype, question, raw_prediction=candidate):
                return True
    return False


def parse_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    if not isinstance(value, str) or not NUMBER.fullmatch(value):
        return None
    token = NUMBER_TOKEN.search(clean_text(value).replace(",", ""))
    return float(token.group()) if token else None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, str) and NA_TEXT.fullmatch(clean_text(value))):
        return False
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if not isinstance(value, str):
        return None
    text = clean_text(value)
    if (NUMBER.fullmatch(text) and re.search(r"[.,]\d|%", text)) or any(
        f"not {word}" in text.casefold() for word in ("yes", "no", "true", "false")
    ):
        return None
    for token in BOOL_TOKEN.finditer(text):
        before = text[: token.start()].strip(" \t\r\n.,!?'\"")
        after = text[token.end() :].strip(" \t\r\n.,!?'\"")
        if not before or not after:
            return token.group(1).casefold() in {"yes", "true", "1"}
    return None


def text_matches(expected: Any, predicted: Any) -> bool:
    expected_text = " ".join(clean_text(str(expected)).split()).casefold()
    predicted_text = " ".join(clean_text(str(predicted)).split()).casefold()
    if expected_text == predicted_text:
        return True
    if f"not {expected_text}" in predicted_text or f"{expected_text} is not" in predicted_text:
        return False
    return bool(expected_text) and (
        (
            predicted_text.startswith(expected_text)
            and len(predicted_text) > len(expected_text)
            and not predicted_text[len(expected_text)].isalnum()
        )
        or (
            predicted_text.endswith(expected_text)
            and len(predicted_text) > len(expected_text)
            and not predicted_text[-len(expected_text) - 1].isalnum()
        )
    )


def is_empty(value: Any) -> bool:
    return value in ("", []) or (isinstance(value, str) and EMPTY_TEXT.fullmatch(clean_text(value)) is not None)


def is_na(value: Any) -> bool:
    return value is None or (isinstance(value, str) and NA_TEXT.fullmatch(clean_text(value)) is not None)


def compare_numeric(num1: float, num2: float) -> bool:
    if num1 == num2:
        return True
    if num1 < 1 and num2 < 1:
        return math.isclose(num1, num2, rel_tol=1e-4, abs_tol=1e-4)
    decimals1 = len(str(num1).split(".")[-1]) if "." in str(num1) else 0
    decimals2 = len(str(num2).split(".")[-1]) if "." in str(num2) else 0
    if round(num1, min(decimals1, decimals2)) == round(num2, min(decimals1, decimals2)):
        return True
    return math.isclose(num1, num2, rel_tol=1e-4, abs_tol=1e-4)


def numbers_match(expected: Any, predicted: Any, exact_int: bool = False) -> bool:
    expected_num, predicted_num = parse_number(expected), parse_number(predicted)
    try:
        # Try the standard math grader first so symbolic numeric forms such as
        # "1/3", "\frac{1}{3}", and "sqrt(4)" get a chance before DABStep's
        # decimal-only closeness rule runs.
        if math_equal(expected, predicted, numeric_precision=15, timeout_seconds=10):
            return True
        if expected_num is not None and predicted_num is not None:
            return compare_numeric(float(expected_num), float(predicted_num))
        return False
    except OverflowError:
        return False


def _additional_normalization(expr: str) -> str:
    """Exact port of nemo_skills.evaluation.math_grader._additional_normalization."""
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    expr = expr.rstrip(".\\")
    return expr


def math_equal(gt_answer: Any, predicted_answer: Any, take_modulo: Optional[int] = None, **kwargs: Any) -> bool:
    """Exact port of nemo_skills.evaluation.math_grader.math_equal -- MCQ-string comparison,
    then normalized-LaTeX fast path, then full symbolic/LaTeX comparison via math_verify."""
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except Exception:
            predicted_answer = None
        return predicted_answer == gt_answer

    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_normalized_equal:
        return True

    text_literal_pattern = r"[a-zA-Z ,]+"
    is_text_literal = re.fullmatch(text_literal_pattern, normalized_gt) and re.fullmatch(
        text_literal_pattern, normalized_pred
    )
    if is_text_literal:
        return False

    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred, **kwargs)


class DABStepSymbolicJudgeResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "dabstep_symbolic_judge"


class DABStepSeedSessionRequest(BaseSeedSessionRequest):
    pass


class DABStepVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    question: Optional[str] = None
    expected_answer_dtype: Optional[str] = None


class DABStepVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    predicted_answer: str
    symbolic_correct: bool


class DABStepSymbolicJudgeResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: DABStepSymbolicJudgeResourcesServerConfig

    async def seed_session(self, body: DABStepSeedSessionRequest) -> BaseSeedSessionResponse:
        # Every DABStep task shares one fixed sandbox image (configured on the agent side), so
        # this is a no-op -- OpenCodeSandboxedAgent._start_sandbox falls through to creating a
        # fresh sandbox from its own sandbox_config, same as itsla_equivalence_judge.
        return BaseSeedSessionResponse()

    async def verify(self, body: DABStepVerifyRequest) -> DABStepVerifyResponse:
        candidate_text = _extract_last_assistant_text(body, extract_regex=None)
        extracted = extract_prediction(candidate_text)
        predicted_answer = extracted if extracted is not None else candidate_text

        expected_answer_raw = _extract_expected_answer(body) or ""
        # The environment file JSON-encodes expected_answer (the base agent request schema
        # requires a plain string; DABStep's real ground truth is natively float/int/list). Undo
        # that here so dabstep_equal/match_answer see the real native type -- otherwise a plain
        # number like "12.91" never reaches the numeric-comparison branch in match_answer (only
        # strings starting with [/{/quote get re-parsed by normalize_answer), silently degrading
        # to exact string matching and losing tolerance/rounding/symbolic-form support.
        try:
            expected_answer = json.loads(expected_answer_raw)
        except (json.JSONDecodeError, TypeError):
            expected_answer = expected_answer_raw
        expected_answer_dtype = body.expected_answer_dtype or (body.metadata or {}).get("expected_answer_dtype")
        question = body.question or (body.metadata or {}).get("question")

        correct = dabstep_equal(expected_answer, predicted_answer, expected_answer_dtype, question=question)

        return DABStepVerifyResponse(
            **body.model_dump(exclude={"expected_answer"}),
            reward=1.0 if correct else 0.0,
            expected_answer=str(expected_answer),
            predicted_answer=str(predicted_answer),
            symbolic_correct=correct,
        )


if __name__ == "__main__":
    DABStepSymbolicJudgeResourcesServer.run_webserver()
