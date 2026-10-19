# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
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

import ast
import json
import math
import re
from typing import Any

from resources_servers.dabstep_rl._math_grader import extract_answer, math_equal

LIST_ITEM_DTYPE = {
    "list_int": "int",
    "list_float": "float",
    "list_str": "str",
    "list_str_float_pair": "str_float_pair",
}

# Regexes are grouped here because these are the riskiest text rules. Each block
# says what the regex is for and gives examples in plain English. The comments
# are intentionally more detailed than the code around them.
#
# ANSWER_LINE detects lines where the model clearly says "answer", "output", or
# "result"
# The label is kept so fallback matching can prefer explicit "answer"
#   "Final answer: 12" -> label="answer", value="12"
#   "Text output = done" -> label="output", value="done"
ANSWER_LINE = re.compile(
    r"\b(?P<label>answer|output|result)\s*\d*\s*(?:\bis\b\s*[^\w\s]*|[^\w\s]+)\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)

# HELPER_CALL unwraps exact helper-style final answers. It accepts print(42),
# tool.print(42), final_answer("A"), and answer("A"). It deliberately does not
# accept wrong_answer(...), wrong_print(...), or blueprint(...).
HELPER_CALL = re.compile(r"(?:^|[.\s])(?:final_answer|print|answer)\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)

# PAIR_SPLIT separates loose key/value pairs only at delimiters followed by a
# new key and colon. This keeps "A: 15,237.49, B: 2" from splitting inside the
# thousands separator, while still parsing "A: 1, B: 2".
PAIR_SPLIT = re.compile(r"\s*[,;]\s*(?=[^,;:\[\]{}]+:)")

# FALLBACK_SPLIT is only used after direct matching fails. It pulls a likely
# answer suffix from prose or labels:
#   "the answer is 17" -> "17"
#   "scheme: GlobalCard" -> "GlobalCard"
FALLBACK_SPLIT = re.compile(r":|\b(?:is|are)\b", re.IGNORECASE)

# NUMBER accepts complete numeric candidates with optional currency, thousands
# separators, decimal/exponent notation, percent sign, or currency suffix:
# "$15,237.49 USD", "-1e-3", and "73.15%" are numeric. "1, 2, 3" is not numeric,
# so comma-separated answers can still become lists.
NUMBER = re.compile(
    r"\s*[$€]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?\s*%?\s*(?:EUR|USD|GBP)?\s*", re.IGNORECASE
)

# NUMBER_TOKEN extracts the numeric core only after NUMBER has already accepted
# the whole candidate. It is not used to fish a number out of a sentence.
NUMBER_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

# BOOL_TOKEN accepts standalone bool tokens. The boundary guards keep words like
# "nobody" and "truest" from becoming "no" or "true".
BOOL_TOKEN = re.compile(r"(?<!\w)(yes|no|true|false|1|0)(?!\w)", re.IGNORECASE)

# CHOICE accepts choice answers like "A", "A. Revenue", and "B - gross margin".
CHOICE = re.compile(r"\s*([A-Z])(?:(?:\s|[^\w\s])+(.+?))?\s*", re.IGNORECASE)

# These are explicit empty/NA spellings seen in DABStep candidate answers and
# model output. They are narrow text conventions, not broad semantic matching.
NA_TEXT = re.compile(r"\s*(?:n/?a|not applicable|not available)\.?\s*", re.IGNORECASE)
EMPTY_TEXT = re.compile(r"\s*(?:none|null|nothing|empty|no answer|no results?)\.?\s*", re.IGNORECASE)

# Narrow sign-flip relaxation for the DABStep wording where a delta can be
# reported with the opposite sign but the same magnitude.
DELTA_RELATIVE_FEE = re.compile(r"\bdelta\b.*\brelative\s+fee\b.*\bchanged\s+to\b", re.IGNORECASE)


def dabstep_equal(
    expected_answer: Any, predicted_answer: Any, expected_answer_dtype: str | None = None, question: str | None = None
) -> bool:
    # Main flow:
    #   1. Normalize the expected answer from the dataset.
    #   2. Normalize the prediction that eval_single already extracted.
    #   3. Match once with dtype-aware recursive equality.
    #   4. Only if that fails, scan the first/last fallback lines.
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
    boxed = extract_answer(text, extract_from_boxed=True)
    if boxed is not None:
        return clean_text(boxed)
    matches = list(ANSWER_LINE.finditer(text))
    if not matches:
        return None
    answer = next((match for match in reversed(matches) if match.group("label").casefold() == "answer"), None)
    return clean_text((answer or matches[-1]).group("value"))


def clean_text(value: str) -> str:
    # Clean presentation noise only. This function does not decide whether a
    # value is a list, dict, number, bool, or string.
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
    # This is the only parser for answer shapes. It recursively cleans lists and
    # dicts, tries real JSON/Python literals, then handles loose DABStep syntax.
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

    # Loose key/value parsing is deliberately consistent:
    #   "scheme: GlobalCard" -> {"scheme": "GlobalCard"}
    #   "{A: 1, B: 2}" -> {"A": "1", "B": "2"}
    #   "A: 1, B: 2" -> [{"A": "1"}, {"B": "2"}]
    #   "[scheme: GlobalCard]" -> [{"scheme": "GlobalCard"}]
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
    # This is the only recursive matcher. Everything above only cleans/extracts
    # or parses shape; all equality decisions live here.
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
    # Fallbacks are intentionally last and narrow. They inspect only the first
    # and last non-empty lines, so scratch values buried in the middle of a log
    # cannot win.
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
    # The whole candidate must be numeric-looking before the numeric core is
    # extracted. This rejects "scratch produced 12.34" as a numeric answer.
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    if not isinstance(value, str) or not NUMBER.fullmatch(value):
        return None
    token = NUMBER_TOKEN.search(clean_text(value).replace(",", ""))
    return float(token.group()) if token else None


def parse_bool(value: Any) -> bool | None:
    # Bool matching accepts exact bools, 0/1 scalars, and a yes/no token at the
    # beginning or end of a sentence. It rejects numeric-only decimals/percents
    # such as "0.9", but still accepts "Yes. The measured rate is 0.1149".
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
    # Text matching allows harmless case/spacing drift and table-name affixes:
    # "aci_C" can match "C"; "abcC", "GlobalCarded", and explicit negations do not.
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
    # Default DABStep numeric rule, adapted from
    # data-explorer-agent/dabstep_agent/metric.py:
    # exact equality wins; small numbers use math.isclose; other numbers are
    # first rounded to the smaller displayed decimal precision, then checked
    # with math.isclose(rel_tol=1e-4, abs_tol=1e-4).
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
