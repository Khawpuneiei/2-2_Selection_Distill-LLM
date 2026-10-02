"""Answer extraction and conservative math matching without eval()."""

from __future__ import annotations

import ast
import math
import operator
import re
from fractions import Fraction


def _read_braced_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _boxed_content(text: str) -> str | None:
    index = text.rfind(r"\boxed")
    if index < 0:
        return None
    start = index + len(r"\boxed")
    while start < len(text) and text[start].isspace():
        start += 1
    group = _read_braced_group(text, start)
    return group[0].strip() if group else None


def extract_final_answer(text: str) -> str:
    """Extract the last GSM8K delimiter, boxed answer, or explicit answer line."""
    value = str(text).strip()
    gsm_match = re.search(r"####\s*(.+?)\s*$", value, flags=re.DOTALL)
    if gsm_match:
        return gsm_match.group(1).strip()
    boxed = _boxed_content(value)
    if boxed is not None:
        return boxed
    answer_lines = re.findall(
        r"(?im)(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", value
    )
    if answer_lines:
        return answer_lines[-1].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else ""


_ANSWER_IS = re.compile(r"(?i)\bthe\s+(?:final\s+)?answer\s+is\s*:?\s*")


def extract_prediction_answer(text: str) -> str:
    """Extract a model's answer from the FIRST answer marker it writes.

    Small base models often state an answer and then keep generating unrelated
    text, so the earliest ``####``, ``\boxed{}``, or "the answer is" marker is
    used. References keep using ``extract_final_answer``.
    """
    value = str(text).strip()
    candidates: list[tuple[int, str]] = []
    gsm_match = re.search(r"####\s*([^\n]+)", value)
    if gsm_match:
        candidates.append((gsm_match.start(), gsm_match.group(1).strip()))
    boxed_index = value.find(r"\boxed")
    if boxed_index >= 0:
        start = boxed_index + len(r"\boxed")
        while start < len(value) and value[start].isspace():
            start += 1
        group = _read_braced_group(value, start)
        if group:
            candidates.append((boxed_index, group[0].strip()))
    answer_match = _ANSWER_IS.search(value)
    if answer_match:
        line = value[answer_match.end():].split("\n", 1)[0].strip()
        boxed = _boxed_content(line)
        candidates.append((answer_match.start(), boxed if boxed is not None else line.rstrip(".")))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return extract_final_answer(value)


def _replace_latex_fractions(text: str) -> str:
    value = text
    while True:
        match = re.search(r"\\(?:dfrac|tfrac|frac)\s*", value)
        if not match:
            return value
        numerator = _read_braced_group(value, match.end())
        if not numerator:
            return value
        denominator = _read_braced_group(value, numerator[1])
        if not denominator:
            return value
        replacement = f"(({numerator[0]})/({denominator[0]}))"
        value = value[: match.start()] + replacement + value[denominator[1] :]


def _normalized_expression(text: str, *, prediction: bool = False) -> str:
    value = (extract_prediction_answer(text) if prediction else extract_final_answer(text)).strip()
    value = re.sub(r"\\(?:left|right|displaystyle|quad|qquad)\b", "", value)
    for spacing in (r"\,", r"\;", r"\!"):
        value = value.replace(spacing, "")
    value = value.replace("$", "").replace(",", "")
    value = value.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    value = value.replace("^", "**")
    value = _replace_latex_fractions(value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_numeric_value(expression: str) -> Fraction | float | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def visit(node: ast.AST) -> Fraction | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Pow):
                if not isinstance(right, Fraction) or right.denominator != 1 or abs(right) > 100:
                    raise ValueError("unsupported exponent")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return math.sqrt(float(visit(node.args[0])))
        raise ValueError("unsupported expression")

    try:
        return visit(tree)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None


def answers_match(prediction: str, reference: str, *, tolerance: float = 1e-9) -> bool:
    """Match common numeric / LaTeX scalar forms; otherwise compare normalized text."""
    predicted = _normalized_expression(prediction, prediction=True)
    expected = _normalized_expression(reference)
    if not predicted or not expected:
        return False
    predicted_number = _safe_numeric_value(predicted)
    expected_number = _safe_numeric_value(expected)
    if predicted_number is not None and expected_number is not None:
        try:
            return math.isclose(
                float(predicted_number), float(expected_number), rel_tol=tolerance, abs_tol=tolerance
            )
        except (OverflowError, ValueError):
            return predicted_number == expected_number
    if predicted.casefold() == expected.casefold():
        return True
    # Free-form answer lines ("The answer is 18."): compare the last number
    # in the extracted answer when the reference itself is a plain number.
    if expected_number is not None and predicted_number is None:
        numbers = _NUMBER.findall(extract_prediction_answer(prediction).replace(",", ""))
        if numbers:
            candidate = _safe_numeric_value(numbers[-1])
            if candidate is not None:
                try:
                    return math.isclose(
                        float(candidate), float(expected_number), rel_tol=tolerance, abs_tol=tolerance
                    )
                except (OverflowError, ValueError):
                    return candidate == expected_number
    return False


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
