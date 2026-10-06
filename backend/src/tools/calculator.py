"""Safe calculation tools available to the LangGraph chat agent."""

import ast
import math
import operator
import re


NUMBER = r"(\d+(?:[.,]\d+)?)"
PERCENT_OF = re.compile(
    rf"{NUMBER}\s*(?:%|prozent)\s*(?:von|of)\s*{NUMBER}",
    re.IGNORECASE,
)
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic or percentage expression without eval()."""
    percentage_match = PERCENT_OF.search(expression)
    if percentage_match:
        percentage = float(percentage_match.group(1).replace(",", "."))
        base = float(percentage_match.group(2).replace(",", "."))
        result = percentage / 100 * base
        return (
            f"{_format(result)} "
            f"(berechnet: {_format(percentage)} % von {_format(base)})"
        )

    ratio_match = re.search(
        rf"{NUMBER}\s+(?:ist|sind)\s*wie viel\s*(?:%|prozent)\s*"
        rf"(?:von)?\s*{NUMBER}",
        expression,
        re.IGNORECASE,
    )
    if ratio_match:
        part = float(ratio_match.group(1).replace(",", "."))
        total = float(ratio_match.group(2).replace(",", "."))
        if total == 0:
            raise ValueError("Division durch null ist nicht möglich.")
        return f"{_format(part / total * 100)} %"

    normalized = expression.replace(",", ".")
    normalized = re.sub(
        rf"({NUMBER[1:-1]})\s*%",
        r"(\1 / 100)",
        normalized,
    )
    normalized = re.sub(
        r"(?<![\w.])0*(\d+)(?:\.(\d+))?(?![\w.])",
        lambda match: (
            f"{match.group(1)}.{match.group(2)}"
            if match.group(2)
            else match.group(1)
        ),
        normalized,
    )
    try:
        tree = ast.parse(normalized, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, TypeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(
            "Die Rechnung konnte nicht gelesen werden. Nutze Zahlen und "
            "Grundrechenarten, zum Beispiel „15 % von 200“."
        ) from error

    if not math.isfinite(result):
        raise ValueError("Das Rechenergebnis ist nicht endlich.")
    return _format(result)


def _evaluate(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Der Exponent ist zu groß.")
        return float(OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return float(UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    raise ValueError("Unerlaubter Ausdruck.")


def _format(value: float) -> str:
    return f"{value:g}".replace(".", ",")
