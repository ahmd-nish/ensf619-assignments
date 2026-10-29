"""Task definition (identical to Assignment 1), prompts, and the below_zero accuracy checker."""
import ast
import doctest
import re
import textwrap
from typing import List

FUNCTION_CODE = '''from typing import List
def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True.
    """
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False'''

PROMPTS = {
    "P-1": (
        "Please act as an expert Python software engineer. Given the function below, I would "
        "appreciate it if you could generate a complete and professional Google-style docstring. "
        "The docstring should not include any extra commentary, only the docstring itself and the "
        "original function. Thank you.\n\n{code}"
    ),
    "P-2": (
        "Generate Google-style docstring for the following Python function. "
        "Only output the docstring and function code.\n\n{code}"
    ),
    "P-3": "Python Docstring: {code}. Output only the new code.",
}

# Reference behaviour of the original function, used to confirm the logic was not altered.
TEST_CASES = [([], False), ([1, 2, -3, 1, 2, -3], False), ([1, 2, -4, 5, 6], True),
              ([-1], True), ([5, -5], False), ([1, -2, 10], True)]


def build_prompt(prompt_id):
    return PROMPTS[prompt_id].format(code=FUNCTION_CODE)


def extract_code(text):
    """Pull the Python code out of a model reply (fenced block if present, else raw text)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        for b in blocks:
            if "def below_zero" in b:
                return b
        return blocks[0]
    # Unterminated fence (generation hit the token limit).
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)
    if m:
        return m.group(1)
    return text


def _body_without_docstring(fn):
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return [ast.dump(n) for n in body]


_ORIGINAL_BODY = None


def check_accuracy(text):
    """Return (passed: bool, reason: str).

    PASS requires all of:
      1. the extracted code parses (syntactically valid Python),
      2. it defines below_zero with a docstring,
      3. the function logic is unchanged (identical AST body and identical results on test cases),
      4. the docstring documents the parameter (`operations`) and the return value (bool / True/False)
         in a recognised structured style (Google `Args:`/`Returns:` or NumPy `Parameters`/`Returns`),
      5. it does not describe the opposite condition (e.g. "above zero"),
      6. any doctest examples in the docstring are correct.
    """
    global _ORIGINAL_BODY
    if _ORIGINAL_BODY is None:
        _ORIGINAL_BODY = _body_without_docstring(
            next(n for n in ast.parse(FUNCTION_CODE).body if isinstance(n, ast.FunctionDef)))

    code = textwrap.dedent(extract_code(text))
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "below_zero"]
    if not fns:
        return False, "function below_zero missing"
    fn = fns[0]
    doc = ast.get_docstring(fn)
    if not doc:
        return False, "no docstring"
    if _body_without_docstring(fn) != _ORIGINAL_BODY:
        return False, "function body changed"

    # Models often return only the function without `from typing import List`; that is not a
    # docstring defect, so the name is provided for the behavioural check.
    ns = {"List": List}
    try:
        exec(compile(code, "<generated>", "exec"), ns)
        f = ns["below_zero"]
        for args, expected in TEST_CASES:
            if f(list(args)) != expected:
                return False, "behaviour changed"
    except Exception as e:  # noqa: BLE001
        return False, f"runtime error: {type(e).__name__}"

    d = doc.lower()
    has_args = re.search(r"^\s*(args|arguments|parameters)\s*:?\s*$", doc, re.M | re.I) or "args:" in d
    has_returns = re.search(r"^\s*returns?\s*:?\s*$", doc, re.M | re.I) or "returns:" in d
    if not (has_args and has_returns):
        return False, "missing Args/Returns sections"
    if "operations" not in d:
        return False, "parameter not documented"
    if not ("bool" in d or "true" in d):
        return False, "return value not documented"
    if re.search(r"above zero|exceeds zero|greater than zero", d) and "below" not in d:
        return False, "describes wrong condition"

    if ">>>" in doc:
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
        try:
            test = doctest.DocTestParser().get_doctest(doc, {"below_zero": f}, "below_zero", None, 0)
        except ValueError:
            return False, "doctest malformed"
        res = runner.run(test, out=lambda s: None)
        if res.failed:
            return False, f"{res.failed} doctest example(s) wrong"
    return True, "ok"
