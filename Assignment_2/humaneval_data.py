"""HumanEval-based docstring corpus used for (a) fine-tuning data and (b) a held-out accuracy check.

The assignment's own function (below_zero) is HumanEval/3, so it is excluded everywhere
to avoid training on the evaluation item.
"""
import ast
import doctest
import re
import textwrap

from datasets import load_dataset

from task import PROMPTS, extract_code

EVAL_TASK_ID = "HumanEval/3"   # below_zero - never used for training


def load_items():
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for r in ds:
        if r["task_id"] == EVAL_TASK_ID:
            continue
        code = (r["prompt"] + r["canonical_solution"]).strip("\n")
        imports = "\n".join(l for l in r["prompt"].splitlines() if l.startswith(("import ", "from ")))
        items.append({"task_id": r["task_id"], "entry_point": r["entry_point"], "code": code,
                      "imports": imports, "test": r["test"]})
    return items


def load_mbpp_items():
    """MBPP (974 short Python functions with assert tests) - extra transfer data, training only."""
    items = []
    for split in ("train", "validation", "test", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            code = r["code"].replace("\r\n", "\n").replace("\t", "    ").strip("\n")
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            called = [n for n in fns if any(n + "(" in t for t in r["test_list"])]
            if not called:
                continue
            # Mirror the assignment's input format (a function with a plain prose docstring): insert the
            # MBPP task description as the docstring of the target function.
            first = fns[called[0]].body[0]
            if first.lineno == fns[called[0]].lineno:
                continue  # one-line def, no room for a docstring
            lines = code.split("\n")
            desc = r["text"].strip().replace('"""', "'''").replace("\\", "\\\\")
            lines.insert(first.lineno - 1, " " * first.col_offset + f'""" {desc}\n' + " " * first.col_offset + '"""')
            code = "\n".join(lines)
            try:
                ast.parse(code)
            except SyntaxError:
                continue
            asserts = "\n".join("    " + t for t in r["test_list"])
            items.append({"task_id": f"MBPP/{r['task_id']}", "entry_point": called[0], "code": code,
                          "imports": r["test_setup_code"] or "",
                          "test": "def check(candidate):\n" + asserts + "\n"})
    return items


def split_items(items):
    """Every 5th item is held out (~20%) for the generalisation check; the rest are training data."""
    train = [it for i, it in enumerate(items) if i % 5 != 0]
    held = [it for i, it in enumerate(items) if i % 5 == 0]
    return train, held


def make_prompt(code, prompt_id):
    return PROMPTS[prompt_id].format(code=code)


def _strip_doc(fn):
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return [ast.dump(n) for n in body]


def _find_fn(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def check_generic(text, item):
    """Same criteria as the below_zero checker, generalised to any HumanEval function:
    valid Python, target function present with a docstring, body unchanged, unit tests still pass,
    every parameter documented in an Args/Parameters section, a Returns section, correct doctests."""
    code = textwrap.dedent(extract_code(text))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax error"
    fn = _find_fn(tree, item["entry_point"])
    if fn is None:
        return False, "function missing"
    doc = ast.get_docstring(fn)
    if not doc:
        return False, "no docstring"
    orig = _find_fn(ast.parse(item["code"]), item["entry_point"])
    if _strip_doc(fn) != _strip_doc(orig):
        return False, "body changed"
    params = [a.arg for a in fn.args.args]
    if params:
        if not re.search(r"^\s*(args|arguments|parameters)\s*:?\s*$", doc, re.M | re.I):
            return False, "no Args section"
        if any(p not in doc for p in params):
            return False, "parameter undocumented"
    if not re.search(r"^\s*returns?\s*:?\s*$", doc, re.M | re.I):
        return False, "no Returns section"
    ns = {}
    try:
        exec(item["imports"] + "\n" + code, ns)
        exec(item["test"], ns)
        ns["check"](ns[item["entry_point"]])
    except Exception:  # noqa: BLE001
        return False, "tests fail"
    if ">>>" in doc:
        try:
            test = doctest.DocTestParser().get_doctest(doc, dict(ns), item["entry_point"], None, 0)
        except ValueError:
            return False, "doctest malformed"
        res = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE).run(test, out=lambda s: None)
        if res.failed:
            return False, "doctest wrong"
    return True, "ok"
