from parse_params import *
from typing import Callable, Any

def generate_json_getter(path: str) -> Callable[[dict], Any]:
    path = path.split(".")

    def getter(j):
        for key in path:
            if type(j) == dict:
                if key not in j or j[key] is None:
                    return None
                j = j[key]
            else:
                d = vars(j)

                if d[key] is None:
                    return None
                j = d[key]
        return j

    return getter


def or_matcher(m: Any, lhs: Callable[[dict], bool], rhs: Callable[[dict], bool]) -> bool:
    l = lhs(m)
    if l:
        return True
    return rhs(m)


def and_matcher(m: Any, lhs: Callable[[dict], bool], rhs: Callable[[dict], bool]) -> bool:
    l = lhs(m)
    if not l:
        return False
    return rhs(m)


def json_matcher(m: Any, op: str, getter: Callable[[dict], Any], value: str):
    result = getter(m)
    if value == "None":
        value = None
    elif type(result) is int:
        value = int(value)
    elif type(result) is float:
        value = float(value)
    if op == "=":
        return result == value
    if op == "!=":
        return result != value
    if op == "<":
        return result is not None and result < value
    if op == "<=":
        return result is not None and result <= value
    if op == ">":
        return result is not None and result > value
    if op == ">=":
        return result is not None and result >= value


def apply_filter(measurements: list[dict], filter: str) -> list[dict]:
    """
    Filter a list of JSON objects based on a filter, which is an expression in Polish notation.
    The (simplified) grammar of the filter is specified as follows:
    ```bnf
    FILTER -> true | false | not FILTER | or FILTER FILTER | and FILTER FILTER | OP_FILTER | LIST_FILTER
    OP_FILTER -> OP GETTER VALUE
    LIST_FILTER GETTER LIST
    OP -> = | != | < | <= | > | >=
    GETTER -> KEY | KEY.GETTER
    KEY -> "arbitrary string excluding ."
    LIST -> VALUE | VALUE,LIST
    VALUE -> "arbitrary string excluding ,"
    ```

    A GETTER retrieves a value from a (possibly nested) dict, e.g. "a.b.c" retrieves the value `d["a"]["b"]["c"]`
    If any key along the path does not exist, the getter retrieves the value `None`.
    A VALUE of "None" is always interpreted as `None` instead of `"None"`.
    If the value obtained by a getter is of type `int` the `VALUE` to compared it to is automatically parsed as an `int`. The same goes for `float`s.
    A LIST_FILTER matches if the value retrieved by the getter is equal to at least one value in the list (after conversion).

    `apply_filter` also works on non-dicts. In this case the `vars` funnction is used instead.

    Args:
        measurements (list[dict]): Dictionaries to filter
        filter (str): Filter

    Returns:
        list[dict]: Dictionaries matching the filter
    """
    filter = [p for p in filter.split(" ") if len(p) > 0]

    index = 0

    def build_matcher():
        nonlocal index
        op = filter[index]
        if op == "true":

            def matcher(m):
                return True

            index += 1
            return matcher
        if op == "false":

            def matcher(m):
                return False

            index += 1
            return matcher
        if op == "not":
            index += 1
            expr = build_matcher()

            def matcher(m):
                return not expr(m)

            return matcher
        if op in ["or", "and"]:
            index += 1
            lhs = build_matcher()
            rhs = build_matcher()

            def matcher(m):
                if op == "or":
                    return or_matcher(m, lhs, rhs)
                return and_matcher(m, lhs, rhs)

            return matcher
        if op in ["=", "!=", "<", "<=", ">", ">="]:
            getter = generate_json_getter(filter[index + 1])
            value = filter[index + 2]

            def matcher(m):
                return json_matcher(m, op, getter, value)

            index += 3
            return matcher
        else:
            getter = generate_json_getter(op)
            values = filter[index + 1].split(",")

            def matcher(m):
                for value in values:
                    if json_matcher(m, "=", getter, value):
                        return True
                return False

            index += 2
            return matcher

    matcher = build_matcher()
    return [m for m in measurements if matcher(m)]
