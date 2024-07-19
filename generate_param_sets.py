import argparse
import json5 as json
import itertools
from typing import Any


def contains(a: Any, b: Any) -> bool:
    if type(a) is dict:
        if type(b) is not dict:
            return False
        for key in b:
            if key not in a or not contains(a[key], b[key]):
                return False
        return True
    return a == b


def spread_json(input_dict: dict) -> list[dict]:
    """
    Takes a (nested) dictionary, which possibly contains lists "unfolds" them.
    Example:
    ```json
    {
        "a": [
            {"b": 0, "c": 1},
            {"b": 1, "c": [1, 2]}
        ]
    }
    ```
    is converted into
    ```json
    [
        {"a": {"b": 0, "c": 0}},
        {"a": {"b": 1, "c": 1}},
        {"a": {"b": 1, "c": 2}}
    ]
    ```

    Args:
        input_dict (dict): The dictionary to unfold

    Returns:
        list[dict]: the list of unfolded dictionaries
    """
    keys = input_dict.keys()
    vals = input_dict.values()
    val_lists = []
    for val in vals:
        if not isinstance(val, list):
            val = [val]
        spread_val = []
        for v in val:
            if isinstance(v, dict):
                spread_val.extend(spread_json(v))
            else:
                spread_val.append(v)
        val_lists.append(spread_val)

    combinations = itertools.product(*val_lists)
    return [dict(zip(keys, comb)) for comb in combinations]


def check_for_duplicates(param_sets: list[Any], out_file: str) -> list[Any]:
    """Filters out parameter sets which are already contained in a output file
    The file must be formated such that each line is a valid JSON representing a param set.
    For a param set O in the output file to match an input param set I, O must be a superset of I, meaning that it can contain extra keys.

    Args:
        param_sets (list[Any]): List of input param sets to check.
        out_file (str): Path to output file, containin param set

    Returns:
        list[Any]: Subset of param_sets: all sets not contained in the out file
    """
    if out_file is None:
        return param_sets
    try:
        with open(out_file) as file:
            lines = file.readlines()
    except FileNotFoundError:
        return param_sets
    duplicates = [json.loads(l) for l in lines]
    output = []
    for param_set in param_sets:
        is_contained = False
        for dup in duplicates:
            if contains(dup, param_set) and dup["value"] is not None:
                is_contained = True
                break
        if not is_contained:
            output.append(param_set)
    return output

