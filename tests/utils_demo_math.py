# utils_demo_math.py

from __future__ import annotations

import math
from typing import Iterable, List


def add(a: float, b: float) -> float:
    return a + b


def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return max(lo, min(hi, x))


def is_even(n: int) -> bool:
    return n % 2 == 0


def mean(values: Iterable[float]) -> float:
    vals: List[float] = list(values)
    if len(vals) == 0:
        raise ValueError("mean of empty iterable")
    return sum(vals) / len(vals)


def safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        raise ValueError("zero vector")
    return [x / norm for x in v]
