# tests/test_utils_demo_math.py

import math
import pytest

from utils_demo_math import (
    add,
    clamp,
    is_even,
    mean,
    safe_div,
    distance_2d,
    normalize,
)


def test_add_integers():
    # Проверяем сложение целых чисел
    assert add(2, 3) == 5


def test_add_floats():
    # Проверяем сложение вещественных чисел
    assert add(0.1, 0.2) == pytest.approx(0.3)


def test_clamp_inside_range():
    # Значение внутри диапазона не меняется
    assert clamp(5, 0, 10) == 5


def test_clamp_below_range():
    # Значение ниже диапазона прижимается к нижней границе
    assert clamp(-5, 0, 10) == 0


def test_clamp_invalid_range_raises():
    # Если lo > hi, должна быть ошибка
    with pytest.raises(ValueError):
        clamp(1, 10, 0)


def test_is_even_true():
    # Чётное число
    assert is_even(4) is True


def test_is_even_false():
    # Нечётное число
    assert is_even(5) is False


def test_mean_basic():
    # Среднее значение набора чисел
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_empty_raises():
    # Для пустого списка должна быть ошибка
    with pytest.raises(ValueError):
        mean([])


def test_safe_div_zero_raises():
    # Деление на ноль должно вызывать исключение
    with pytest.raises(ZeroDivisionError):
        safe_div(10, 0)


def test_distance_2d():
    # Проверяем расстояние 2D (3-4-5 треугольник)
    assert distance_2d(0, 0, 3, 4) == 5.0


def test_normalize_unit_length():
    # Нормализованный вектор должен иметь длину 1
    v = normalize([3, 4])
    length = math.sqrt(v[0] ** 2 + v[1] ** 2)
    assert length == pytest.approx(1.0)


def test_normalize_zero_vector_raises():
    # Нулевой вектор нормализовать нельзя
    with pytest.raises(ValueError):
        normalize([0, 0, 0])
