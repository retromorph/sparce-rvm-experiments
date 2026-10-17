"""Real breaking API changes: numpy 1.26.4 -> 2.1.3 and scipy 1.10.1 -> 1.15.3.

Every change below is documented in the official release notes (see each
``change_url``) and every program was verified by execution on both pins:
``old_code`` passes the hidden test on the old pin and fails on the new pin,
``new_code`` passes on the new pin, sibling programs pass on both pins.
"""

from __future__ import annotations

from textwrap import dedent

from sparce.catalog import ApiChange, Program, SiblingProgram

NP_OLD = "1.26.4"
NP_NEW = "2.1.3"
NP_DATE = "2024-06-16"
NP_URL = "https://numpy.org/doc/stable/release/2.0.0-notes.html"
NP_MIGRATION_URL = "https://numpy.org/devdocs/numpy_2_0_migration_guide.html"

SP_OLD = "1.10.1"
SP_NEW = "1.15.3"
# numpy 1.26.4 satisfies both scipy pins; pinning it keeps both scipy
# environments deterministic and free of numpy-2 confounds.
SP_EXTRA = ("numpy==1.26.4",)
SP_URL_112 = "https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html"
SP_URL_114 = "https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html"
SP_URL_115 = "https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html"
SP_DATE_112 = "2024-01-20"
SP_DATE_114 = "2024-06-24"
SP_DATE_115 = "2025-01-03"


def _code(text: str) -> str:
    return dedent(text).strip() + "\n"


def _prog(name: str, old: str, new: str, test: str, description: str = "") -> Program:
    return Program(name=name, old_code=_code(old), new_code=_code(new), test_code=_code(test), description=description)


def _sib(name: str, code: str, test: str, description: str = "") -> SiblingProgram:
    return SiblingProgram(name=name, code=_code(code), test_code=_code(test), description=description)


# ---------------------------------------------------------------------------
# numpy.product -> numpy.prod
# ---------------------------------------------------------------------------

_NP_PRODUCT = ApiChange(
    package="numpy",
    api_family="numpy.product",
    old_api="np.product",
    new_api="np.prod",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_MIGRATION_URL,
    summary="np.product alias removed in NumPy 2.0; use np.prod.",
    programs=(
        _prog(
            "geometric_mean",
            """
            import numpy as np

            def geometric_mean(values):
                arr = np.asarray(values, dtype=float)
                return float(np.product(arr) ** (1.0 / arr.size))
            """,
            """
            import numpy as np

            def geometric_mean(values):
                arr = np.asarray(values, dtype=float)
                return float(np.prod(arr) ** (1.0 / arr.size))
            """,
            """
            import solution
            assert abs(solution.geometric_mean([1, 2, 4]) - 2.0) < 1e-9
            assert abs(solution.geometric_mean([3, 3, 3]) - 3.0) < 1e-9
            assert abs(solution.geometric_mean([2, 8]) - 4.0) < 1e-9
            """,
            "Geometric mean of a list of positive numbers.",
        ),
        _prog(
            "box_volume",
            """
            import numpy as np

            class Box:
                def __init__(self, dims):
                    self.dims = np.asarray(dims, dtype=float)

                def volume(self):
                    return float(np.product(self.dims))

                def scaled(self, factor):
                    return Box(self.dims * factor)
            """,
            """
            import numpy as np

            class Box:
                def __init__(self, dims):
                    self.dims = np.asarray(dims, dtype=float)

                def volume(self):
                    return float(np.prod(self.dims))

                def scaled(self, factor):
                    return Box(self.dims * factor)
            """,
            """
            import solution
            box = solution.Box([2, 3, 4])
            assert box.volume() == 24.0
            assert box.scaled(2).volume() == 192.0
            assert solution.Box([1.5, 2]).volume() == 3.0
            """,
            "Volume of an axis-aligned box from its dimensions.",
        ),
        _prog(
            "row_products",
            """
            import numpy as np

            def row_products(matrix):
                return np.product(np.asarray(matrix), axis=1).tolist()
            """,
            """
            import numpy as np

            def row_products(matrix):
                return np.prod(np.asarray(matrix), axis=1).tolist()
            """,
            """
            import solution
            assert solution.row_products([[1, 2, 3], [4, 5, 6]]) == [6, 120]
            assert solution.row_products([[2, 2], [3, 0]]) == [4, 0]
            """,
            "Product of each row of a matrix.",
        ),
        _prog(
            "cell_count",
            """
            import numpy as np

            def cell_count(shape):
                return int(np.product(shape))

            def bytes_needed(shape, itemsize):
                return cell_count(shape) * itemsize
            """,
            """
            import numpy as np

            def cell_count(shape):
                return int(np.prod(shape))

            def bytes_needed(shape, itemsize):
                return cell_count(shape) * itemsize
            """,
            """
            import solution
            assert solution.cell_count((3, 4, 5)) == 60
            assert solution.cell_count(()) == 1
            assert solution.bytes_needed((2, 3), 8) == 48
            """,
            "Number of cells and bytes for an array shape.",
        ),
        _prog(
            "joint_probability",
            """
            import numpy as np

            def joint_probability(probs):
                probs = np.asarray(probs, dtype=float)
                if np.any(probs < 0) or np.any(probs > 1):
                    raise ValueError("probabilities must be in [0, 1]")
                return float(np.product(probs))
            """,
            """
            import numpy as np

            def joint_probability(probs):
                probs = np.asarray(probs, dtype=float)
                if np.any(probs < 0) or np.any(probs > 1):
                    raise ValueError("probabilities must be in [0, 1]")
                return float(np.prod(probs))
            """,
            """
            import solution
            assert abs(solution.joint_probability([0.5, 0.5, 0.5]) - 0.125) < 1e-12
            assert abs(solution.joint_probability([1.0, 0.2]) - 0.2) < 1e-12
            try:
                solution.joint_probability([1.5])
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """,
            "Joint probability of independent events.",
        ),
        _prog(
            "factorial_table",
            """
            import numpy as np

            def factorial_table(n):
                return [int(np.product(np.arange(1, k + 1))) for k in range(1, n + 1)]
            """,
            """
            import numpy as np

            def factorial_table(n):
                return [int(np.prod(np.arange(1, k + 1))) for k in range(1, n + 1)]
            """,
            """
            import solution
            assert solution.factorial_table(5) == [1, 2, 6, 24, 120]
            assert solution.factorial_table(1) == [1]
            assert solution.factorial_table(0) == []
            """,
            "Table of factorials 1! .. n!.",
        ),
        _prog(
            "compound_growth",
            """
            import numpy as np

            def compound_growth(rates):
                return float(np.product(1.0 + np.asarray(rates, dtype=float)) - 1.0)
            """,
            """
            import numpy as np

            def compound_growth(rates):
                return float(np.prod(1.0 + np.asarray(rates, dtype=float)) - 1.0)
            """,
            """
            import solution
            assert abs(solution.compound_growth([0.1, 0.1]) - 0.21) < 1e-12
            assert abs(solution.compound_growth([0.0, 0.5]) - 0.5) < 1e-12
            assert solution.compound_growth([]) == 0.0
            """,
            "Total compound growth from a sequence of period rates.",
        ),
    ),
    siblings=(
        _sib(
            "nanprod_total",
            """
            import numpy as np

            def product_ignoring_missing(values):
                return float(np.nanprod(np.asarray(values, dtype=float)))
            """,
            """
            import solution
            assert solution.product_ignoring_missing([2, float("nan"), 3]) == 6.0
            assert solution.product_ignoring_missing([float("nan")]) == 1.0
            """,
            "Product that skips NaN entries (np.nanprod, unchanged).",
        ),
        _sib(
            "multiply_reduce_columns",
            """
            import numpy as np

            def column_products(matrix):
                return np.multiply.reduce(np.asarray(matrix), axis=0).tolist()
            """,
            """
            import solution
            assert solution.column_products([[1, 2], [3, 4]]) == [3, 8]
            assert solution.column_products([[5, 0]]) == [5, 0]
            """,
            "Column products via np.multiply.reduce (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.cumproduct -> numpy.cumprod
# ---------------------------------------------------------------------------

_NP_CUMPRODUCT = ApiChange(
    package="numpy",
    api_family="numpy.cumproduct",
    old_api="np.cumproduct",
    new_api="np.cumprod",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_MIGRATION_URL,
    summary="np.cumproduct alias removed in NumPy 2.0; use np.cumprod.",
    programs=(
        _prog(
            "running_balance",
            """
            import numpy as np

            def running_balance(start, rates):
                growth = np.cumproduct(1.0 + np.asarray(rates, dtype=float))
                return [round(float(v), 6) for v in start * growth]
            """,
            """
            import numpy as np

            def running_balance(start, rates):
                growth = np.cumprod(1.0 + np.asarray(rates, dtype=float))
                return [round(float(v), 6) for v in start * growth]
            """,
            """
            import solution
            assert solution.running_balance(100, [0.1, 0.1]) == [110.0, 121.0]
            assert solution.running_balance(50, [-0.5, 1.0]) == [25.0, 50.0]
            """,
            "Account balance after each compounding period.",
        ),
        _prog(
            "cumulative_factorials",
            """
            import numpy as np

            def cumulative_factorials(n):
                return np.cumproduct(np.arange(1, n + 1)).tolist()
            """,
            """
            import numpy as np

            def cumulative_factorials(n):
                return np.cumprod(np.arange(1, n + 1)).tolist()
            """,
            """
            import solution
            assert solution.cumulative_factorials(5) == [1, 2, 6, 24, 120]
            assert solution.cumulative_factorials(0) == []
            """,
            "Factorials via a cumulative product.",
        ),
        _prog(
            "survival_curve",
            """
            import numpy as np

            def survival_curve(hazards):
                hazards = np.asarray(hazards, dtype=float)
                return np.cumproduct(1.0 - hazards).round(6).tolist()
            """,
            """
            import numpy as np

            def survival_curve(hazards):
                hazards = np.asarray(hazards, dtype=float)
                return np.cumprod(1.0 - hazards).round(6).tolist()
            """,
            """
            import solution
            assert solution.survival_curve([0.1, 0.1, 0.5]) == [0.9, 0.81, 0.405]
            assert solution.survival_curve([0.0]) == [1.0]
            """,
            "Survival probabilities from per-period hazards.",
        ),
        _prog(
            "scale_chain",
            """
            import numpy as np

            class ScaleChain:
                def __init__(self, factors):
                    self.factors = np.asarray(factors, dtype=float)

                def cumulative(self):
                    return np.cumproduct(self.factors).tolist()

                def final_scale(self):
                    return float(self.cumulative()[-1])
            """,
            """
            import numpy as np

            class ScaleChain:
                def __init__(self, factors):
                    self.factors = np.asarray(factors, dtype=float)

                def cumulative(self):
                    return np.cumprod(self.factors).tolist()

                def final_scale(self):
                    return float(self.cumulative()[-1])
            """,
            """
            import solution
            chain = solution.ScaleChain([2, 3, 4])
            assert chain.cumulative() == [2.0, 6.0, 24.0]
            assert chain.final_scale() == 24.0
            """,
            "Cumulative scale factors of a transform chain.",
        ),
        _prog(
            "decay_steps",
            """
            import numpy as np

            def decay_steps(initial, factor, steps):
                return (initial * np.cumproduct(np.full(steps, factor))).tolist()
            """,
            """
            import numpy as np

            def decay_steps(initial, factor, steps):
                return (initial * np.cumprod(np.full(steps, factor))).tolist()
            """,
            """
            import solution
            assert solution.decay_steps(16, 0.5, 3) == [8.0, 4.0, 2.0]
            assert solution.decay_steps(1, 0.1, 2) == [0.1, 0.010000000000000002]
            """,
            "Geometric decay sequence.",
        ),
        _prog(
            "suffix_products",
            """
            import numpy as np

            def suffix_products(values):
                arr = np.asarray(values)
                return np.cumproduct(arr[::-1])[::-1].tolist()
            """,
            """
            import numpy as np

            def suffix_products(values):
                arr = np.asarray(values)
                return np.cumprod(arr[::-1])[::-1].tolist()
            """,
            """
            import solution
            assert solution.suffix_products([1, 2, 3, 4]) == [24, 24, 12, 4]
            assert solution.suffix_products([5]) == [5]
            """,
            "Product of each suffix of a sequence.",
        ),
        _prog(
            "row_cumulative",
            """
            import numpy as np

            def row_cumulative(matrix):
                return np.cumproduct(np.asarray(matrix), axis=1).tolist()
            """,
            """
            import numpy as np

            def row_cumulative(matrix):
                return np.cumprod(np.asarray(matrix), axis=1).tolist()
            """,
            """
            import solution
            assert solution.row_cumulative([[1, 2, 3], [2, 2, 2]]) == [[1, 2, 6], [2, 4, 8]]
            """,
            "Cumulative product along each row.",
        ),
    ),
    siblings=(
        _sib(
            "running_total",
            """
            import numpy as np

            def running_total(values):
                return np.cumsum(values).tolist()
            """,
            """
            import solution
            assert solution.running_total([1, 2, 3]) == [1, 3, 6]
            assert solution.running_total([]) == []
            """,
            "Running sum via np.cumsum (unchanged).",
        ),
        _sib(
            "nancumprod_chain",
            """
            import numpy as np

            def cumulative_ignoring_missing(values):
                return np.nancumprod(np.asarray(values, dtype=float)).tolist()
            """,
            """
            import solution
            assert solution.cumulative_ignoring_missing([2, float("nan"), 3]) == [2.0, 2.0, 6.0]
            """,
            "Cumulative product skipping NaN (np.nancumprod, unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.alltrue -> numpy.all
# ---------------------------------------------------------------------------

_NP_ALLTRUE = ApiChange(
    package="numpy",
    api_family="numpy.alltrue",
    old_api="np.alltrue",
    new_api="np.all",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_MIGRATION_URL,
    summary="np.alltrue alias removed in NumPy 2.0; use np.all.",
    programs=(
        _prog(
            "is_sorted",
            """
            import numpy as np

            def is_sorted(values):
                arr = np.asarray(values)
                return bool(np.alltrue(np.diff(arr) >= 0))
            """,
            """
            import numpy as np

            def is_sorted(values):
                arr = np.asarray(values)
                return bool(np.all(np.diff(arr) >= 0))
            """,
            """
            import solution
            assert solution.is_sorted([1, 2, 2, 3]) is True
            assert solution.is_sorted([3, 1]) is False
            assert solution.is_sorted([5]) is True
            """,
            "Check that a sequence is non-decreasing.",
        ),
        _prog(
            "within_tolerance",
            """
            import numpy as np

            def all_within_tolerance(actual, expected, tol):
                a = np.asarray(actual, dtype=float)
                e = np.asarray(expected, dtype=float)
                return bool(np.alltrue(np.abs(a - e) <= tol))
            """,
            """
            import numpy as np

            def all_within_tolerance(actual, expected, tol):
                a = np.asarray(actual, dtype=float)
                e = np.asarray(expected, dtype=float)
                return bool(np.all(np.abs(a - e) <= tol))
            """,
            """
            import solution
            assert solution.all_within_tolerance([1.0, 2.05], [1.0, 2.0], 0.1) is True
            assert solution.all_within_tolerance([1.0, 2.5], [1.0, 2.0], 0.1) is False
            """,
            "Elementwise tolerance check between two vectors.",
        ),
        _prog(
            "range_validator",
            """
            import numpy as np

            class RangeValidator:
                def __init__(self, low, high):
                    self.low = low
                    self.high = high

                def in_range(self, values):
                    arr = np.asarray(values)
                    return bool(np.alltrue((arr >= self.low) & (arr <= self.high)))

                def report(self, values):
                    return "ok" if self.in_range(values) else "out of range"
            """,
            """
            import numpy as np

            class RangeValidator:
                def __init__(self, low, high):
                    self.low = low
                    self.high = high

                def in_range(self, values):
                    arr = np.asarray(values)
                    return bool(np.all((arr >= self.low) & (arr <= self.high)))

                def report(self, values):
                    return "ok" if self.in_range(values) else "out of range"
            """,
            """
            import solution
            v = solution.RangeValidator(0, 10)
            assert v.in_range([0, 5, 10]) is True
            assert v.in_range([0, 11]) is False
            assert v.report([1, 2]) == "ok"
            assert v.report([-1]) == "out of range"
            """,
            "Validate that all readings fall inside a closed range.",
        ),
        _prog(
            "rows_all_nonzero",
            """
            import numpy as np

            def rows_all_nonzero(matrix):
                return np.alltrue(np.asarray(matrix) != 0, axis=1).tolist()
            """,
            """
            import numpy as np

            def rows_all_nonzero(matrix):
                return np.all(np.asarray(matrix) != 0, axis=1).tolist()
            """,
            """
            import solution
            assert solution.rows_all_nonzero([[1, 2], [0, 3], [4, 5]]) == [True, False, True]
            assert solution.rows_all_nonzero([[0, 0]]) == [False]
            """,
            "Flag rows of a matrix with no zero entries.",
        ),
        _prog(
            "is_identity",
            """
            import numpy as np

            def is_identity(matrix):
                m = np.asarray(matrix)
                if m.ndim != 2 or m.shape[0] != m.shape[1]:
                    return False
                return bool(np.alltrue(m == np.eye(m.shape[0])))
            """,
            """
            import numpy as np

            def is_identity(matrix):
                m = np.asarray(matrix)
                if m.ndim != 2 or m.shape[0] != m.shape[1]:
                    return False
                return bool(np.all(m == np.eye(m.shape[0])))
            """,
            """
            import solution
            assert solution.is_identity([[1, 0], [0, 1]]) is True
            assert solution.is_identity([[1, 1], [0, 1]]) is False
            assert solution.is_identity([[1, 0, 0]]) is False
            """,
            "Check whether a matrix is the identity.",
        ),
        _prog(
            "flags_required",
            """
            import numpy as np

            def all_flags_set(flags, required):
                arr = np.asarray(flags)
                return bool(np.alltrue((arr & required) == required))
            """,
            """
            import numpy as np

            def all_flags_set(flags, required):
                arr = np.asarray(flags)
                return bool(np.all((arr & required) == required))
            """,
            """
            import solution
            assert solution.all_flags_set([0b011, 0b111], 0b011) is True
            assert solution.all_flags_set([0b001, 0b011], 0b011) is False
            """,
            "Check a required bitmask is set on every flag word.",
        ),
    ),
    siblings=(
        _sib(
            "allclose_match",
            """
            import numpy as np

            def arrays_match(a, b):
                return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
            """,
            """
            import solution
            assert solution.arrays_match([1.0, 2.0], [1.0, 2.0 + 1e-12]) is True
            assert solution.arrays_match([1.0, 2.0], [1.0, 3.0]) is False
            """,
            "Approximate equality via np.allclose (unchanged).",
        ),
        _sib(
            "all_method_finite",
            """
            import numpy as np

            def no_missing(values):
                arr = np.asarray(values, dtype=float)
                return bool(np.isfinite(arr).all())
            """,
            """
            import solution
            assert solution.no_missing([1, 2, 3]) is True
            assert solution.no_missing([1, float("nan")]) is False
            """,
            "ndarray.all() method (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.sometrue -> numpy.any
# ---------------------------------------------------------------------------

_NP_SOMETRUE = ApiChange(
    package="numpy",
    api_family="numpy.sometrue",
    old_api="np.sometrue",
    new_api="np.any",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_MIGRATION_URL,
    summary="np.sometrue alias removed in NumPy 2.0; use np.any.",
    programs=(
        _prog(
            "has_outlier",
            """
            import numpy as np

            def has_outlier(values, threshold):
                arr = np.asarray(values, dtype=float)
                z = (arr - arr.mean()) / arr.std()
                return bool(np.sometrue(np.abs(z) > threshold))
            """,
            """
            import numpy as np

            def has_outlier(values, threshold):
                arr = np.asarray(values, dtype=float)
                z = (arr - arr.mean()) / arr.std()
                return bool(np.any(np.abs(z) > threshold))
            """,
            """
            import solution
            assert solution.has_outlier([1, 1, 1, 1, 10], 1.5) is True
            assert solution.has_outlier([1, 2, 3], 2.0) is False
            """,
            "Detect a z-score outlier in a sample.",
        ),
        _prog(
            "columns_with_missing",
            """
            import numpy as np

            def columns_with_missing(matrix):
                m = np.asarray(matrix, dtype=float)
                return np.sometrue(np.isnan(m), axis=0).tolist()
            """,
            """
            import numpy as np

            def columns_with_missing(matrix):
                m = np.asarray(matrix, dtype=float)
                return np.any(np.isnan(m), axis=0).tolist()
            """,
            """
            import solution
            nan = float("nan")
            assert solution.columns_with_missing([[1, nan], [2, 3]]) == [False, True]
            assert solution.columns_with_missing([[1, 2]]) == [False, False]
            """,
            "Flag columns containing at least one NaN.",
        ),
        _prog(
            "negative_scan",
            """
            import numpy as np

            def contains_negative(values):
                return bool(np.sometrue(np.asarray(values) < 0))

            def first_negative_index(values):
                arr = np.asarray(values)
                if not np.sometrue(arr < 0):
                    return -1
                return int(np.argmax(arr < 0))
            """,
            """
            import numpy as np

            def contains_negative(values):
                return bool(np.any(np.asarray(values) < 0))

            def first_negative_index(values):
                arr = np.asarray(values)
                if not np.any(arr < 0):
                    return -1
                return int(np.argmax(arr < 0))
            """,
            """
            import solution
            assert solution.contains_negative([1, -2, 3]) is True
            assert solution.contains_negative([0, 1]) is False
            assert solution.first_negative_index([4, 5, -1, -2]) == 2
            assert solution.first_negative_index([4, 5]) == -1
            """,
            "Find negative entries in a sequence.",
        ),
        _prog(
            "sensor_log",
            """
            import numpy as np

            class SensorLog:
                def __init__(self):
                    self.readings = []

                def record(self, value):
                    self.readings.append(value)

                def any_above(self, limit):
                    return bool(np.sometrue(np.asarray(self.readings) > limit))
            """,
            """
            import numpy as np

            class SensorLog:
                def __init__(self):
                    self.readings = []

                def record(self, value):
                    self.readings.append(value)

                def any_above(self, limit):
                    return bool(np.any(np.asarray(self.readings) > limit))
            """,
            """
            import solution
            log = solution.SensorLog()
            for v in [20.5, 21.0, 25.5]:
                log.record(v)
            assert log.any_above(25) is True
            assert log.any_above(30) is False
            """,
            "Sensor log that checks whether any reading exceeded a limit.",
        ),
        _prog(
            "interval_overlap",
            """
            import numpy as np

            def overlaps_any(start, end, intervals):
                arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
                return bool(np.sometrue((arr[:, 0] < end) & (arr[:, 1] > start)))
            """,
            """
            import numpy as np

            def overlaps_any(start, end, intervals):
                arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
                return bool(np.any((arr[:, 0] < end) & (arr[:, 1] > start)))
            """,
            """
            import solution
            assert solution.overlaps_any(2, 4, [[0, 1], [3, 5]]) is True
            assert solution.overlaps_any(2, 4, [[0, 2], [4, 6]]) is False
            assert solution.overlaps_any(2, 4, []) is False
            """,
            "Check whether an interval overlaps any in a list.",
        ),
        _prog(
            "bit_probe",
            """
            import numpy as np

            def any_bit_set(values, bit):
                return bool(np.sometrue(np.asarray(values) & (1 << bit)))
            """,
            """
            import numpy as np

            def any_bit_set(values, bit):
                return bool(np.any(np.asarray(values) & (1 << bit)))
            """,
            """
            import solution
            assert solution.any_bit_set([1, 2, 4], 1) is True
            assert solution.any_bit_set([1, 4], 1) is False
            """,
            "Check whether a given bit is set in any word.",
        ),
    ),
    siblings=(
        _sib(
            "any_method_threshold",
            """
            import numpy as np

            def exceeds(values, limit):
                return bool((np.asarray(values) > limit).any())
            """,
            """
            import solution
            assert solution.exceeds([1, 5], 4) is True
            assert solution.exceeds([1, 2], 4) is False
            """,
            "ndarray.any() method (unchanged).",
        ),
        _sib(
            "count_nonzero_hits",
            """
            import numpy as np

            def hit_count(values, target):
                return int(np.count_nonzero(np.asarray(values) == target))
            """,
            """
            import solution
            assert solution.hit_count([1, 2, 1, 1], 1) == 3
            assert solution.hit_count([1, 2], 9) == 0
            """,
            "np.count_nonzero (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.NaN / numpy.Inf / numpy.infty / numpy.PINF / numpy.NINF -> np.nan / np.inf
# ---------------------------------------------------------------------------

_NP_NAN_ALIASES = ApiChange(
    package="numpy",
    api_family="numpy.NaN",
    old_api="np.NaN / np.Inf / np.infty / np.PINF / np.NINF",
    new_api="np.nan / np.inf / -np.inf",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="Capitalised constant aliases np.NaN, np.Inf, np.infty, np.PINF, np.NINF removed in NumPy 2.0; use np.nan, np.inf, -np.inf.",
    programs=(
        _prog(
            "mark_missing",
            """
            import numpy as np

            def mark_missing(values, sentinel):
                arr = np.asarray(values, dtype=float)
                arr[arr == sentinel] = np.NaN
                return arr

            def count_missing(values, sentinel):
                return int(np.isnan(mark_missing(values, sentinel)).sum())
            """,
            """
            import numpy as np

            def mark_missing(values, sentinel):
                arr = np.asarray(values, dtype=float)
                arr[arr == sentinel] = np.nan
                return arr

            def count_missing(values, sentinel):
                return int(np.isnan(mark_missing(values, sentinel)).sum())
            """,
            """
            import solution
            import math
            out = solution.mark_missing([1, -999, 3], -999)
            assert out[0] == 1.0 and out[2] == 3.0
            assert math.isnan(out[1])
            assert solution.count_missing([-999, -999, 2], -999) == 2
            """,
            "Replace a sentinel value with NaN.",
        ),
        _prog(
            "clip_upper_default",
            """
            import numpy as np

            def clip_upper(values, limit=np.Inf):
                return np.minimum(np.asarray(values, dtype=float), limit).tolist()
            """,
            """
            import numpy as np

            def clip_upper(values, limit=np.inf):
                return np.minimum(np.asarray(values, dtype=float), limit).tolist()
            """,
            """
            import solution
            assert solution.clip_upper([1, 5, 9], 4) == [1.0, 4.0, 4.0]
            assert solution.clip_upper([1, 5]) == [1.0, 5.0]
            """,
            "Clip values from above with an infinite default limit.",
        ),
        _prog(
            "safe_ratio",
            """
            import numpy as np

            def safe_ratio(numerators, denominators):
                num = np.asarray(numerators, dtype=float)
                den = np.asarray(denominators, dtype=float)
                out = np.full(num.shape, np.infty)
                nonzero = den != 0
                out[nonzero] = num[nonzero] / den[nonzero]
                return out
            """,
            """
            import numpy as np

            def safe_ratio(numerators, denominators):
                num = np.asarray(numerators, dtype=float)
                den = np.asarray(denominators, dtype=float)
                out = np.full(num.shape, np.inf)
                nonzero = den != 0
                out[nonzero] = num[nonzero] / den[nonzero]
                return out
            """,
            """
            import solution
            import math
            r = solution.safe_ratio([1, 2, 3], [2, 0, 3])
            assert r[0] == 0.5
            assert math.isinf(r[1]) and r[1] > 0
            assert r[2] == 1.0
            """,
            "Elementwise division that maps division by zero to +inf.",
        ),
        _prog(
            "running_extremes",
            """
            import numpy as np

            def running_extremes(values):
                lo, hi = np.PINF, np.NINF
                result = []
                for v in values:
                    lo = min(lo, v)
                    hi = max(hi, v)
                    result.append((lo, hi))
                return result
            """,
            """
            import numpy as np

            def running_extremes(values):
                lo, hi = np.inf, -np.inf
                result = []
                for v in values:
                    lo = min(lo, v)
                    hi = max(hi, v)
                    result.append((lo, hi))
                return result
            """,
            """
            import solution
            assert solution.running_extremes([3, 1, 4]) == [(3, 3), (1, 3), (1, 4)]
            assert solution.running_extremes([]) == []
            """,
            "Running minimum and maximum of a stream.",
        ),
        _prog(
            "interval_bounds",
            """
            import numpy as np

            class Interval:
                def __init__(self, low=-np.Inf, high=np.Inf):
                    self.low = low
                    self.high = high

                def contains(self, x):
                    return self.low <= x <= self.high

                def is_bounded(self):
                    return bool(np.isfinite(self.low) and np.isfinite(self.high))
            """,
            """
            import numpy as np

            class Interval:
                def __init__(self, low=-np.inf, high=np.inf):
                    self.low = low
                    self.high = high

                def contains(self, x):
                    return self.low <= x <= self.high

                def is_bounded(self):
                    return bool(np.isfinite(self.low) and np.isfinite(self.high))
            """,
            """
            import solution
            assert solution.Interval().contains(1e12) is True
            assert solution.Interval(0, 1).contains(2) is False
            assert solution.Interval().is_bounded() is False
            assert solution.Interval(0, 1).is_bounded() is True
            """,
            "Interval with unbounded defaults.",
        ),
        _prog(
            "sentinel_mean",
            """
            import numpy as np

            def mean_ignoring_sentinel(values, sentinel):
                arr = np.asarray(values, dtype=float)
                arr = np.where(arr == sentinel, np.NaN, arr)
                return float(np.nanmean(arr))

            def max_or_default(values):
                arr = np.asarray(values, dtype=float)
                return float(np.max(arr)) if arr.size else np.NINF
            """,
            """
            import numpy as np

            def mean_ignoring_sentinel(values, sentinel):
                arr = np.asarray(values, dtype=float)
                arr = np.where(arr == sentinel, np.nan, arr)
                return float(np.nanmean(arr))

            def max_or_default(values):
                arr = np.asarray(values, dtype=float)
                return float(np.max(arr)) if arr.size else -np.inf
            """,
            """
            import solution
            import math
            assert solution.mean_ignoring_sentinel([1, -1, 3], -1) == 2.0
            assert solution.max_or_default([1, 7, 3]) == 7.0
            m = solution.max_or_default([])
            assert math.isinf(m) and m < 0
            """,
            "Mean that ignores a sentinel, and a max with -inf default.",
        ),
    ),
    siblings=(
        _sib(
            "nan_to_num_clean",
            """
            import numpy as np

            def clean(values):
                arr = np.asarray(values, dtype=float)
                return np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9).tolist()
            """,
            """
            import solution
            assert solution.clean([1.0, float("nan"), float("inf"), -float("inf")]) == [1.0, 0.0, 1e9, -1e9]
            """,
            "np.nan_to_num (unchanged).",
        ),
        _sib(
            "finite_counter",
            """
            import numpy as np

            def count_finite(values):
                arr = np.asarray(values, dtype=float)
                return int(np.isfinite(arr).sum())

            def count_neg_inf(values):
                return int(np.isneginf(np.asarray(values, dtype=float)).sum())
            """,
            """
            import solution
            assert solution.count_finite([1.0, float("nan"), float("inf"), 2.0]) == 2
            assert solution.count_neg_inf([-float("inf"), 1.0, -float("inf")]) == 2
            """,
            "np.isfinite / np.isneginf (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.float_ / complex_ / unicode_ / string_ -> float64 / complex128 / str_ / bytes_
# ---------------------------------------------------------------------------

_NP_FLOAT_ = ApiChange(
    package="numpy",
    api_family="numpy.float_",
    old_api="np.float_ / np.complex_ / np.unicode_ / np.string_",
    new_api="np.float64 / np.complex128 / np.str_ / np.bytes_",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="Scalar type aliases np.float_, np.complex_, np.unicode_, np.string_ removed in NumPy 2.0.",
    programs=(
        _prog(
            "float_array",
            """
            import numpy as np

            def as_float_array(values):
                return np.array(values, dtype=np.float_)

            def halves(values):
                return (as_float_array(values) / 2).tolist()
            """,
            """
            import numpy as np

            def as_float_array(values):
                return np.array(values, dtype=np.float64)

            def halves(values):
                return (as_float_array(values) / 2).tolist()
            """,
            """
            import solution
            import numpy as np
            arr = solution.as_float_array([1, 2])
            assert arr.dtype == np.float64
            assert arr.tolist() == [1.0, 2.0]
            assert solution.halves([1, 3]) == [0.5, 1.5]
            """,
            "Build a float64 array from a list.",
        ),
        _prog(
            "complex_magnitudes",
            """
            import numpy as np

            def to_complex(values):
                return np.asarray(values, dtype=np.complex_)

            def magnitudes(values):
                return np.abs(to_complex(values)).tolist()
            """,
            """
            import numpy as np

            def to_complex(values):
                return np.asarray(values, dtype=np.complex128)

            def magnitudes(values):
                return np.abs(to_complex(values)).tolist()
            """,
            """
            import solution
            import numpy as np
            assert solution.to_complex([1, 2j]).dtype == np.complex128
            assert solution.magnitudes([3 + 4j, 1]) == [5.0, 1.0]
            """,
            "Complex array conversion and magnitudes.",
        ),
        _prog(
            "name_table",
            """
            import numpy as np

            def names_table(names):
                return np.array(names, dtype=np.unicode_)

            def longest_name(names):
                arr = names_table(names)
                lengths = np.char.str_len(arr)
                return str(arr[int(np.argmax(lengths))])
            """,
            """
            import numpy as np

            def names_table(names):
                return np.array(names, dtype=np.str_)

            def longest_name(names):
                arr = names_table(names)
                lengths = np.char.str_len(arr)
                return str(arr[int(np.argmax(lengths))])
            """,
            """
            import solution
            assert solution.names_table(["ann", "bo"]).dtype.kind == "U"
            assert solution.longest_name(["ann", "robert", "li"]) == "robert"
            """,
            "Unicode name table and longest-name lookup.",
        ),
        _prog(
            "byte_tags",
            """
            import numpy as np

            def encode_tags(tags):
                return np.array([t.encode("ascii") for t in tags], dtype=np.string_)

            def tag_lengths(tags):
                return [len(t) for t in encode_tags(tags)]
            """,
            """
            import numpy as np

            def encode_tags(tags):
                return np.array([t.encode("ascii") for t in tags], dtype=np.bytes_)

            def tag_lengths(tags):
                return [len(t) for t in encode_tags(tags)]
            """,
            """
            import solution
            arr = solution.encode_tags(["a", "bcd"])
            assert arr.dtype.kind == "S"
            assert arr[0] == b"a"
            assert solution.tag_lengths(["a", "bcd"]) == [1, 3]
            """,
            "Byte-string tag array.",
        ),
        _prog(
            "describe_scalar",
            """
            import numpy as np

            def describe_scalar(x):
                if isinstance(x, np.float_):
                    return "float64"
                if isinstance(x, np.complex_):
                    return "complex128"
                return "other"
            """,
            """
            import numpy as np

            def describe_scalar(x):
                if isinstance(x, np.float64):
                    return "float64"
                if isinstance(x, np.complex128):
                    return "complex128"
                return "other"
            """,
            """
            import solution
            import numpy as np
            assert solution.describe_scalar(np.float64(1.0)) == "float64"
            assert solution.describe_scalar(np.complex128(1j)) == "complex128"
            assert solution.describe_scalar(3) == "other"
            assert solution.describe_scalar(np.int64(3)) == "other"
            """,
            "Classify a NumPy scalar by type.",
        ),
        _prog(
            "score_records",
            """
            import numpy as np

            RECORD = np.dtype([("name", np.unicode_, 16), ("score", np.float_)])

            def make_records(pairs):
                return np.array(pairs, dtype=RECORD)

            def best(pairs):
                rec = make_records(pairs)
                return str(rec["name"][int(np.argmax(rec["score"]))])
            """,
            """
            import numpy as np

            RECORD = np.dtype([("name", np.str_, 16), ("score", np.float64)])

            def make_records(pairs):
                return np.array(pairs, dtype=RECORD)

            def best(pairs):
                rec = make_records(pairs)
                return str(rec["name"][int(np.argmax(rec["score"]))])
            """,
            """
            import solution
            import numpy as np
            rec = solution.make_records([("a", 1.0), ("b", 2.5)])
            assert rec["score"].dtype == np.float64
            assert rec["score"].tolist() == [1.0, 2.5]
            assert solution.best([("a", 1.0), ("b", 2.5), ("c", 0.1)]) == "b"
            """,
            "Structured record array of names and scores.",
        ),
        _prog(
            "mean_scalar",
            """
            import numpy as np

            def mean_as_float(values):
                return np.float_(np.mean(values))
            """,
            """
            import numpy as np

            def mean_as_float(values):
                return np.float64(np.mean(values))
            """,
            """
            import solution
            import numpy as np
            m = solution.mean_as_float([1, 2, 3])
            assert type(m) is np.float64
            assert m == 2.0
            """,
            "Mean returned as a NumPy float64 scalar.",
        ),
    ),
    siblings=(
        _sib(
            "int_array",
            """
            import numpy as np

            def as_int_array(values):
                return np.array(values, dtype=np.int_)

            def doubled(values):
                return (as_int_array(values) * 2).tolist()
            """,
            """
            import solution
            assert solution.as_int_array([1, 2]).dtype.kind == "i"
            assert solution.doubled([1, 2]) == [2, 4]
            """,
            "np.int_ dtype (unchanged).",
        ),
        _sib(
            "str_scalar_join",
            """
            import numpy as np

            def label(parts):
                return np.str_("-").join(np.asarray(parts, dtype=np.str_))

            def is_text(x):
                return isinstance(x, np.str_)
            """,
            """
            import solution
            import numpy as np
            assert solution.label(["a", "b", "c"]) == "a-b-c"
            assert solution.is_text(np.str_("x")) is True
            assert solution.is_text(3) is False
            """,
            "np.str_ scalar (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.round_ -> numpy.round
# ---------------------------------------------------------------------------

_NP_ROUND_ = ApiChange(
    package="numpy",
    api_family="numpy.round_",
    old_api="np.round_",
    new_api="np.round",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="np.round_ removed in NumPy 2.0; use np.round.",
    programs=(
        _prog(
            "round_prices",
            """
            import numpy as np

            def round_prices(prices):
                return np.round_(np.asarray(prices, dtype=float), 2).tolist()
            """,
            """
            import numpy as np

            def round_prices(prices):
                return np.round(np.asarray(prices, dtype=float), 2).tolist()
            """,
            """
            import solution
            assert solution.round_prices([1.234, 5.678]) == [1.23, 5.68]
            assert solution.round_prices([2.0]) == [2.0]
            """,
            "Round prices to cents.",
        ),
        _prog(
            "round_to_tens",
            """
            import numpy as np

            def round_to_tens(values):
                return np.round_(np.asarray(values), -1).tolist()
            """,
            """
            import numpy as np

            def round_to_tens(values):
                return np.round(np.asarray(values), -1).tolist()
            """,
            """
            import solution
            assert solution.round_to_tens([13, 27, 48]) == [10, 30, 50]
            assert solution.round_to_tens([101]) == [100]
            """,
            "Round integers to the nearest ten.",
        ),
        _prog(
            "stats_reporter",
            """
            import numpy as np

            class Reporter:
                def __init__(self, decimals):
                    self.decimals = decimals

                def summarize(self, values):
                    arr = np.asarray(values, dtype=float)
                    return {
                        "mean": float(np.round_(arr.mean(), self.decimals)),
                        "std": float(np.round_(arr.std(), self.decimals)),
                    }
            """,
            """
            import numpy as np

            class Reporter:
                def __init__(self, decimals):
                    self.decimals = decimals

                def summarize(self, values):
                    arr = np.asarray(values, dtype=float)
                    return {
                        "mean": float(np.round(arr.mean(), self.decimals)),
                        "std": float(np.round(arr.std(), self.decimals)),
                    }
            """,
            """
            import solution
            assert solution.Reporter(2).summarize([1, 2, 3, 4]) == {"mean": 2.5, "std": 1.12}
            assert solution.Reporter(0).summarize([1, 2, 3, 4]) == {"mean": 2.0, "std": 1.0}
            """,
            "Rounded summary statistics.",
        ),
        _prog(
            "percentages",
            """
            import numpy as np

            def percentages(counts):
                arr = np.asarray(counts, dtype=float)
                return np.round_(100.0 * arr / arr.sum(), 1).tolist()
            """,
            """
            import numpy as np

            def percentages(counts):
                arr = np.asarray(counts, dtype=float)
                return np.round(100.0 * arr / arr.sum(), 1).tolist()
            """,
            """
            import solution
            assert solution.percentages([1, 2, 1]) == [25.0, 50.0, 25.0]
            assert solution.percentages([1, 2]) == [33.3, 66.7]
            """,
            "Percent shares rounded to one decimal.",
        ),
        _prog(
            "nearest_integers",
            """
            import numpy as np

            def nearest_integers(values):
                return np.round_(np.asarray(values, dtype=float)).astype(int).tolist()
            """,
            """
            import numpy as np

            def nearest_integers(values):
                return np.round(np.asarray(values, dtype=float)).astype(int).tolist()
            """,
            """
            import solution
            assert solution.nearest_integers([1.2, 2.7, -0.6]) == [1, 3, -1]
            assert solution.nearest_integers([]) == []
            """,
            "Round floats to the nearest integer.",
        ),
        _prog(
            "round_matrix",
            """
            import numpy as np

            def round_matrix(matrix, decimals):
                return np.round_(np.asarray(matrix, dtype=float), decimals=decimals).tolist()
            """,
            """
            import numpy as np

            def round_matrix(matrix, decimals):
                return np.round(np.asarray(matrix, dtype=float), decimals=decimals).tolist()
            """,
            """
            import solution
            assert solution.round_matrix([[1.2345, 2.3456], [0.1111, 9.8765]], 2) == [[1.23, 2.35], [0.11, 9.88]]
            assert solution.round_matrix([[1.26]], 1) == [[1.3]]
            """,
            "Round every entry of a matrix.",
        ),
    ),
    siblings=(
        _sib(
            "around_values",
            """
            import numpy as np

            def round_half_even(values, decimals):
                return np.around(np.asarray(values, dtype=float), decimals).tolist()
            """,
            """
            import solution
            assert solution.round_half_even([1.234, 5.678], 2) == [1.23, 5.68]
            assert solution.round_half_even([0.5, 1.5, 2.5], 0) == [0.0, 2.0, 2.0]
            """,
            "np.around (unchanged).",
        ),
        _sib(
            "rint_snap",
            """
            import numpy as np

            def snap(values):
                return np.rint(np.asarray(values, dtype=float)).tolist()
            """,
            """
            import solution
            assert solution.snap([1.2, 2.7, -0.6]) == [1.0, 3.0, -1.0]
            """,
            "np.rint (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.asfarray -> numpy.asarray(x, dtype=float)
# ---------------------------------------------------------------------------

_NP_ASFARRAY = ApiChange(
    package="numpy",
    api_family="numpy.asfarray",
    old_api="np.asfarray",
    new_api="np.asarray(x, dtype=float)",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="np.asfarray removed in NumPy 2.0; use np.asarray with a float dtype.",
    programs=(
        _prog(
            "normalize_weights",
            """
            import numpy as np

            def normalize(values):
                arr = np.asfarray(values)
                arr /= arr.sum()
                return arr.tolist()
            """,
            """
            import numpy as np

            def normalize(values):
                arr = np.asarray(values, dtype=float)
                arr /= arr.sum()
                return arr.tolist()
            """,
            """
            import solution
            assert solution.normalize([1, 1, 2]) == [0.25, 0.25, 0.5]
            assert solution.normalize([5]) == [1.0]
            """,
            "Normalize integer weights to sum to one (in place).",
        ),
        _prog(
            "centered",
            """
            import numpy as np

            def centered(values):
                arr = np.asfarray(values)
                arr -= arr.mean()
                return arr
            """,
            """
            import numpy as np

            def centered(values):
                arr = np.asarray(values, dtype=float)
                arr -= arr.mean()
                return arr
            """,
            """
            import solution
            import numpy as np
            out = solution.centered([1, 2, 3])
            assert out.dtype == np.float64
            assert out.tolist() == [-1.0, 0.0, 1.0]
            """,
            "Subtract the mean from integer samples.",
        ),
        _prog(
            "float_matrix",
            """
            import numpy as np

            def as_float_matrix(rows):
                m = np.asfarray(rows)
                if m.ndim != 2:
                    raise ValueError("expected a 2-D table")
                return m
            """,
            """
            import numpy as np

            def as_float_matrix(rows):
                m = np.asarray(rows, dtype=float)
                if m.ndim != 2:
                    raise ValueError("expected a 2-D table")
                return m
            """,
            """
            import solution
            import numpy as np
            m = solution.as_float_matrix([[1, 2], [3, 4]])
            assert m.dtype == np.float64
            assert m.shape == (2, 2)
            assert m.tolist() == [[1.0, 2.0], [3.0, 4.0]]
            try:
                solution.as_float_matrix([1, 2])
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """,
            "Coerce nested lists to a float matrix.",
        ),
        _prog(
            "series_scale",
            """
            import numpy as np

            class Series:
                def __init__(self, data):
                    self.data = np.asfarray(data)

                def scale(self, factor):
                    self.data *= factor
                    return self

                def values(self):
                    return self.data.tolist()
            """,
            """
            import numpy as np

            class Series:
                def __init__(self, data):
                    self.data = np.asarray(data, dtype=float)

                def scale(self, factor):
                    self.data *= factor
                    return self

                def values(self):
                    return self.data.tolist()
            """,
            """
            import solution
            assert solution.Series([1, 2]).scale(0.5).values() == [0.5, 1.0]
            assert solution.Series([3]).values() == [3.0]
            """,
            "Series wrapper with in-place scaling.",
        ),
        _prog(
            "weighted_average",
            """
            import numpy as np

            def weighted_average(values, weights):
                v = np.asfarray(values)
                w = np.asfarray(weights)
                return float((v * w).sum() / w.sum())
            """,
            """
            import numpy as np

            def weighted_average(values, weights):
                v = np.asarray(values, dtype=float)
                w = np.asarray(weights, dtype=float)
                return float((v * w).sum() / w.sum())
            """,
            """
            import solution
            assert solution.weighted_average([1, 3], [1, 1]) == 2.0
            assert solution.weighted_average([10, 20], [3, 1]) == 12.5
            """,
            "Weighted average of integer samples.",
        ),
        _prog(
            "midpoints",
            """
            import numpy as np

            def midpoints(values):
                a = np.asfarray(values)
                return (a[1:] + a[:-1]) / 2.0

            def midpoint_dtype(values):
                return midpoints(values).dtype.name
            """,
            """
            import numpy as np

            def midpoints(values):
                a = np.asarray(values, dtype=float)
                return (a[1:] + a[:-1]) / 2.0

            def midpoint_dtype(values):
                return midpoints(values).dtype.name
            """,
            """
            import solution
            assert solution.midpoints([1, 2, 4]).tolist() == [1.5, 3.0]
            assert solution.midpoint_dtype([1, 2]) == "float64"
            """,
            "Midpoints between consecutive samples.",
        ),
    ),
    siblings=(
        _sib(
            "chkfinite_floats",
            """
            import numpy as np

            def strict_floats(values):
                return np.asarray_chkfinite(values, dtype=float)

            def has_only_finite(values):
                try:
                    strict_floats(values)
                except ValueError:
                    return False
                return True
            """,
            """
            import solution
            assert solution.strict_floats([1, 2]).tolist() == [1.0, 2.0]
            assert solution.has_only_finite([1.0, float("inf")]) is False
            assert solution.has_only_finite([1.0, 2.0]) is True
            """,
            "np.asarray_chkfinite (unchanged).",
        ),
        _sib(
            "contiguous_floats",
            """
            import numpy as np

            def contiguous_floats(values):
                arr = np.ascontiguousarray(values, dtype=float)
                return arr.flags["C_CONTIGUOUS"], arr.tolist()
            """,
            """
            import solution
            assert solution.contiguous_floats([[1, 2], [3, 4]]) == (True, [[1.0, 2.0], [3.0, 4.0]])
            """,
            "np.ascontiguousarray with dtype (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.msort -> numpy.sort(x, axis=0)
# ---------------------------------------------------------------------------

_NP_MSORT = ApiChange(
    package="numpy",
    api_family="numpy.msort",
    old_api="np.msort",
    new_api="np.sort(x, axis=0)",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="np.msort removed in NumPy 2.0 (expired deprecation); use np.sort(a, axis=0).",
    programs=(
        _prog(
            "sort_columns",
            """
            import numpy as np

            def sort_columns(matrix):
                return np.msort(np.asarray(matrix)).tolist()
            """,
            """
            import numpy as np

            def sort_columns(matrix):
                return np.sort(np.asarray(matrix), axis=0).tolist()
            """,
            """
            import solution
            assert solution.sort_columns([[3, 1], [1, 2], [2, 0]]) == [[1, 0], [2, 1], [3, 2]]
            assert solution.sort_columns([[5, 4]]) == [[5, 4]]
            """,
            "Sort each column of a matrix independently.",
        ),
        _prog(
            "median",
            """
            import numpy as np

            def median(values):
                s = np.msort(np.asarray(values, dtype=float))
                n = s.size
                mid = n // 2
                if n % 2:
                    return float(s[mid])
                return float((s[mid - 1] + s[mid]) / 2.0)
            """,
            """
            import numpy as np

            def median(values):
                s = np.sort(np.asarray(values, dtype=float), axis=0)
                n = s.size
                mid = n // 2
                if n % 2:
                    return float(s[mid])
                return float((s[mid - 1] + s[mid]) / 2.0)
            """,
            """
            import solution
            assert solution.median([3, 1, 2]) == 2.0
            assert solution.median([4, 1, 3, 2]) == 2.5
            """,
            "Median via a full sort.",
        ),
        _prog(
            "k_smallest",
            """
            import numpy as np

            def k_smallest(values, k):
                return np.msort(np.asarray(values))[:k].tolist()
            """,
            """
            import numpy as np

            def k_smallest(values, k):
                return np.sort(np.asarray(values), axis=0)[:k].tolist()
            """,
            """
            import solution
            assert solution.k_smallest([9, 4, 7, 1], 2) == [1, 4]
            assert solution.k_smallest([2, 1], 5) == [1, 2]
            """,
            "The k smallest values of a sequence.",
        ),
        _prog(
            "leaderboard",
            """
            import numpy as np

            class Leaderboard:
                def __init__(self):
                    self.scores = []

                def add(self, score):
                    self.scores.append(score)

                def ranked(self):
                    return np.msort(np.asarray(self.scores))[::-1].tolist()

                def best(self):
                    return self.ranked()[0]
            """,
            """
            import numpy as np

            class Leaderboard:
                def __init__(self):
                    self.scores = []

                def add(self, score):
                    self.scores.append(score)

                def ranked(self):
                    return np.sort(np.asarray(self.scores), axis=0)[::-1].tolist()

                def best(self):
                    return self.ranked()[0]
            """,
            """
            import solution
            board = solution.Leaderboard()
            for s in [40, 90, 65]:
                board.add(s)
            assert board.ranked() == [90, 65, 40]
            assert board.best() == 90
            """,
            "Leaderboard with descending scores.",
        ),
        _prog(
            "nearest_rank_percentile",
            """
            import numpy as np

            def percentile_nearest_rank(values, p):
                s = np.msort(np.asarray(values, dtype=float))
                rank = int(np.ceil(p / 100.0 * s.size)) - 1
                return float(s[max(rank, 0)])
            """,
            """
            import numpy as np

            def percentile_nearest_rank(values, p):
                s = np.sort(np.asarray(values, dtype=float), axis=0)
                rank = int(np.ceil(p / 100.0 * s.size)) - 1
                return float(s[max(rank, 0)])
            """,
            """
            import solution
            data = [35, 20, 15, 50, 40]
            assert solution.percentile_nearest_rank(data, 40) == 20.0
            assert solution.percentile_nearest_rank(data, 100) == 50.0
            assert solution.percentile_nearest_rank(data, 5) == 15.0
            """,
            "Nearest-rank percentile.",
        ),
        _prog(
            "feature_extremes",
            """
            import numpy as np

            def per_feature_extremes(table):
                s = np.msort(np.asarray(table, dtype=float))
                return {"min": s[0].tolist(), "max": s[-1].tolist()}
            """,
            """
            import numpy as np

            def per_feature_extremes(table):
                s = np.sort(np.asarray(table, dtype=float), axis=0)
                return {"min": s[0].tolist(), "max": s[-1].tolist()}
            """,
            """
            import solution
            out = solution.per_feature_extremes([[1, 9], [5, 2], [3, 7]])
            assert out == {"min": [1.0, 2.0], "max": [5.0, 9.0]}
            """,
            "Per-column minimum and maximum via column sort.",
        ),
    ),
    siblings=(
        _sib(
            "sort_rows",
            """
            import numpy as np

            def sort_rows(matrix):
                return np.sort(np.asarray(matrix), axis=-1).tolist()
            """,
            """
            import solution
            assert solution.sort_rows([[3, 1], [2, 0]]) == [[1, 3], [0, 2]]
            """,
            "np.sort along the last axis (unchanged).",
        ),
        _sib(
            "argsort_ranks",
            """
            import numpy as np

            def rank_order(values):
                return np.argsort(np.asarray(values)).tolist()
            """,
            """
            import solution
            assert solution.rank_order([30, 10, 20]) == [1, 2, 0]
            """,
            "np.argsort (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.array(x, copy=False) -> numpy.asarray(x)
# ---------------------------------------------------------------------------

_NP_COPY_FALSE = ApiChange(
    package="numpy",
    api_family="numpy.array.copy_false",
    old_api="np.array(x, copy=False)",
    new_api="np.asarray(x)",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="In NumPy 2.0 copy=False means never copy and raises ValueError when a copy is needed; use np.asarray.",
    programs=(
        _prog(
            "float_total",
            """
            import numpy as np

            def to_float_array(values):
                return np.array(values, dtype=float, copy=False)

            def total(values):
                return float(to_float_array(values).sum())
            """,
            """
            import numpy as np

            def to_float_array(values):
                return np.asarray(values, dtype=float)

            def total(values):
                return float(to_float_array(values).sum())
            """,
            """
            import solution
            assert solution.to_float_array([1, 2]).tolist() == [1.0, 2.0]
            assert solution.total([1, 2, 3]) == 6.0
            """,
            "Float conversion of a list without an explicit copy.",
        ),
        _prog(
            "as_table",
            """
            import numpy as np

            def as_table(rows):
                table = np.array(rows, copy=False)
                return table.shape, table.tolist()
            """,
            """
            import numpy as np

            def as_table(rows):
                table = np.asarray(rows)
                return table.shape, table.tolist()
            """,
            """
            import solution
            assert solution.as_table([[1, 2], [3, 4]]) == ((2, 2), [[1, 2], [3, 4]])
            assert solution.as_table([7]) == ((1,), [7])
            """,
            "Shape and contents of nested lists as a table.",
        ),
        _prog(
            "matrix_wrapper",
            """
            import numpy as np

            class Matrix:
                def __init__(self, data):
                    self.data = np.array(data, copy=False)

                def transpose(self):
                    return Matrix(self.data.T)

                def tolist(self):
                    return self.data.tolist()
            """,
            """
            import numpy as np

            class Matrix:
                def __init__(self, data):
                    self.data = np.asarray(data)

                def transpose(self):
                    return Matrix(self.data.T)

                def tolist(self):
                    return self.data.tolist()
            """,
            """
            import solution
            m = solution.Matrix([[1, 2], [3, 4]])
            assert m.transpose().tolist() == [[1, 3], [2, 4]]
            assert m.tolist() == [[1, 2], [3, 4]]
            """,
            "Lightweight matrix wrapper built from lists.",
        ),
        _prog(
            "int32_checksum",
            """
            import numpy as np

            def to_int32(values):
                return np.array(values, dtype=np.int32, copy=False)

            def checksum(values):
                return int(to_int32(values).sum())
            """,
            """
            import numpy as np

            def to_int32(values):
                return np.asarray(values, dtype=np.int32)

            def checksum(values):
                return int(to_int32(values).sum())
            """,
            """
            import solution
            import numpy as np
            out = solution.to_int32(np.arange(3, dtype=np.int64))
            assert out.dtype == np.int32
            assert out.tolist() == [0, 1, 2]
            assert solution.checksum([1, 2, 3]) == 6
            """,
            "Down-cast to int32 and checksum.",
        ),
        _prog(
            "pair_sums",
            """
            import numpy as np

            def pair_sums(pairs):
                arr = np.array(pairs, copy=False)
                return arr.sum(axis=1).tolist()
            """,
            """
            import numpy as np

            def pair_sums(pairs):
                arr = np.asarray(pairs)
                return arr.sum(axis=1).tolist()
            """,
            """
            import solution
            assert solution.pair_sums([(1, 2), (3, 4)]) == [3, 7]
            assert solution.pair_sums([(0, 0)]) == [0]
            """,
            "Sum of each pair in a list of tuples.",
        ),
        _prog(
            "flatten_nested",
            """
            import numpy as np

            def flatten(values):
                return np.array(values, copy=False).ravel().tolist()

            def count_items(values):
                return len(flatten(values))
            """,
            """
            import numpy as np

            def flatten(values):
                return np.asarray(values).ravel().tolist()

            def count_items(values):
                return len(flatten(values))
            """,
            """
            import solution
            assert solution.flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
            assert solution.count_items([[1, 2, 3]]) == 3
            """,
            "Flatten nested lists.",
        ),
    ),
    siblings=(
        _sib(
            "snapshot_copy",
            """
            import numpy as np

            def snapshot(values):
                return np.array(values, copy=True)
            """,
            """
            import solution
            import numpy as np
            src = np.array([1, 2, 3])
            snap = solution.snapshot(src)
            src[0] = 99
            assert snap.tolist() == [1, 2, 3]
            assert solution.snapshot([4, 5]).tolist() == [4, 5]
            """,
            "np.array(copy=True) (unchanged).",
        ),
        _sib(
            "require_contiguous",
            """
            import numpy as np

            def contiguous_floats(values):
                arr = np.require(values, dtype=float, requirements="C")
                return arr.dtype.name, arr.tolist()
            """,
            """
            import solution
            assert solution.contiguous_floats([1, 2]) == ("float64", [1.0, 2.0])
            """,
            "np.require (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.find_common_type -> numpy.result_type
# ---------------------------------------------------------------------------

_NP_FIND_COMMON_TYPE = ApiChange(
    package="numpy",
    api_family="numpy.find_common_type",
    old_api="np.find_common_type",
    new_api="np.result_type",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="np.find_common_type removed in NumPy 2.0; use np.result_type or np.promote_types.",
    programs=(
        _prog(
            "common_dtype_name",
            """
            import numpy as np

            def common_dtype(arrays):
                return np.find_common_type([a.dtype for a in arrays], []).name
            """,
            """
            import numpy as np

            def common_dtype(arrays):
                return np.result_type(*[a.dtype for a in arrays]).name
            """,
            """
            import solution
            import numpy as np
            assert solution.common_dtype([np.zeros(2, np.int32), np.zeros(2, np.float32)]) == "float64"
            assert solution.common_dtype([np.zeros(1, np.int8), np.zeros(1, np.int16)]) == "int16"
            assert solution.common_dtype([np.zeros(1, np.uint8), np.zeros(1, np.int8)]) == "int16"
            """,
            "Name of the common dtype of several arrays.",
        ),
        _prog(
            "concat_promoted",
            """
            import numpy as np

            def concat_promoted(arrays):
                dtype = np.find_common_type([a.dtype for a in arrays], [])
                return np.concatenate([a.astype(dtype) for a in arrays])
            """,
            """
            import numpy as np

            def concat_promoted(arrays):
                dtype = np.result_type(*[a.dtype for a in arrays])
                return np.concatenate([a.astype(dtype) for a in arrays])
            """,
            """
            import solution
            import numpy as np
            out = solution.concat_promoted([np.array([1, 2], np.int16), np.array([0.5], np.float32)])
            assert out.dtype == np.float32
            assert out.tolist() == [1.0, 2.0, 0.5]
            out2 = solution.concat_promoted([np.array([1], np.int32), np.array([0.5], np.float32)])
            assert out2.dtype == np.float64
            """,
            "Concatenate arrays after promoting to a common dtype.",
        ),
        _prog(
            "promote_names",
            """
            import numpy as np

            def promote_names(name_a, name_b):
                return np.find_common_type([np.dtype(name_a), np.dtype(name_b)], []).name
            """,
            """
            import numpy as np

            def promote_names(name_a, name_b):
                return np.result_type(np.dtype(name_a), np.dtype(name_b)).name
            """,
            """
            import solution
            assert solution.promote_names("int32", "float32") == "float64"
            assert solution.promote_names("bool", "int8") == "int8"
            assert solution.promote_names("float32", "complex64") == "complex64"
            """,
            "Promote two dtype names.",
        ),
        _prog(
            "chunk_accumulator",
            """
            import numpy as np

            class Accumulator:
                def __init__(self):
                    self.chunks = []

                def add(self, chunk):
                    self.chunks.append(np.asarray(chunk))

                def dtype(self):
                    return np.find_common_type([c.dtype for c in self.chunks], [])

                def combined(self):
                    return np.concatenate(self.chunks).astype(self.dtype())
            """,
            """
            import numpy as np

            class Accumulator:
                def __init__(self):
                    self.chunks = []

                def add(self, chunk):
                    self.chunks.append(np.asarray(chunk))

                def dtype(self):
                    return np.result_type(*[c.dtype for c in self.chunks])

                def combined(self):
                    return np.concatenate(self.chunks).astype(self.dtype())
            """,
            """
            import solution
            import numpy as np
            acc = solution.Accumulator()
            acc.add(np.array([1, 2], dtype=np.int32))
            acc.add(np.array([0.5], dtype=np.float32))
            assert acc.dtype() == np.float64
            assert acc.combined().tolist() == [1.0, 2.0, 0.5]
            small = solution.Accumulator()
            small.add(np.array([1], dtype=np.int8))
            small.add(np.array([2], dtype=np.uint8))
            assert small.dtype() == np.int16
            """,
            "Accumulate chunks and combine them with a common dtype.",
        ),
        _prog(
            "storage_dtype",
            """
            import numpy as np

            def storage_dtype(samples):
                dtypes = [np.min_scalar_type(s) for s in samples]
                return np.find_common_type(dtypes, []).name
            """,
            """
            import numpy as np

            def storage_dtype(samples):
                dtypes = [np.min_scalar_type(s) for s in samples]
                return np.result_type(*dtypes).name
            """,
            """
            import solution
            assert solution.storage_dtype([1, 200]) == "uint8"
            assert solution.storage_dtype([-1, -5]) == "int8"
            assert solution.storage_dtype([200, -1]) == "int16"
            assert solution.storage_dtype([1.5, 1]) == "float16"
            """,
            "Smallest dtype able to hold all samples.",
        ),
        _prog(
            "merge_columns",
            """
            import numpy as np

            def merge_columns(columns):
                arrays = [np.asarray(c) for c in columns]
                dtype = np.find_common_type([a.dtype for a in arrays], [])
                return np.stack(arrays, axis=1).astype(dtype).tolist(), dtype.name
            """,
            """
            import numpy as np

            def merge_columns(columns):
                arrays = [np.asarray(c) for c in columns]
                dtype = np.result_type(*[a.dtype for a in arrays])
                return np.stack(arrays, axis=1).astype(dtype).tolist(), dtype.name
            """,
            """
            import solution
            import numpy as np
            rows, name = solution.merge_columns([np.array([1, 2], np.int8), np.array([0.5, 1.5], np.float32)])
            assert rows == [[1.0, 0.5], [2.0, 1.5]]
            assert name == "float32"
            """,
            "Merge typed columns into rows with a common dtype.",
        ),
    ),
    siblings=(
        _sib(
            "promote_types_pair",
            """
            import numpy as np

            def promoted_name(a, b):
                return np.promote_types(a, b).name
            """,
            """
            import solution
            assert solution.promoted_name("int32", "float32") == "float64"
            assert solution.promoted_name("uint8", "int8") == "int16"
            """,
            "np.promote_types (unchanged).",
        ),
        _sib(
            "can_cast_check",
            """
            import numpy as np

            def fits_without_loss(src, dst):
                return bool(np.can_cast(np.dtype(src), np.dtype(dst), casting="safe"))
            """,
            """
            import solution
            assert solution.fits_without_loss("int8", "int16") is True
            assert solution.fits_without_loss("int16", "int8") is False
            """,
            "np.can_cast (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# numpy.cast[dtype](x) -> x.astype(dtype)
# ---------------------------------------------------------------------------

_NP_CAST = ApiChange(
    package="numpy",
    api_family="numpy.cast",
    old_api="np.cast[dtype](x)",
    new_api="np.asarray(x).astype(dtype)",
    old_pin=NP_OLD,
    new_pin=NP_NEW,
    change_date=NP_DATE,
    change_url=NP_URL,
    summary="np.cast dictionary removed in NumPy 2.0; use np.asarray(x, dtype=dtype) or ndarray.astype.",
    programs=(
        _prog(
            "single_precision",
            """
            import numpy as np

            def to_single(values):
                return np.cast[np.float32](values)

            def single_sum(values):
                return float(to_single(values).sum())
            """,
            """
            import numpy as np

            def to_single(values):
                return np.asarray(values).astype(np.float32)

            def single_sum(values):
                return float(to_single(values).sum())
            """,
            """
            import solution
            import numpy as np
            out = solution.to_single([1, 2.5])
            assert out.dtype == np.float32
            assert out.tolist() == [1.0, 2.5]
            assert solution.single_sum([1, 2, 3]) == 6.0
            """,
            "Convert to float32.",
        ),
        _prog(
            "quantize_uint8",
            """
            import numpy as np

            def quantize(values):
                scaled = np.clip(np.asarray(values, dtype=float) * 255.0, 0, 255)
                return np.cast[np.uint8](np.rint(scaled))
            """,
            """
            import numpy as np

            def quantize(values):
                scaled = np.clip(np.asarray(values, dtype=float) * 255.0, 0, 255)
                return np.rint(scaled).astype(np.uint8)
            """,
            """
            import solution
            import numpy as np
            out = solution.quantize([0.0, 0.5, 1.0, 2.0])
            assert out.dtype == np.uint8
            assert out.tolist() == [0, 128, 255, 255]
            """,
            "Quantize unit-interval values to 8-bit.",
        ),
        _prog(
            "nonzero_mask",
            """
            import numpy as np

            def nonzero_mask(values):
                return np.cast[np.bool_](values)

            def count_active(values):
                return int(nonzero_mask(values).sum())
            """,
            """
            import numpy as np

            def nonzero_mask(values):
                return np.asarray(values).astype(np.bool_)

            def count_active(values):
                return int(nonzero_mask(values).sum())
            """,
            """
            import solution
            assert solution.nonzero_mask([0, 2, 0, 5]).tolist() == [False, True, False, True]
            assert solution.count_active([0, 2, 0, 5]) == 2
            """,
            "Boolean mask of nonzero entries.",
        ),
        _prog(
            "half_precision",
            """
            import numpy as np

            def compress(values):
                return np.cast[np.float16](values)

            def compression_error(values):
                arr = np.asarray(values, dtype=float)
                return float(np.max(np.abs(arr - compress(values).astype(float))))
            """,
            """
            import numpy as np

            def compress(values):
                return np.asarray(values).astype(np.float16)

            def compression_error(values):
                arr = np.asarray(values, dtype=float)
                return float(np.max(np.abs(arr - compress(values).astype(float))))
            """,
            """
            import solution
            import numpy as np
            assert solution.compress([1.0, 0.1]).dtype == np.float16
            assert solution.compression_error([1.0, 2.0]) == 0.0
            err = solution.compression_error([0.1])
            assert 0.0 < err < 1e-3
            """,
            "Lossy float16 compression and its error.",
        ),
        _prog(
            "grid_cells",
            """
            import numpy as np

            def grid_cell(points, cell_size):
                pts = np.asarray(points, dtype=float)
                return np.cast[np.int64](np.floor(pts / cell_size)).tolist()
            """,
            """
            import numpy as np

            def grid_cell(points, cell_size):
                pts = np.asarray(points, dtype=float)
                return np.floor(pts / cell_size).astype(np.int64).tolist()
            """,
            """
            import solution
            assert solution.grid_cell([[1.5, 2.7], [-0.5, 4.0]], 1.0) == [[1, 2], [-1, 4]]
            assert solution.grid_cell([[9.9]], 5.0) == [[1]]
            """,
            "Integer grid cell indices of points.",
        ),
        _prog(
            "histogram_buckets",
            """
            import numpy as np

            class Histogram:
                def __init__(self, width):
                    self.width = width

                def bucket_ids(self, values):
                    return np.cast[np.int32](np.asarray(values) // self.width)

                def counts(self, values):
                    return np.bincount(self.bucket_ids(values)).tolist()
            """,
            """
            import numpy as np

            class Histogram:
                def __init__(self, width):
                    self.width = width

                def bucket_ids(self, values):
                    return (np.asarray(values) // self.width).astype(np.int32)

                def counts(self, values):
                    return np.bincount(self.bucket_ids(values)).tolist()
            """,
            """
            import solution
            import numpy as np
            h = solution.Histogram(10)
            ids = h.bucket_ids([3, 15, 27])
            assert ids.dtype == np.int32
            assert ids.tolist() == [0, 1, 2]
            assert h.counts([3, 15, 27, 18]) == [1, 2, 1]
            """,
            "Fixed-width histogram bucket ids and counts.",
        ),
    ),
    siblings=(
        _sib(
            "can_cast_values",
            """
            import numpy as np

            def safe_to_cast(values, dtype):
                return bool(np.can_cast(np.asarray(values).dtype, dtype, casting="safe"))
            """,
            """
            import solution
            import numpy as np
            assert solution.safe_to_cast([1, 2], np.float64) is True
            assert solution.safe_to_cast([1.5], np.int32) is False
            """,
            "np.can_cast on array dtypes (unchanged).",
        ),
        _sib(
            "clip_to_dtype",
            """
            import numpy as np

            def clip_to_dtype(values, dtype):
                info = np.iinfo(dtype)
                return np.clip(np.asarray(values), info.min, info.max).astype(dtype)
            """,
            """
            import solution
            import numpy as np
            out = solution.clip_to_dtype([-5, 100, 300], np.uint8)
            assert out.dtype == np.uint8
            assert out.tolist() == [0, 100, 255]
            """,
            "np.iinfo plus astype (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.integrate.simps -> scipy.integrate.simpson
# ---------------------------------------------------------------------------

_SP_SIMPS = ApiChange(
    package="scipy",
    api_family="scipy.integrate.simps",
    old_api="scipy.integrate.simps",
    new_api="scipy.integrate.simpson",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_114,
    change_url=SP_URL_114,
    summary="scipy.integrate.simps removed in SciPy 1.14.0; use simpson (x is keyword-only).",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "area_under_samples",
            """
            import numpy as np
            from scipy.integrate import simps

            def area_under_samples(y, x):
                return float(simps(np.asarray(y, dtype=float), x=np.asarray(x, dtype=float)))
            """,
            """
            import numpy as np
            from scipy.integrate import simpson

            def area_under_samples(y, x):
                return float(simpson(np.asarray(y, dtype=float), x=np.asarray(x, dtype=float)))
            """,
            """
            import solution
            x = [0.0, 0.25, 0.5, 0.75, 1.0]
            y = [v * v for v in x]
            assert abs(solution.area_under_samples(y, x) - 1.0 / 3.0) < 1e-9
            assert abs(solution.area_under_samples([1, 3, 5], [0, 1, 2]) - 6.0) < 1e-9
            """,
            "Simpson integration of sampled data.",
        ),
        _prog(
            "work_done",
            """
            from scipy.integrate import simps

            def work_done(forces, positions):
                return float(simps(forces, x=positions))
            """,
            """
            from scipy.integrate import simpson

            def work_done(forces, positions):
                return float(simpson(forces, x=positions))
            """,
            """
            import solution
            assert abs(solution.work_done([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) - 8.0) < 1e-9
            assert abs(solution.work_done([2, 2, 2], [0, 0.5, 1.0]) - 2.0) < 1e-9
            """,
            "Work from force samples along a path.",
        ),
        _prog(
            "radial_profile_mass",
            """
            import numpy as np
            from scipy.integrate import simps

            class RadialProfile:
                def __init__(self, radii, density):
                    self.r = np.asarray(radii, dtype=float)
                    self.rho = np.asarray(density, dtype=float)

                def total_mass(self):
                    integrand = 4.0 * np.pi * self.r ** 2 * self.rho
                    return float(simps(integrand, x=self.r))
            """,
            """
            import numpy as np
            from scipy.integrate import simpson

            class RadialProfile:
                def __init__(self, radii, density):
                    self.r = np.asarray(radii, dtype=float)
                    self.rho = np.asarray(density, dtype=float)

                def total_mass(self):
                    integrand = 4.0 * np.pi * self.r ** 2 * self.rho
                    return float(simpson(integrand, x=self.r))
            """,
            """
            import solution
            import math
            prof = solution.RadialProfile([0, 0.5, 1.0, 1.5, 2.0], [1, 1, 1, 1, 1])
            assert abs(prof.total_mass() - 32.0 * math.pi / 3.0) < 1e-9
            """,
            "Mass of a sphere from a sampled radial density profile.",
        ),
        _prog(
            "mean_value",
            """
            import numpy as np
            from scipy.integrate import simps

            def mean_value(samples, dx):
                samples = np.asarray(samples, dtype=float)
                length = dx * (samples.size - 1)
                return float(simps(samples, dx=dx) / length)
            """,
            """
            import numpy as np
            from scipy.integrate import simpson

            def mean_value(samples, dx):
                samples = np.asarray(samples, dtype=float)
                length = dx * (samples.size - 1)
                return float(simpson(samples, dx=dx) / length)
            """,
            """
            import solution
            assert abs(solution.mean_value([0, 1, 4, 9, 16], 1.0) - 16.0 / 3.0) < 1e-9
            assert abs(solution.mean_value([3, 3, 3], 0.5) - 3.0) < 1e-9
            """,
            "Mean value of a uniformly sampled function.",
        ),
        _prog(
            "rowwise_area",
            """
            import numpy as np
            from scipy.integrate import simps

            def rowwise_area(matrix, dx):
                return simps(np.asarray(matrix, dtype=float), dx=dx, axis=1).tolist()
            """,
            """
            import numpy as np
            from scipy.integrate import simpson

            def rowwise_area(matrix, dx):
                return simpson(np.asarray(matrix, dtype=float), dx=dx, axis=1).tolist()
            """,
            """
            import solution
            out = solution.rowwise_area([[1, 2, 3], [2, 4, 6]], 0.5)
            assert len(out) == 2
            assert abs(out[0] - 2.0) < 1e-9 and abs(out[1] - 4.0) < 1e-9
            """,
            "Simpson integral of every row.",
        ),
        _prog(
            "signal_energy",
            """
            import numpy as np
            from scipy.integrate import simps

            def signal_energy(samples, dt):
                s = np.asarray(samples, dtype=float)
                return float(simps(s ** 2, dx=dt))
            """,
            """
            import numpy as np
            from scipy.integrate import simpson

            def signal_energy(samples, dt):
                s = np.asarray(samples, dtype=float)
                return float(simpson(s ** 2, dx=dt))
            """,
            """
            import solution
            assert abs(solution.signal_energy([0, 1, 2], 1.0) - 8.0 / 3.0) < 1e-9
            assert abs(solution.signal_energy([1, 1, 1, 1, 1], 0.5) - 2.0) < 1e-9
            """,
            "Energy of a sampled signal.",
        ),
    ),
    siblings=(
        _sib(
            "fixed_quad_poly",
            """
            import numpy as np
            from scipy.integrate import fixed_quad

            def poly_integral(coeffs, a, b):
                value, _ = fixed_quad(lambda x: np.polyval(coeffs, x), a, b, n=5)
                return float(value)
            """,
            """
            import solution
            assert abs(solution.poly_integral([1, 0, 0], 0, 1) - 1.0 / 3.0) < 1e-9
            assert abs(solution.poly_integral([2, 1], 0, 3) - 12.0) < 1e-9
            """,
            "scipy.integrate.fixed_quad (unchanged).",
        ),
        _sib(
            "romb_samples",
            """
            import numpy as np
            from scipy.integrate import romb

            def romberg_area(samples, dx):
                return float(romb(np.asarray(samples, dtype=float), dx=dx))
            """,
            """
            import solution
            assert abs(solution.romberg_area([0, 1, 4, 9, 16], 1.0) - 64.0 / 3.0) < 1e-9
            """,
            "scipy.integrate.romb (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.integrate.cumtrapz -> scipy.integrate.cumulative_trapezoid
# ---------------------------------------------------------------------------

_SP_CUMTRAPZ = ApiChange(
    package="scipy",
    api_family="scipy.integrate.cumtrapz",
    old_api="scipy.integrate.cumtrapz",
    new_api="scipy.integrate.cumulative_trapezoid",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_114,
    change_url=SP_URL_114,
    summary="scipy.integrate.cumtrapz removed in SciPy 1.14.0; use cumulative_trapezoid.",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "cumulative_distance",
            """
            import numpy as np
            from scipy.integrate import cumtrapz

            def cumulative_distance(speeds, times):
                v = np.asarray(speeds, dtype=float)
                t = np.asarray(times, dtype=float)
                return cumtrapz(v, t, initial=0).tolist()
            """,
            """
            import numpy as np
            from scipy.integrate import cumulative_trapezoid

            def cumulative_distance(speeds, times):
                v = np.asarray(speeds, dtype=float)
                t = np.asarray(times, dtype=float)
                return cumulative_trapezoid(v, t, initial=0).tolist()
            """,
            """
            import solution
            assert solution.cumulative_distance([0, 1, 2, 3], [0, 1, 2, 3]) == [0.0, 0.5, 2.0, 4.5]
            assert solution.cumulative_distance([2, 2], [0, 5]) == [0.0, 10.0]
            """,
            "Distance travelled over time from speed samples.",
        ),
        _prog(
            "cdf_from_pdf",
            """
            from scipy.integrate import cumtrapz

            def cdf_from_pdf(pdf, x):
                cdf = cumtrapz(pdf, x, initial=0)
                return (cdf / cdf[-1]).tolist()
            """,
            """
            from scipy.integrate import cumulative_trapezoid

            def cdf_from_pdf(pdf, x):
                cdf = cumulative_trapezoid(pdf, x, initial=0)
                return (cdf / cdf[-1]).tolist()
            """,
            """
            import solution
            assert solution.cdf_from_pdf([1, 1, 1], [0, 1, 2]) == [0.0, 0.5, 1.0]
            assert solution.cdf_from_pdf([0, 2, 0], [0, 1, 2]) == [0.0, 0.5, 1.0]
            """,
            "Normalised CDF from sampled density values.",
        ),
        _prog(
            "displacement",
            """
            import numpy as np
            from scipy.integrate import cumtrapz

            def displacement(velocity, dt):
                v = np.asarray(velocity, dtype=float)
                return float(cumtrapz(v, dx=dt, initial=0)[-1])
            """,
            """
            import numpy as np
            from scipy.integrate import cumulative_trapezoid

            def displacement(velocity, dt):
                v = np.asarray(velocity, dtype=float)
                return float(cumulative_trapezoid(v, dx=dt, initial=0)[-1])
            """,
            """
            import solution
            assert solution.displacement([2, 2, 2], 0.5) == 2.0
            assert solution.displacement([0, 4], 1.0) == 2.0
            """,
            "Final displacement from uniformly sampled velocity.",
        ),
        _prog(
            "running_integral",
            """
            import numpy as np
            from scipy.integrate import cumtrapz

            class RunningIntegral:
                def __init__(self, x):
                    self.x = np.asarray(x, dtype=float)

                def of(self, y):
                    return cumtrapz(np.asarray(y, dtype=float), self.x).tolist()
            """,
            """
            import numpy as np
            from scipy.integrate import cumulative_trapezoid

            class RunningIntegral:
                def __init__(self, x):
                    self.x = np.asarray(x, dtype=float)

                def of(self, y):
                    return cumulative_trapezoid(np.asarray(y, dtype=float), self.x).tolist()
            """,
            """
            import solution
            ri = solution.RunningIntegral([0, 1, 2])
            assert ri.of([0, 2, 4]) == [1.0, 4.0]
            assert ri.of([1, 1, 1]) == [1.0, 2.0]
            """,
            "Running integral on a fixed grid (no initial value).",
        ),
        _prog(
            "running_row_area",
            """
            import numpy as np
            from scipy.integrate import cumtrapz

            def running_row_area(matrix, dx):
                m = np.asarray(matrix, dtype=float)
                return cumtrapz(m, dx=dx, axis=1, initial=0).tolist()
            """,
            """
            import numpy as np
            from scipy.integrate import cumulative_trapezoid

            def running_row_area(matrix, dx):
                m = np.asarray(matrix, dtype=float)
                return cumulative_trapezoid(m, dx=dx, axis=1, initial=0).tolist()
            """,
            """
            import solution
            assert solution.running_row_area([[1, 1, 1], [0, 2, 4]], 1.0) == [[0.0, 1.0, 2.0], [0.0, 1.0, 4.0]]
            """,
            "Running trapezoid area along each row.",
        ),
        _prog(
            "charge_profile",
            """
            import numpy as np
            from scipy.integrate import cumtrapz

            def charge_profile(current, t):
                q = cumtrapz(current, t, initial=0)
                return {"final": float(q[-1]), "peak_index": int(np.argmax(q))}
            """,
            """
            import numpy as np
            from scipy.integrate import cumulative_trapezoid

            def charge_profile(current, t):
                q = cumulative_trapezoid(current, t, initial=0)
                return {"final": float(q[-1]), "peak_index": int(np.argmax(q))}
            """,
            """
            import solution
            assert solution.charge_profile([0, 2, 2, 0], [0, 1, 2, 3]) == {"final": 4.0, "peak_index": 3}
            assert solution.charge_profile([2, 2, -4, -4], [0, 1, 2, 3]) == {"final": -3.0, "peak_index": 1}
            """,
            "Accumulated charge from current samples.",
        ),
    ),
    siblings=(
        _sib(
            "trapezoid_total",
            """
            import numpy as np
            from scipy.integrate import trapezoid

            def total_area(y, x):
                return float(trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))
            """,
            """
            import solution
            assert solution.total_area([1, 2, 3], [0, 1, 2]) == 4.0
            """,
            "scipy.integrate.trapezoid (unchanged).",
        ),
        _sib(
            "fixed_quad_running",
            """
            from scipy.integrate import fixed_quad

            def running_integral(f, points):
                out = [0.0]
                for a, b in zip(points[:-1], points[1:]):
                    out.append(out[-1] + float(fixed_quad(f, a, b, n=4)[0]))
                return out
            """,
            """
            import solution
            out = solution.running_integral(lambda x: 2 * x, [0, 1, 2])
            assert abs(out[1] - 1.0) < 1e-9 and abs(out[2] - 4.0) < 1e-9
            """,
            "Running integral via fixed_quad (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.integrate.trapz -> scipy.integrate.trapezoid
# ---------------------------------------------------------------------------

_SP_TRAPZ = ApiChange(
    package="scipy",
    api_family="scipy.integrate.trapz",
    old_api="scipy.integrate.trapz",
    new_api="scipy.integrate.trapezoid",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_114,
    change_url=SP_URL_114,
    summary="scipy.integrate.trapz removed in SciPy 1.14.0; use trapezoid.",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "total_rainfall",
            """
            from scipy.integrate import trapz

            def total_rainfall(rates, hours):
                return float(trapz(rates, hours))
            """,
            """
            from scipy.integrate import trapezoid

            def total_rainfall(rates, hours):
                return float(trapezoid(rates, hours))
            """,
            """
            import solution
            assert solution.total_rainfall([1, 2, 3], [0, 1, 2]) == 4.0
            assert solution.total_rainfall([0, 4], [0, 0.5]) == 1.0
            """,
            "Total rainfall from hourly rate samples.",
        ),
        _prog(
            "roc_auc",
            """
            import numpy as np
            from scipy.integrate import trapz

            def auc(fpr, tpr):
                order = np.argsort(fpr)
                return float(trapz(np.asarray(tpr)[order], np.asarray(fpr)[order]))
            """,
            """
            import numpy as np
            from scipy.integrate import trapezoid

            def auc(fpr, tpr):
                order = np.argsort(fpr)
                return float(trapezoid(np.asarray(tpr)[order], np.asarray(fpr)[order]))
            """,
            """
            import solution
            assert solution.auc([0, 0.5, 1], [0, 1, 1]) == 0.75
            assert solution.auc([1, 0], [1, 0]) == 0.5
            """,
            "Area under a ROC curve.",
        ),
        _prog(
            "average_power",
            """
            import numpy as np
            from scipy.integrate import trapz

            def average_power(power, t):
                t = np.asarray(t, dtype=float)
                return float(trapz(power, t) / (t[-1] - t[0]))
            """,
            """
            import numpy as np
            from scipy.integrate import trapezoid

            def average_power(power, t):
                t = np.asarray(t, dtype=float)
                return float(trapezoid(power, t) / (t[-1] - t[0]))
            """,
            """
            import solution
            assert solution.average_power([2, 4], [0, 2]) == 3.0
            assert solution.average_power([1, 1, 1], [0, 1, 4]) == 1.0
            """,
            "Time-averaged power.",
        ),
        _prog(
            "sampled_signal",
            """
            import numpy as np
            from scipy.integrate import trapz

            class SampledSignal:
                def __init__(self, dt, samples):
                    self.dt = dt
                    self.samples = np.asarray(samples, dtype=float)

                def area(self):
                    return float(trapz(self.samples, dx=self.dt))

                def absolute_area(self):
                    return float(trapz(np.abs(self.samples), dx=self.dt))
            """,
            """
            import numpy as np
            from scipy.integrate import trapezoid

            class SampledSignal:
                def __init__(self, dt, samples):
                    self.dt = dt
                    self.samples = np.asarray(samples, dtype=float)

                def area(self):
                    return float(trapezoid(self.samples, dx=self.dt))

                def absolute_area(self):
                    return float(trapezoid(np.abs(self.samples), dx=self.dt))
            """,
            """
            import solution
            sig = solution.SampledSignal(0.5, [1, -1, 1])
            assert sig.area() == 0.0
            assert sig.absolute_area() == 1.0
            """,
            "Signed and absolute area of a sampled signal.",
        ),
        _prog(
            "column_integrals",
            """
            import numpy as np
            from scipy.integrate import trapz

            def column_integrals(matrix, dx):
                return trapz(np.asarray(matrix, dtype=float), dx=dx, axis=0).tolist()
            """,
            """
            import numpy as np
            from scipy.integrate import trapezoid

            def column_integrals(matrix, dx):
                return trapezoid(np.asarray(matrix, dtype=float), dx=dx, axis=0).tolist()
            """,
            """
            import solution
            assert solution.column_integrals([[0, 1], [2, 3], [4, 5]], 1.0) == [4.0, 6.0]
            """,
            "Trapezoid integral of each column.",
        ),
        _prog(
            "impulse",
            """
            from scipy.integrate import trapz

            def impulse(force, dt):
                return float(trapz(force, dx=dt))

            def average_force(force, dt):
                return impulse(force, dt) / (dt * (len(force) - 1))
            """,
            """
            from scipy.integrate import trapezoid

            def impulse(force, dt):
                return float(trapezoid(force, dx=dt))

            def average_force(force, dt):
                return impulse(force, dt) / (dt * (len(force) - 1))
            """,
            """
            import solution
            assert abs(solution.impulse([0, 10, 0], 0.1) - 1.0) < 1e-12
            assert abs(solution.average_force([0, 10, 0], 0.1) - 5.0) < 1e-12
            """,
            "Impulse and average force from force samples.",
        ),
    ),
    siblings=(
        _sib(
            "simpson_keyword",
            """
            import numpy as np
            from scipy.integrate import simpson

            def simpson_area(y, x):
                return float(simpson(np.asarray(y, dtype=float), x=np.asarray(x, dtype=float)))
            """,
            """
            import solution
            assert abs(solution.simpson_area([0, 1, 4], [0, 1, 2]) - 8.0 / 3.0) < 1e-9
            """,
            "scipy.integrate.simpson with keyword x (unchanged).",
        ),
        _sib(
            "cumulative_trapezoid_curve",
            """
            from scipy.integrate import cumulative_trapezoid

            def running_area(y, x):
                return cumulative_trapezoid(y, x, initial=0).tolist()
            """,
            """
            import solution
            assert solution.running_area([1, 1, 1], [0, 1, 2]) == [0.0, 1.0, 2.0]
            """,
            "scipy.integrate.cumulative_trapezoid (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.stats.binom_test -> scipy.stats.binomtest(...).pvalue
# ---------------------------------------------------------------------------

_SP_BINOM_TEST = ApiChange(
    package="scipy",
    api_family="scipy.stats.binom_test",
    old_api="scipy.stats.binom_test",
    new_api="scipy.stats.binomtest(...).pvalue",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_112,
    change_url=SP_URL_112,
    summary="scipy.stats.binom_test removed in SciPy 1.12.0; use binomtest, which returns a result object with .pvalue.",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "coin_fairness",
            """
            from scipy.stats import binom_test

            def coin_fairness_pvalue(heads, flips):
                return float(binom_test(heads, flips, 0.5))
            """,
            """
            from scipy.stats import binomtest

            def coin_fairness_pvalue(heads, flips):
                return float(binomtest(heads, flips, 0.5).pvalue)
            """,
            """
            import solution
            assert abs(solution.coin_fairness_pvalue(3, 10) - 0.34375) < 1e-12
            assert abs(solution.coin_fairness_pvalue(9, 10) - 0.021484375) < 1e-12
            assert abs(solution.coin_fairness_pvalue(5, 10) - 1.0) < 1e-12
            """,
            "Two-sided p-value for a fair-coin hypothesis.",
        ),
        _prog(
            "is_biased",
            """
            from scipy.stats import binom_test

            def is_biased(successes, trials, alpha=0.05):
                return bool(binom_test(successes, trials) < alpha)
            """,
            """
            from scipy.stats import binomtest

            def is_biased(successes, trials, alpha=0.05):
                return bool(binomtest(successes, trials).pvalue < alpha)
            """,
            """
            import solution
            assert solution.is_biased(9, 10) is True
            assert solution.is_biased(6, 10) is False
            assert solution.is_biased(8, 10, alpha=0.2) is True
            """,
            "Decide whether a success count is inconsistent with p=0.5.",
        ),
        _prog(
            "conversion_lift",
            """
            import scipy.stats as st

            def conversion_lift_pvalue(conversions, visitors, baseline):
                return float(st.binom_test(conversions, n=visitors, p=baseline, alternative="greater"))
            """,
            """
            import scipy.stats as st

            def conversion_lift_pvalue(conversions, visitors, baseline):
                return float(st.binomtest(conversions, n=visitors, p=baseline, alternative="greater").pvalue)
            """,
            """
            import solution
            assert abs(solution.conversion_lift_pvalue(4, 20, 0.1) - 0.1329533234343351) < 1e-9
            assert abs(solution.conversion_lift_pvalue(6, 50, 0.05) - 0.03777617298977731) < 1e-9
            """,
            "One-sided p-value that a conversion rate exceeds a baseline.",
        ),
        _prog(
            "defect_monitor",
            """
            from scipy.stats import binom_test

            class DefectMonitor:
                def __init__(self, max_rate, alpha):
                    self.max_rate = max_rate
                    self.alpha = alpha

                def pvalue(self, defects, inspected):
                    return float(binom_test(defects, inspected, self.max_rate, alternative="greater"))

                def alarm(self, defects, inspected):
                    return self.pvalue(defects, inspected) < self.alpha
            """,
            """
            from scipy.stats import binomtest

            class DefectMonitor:
                def __init__(self, max_rate, alpha):
                    self.max_rate = max_rate
                    self.alpha = alpha

                def pvalue(self, defects, inspected):
                    return float(binomtest(defects, inspected, self.max_rate, alternative="greater").pvalue)

                def alarm(self, defects, inspected):
                    return self.pvalue(defects, inspected) < self.alpha
            """,
            """
            import solution
            mon = solution.DefectMonitor(0.05, 0.05)
            assert abs(mon.pvalue(6, 50) - 0.03777617298977731) < 1e-9
            assert mon.alarm(6, 50) is True
            assert mon.alarm(1, 50) is False
            """,
            "Quality-control alarm based on a one-sided binomial test.",
        ),
        _prog(
            "pvalue_table",
            """
            from scipy.stats import binom_test

            def pvalue_table(results, p=0.5):
                return [float(binom_test(k, n, p)) for k, n in results]
            """,
            """
            from scipy.stats import binomtest

            def pvalue_table(results, p=0.5):
                return [float(binomtest(k, n, p).pvalue) for k, n in results]
            """,
            """
            import solution
            out = solution.pvalue_table([(3, 10), (12, 20), (0, 8)])
            expected = [0.34375, 0.5034446716308594, 0.0078125]
            assert len(out) == 3
            assert all(abs(a - b) < 1e-9 for a, b in zip(out, expected))
            """,
            "Two-sided p-values for a batch of (successes, trials) pairs.",
        ),
        _prog(
            "tail_pvalues",
            """
            from scipy.stats import binom_test

            def tail_pvalues(k, n, p):
                return {
                    "less": float(binom_test(k, n, p, alternative="less")),
                    "greater": float(binom_test(k, n, p, alternative="greater")),
                    "two-sided": float(binom_test(k, n, p, alternative="two-sided")),
                }
            """,
            """
            from scipy.stats import binomtest

            def tail_pvalues(k, n, p):
                return {
                    "less": float(binomtest(k, n, p, alternative="less").pvalue),
                    "greater": float(binomtest(k, n, p, alternative="greater").pvalue),
                    "two-sided": float(binomtest(k, n, p, alternative="two-sided").pvalue),
                }
            """,
            """
            import solution
            out = solution.tail_pvalues(3, 10, 0.5)
            assert abs(out["less"] - 0.171875) < 1e-12
            assert abs(out["greater"] - 0.9453125) < 1e-12
            assert abs(out["two-sided"] - 0.34375) < 1e-12
            """,
            "All three alternative p-values for one observation.",
        ),
    ),
    siblings=(
        _sib(
            "binom_tail_probability",
            """
            from scipy.stats import binom

            def prob_at_least(k, n, p):
                return float(binom.sf(k - 1, n, p))
            """,
            """
            import solution
            assert abs(solution.prob_at_least(7, 10, 0.5) - 0.171875) < 1e-12
            assert abs(solution.prob_at_least(0, 5, 0.5) - 1.0) < 1e-12
            """,
            "scipy.stats.binom.sf (unchanged).",
        ),
        _sib(
            "most_likely_count",
            """
            import numpy as np
            from scipy.stats import binom

            def most_likely_count(n, p):
                ks = np.arange(n + 1)
                return int(ks[np.argmax(binom.pmf(ks, n, p))])
            """,
            """
            import solution
            assert solution.most_likely_count(10, 0.5) == 5
            assert solution.most_likely_count(10, 0.3) == 3
            """,
            "scipy.stats.binom.pmf (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.interpolate.interp2d -> RegularGridInterpolator / RectBivariateSpline
# ---------------------------------------------------------------------------

_SP_INTERP2D = ApiChange(
    package="scipy",
    api_family="scipy.interpolate.interp2d",
    old_api="scipy.interpolate.interp2d",
    new_api="scipy.interpolate.RegularGridInterpolator / RectBivariateSpline",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_114,
    change_url=SP_URL_114,
    summary="scipy.interpolate.interp2d replaced by a stub raising NotImplementedError in SciPy 1.14.0; use RegularGridInterpolator or RectBivariateSpline.",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "bilinear_lookup",
            """
            import numpy as np
            from scipy.interpolate import interp2d

            def bilinear_lookup(x, y, table, xq, yq):
                f = interp2d(x, y, table, kind="linear")
                return float(f(xq, yq)[0])
            """,
            """
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            def bilinear_lookup(x, y, table, xq, yq):
                grid = (np.asarray(y, dtype=float), np.asarray(x, dtype=float))
                f = RegularGridInterpolator(grid, np.asarray(table, dtype=float))
                return float(f([[yq, xq]])[0])
            """,
            """
            import solution
            x = [0, 1, 2]
            y = [0, 10]
            table = [[0, 1, 2], [10, 11, 12]]
            assert abs(solution.bilinear_lookup(x, y, table, 0.5, 5) - 5.5) < 1e-9
            assert abs(solution.bilinear_lookup(x, y, table, 2, 10) - 12.0) < 1e-9
            assert abs(solution.bilinear_lookup(x, y, table, 1.5, 2.5) - 4.0) < 1e-9
            """,
            "Bilinear lookup in a 2-D table (z indexed as [y][x]).",
        ),
        _prog(
            "height_map",
            """
            import numpy as np
            from scipy.interpolate import interp2d

            class HeightMap:
                def __init__(self, xs, ys, heights):
                    self._f = interp2d(xs, ys, heights, kind="linear")

                def height_at(self, px, py):
                    return float(self._f(px, py)[0])

                def slope_between(self, p, q):
                    rise = self.height_at(*q) - self.height_at(*p)
                    run = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                    return rise / run
            """,
            """
            import numpy as np
            from scipy.interpolate import RectBivariateSpline

            class HeightMap:
                def __init__(self, xs, ys, heights):
                    z = np.asarray(heights, dtype=float).T
                    self._f = RectBivariateSpline(xs, ys, z, kx=1, ky=1)

                def height_at(self, px, py):
                    return float(self._f(px, py)[0, 0])

                def slope_between(self, p, q):
                    rise = self.height_at(*q) - self.height_at(*p)
                    run = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                    return rise / run
            """,
            """
            import solution
            hm = solution.HeightMap([0, 2], [0, 2], [[0, 2], [4, 6]])
            assert abs(hm.height_at(1, 1) - 3.0) < 1e-9
            assert abs(hm.height_at(2, 2) - 6.0) < 1e-9
            assert abs(hm.slope_between((0, 0), (2, 0)) - 1.0) < 1e-9
            """,
            "Height map with bilinear sampling and slope.",
        ),
        _prog(
            "resample_grid",
            """
            from scipy.interpolate import interp2d

            def resample_grid(x, y, z, new_x, new_y):
                f = interp2d(x, y, z, kind="linear")
                return f(new_x, new_y).tolist()
            """,
            """
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            def resample_grid(x, y, z, new_x, new_y):
                f = RegularGridInterpolator((np.asarray(y, dtype=float), np.asarray(x, dtype=float)), np.asarray(z, dtype=float))
                yy, xx = np.meshgrid(new_y, new_x, indexing="ij")
                pts = np.stack([yy.ravel(), xx.ravel()], axis=1)
                return f(pts).reshape(len(new_y), len(new_x)).tolist()
            """,
            """
            import solution
            out = solution.resample_grid([0, 1], [0, 1], [[0, 1], [1, 2]], [0, 0.5, 1], [0, 1])
            assert len(out) == 2 and len(out[0]) == 3
            expected = [[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]]
            for row, erow in zip(out, expected):
                for a, b in zip(row, erow):
                    assert abs(a - b) < 1e-9
            """,
            "Resample a 2-D grid onto new axes (rows follow new_y).",
        ),
        _prog(
            "temperature_points",
            """
            from scipy.interpolate import interp2d

            def temperatures_at(xs, ys, temps, points):
                f = interp2d(xs, ys, temps, kind="linear")
                return [float(f(px, py)[0]) for px, py in points]
            """,
            """
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            def temperatures_at(xs, ys, temps, points):
                f = RegularGridInterpolator((np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)), np.asarray(temps, dtype=float))
                return [float(f([[py, px]])[0]) for px, py in points]
            """,
            """
            import solution
            out = solution.temperatures_at([0, 10], [0, 10], [[20, 30], [40, 50]], [(5, 5), (10, 0), (0, 10)])
            expected = [35.0, 30.0, 40.0]
            assert all(abs(a - b) < 1e-9 for a, b in zip(out, expected)) and len(out) == 3
            """,
            "Interpolated temperatures at query points.",
        ),
        _prog(
            "calibration_gain",
            """
            from scipy.interpolate import interp2d

            class Calibration:
                def __init__(self, voltages, temperatures, gain_table):
                    self._gain = interp2d(voltages, temperatures, gain_table, kind="linear")

                def corrected(self, raw, voltage, temperature):
                    return raw * float(self._gain(voltage, temperature)[0])
            """,
            """
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            class Calibration:
                def __init__(self, voltages, temperatures, gain_table):
                    grid = (np.asarray(temperatures, dtype=float), np.asarray(voltages, dtype=float))
                    self._gain = RegularGridInterpolator(grid, np.asarray(gain_table, dtype=float))

                def corrected(self, raw, voltage, temperature):
                    return raw * float(self._gain([[temperature, voltage]])[0])
            """,
            """
            import solution
            cal = solution.Calibration([1, 3], [0, 100], [[1, 2], [2, 3]])
            assert abs(cal.corrected(10, 2, 50) - 20.0) < 1e-9
            assert abs(cal.corrected(1, 3, 100) - 3.0) < 1e-9
            assert abs(cal.corrected(4, 1, 0) - 4.0) < 1e-9
            """,
            "Apply a bilinearly interpolated calibration gain.",
        ),
        _prog(
            "upsample_image",
            """
            import numpy as np
            from scipy.interpolate import interp2d

            def upsample(image, factor):
                img = np.asarray(image, dtype=float)
                h, w = img.shape
                f = interp2d(np.arange(w), np.arange(h), img, kind="linear")
                new_x = np.linspace(0, w - 1, (w - 1) * factor + 1)
                new_y = np.linspace(0, h - 1, (h - 1) * factor + 1)
                return f(new_x, new_y).tolist()
            """,
            """
            import numpy as np
            from scipy.interpolate import RectBivariateSpline

            def upsample(image, factor):
                img = np.asarray(image, dtype=float)
                h, w = img.shape
                f = RectBivariateSpline(np.arange(w), np.arange(h), img.T, kx=1, ky=1)
                new_x = np.linspace(0, w - 1, (w - 1) * factor + 1)
                new_y = np.linspace(0, h - 1, (h - 1) * factor + 1)
                return f(new_x, new_y).T.tolist()
            """,
            """
            import solution
            out = solution.upsample([[0, 2], [4, 6]], 2)
            expected = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
            assert len(out) == 3 and all(len(r) == 3 for r in out)
            for row, erow in zip(out, expected):
                for a, b in zip(row, erow):
                    assert abs(a - b) < 1e-9
            """,
            "Bilinear upsampling of a small image.",
        ),
    ),
    siblings=(
        _sib(
            "interp1d_resample",
            """
            from scipy.interpolate import interp1d

            def resample_1d(x, y, new_x):
                return interp1d(x, y)(new_x).tolist()
            """,
            """
            import solution
            assert solution.resample_1d([0, 1, 2], [0, 10, 20], [0.5, 1.5]) == [5.0, 15.0]
            """,
            "scipy.interpolate.interp1d (unchanged).",
        ),
        _sib(
            "interpn_lookup",
            """
            import numpy as np
            from scipy.interpolate import interpn

            def lookup(x, y, table, points):
                grid = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                return interpn(grid, np.asarray(table, dtype=float), np.asarray(points, dtype=float)).tolist()
            """,
            """
            import solution
            out = solution.lookup([0, 1], [0, 1], [[0, 1], [2, 3]], [[0.5, 0.5], [1, 0]])
            assert all(abs(a - b) < 1e-9 for a, b in zip(out, [1.5, 2.0]))
            """,
            "scipy.interpolate.interpn (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.integrate.romberg / quadrature -> scipy.integrate.quad
# ---------------------------------------------------------------------------

_SP_ROMBERG = ApiChange(
    package="scipy",
    api_family="scipy.integrate.romberg",
    old_api="scipy.integrate.romberg / scipy.integrate.quadrature",
    new_api="scipy.integrate.quad",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_115,
    change_url=SP_URL_115,
    summary="scipy.integrate.romberg and quadrature removed in SciPy 1.15.0; use quad (returns (value, error)).",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "integral_of_square",
            """
            from scipy.integrate import romberg

            def integral_of_square(a, b):
                return float(romberg(lambda x: x * x, a, b))
            """,
            """
            from scipy.integrate import quad

            def integral_of_square(a, b):
                return float(quad(lambda x: x * x, a, b)[0])
            """,
            """
            import solution
            assert abs(solution.integral_of_square(0, 1) - 1.0 / 3.0) < 1e-8
            assert abs(solution.integral_of_square(1, 2) - 7.0 / 3.0) < 1e-8
            """,
            "Integral of x^2 over an interval.",
        ),
        _prog(
            "normal_mass",
            """
            import math
            import numpy as np
            from scipy.integrate import quadrature

            def probability_within(sigma, width):
                def pdf(x):
                    return np.exp(-x * x / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
                value, _err = quadrature(pdf, -width, width)
                return float(value)
            """,
            """
            import math
            import numpy as np
            from scipy.integrate import quad

            def probability_within(sigma, width):
                def pdf(x):
                    return np.exp(-x * x / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
                value, _err = quad(pdf, -width, width)
                return float(value)
            """,
            """
            import solution
            import math
            assert abs(solution.probability_within(1.0, 1.0) - math.erf(1 / math.sqrt(2))) < 1e-6
            assert abs(solution.probability_within(2.0, 4.0) - math.erf(2 / math.sqrt(2))) < 1e-6
            """,
            "Probability mass of a normal distribution within +-width.",
        ),
        _prog(
            "exp_log_areas",
            """
            import math
            from scipy.integrate import romberg

            def exp_area(a, b):
                return float(romberg(math.exp, a, b))

            def log_area(a, b):
                return float(romberg(lambda x: 1.0 / x, a, b))
            """,
            """
            import math
            from scipy.integrate import quad

            def exp_area(a, b):
                return float(quad(math.exp, a, b)[0])

            def log_area(a, b):
                return float(quad(lambda x: 1.0 / x, a, b)[0])
            """,
            """
            import solution
            import math
            assert abs(solution.exp_area(0, 1) - (math.e - 1)) < 1e-8
            assert abs(solution.log_area(1, 2) - math.log(2)) < 1e-8
            """,
            "Areas under exp(x) and 1/x.",
        ),
        _prog(
            "spring_work",
            """
            from scipy.integrate import quadrature

            class SpringWork:
                def __init__(self, k):
                    self.k = k

                def work(self, x0, x1):
                    value, _err = quadrature(lambda x: self.k * x, x0, x1)
                    return float(value)
            """,
            """
            from scipy.integrate import quad

            class SpringWork:
                def __init__(self, k):
                    self.k = k

                def work(self, x0, x1):
                    value, _err = quad(lambda x: self.k * x, x0, x1)
                    return float(value)
            """,
            """
            import solution
            assert abs(solution.SpringWork(2).work(0, 3) - 9.0) < 1e-9
            assert abs(solution.SpringWork(5).work(1, 2) - 7.5) < 1e-9
            """,
            "Work done stretching a spring.",
        ),
        _prog(
            "function_mean",
            """
            from scipy.integrate import quadrature

            def mean_value(f, a, b):
                total, _ = quadrature(f, a, b)
                return float(total) / (b - a)
            """,
            """
            from scipy.integrate import quad

            def mean_value(f, a, b):
                total, _ = quad(f, a, b)
                return float(total) / (b - a)
            """,
            """
            import solution
            import math
            import numpy as np
            assert abs(solution.mean_value(np.sin, 0, math.pi) - 2.0 / math.pi) < 1e-8
            assert abs(solution.mean_value(lambda x: 2 * x + 1, 0, 3) - 4.0) < 1e-9
            """,
            "Mean value of a function over an interval.",
        ),
        _prog(
            "parabola_arc_length",
            """
            import math
            from scipy.integrate import romberg

            def parabola_arc_length(a, b):
                return float(romberg(lambda x: math.sqrt(1.0 + 4.0 * x * x), a, b))
            """,
            """
            import math
            from scipy.integrate import quad

            def parabola_arc_length(a, b):
                return float(quad(lambda x: math.sqrt(1.0 + 4.0 * x * x), a, b)[0])
            """,
            """
            import solution
            import math
            exact = (2 * math.sqrt(5) + math.asinh(2)) / 4
            assert abs(solution.parabola_arc_length(0, 1) - exact) < 1e-6
            assert abs(solution.parabola_arc_length(0, 0) - 0.0) < 1e-12
            """,
            "Arc length of y = x^2.",
        ),
    ),
    siblings=(
        _sib(
            "fixed_quad_cubic",
            """
            from scipy.integrate import fixed_quad

            def cubic_area(a, b):
                return float(fixed_quad(lambda x: x ** 3, a, b, n=3)[0])
            """,
            """
            import solution
            assert abs(solution.cubic_area(0, 2) - 4.0) < 1e-9
            """,
            "scipy.integrate.fixed_quad (unchanged).",
        ),
        _sib(
            "dblquad_rectangle",
            """
            from scipy.integrate import dblquad

            def rectangle_mass(density, x0, x1, y0, y1):
                value, _ = dblquad(lambda y, x: density(x, y), x0, x1, y0, y1)
                return float(value)
            """,
            """
            import solution
            assert abs(solution.rectangle_mass(lambda x, y: x + y, 0, 1, 0, 1) - 1.0) < 1e-9
            """,
            "scipy.integrate.dblquad (unchanged).",
        ),
    ),
)


# ---------------------------------------------------------------------------
# scipy.sparse matrix .A -> .toarray()
# ---------------------------------------------------------------------------

_SP_SPARSE_A = ApiChange(
    package="scipy",
    api_family="scipy.sparse.spmatrix.A",
    old_api="sparse_matrix.A",
    new_api="sparse_matrix.toarray()",
    old_pin=SP_OLD,
    new_pin=SP_NEW,
    change_date=SP_DATE_114,
    change_url=SP_URL_114,
    summary="The .A (and .H) attributes of scipy.sparse matrices/arrays were removed in SciPy 1.14.0; use .toarray().",
    extra_requirements=SP_EXTRA,
    programs=(
        _prog(
            "dense_from_triplets",
            """
            from scipy.sparse import coo_matrix

            def dense_from_triplets(rows, cols, vals, shape):
                return coo_matrix((vals, (rows, cols)), shape=shape).A.tolist()
            """,
            """
            from scipy.sparse import coo_matrix

            def dense_from_triplets(rows, cols, vals, shape):
                return coo_matrix((vals, (rows, cols)), shape=shape).toarray().tolist()
            """,
            """
            import solution
            assert solution.dense_from_triplets([0, 1], [1, 0], [5, 7], (2, 2)) == [[0, 5], [7, 0]]
            assert solution.dense_from_triplets([0, 0], [0, 0], [1, 2], (1, 2)) == [[3, 0]]
            """,
            "Dense matrix from COO triplets (duplicates summed).",
        ),
        _prog(
            "adjacency_dense",
            """
            from scipy.sparse import lil_matrix

            def adjacency_dense(edges, n):
                m = lil_matrix((n, n), dtype=int)
                for a, b in edges:
                    m[a, b] = 1
                    m[b, a] = 1
                return m.A.tolist()
            """,
            """
            from scipy.sparse import lil_matrix

            def adjacency_dense(edges, n):
                m = lil_matrix((n, n), dtype=int)
                for a, b in edges:
                    m[a, b] = 1
                    m[b, a] = 1
                return m.toarray().tolist()
            """,
            """
            import solution
            assert solution.adjacency_dense([(0, 1), (1, 2)], 3) == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            assert solution.adjacency_dense([], 2) == [[0, 0], [0, 0]]
            """,
            "Dense adjacency matrix of an undirected graph.",
        ),
        _prog(
            "cooccurrence_counter",
            """
            from scipy.sparse import dok_matrix

            class CooccurrenceCounter:
                def __init__(self, size):
                    self.m = dok_matrix((size, size), dtype=int)

                def add(self, i, j):
                    self.m[i, j] += 1

                def dense(self):
                    return self.m.A.tolist()
            """,
            """
            from scipy.sparse import dok_matrix

            class CooccurrenceCounter:
                def __init__(self, size):
                    self.m = dok_matrix((size, size), dtype=int)

                def add(self, i, j):
                    self.m[i, j] += 1

                def dense(self):
                    return self.m.toarray().tolist()
            """,
            """
            import solution
            c = solution.CooccurrenceCounter(2)
            c.add(0, 1)
            c.add(0, 1)
            c.add(1, 1)
            assert c.dense() == [[0, 2], [0, 1]]
            """,
            "Sparse co-occurrence counter with dense export.",
        ),
        _prog(
            "sparse_square",
            """
            import numpy as np
            from scipy.sparse import csr_matrix

            def sparse_square(matrix):
                m = csr_matrix(np.asarray(matrix))
                return (m @ m).A.tolist()
            """,
            """
            import numpy as np
            from scipy.sparse import csr_matrix

            def sparse_square(matrix):
                m = csr_matrix(np.asarray(matrix))
                return (m @ m).toarray().tolist()
            """,
            """
            import solution
            assert solution.sparse_square([[1, 1], [0, 1]]) == [[1, 2], [0, 1]]
            assert solution.sparse_square([[2, 0], [0, 3]]) == [[4, 0], [0, 9]]
            """,
            "Square of a matrix computed sparsely.",
        ),
        _prog(
            "normalize_rows_sparse",
            """
            import numpy as np
            from scipy.sparse import csr_matrix, diags

            def normalize_rows(matrix):
                m = csr_matrix(np.asarray(matrix, dtype=float))
                sums = np.asarray(m.sum(axis=1)).ravel()
                scale = diags(1.0 / sums)
                return (scale @ m).A.tolist()
            """,
            """
            import numpy as np
            from scipy.sparse import csr_matrix, diags

            def normalize_rows(matrix):
                m = csr_matrix(np.asarray(matrix, dtype=float))
                sums = np.asarray(m.sum(axis=1)).ravel()
                scale = diags(1.0 / sums)
                return (scale @ m).toarray().tolist()
            """,
            """
            import solution
            assert solution.normalize_rows([[1, 1], [2, 6]]) == [[0.5, 0.5], [0.25, 0.75]]
            """,
            "Row-normalise a matrix using sparse operations.",
        ),
        _prog(
            "tridiagonal",
            """
            from scipy.sparse import diags

            def tridiagonal(n, main, off):
                bands = [[off] * (n - 1), [main] * n, [off] * (n - 1)]
                return diags(bands, [-1, 0, 1]).A.tolist()
            """,
            """
            from scipy.sparse import diags

            def tridiagonal(n, main, off):
                bands = [[off] * (n - 1), [main] * n, [off] * (n - 1)]
                return diags(bands, [-1, 0, 1]).toarray().tolist()
            """,
            """
            import solution
            assert solution.tridiagonal(3, 2, -1) == [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
            assert solution.tridiagonal(1, 4, 9) == [[4]]
            """,
            "Dense tridiagonal matrix built from sparse bands.",
        ),
    ),
    siblings=(
        _sib(
            "todense_export",
            """
            import numpy as np
            from scipy.sparse import csr_matrix

            def dense_rows(matrix):
                return np.asarray(csr_matrix(np.asarray(matrix)).todense()).tolist()
            """,
            """
            import solution
            assert solution.dense_rows([[1, 0], [0, 2]]) == [[1, 0], [0, 2]]
            """,
            "scipy.sparse .todense() (unchanged).",
        ),
        _sib(
            "diagonal_and_nnz",
            """
            import numpy as np
            from scipy.sparse import csr_matrix

            def sparse_summary(matrix):
                m = csr_matrix(np.asarray(matrix))
                return m.diagonal().tolist(), int(m.nnz)
            """,
            """
            import solution
            assert solution.sparse_summary([[1, 0], [3, 2]]) == ([1, 2], 3)
            """,
            "scipy.sparse .diagonal() and .nnz (unchanged).",
        ),
    ),
)


CHANGES: list[ApiChange] = [
    _NP_PRODUCT,
    _NP_CUMPRODUCT,
    _NP_ALLTRUE,
    _NP_SOMETRUE,
    _NP_NAN_ALIASES,
    _NP_FLOAT_,
    _NP_ROUND_,
    _NP_ASFARRAY,
    _NP_MSORT,
    _NP_COPY_FALSE,
    _NP_FIND_COMMON_TYPE,
    _NP_CAST,
    _SP_SIMPS,
    _SP_CUMTRAPZ,
    _SP_TRAPZ,
    _SP_BINOM_TEST,
    _SP_INTERP2D,
    _SP_ROMBERG,
    _SP_SPARSE_A,
]
