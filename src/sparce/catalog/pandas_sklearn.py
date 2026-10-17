"""Real breaking API changes: pandas 1.5.3 -> 2.2.3 and scikit-learn 1.1.3 -> 1.6.1.

Every change below is documented in the official release notes (``change_url``)
and every program was verified by execution on both pins (see the group brief).
numpy is pinned to 1.26.4 in both environments so that numpy itself is never a
confound (pandas 1.5.3 and scikit-learn 1.1.3 do not support numpy 2).
"""

from __future__ import annotations

from textwrap import dedent as _dedent

from sparce.catalog import ApiChange, Program, SiblingProgram


def _p(name: str, old: str, new: str, test: str, description: str = "") -> Program:
    return Program(
        name=name,
        old_code=_dedent(old).strip() + "\n",
        new_code=_dedent(new).strip() + "\n",
        test_code=_dedent(test).strip() + "\n",
        description=description,
    )


def _s(name: str, code: str, test: str, description: str = "") -> SiblingProgram:
    return SiblingProgram(
        name=name,
        code=_dedent(code).strip() + "\n",
        test_code=_dedent(test).strip() + "\n",
        description=description,
    )


PANDAS_OLD = "1.5.3"
PANDAS_NEW = "2.2.3"
PANDAS_DATE = "2023-04-03"
PANDAS_URL = "https://pandas.pydata.org/docs/whatsnew/v2.0.0.html"
PANDAS_EXTRA = ("numpy==1.26.4",)

SKLEARN_OLD = "1.1.3"
SKLEARN_NEW = "1.6.1"
SKLEARN_EXTRA = ("numpy==1.26.4", "scipy==1.10.1")
SKLEARN_URL_12 = "https://scikit-learn.org/stable/whats_new/v1.2.html"
SKLEARN_URL_13 = "https://scikit-learn.org/stable/whats_new/v1.3.html"
SKLEARN_URL_14 = "https://scikit-learn.org/stable/whats_new/v1.4.html"
SKLEARN_URL_16 = "https://scikit-learn.org/stable/whats_new/v1.6.html"
SKLEARN_DATE_12 = "2022-12-08"
SKLEARN_DATE_13 = "2023-06-30"
SKLEARN_DATE_14 = "2024-01-18"
SKLEARN_DATE_16 = "2024-12-09"


def _pandas_change(api_family, old_api, new_api, summary, programs, siblings) -> ApiChange:
    return ApiChange(
        package="pandas",
        api_family=api_family,
        old_api=old_api,
        new_api=new_api,
        old_pin=PANDAS_OLD,
        new_pin=PANDAS_NEW,
        change_date=PANDAS_DATE,
        change_url=PANDAS_URL,
        summary=summary,
        programs=tuple(programs),
        siblings=tuple(siblings),
        extra_requirements=PANDAS_EXTRA,
    )


def _sklearn_change(api_family, old_api, new_api, date, url, summary, programs, siblings) -> ApiChange:
    return ApiChange(
        package="scikit-learn",
        api_family=api_family,
        old_api=old_api,
        new_api=new_api,
        old_pin=SKLEARN_OLD,
        new_pin=SKLEARN_NEW,
        change_date=date,
        change_url=url,
        summary=summary,
        programs=tuple(programs),
        siblings=tuple(siblings),
        extra_requirements=SKLEARN_EXTRA,
    )


# ---------------------------------------------------------------------------
# pandas: DataFrame.append removed (2.0) -> pd.concat
# ---------------------------------------------------------------------------

DF_APPEND = _pandas_change(
    "pandas.DataFrame.append",
    "DataFrame.append(other)",
    "pd.concat([df, other])",
    "DataFrame.append was removed in pandas 2.0; use pandas.concat instead.",
    [
        _p(
            "event_log",
            """
            import pandas as pd


            class EventLog:
                def __init__(self):
                    self.frame = pd.DataFrame(columns=["event", "value"])

                def add(self, event, value):
                    row = {"event": event, "value": value}
                    self.frame = self.frame.append(row, ignore_index=True)

                def total(self):
                    return float(self.frame["value"].sum())
            """,
            """
            import pandas as pd


            class EventLog:
                def __init__(self):
                    self.frame = pd.DataFrame(columns=["event", "value"])

                def add(self, event, value):
                    row = {"event": event, "value": value}
                    self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)

                def total(self):
                    return float(self.frame["value"].sum())
            """,
            """
            import solution

            log = solution.EventLog()
            log.add("click", 2)
            log.add("buy", 5)
            assert len(log.frame) == 2
            assert list(log.frame["event"]) == ["click", "buy"]
            assert log.total() == 7.0
            """,
            "Accumulate event rows into a frame and sum a column.",
        ),
        _p(
            "stack_frames",
            """
            import pandas as pd


            def stack(frames):
                out = frames[0]
                for frame in frames[1:]:
                    out = out.append(frame)
                return out
            """,
            """
            import pandas as pd


            def stack(frames):
                stacked = pd.concat(frames)
                return stacked
            """,
            """
            import pandas as pd
            import solution

            a = pd.DataFrame({"x": [1, 2]}, index=["r1", "r2"])
            b = pd.DataFrame({"x": [3]}, index=["r3"])
            c = pd.DataFrame({"x": [4]}, index=["r4"])
            out = solution.stack([a, b, c])
            assert list(out.index) == ["r1", "r2", "r3", "r4"]
            assert int(out["x"].sum()) == 10
            """,
            "Vertically stack a list of frames preserving their indexes.",
        ),
        _p(
            "total_row",
            """
            import pandas as pd


            def with_total(df):
                total = df.sum().rename("total")
                return df.append(total)
            """,
            """
            import pandas as pd


            def with_total(df):
                total = df.sum().rename("total")
                return pd.concat([df, total.to_frame().T])
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["x", "y", "z"])
            out = solution.with_total(df)
            assert list(out.index) == ["x", "y", "z", "total"]
            assert int(out.loc["total", "a"]) == 6
            assert int(out.loc["total", "b"]) == 60
            """,
            "Add a column-sum row named 'total' to a frame.",
        ),
        _p(
            "sensor_batches",
            """
            import pandas as pd


            def mean_by_sensor(batches):
                out = pd.DataFrame({"sensor": [], "value": []})
                for batch in batches:
                    out = out.append(pd.DataFrame(batch), ignore_index=True)
                return out.groupby("sensor")["value"].mean().to_dict()
            """,
            """
            import pandas as pd


            def mean_by_sensor(batches):
                out = pd.DataFrame({"sensor": [], "value": []})
                for batch in batches:
                    out = pd.concat([out, pd.DataFrame(batch)], ignore_index=True)
                return out.groupby("sensor")["value"].mean().to_dict()
            """,
            """
            import solution

            batches = [
                {"sensor": ["a", "b"], "value": [1.0, 10.0]},
                {"sensor": ["a", "b"], "value": [3.0, 30.0]},
            ]
            out = solution.mean_by_sensor(batches)
            assert out == {"a": 2.0, "b": 20.0}
            """,
            "Append sensor batches and compute the per-sensor mean.",
        ),
        _p(
            "dedupe_union",
            """
            import pandas as pd


            def union_rows(a, b):
                merged = a.append(b)
                return merged.drop_duplicates().reset_index(drop=True)
            """,
            """
            import pandas as pd


            def union_rows(a, b):
                merged = pd.concat([a, b])
                return merged.drop_duplicates().reset_index(drop=True)
            """,
            """
            import pandas as pd
            import solution

            a = pd.DataFrame({"id": [1, 2], "tag": ["p", "q"]})
            b = pd.DataFrame({"id": [2, 3], "tag": ["q", "r"]})
            out = solution.union_rows(a, b)
            assert len(out) == 3
            assert list(out["id"]) == [1, 2, 3]
            assert list(out.index) == [0, 1, 2]
            """,
            "Union two frames and drop duplicate rows.",
        ),
        _p(
            "fill_missing_days",
            """
            import pandas as pd


            def complete_days(df, all_days):
                missing = [d for d in all_days if d not in set(df["day"])]
                filler = pd.DataFrame({"day": missing, "count": [0] * len(missing)})
                out = df.append(filler, ignore_index=True)
                return out.sort_values("day").reset_index(drop=True)
            """,
            """
            import pandas as pd


            def complete_days(df, all_days):
                missing = [d for d in all_days if d not in set(df["day"])]
                filler = pd.DataFrame({"day": missing, "count": [0] * len(missing)})
                out = pd.concat([df, filler], ignore_index=True)
                return out.sort_values("day").reset_index(drop=True)
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"day": [1, 3], "count": [5, 7]})
            out = solution.complete_days(df, [1, 2, 3, 4])
            assert list(out["day"]) == [1, 2, 3, 4]
            assert [int(v) for v in out["count"]] == [5, 0, 7, 0]
            """,
            "Fill missing day rows with zero counts.",
        ),
    ],
    [
        _s(
            "merge_lookup",
            """
            import pandas as pd


            def attach_prices(orders, prices):
                return orders.merge(prices, on="item", how="left")
            """,
            """
            import pandas as pd
            import solution

            orders = pd.DataFrame({"item": ["a", "b", "a"], "qty": [1, 2, 3]})
            prices = pd.DataFrame({"item": ["a", "b"], "price": [10, 20]})
            out = solution.attach_prices(orders, prices)
            assert list(out["price"]) == [10, 20, 10]
            assert list(out.columns) == ["item", "qty", "price"]
            """,
            "Left-merge prices onto orders (DataFrame.merge is unchanged).",
        ),
        _s(
            "combine_first_fill",
            """
            import pandas as pd


            def patch(primary, fallback):
                return primary.combine_first(fallback)
            """,
            """
            import pandas as pd
            import solution

            primary = pd.DataFrame({"v": [1.0, None, 3.0]})
            fallback = pd.DataFrame({"v": [9.0, 2.0, 9.0]})
            out = solution.patch(primary, fallback)
            assert list(out["v"]) == [1.0, 2.0, 3.0]
            """,
            "Fill gaps in one frame from another (combine_first is unchanged).",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: Series.append removed (2.0) -> pd.concat
# ---------------------------------------------------------------------------

SERIES_APPEND = _pandas_change(
    "pandas.Series.append",
    "Series.append(other)",
    "pd.concat([s, other])",
    "Series.append was removed in pandas 2.0; use pandas.concat instead.",
    [
        _p(
            "extend_series",
            """
            import pandas as pd


            def extend(values, extra):
                s = pd.Series(values)
                return s.append(pd.Series(extra), ignore_index=True)
            """,
            """
            import pandas as pd


            def extend(values, extra):
                s = pd.Series(values)
                return pd.concat([s, pd.Series(extra)], ignore_index=True)
            """,
            """
            import solution

            out = solution.extend([1, 2], [3, 4])
            assert list(out) == [1, 2, 3, 4]
            assert list(out.index) == [0, 1, 2, 3]
            """,
            "Extend a series with extra values and a fresh index.",
        ),
        _p(
            "prepend_header",
            """
            import pandas as pd


            def with_header(label, s):
                head = pd.Series([label], index=["header"])
                return head.append(s)
            """,
            """
            import pandas as pd


            def with_header(label, s):
                head = pd.Series([label], index=["header"])
                return pd.concat([head, s])
            """,
            """
            import pandas as pd
            import solution

            s = pd.Series(["x", "y"], index=["r1", "r2"])
            out = solution.with_header("title", s)
            assert list(out.index) == ["header", "r1", "r2"]
            assert list(out) == ["title", "x", "y"]
            """,
            "Prepend a labelled header entry to a series.",
        ),
        _p(
            "accumulator",
            """
            import pandas as pd


            class Accumulator:
                def __init__(self):
                    self.values = pd.Series([], dtype="float64")

                def push(self, chunk):
                    self.values = self.values.append(pd.Series(chunk, dtype="float64"), ignore_index=True)

                def total(self):
                    return float(self.values.sum())

                def size(self):
                    return int(len(self.values))
            """,
            """
            import pandas as pd


            class Accumulator:
                def __init__(self):
                    self.values = pd.Series([], dtype="float64")

                def push(self, chunk):
                    self.values = pd.concat([self.values, pd.Series(chunk, dtype="float64")], ignore_index=True)

                def total(self):
                    return float(self.values.sum())

                def size(self):
                    return int(len(self.values))
            """,
            """
            import solution

            acc = solution.Accumulator()
            acc.push([1.5, 2.5])
            acc.push([4.0])
            assert acc.size() == 3
            assert acc.total() == 8.0
            """,
            "Accumulate numeric chunks into one series.",
        ),
        _p(
            "interleave_sorted",
            """
            import pandas as pd


            def interleave(a, b):
                merged = a.append(b)
                return merged.sort_index()
            """,
            """
            import pandas as pd


            def interleave(a, b):
                merged = pd.concat([a, b])
                return merged.sort_index()
            """,
            """
            import pandas as pd
            import solution

            a = pd.Series([10, 30], index=[1, 3])
            b = pd.Series([20, 40], index=[2, 4])
            out = solution.interleave(a, b)
            assert list(out.index) == [1, 2, 3, 4]
            assert list(out) == [10, 20, 30, 40]
            """,
            "Combine two series and sort by index.",
        ),
        _p(
            "unique_ids",
            """
            import pandas as pd


            def register(ids, new_ids):
                merged = ids.append(pd.Series(new_ids), ignore_index=True)
                return merged.drop_duplicates().reset_index(drop=True)
            """,
            """
            import pandas as pd


            def register(ids, new_ids):
                merged = pd.concat([ids, pd.Series(new_ids)], ignore_index=True)
                return merged.drop_duplicates().reset_index(drop=True)
            """,
            """
            import pandas as pd
            import solution

            ids = pd.Series([101, 102])
            out = solution.register(ids, [102, 103, 101])
            assert list(out) == [101, 102, 103]
            """,
            "Register new ids, keeping the series unique.",
        ),
        _p(
            "join_labels",
            """
            import pandas as pd


            def join_labels(first, second, third):
                joined = first.append([second, third])
                return joined.tolist()
            """,
            """
            import pandas as pd


            def join_labels(first, second, third):
                joined = pd.concat([first, second, third])
                return joined.tolist()
            """,
            """
            import pandas as pd
            import solution

            out = solution.join_labels(pd.Series(["a"]), pd.Series(["b", "c"]), pd.Series(["d"]))
            assert out == ["a", "b", "c", "d"]
            """,
            "Join three label series into one list (list argument form).",
        ),
    ],
    [
        _s(
            "combine_first_series",
            """
            import pandas as pd


            def fill_from(primary, backup):
                return primary.combine_first(backup)
            """,
            """
            import pandas as pd
            import solution

            out = solution.fill_from(pd.Series([1.0, None]), pd.Series([5.0, 2.0]))
            assert list(out) == [1.0, 2.0]
            """,
            "Series.combine_first is unchanged.",
        ),
        _s(
            "repeat_values",
            """
            import pandas as pd


            def repeat_each(values, n):
                return pd.Series(values).repeat(n).reset_index(drop=True).tolist()
            """,
            """
            import solution

            assert solution.repeat_each(["a", "b"], 2) == ["a", "a", "b", "b"]
            """,
            "Series.repeat is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: Series.iteritems / DataFrame.iteritems removed (2.0) -> .items()
# ---------------------------------------------------------------------------

ITERITEMS = _pandas_change(
    "pandas.Series.iteritems",
    "Series.iteritems() / DataFrame.iteritems()",
    ".items()",
    "Series.iteritems and DataFrame.iteritems were removed in pandas 2.0; use .items().",
    [
        _p(
            "label_pairs",
            """
            import pandas as pd


            def label_pairs(values, labels):
                s = pd.Series(values, index=labels)
                return [(k, int(v)) for k, v in s.iteritems()]
            """,
            """
            import pandas as pd


            def label_pairs(values, labels):
                s = pd.Series(values, index=labels)
                return [(k, int(v)) for k, v in s.items()]
            """,
            """
            import solution

            assert solution.label_pairs([3, 4], ["x", "y"]) == [("x", 3), ("y", 4)]
            """,
            "Turn a labelled series into (label, value) pairs.",
        ),
        _p(
            "column_sums",
            """
            import pandas as pd


            def column_sums(df):
                sums = {name: float(col.sum()) for name, col in df.iteritems()}
                return sums
            """,
            """
            import pandas as pd


            def column_sums(df):
                sums = {name: float(col.sum()) for name, col in df.items()}
                return sums
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.5]})
            assert solution.column_sums(df) == {"a": 3.0, "b": 1.0}
            """,
            "Sum every column of a frame into a dict.",
        ),
        _p(
            "first_positive",
            """
            import pandas as pd


            def first_positive_label(s):
                for label, value in s.iteritems():
                    if value > 0:
                        return label
                return None
            """,
            """
            import pandas as pd


            def first_positive_label(s):
                for label, value in s.items():
                    if value > 0:
                        return label
                return None
            """,
            """
            import pandas as pd
            import solution

            s = pd.Series([-1, 0, 3, 5], index=["a", "b", "c", "d"])
            assert solution.first_positive_label(s) == "c"
            assert solution.first_positive_label(pd.Series([-1], index=["z"])) is None
            """,
            "Find the label of the first positive entry.",
        ),
        _p(
            "column_kinds",
            """
            import pandas as pd


            def column_kinds(df):
                kinds = {}
                for name, col in df.iteritems():
                    kinds[name] = col.dtype.kind
                return kinds
            """,
            """
            import pandas as pd


            def column_kinds(df):
                kinds = {}
                for name, col in df.items():
                    kinds[name] = col.dtype.kind
                return kinds
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"n": [1, 2], "f": [1.0, 2.0], "t": ["a", "b"]})
            assert solution.column_kinds(df) == {"n": "i", "f": "f", "t": "O"}
            """,
            "Report the numpy dtype kind of every column.",
        ),
        _p(
            "ledger_report",
            """
            import pandas as pd


            class Ledger:
                def __init__(self, entries):
                    self.balances = pd.Series(entries)

                def report(self):
                    lines = []
                    for account, amount in self.balances.iteritems():
                        lines.append(account + ":" + str(int(amount)))
                    return ";".join(lines)
            """,
            """
            import pandas as pd


            class Ledger:
                def __init__(self, entries):
                    self.balances = pd.Series(entries)

                def report(self):
                    lines = []
                    for account, amount in self.balances.items():
                        lines.append(account + ":" + str(int(amount)))
                    return ";".join(lines)
            """,
            """
            import solution

            ledger = solution.Ledger({"cash": 100, "bank": 250})
            assert ledger.report() == "cash:100;bank:250"
            """,
            "Render account balances as a compact string.",
        ),
        _p(
            "argmax_columns",
            """
            import pandas as pd


            def max_positions(df):
                positions = {name: int(col.idxmax()) for name, col in df.iteritems()}
                return positions
            """,
            """
            import pandas as pd


            def max_positions(df):
                positions = {name: int(col.idxmax()) for name, col in df.items()}
                return positions
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"p": [1, 9, 3], "q": [7, 2, 1]})
            assert solution.max_positions(df) == {"p": 1, "q": 0}
            """,
            "Row position of the maximum in each column.",
        ),
    ],
    [
        _s(
            "iterrows_total",
            """
            import pandas as pd


            def row_totals(df):
                return [float(row.sum()) for _, row in df.iterrows()]
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
            assert solution.row_totals(df) == [11.0, 22.0]
            """,
            "DataFrame.iterrows is unchanged.",
        ),
        _s(
            "itertuples_pairs",
            """
            import pandas as pd


            def as_pairs(df):
                return [(t.a, t.b) for t in df.itertuples(index=False)]
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            assert solution.as_pairs(df) == [(1, "x"), (2, "y")]
            """,
            "DataFrame.itertuples is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: DataFrame.lookup removed (2.0) -> numpy fancy indexing
# ---------------------------------------------------------------------------

LOOKUP = _pandas_change(
    "pandas.DataFrame.lookup",
    "DataFrame.lookup(row_labels, col_labels)",
    "df.to_numpy()[df.index.get_indexer(rows), df.columns.get_indexer(cols)]",
    "DataFrame.lookup was removed in pandas 2.0; use numpy fancy indexing on to_numpy().",
    [
        _p(
            "pick_cells",
            """
            import pandas as pd


            def pick_cells(df, rows, cols):
                values = df.lookup(rows, cols)
                return values.tolist()
            """,
            """
            import pandas as pd


            def pick_cells(df, rows, cols):
                ri = df.index.get_indexer(rows)
                ci = df.columns.get_indexer(cols)
                return df.to_numpy()[ri, ci].tolist()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["x", "y", "z"])
            assert solution.pick_cells(df, ["x", "z", "y"], ["b", "a", "b"]) == [10, 3, 20]
            """,
            "Pick one cell per (row label, column label) pair.",
        ),
        _p(
            "chosen_price",
            """
            import pandas as pd


            def chosen_prices(df):
                picked = df.lookup(df.index, df["choice"])
                return [float(v) for v in picked]
            """,
            """
            import pandas as pd


            def chosen_prices(df):
                picked = [df.at[i, c] for i, c in zip(df.index, df["choice"])]
                return [float(v) for v in picked]
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({
                "basic": [5.0, 6.0, 7.0],
                "premium": [15.0, 16.0, 17.0],
                "choice": ["basic", "premium", "basic"],
            })
            assert solution.chosen_prices(df) == [5.0, 16.0, 7.0]
            """,
            "Read the price from the column named in a per-row 'choice' column.",
        ),
        _p(
            "grade_table",
            """
            import pandas as pd


            class GradeBook:
                def __init__(self, table):
                    self.table = pd.DataFrame(table)

                def grades(self, students, subjects):
                    return [int(g) for g in self.table.lookup(students, subjects)]
            """,
            """
            import pandas as pd


            class GradeBook:
                def __init__(self, table):
                    self.table = pd.DataFrame(table)

                def grades(self, students, subjects):
                    ri = self.table.index.get_indexer(students)
                    ci = self.table.columns.get_indexer(subjects)
                    return [int(g) for g in self.table.to_numpy()[ri, ci]]
            """,
            """
            import solution

            book = solution.GradeBook({"math": {"ann": 90, "bob": 70}, "art": {"ann": 60, "bob": 85}})
            assert book.grades(["ann", "bob"], ["art", "math"]) == [60, 70]
            assert book.grades(["bob"], ["art"]) == [85]
            """,
            "Look up grades for student/subject pairs.",
        ),
        _p(
            "matrix_diagonal",
            """
            import pandas as pd


            def diagonal(df):
                values = df.lookup(df.index, df.columns)
                return values.tolist()
            """,
            """
            import pandas as pd


            def diagonal(df):
                values = df.to_numpy()
                n = len(df)
                return values[range(n), range(n)].tolist()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["r", "s", "t"], columns=["c1", "c2", "c3"])
            assert solution.diagonal(df) == [1, 5, 9]
            """,
            "Main diagonal of a square labelled frame.",
        ),
        _p(
            "select_by_map",
            """
            import pandas as pd


            def select(df, mapping):
                rows = list(mapping.keys())
                cols = [mapping[r] for r in rows]
                return dict(zip(rows, df.lookup(rows, cols).tolist()))
            """,
            """
            import pandas as pd


            def select(df, mapping):
                rows = list(mapping.keys())
                cols = [mapping[r] for r in rows]
                values = df.to_numpy()[df.index.get_indexer(rows), df.columns.get_indexer(cols)]
                return dict(zip(rows, values.tolist()))
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"lo": [1, 2], "hi": [9, 8]}, index=[10, 20])
            assert solution.select(df, {10: "hi", 20: "lo"}) == {10: 9, 20: 2}
            """,
            "Select one cell per row according to a row->column mapping.",
        ),
        _p(
            "realized_value",
            """
            import pandas as pd


            def realized(df, column):
                months = df[column].tolist()
                return float(sum(df.lookup(df.index, months)))
            """,
            """
            import pandas as pd


            def realized(df, column):
                months = df[column].tolist()
                values = df.to_numpy()[range(len(df)), df.columns.get_indexer(months)]
                return float(sum(values))
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({
                "jan": [1, 2, 3],
                "feb": [10, 20, 30],
                "realized_in": ["jan", "feb", "feb"],
            })
            assert solution.realized(df, "realized_in") == 51.0
            """,
            "Sum the value realised in the month named per row.",
        ),
    ],
    [
        _s(
            "at_lookup",
            """
            import pandas as pd


            def cell(df, row, col):
                return df.at[row, col]
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
            assert solution.cell(df, "y", "a") == 2
            """,
            "DataFrame.at scalar access is unchanged.",
        ),
        _s(
            "loc_block",
            """
            import pandas as pd


            def block(df, rows, cols):
                return df.loc[rows, cols].to_numpy().tolist()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
            assert solution.block(df, ["z", "x"], ["b"]) == [[6], [4]]
            """,
            "DataFrame.loc list indexing is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: Series.is_monotonic / Index.is_monotonic removed (2.0)
# ---------------------------------------------------------------------------

IS_MONOTONIC = _pandas_change(
    "pandas.Series.is_monotonic",
    "Series.is_monotonic / Index.is_monotonic",
    "is_monotonic_increasing",
    "Series.is_monotonic and Index.is_monotonic were removed in pandas 2.0; use is_monotonic_increasing.",
    [
        _p(
            "is_sorted",
            """
            import pandas as pd


            def is_sorted(values):
                s = pd.Series(values)
                return bool(s.is_monotonic)
            """,
            """
            import pandas as pd


            def is_sorted(values):
                s = pd.Series(values)
                return bool(s.is_monotonic_increasing)
            """,
            """
            import solution

            assert solution.is_sorted([1, 2, 2, 5]) is True
            assert solution.is_sorted([1, 3, 2]) is False
            """,
            "Check whether a list is non-decreasing.",
        ),
        _p(
            "time_index_check",
            """
            import pandas as pd


            def index_ordered(df):
                idx = df.index
                return bool(idx.is_monotonic)
            """,
            """
            import pandas as pd


            def index_ordered(df):
                idx = df.index
                return bool(idx.is_monotonic_increasing)
            """,
            """
            import pandas as pd
            import solution

            ok = pd.DataFrame({"v": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
            bad = pd.DataFrame({"v": [1, 2]}, index=pd.to_datetime(["2020-01-02", "2020-01-01"]))
            assert solution.index_ordered(ok) is True
            assert solution.index_ordered(bad) is False
            """,
            "Check that a frame's datetime index is ordered.",
        ),
        _p(
            "validator",
            """
            import pandas as pd


            class SeriesValidator:
                def __init__(self, s):
                    self.s = s

                def problems(self):
                    out = []
                    if not self.s.is_monotonic:
                        out.append("not_sorted")
                    if self.s.isna().any():
                        out.append("has_nan")
                    return out
            """,
            """
            import pandas as pd


            class SeriesValidator:
                def __init__(self, s):
                    self.s = s

                def problems(self):
                    out = []
                    if not self.s.is_monotonic_increasing:
                        out.append("not_sorted")
                    if self.s.isna().any():
                        out.append("has_nan")
                    return out
            """,
            """
            import pandas as pd
            import solution

            assert solution.SeriesValidator(pd.Series([1.0, 2.0])).problems() == []
            assert solution.SeriesValidator(pd.Series([2.0, 1.0, None])).problems() == ["not_sorted", "has_nan"]
            """,
            "Collect validation problems for a series.",
        ),
        _p(
            "monotone_columns",
            """
            import pandas as pd


            def monotone_columns(df):
                cols = [c for c in df.columns if df[c].is_monotonic]
                return cols
            """,
            """
            import pandas as pd


            def monotone_columns(df):
                cols = [c for c in df.columns if df[c].is_monotonic_increasing]
                return cols
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"up": [1, 2, 3], "down": [3, 2, 1], "flat": [1, 1, 1]})
            assert solution.monotone_columns(df) == ["up", "flat"]
            """,
            "List the non-decreasing columns of a frame.",
        ),
        _p(
            "require_sorted_index",
            """
            import pandas as pd


            def sorted_lookup(idx_values, key):
                idx = pd.Index(idx_values)
                if not idx.is_monotonic:
                    raise ValueError("index must be sorted")
                return int(idx.get_loc(key))
            """,
            """
            import pandas as pd


            def sorted_lookup(idx_values, key):
                idx = pd.Index(idx_values)
                if not idx.is_monotonic_increasing:
                    raise ValueError("index must be sorted")
                return int(idx.get_loc(key))
            """,
            """
            import solution

            assert solution.sorted_lookup([10, 20, 30], 20) == 1
            try:
                solution.sorted_lookup([30, 10], 10)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """,
            "Position lookup that insists on a sorted index.",
        ),
        _p(
            "first_break",
            """
            import pandas as pd


            def first_break(values):
                s = pd.Series(values)
                if s.is_monotonic:
                    return -1
                drops = s.diff() < 0
                return int(drops.idxmax())
            """,
            """
            import pandas as pd


            def first_break(values):
                s = pd.Series(values)
                if s.is_monotonic_increasing:
                    return -1
                drops = s.diff() < 0
                return int(drops.idxmax())
            """,
            """
            import solution

            assert solution.first_break([1, 2, 3]) == -1
            assert solution.first_break([1, 5, 4, 9]) == 2
            """,
            "Position of the first decrease in a sequence, or -1.",
        ),
    ],
    [
        _s(
            "decreasing_check",
            """
            import pandas as pd


            def is_descending(values):
                return bool(pd.Series(values).is_monotonic_decreasing)
            """,
            """
            import solution

            assert solution.is_descending([5, 3, 3, 1]) is True
            assert solution.is_descending([1, 2]) is False
            """,
            "is_monotonic_decreasing is unchanged.",
        ),
        _s(
            "unique_check",
            """
            import pandas as pd


            def all_unique(values):
                return bool(pd.Index(values).is_unique)
            """,
            """
            import solution

            assert solution.all_unique([1, 2, 3]) is True
            assert solution.all_unique([1, 1]) is False
            """,
            "Index.is_unique is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: DataFrame.mad / Series.mad removed (2.0)
# ---------------------------------------------------------------------------

MAD = _pandas_change(
    "pandas.DataFrame.mad",
    "DataFrame.mad() / Series.mad()",
    "(df - df.mean()).abs().mean()",
    "DataFrame.mad, Series.mad and GroupBy.mad were removed in pandas 2.0; compute (x - x.mean()).abs().mean().",
    [
        _p(
            "dispersion",
            """
            import pandas as pd


            def dispersion(values):
                s = pd.Series(values)
                return float(s.mad())
            """,
            """
            import pandas as pd


            def dispersion(values):
                s = pd.Series(values)
                return float((s - s.mean()).abs().mean())
            """,
            """
            import solution

            assert solution.dispersion([1.0, 2.0, 3.0, 4.0]) == 1.0
            assert solution.dispersion([5.0, 5.0]) == 0.0
            """,
            "Mean absolute deviation of a list.",
        ),
        _p(
            "column_mad",
            """
            import pandas as pd


            def column_mad(df):
                mad = df.mad()
                return {k: round(float(v), 6) for k, v in mad.items()}
            """,
            """
            import pandas as pd


            def column_mad(df):
                mad = (df - df.mean()).abs().mean()
                return {k: round(float(v), 6) for k, v in mad.items()}
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 10.0, ], "c": [0.0, 2.0]})
            assert solution.column_mad(df) == {"a": 1.0, "b": 0.0, "c": 1.0}
            """,
            "Per-column mean absolute deviation as a dict.",
        ),
        _p(
            "row_spread",
            """
            import pandas as pd


            class Panel:
                def __init__(self, rows):
                    self.df = pd.DataFrame(rows)

                def row_spread(self):
                    return self.df.mad(axis=1).round(6).tolist()
            """,
            """
            import pandas as pd


            class Panel:
                def __init__(self, rows):
                    self.df = pd.DataFrame(rows)

                def row_spread(self):
                    centered = self.df.sub(self.df.mean(axis=1), axis=0)
                    return centered.abs().mean(axis=1).round(6).tolist()
            """,
            """
            import solution

            panel = solution.Panel([[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
            assert panel.row_spread() == [1.0, 0.0, 2.0]
            """,
            "Row-wise mean absolute deviation.",
        ),
        _p(
            "mad_ratio",
            """
            import pandas as pd


            def mad_ratio(df, numerator, denominator):
                mads = df.mad()
                return round(float(mads[numerator] / mads[denominator]), 6)
            """,
            """
            import pandas as pd


            def mad_ratio(df, numerator, denominator):
                mads = (df - df.mean()).abs().mean()
                return round(float(mads[numerator] / mads[denominator]), 6)
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"x": [0.0, 4.0], "y": [0.0, 1.0]})
            assert solution.mad_ratio(df, "x", "y") == 4.0
            """,
            "Ratio of two columns' mean absolute deviations.",
        ),
        _p(
            "robust_zscore",
            """
            import pandas as pd


            def scaled(values):
                s = pd.Series(values)
                return ((s - s.mean()) / s.mad()).round(6).tolist()
            """,
            """
            import pandas as pd


            def scaled(values):
                s = pd.Series(values)
                mad = (s - s.mean()).abs().mean()
                return ((s - s.mean()) / mad).round(6).tolist()
            """,
            """
            import solution

            assert solution.scaled([1.0, 3.0]) == [-1.0, 1.0]
            assert solution.scaled([0.0, 2.0, 4.0]) == [-1.5, 0.0, 1.5]
            """,
            "Scale values by their mean absolute deviation.",
        ),
        _p(
            "grouped_mad",
            """
            import pandas as pd


            def group_dispersion(df):
                mads = df.groupby("g")["x"].mad()
                return mads.round(6).to_dict()
            """,
            """
            import pandas as pd


            def group_dispersion(df):
                def _mad(s):
                    return (s - s.mean()).abs().mean()
                return df.groupby("g")["x"].apply(_mad).round(6).to_dict()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
            assert solution.group_dispersion(df) == {"a": 1.0, "b": 0.0}
            """,
            "Mean absolute deviation per group.",
        ),
    ],
    [
        _s(
            "population_std",
            """
            import pandas as pd


            def spread(values):
                return float(pd.Series(values).std(ddof=0))
            """,
            """
            import solution

            assert solution.spread([1.0, 3.0]) == 1.0
            """,
            "Series.std is unchanged.",
        ),
        _s(
            "column_median",
            """
            import pandas as pd


            def medians(df):
                return df.median().to_dict()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"a": [1.0, 2.0, 9.0], "b": [4.0, 0.0, 2.0]})
            assert solution.medians(df) == {"a": 2.0, "b": 2.0}
            """,
            "DataFrame.median is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: Int64Index / Float64Index removed (2.0) -> pd.Index(dtype=...)
# ---------------------------------------------------------------------------

NUMERIC_INDEX = _pandas_change(
    "pandas.Int64Index",
    "pd.Int64Index / pd.Float64Index",
    "pd.Index(values, dtype=...)",
    "pandas.Int64Index, UInt64Index and Float64Index were removed in pandas 2.0; use pandas.Index with a dtype.",
    [
        _p(
            "make_int_index",
            """
            import pandas as pd


            def make_index(values):
                idx = pd.Int64Index(values)
                return idx
            """,
            """
            import pandas as pd


            def make_index(values):
                idx = pd.Index(values, dtype="int64")
                return idx
            """,
            """
            import solution

            idx = solution.make_index([3, 1, 2])
            assert list(idx) == [3, 1, 2]
            assert str(idx.dtype) == "int64"
            """,
            "Build an int64 index from a list.",
        ),
        _p(
            "bucket_edges",
            """
            import numpy as np
            import pandas as pd


            def edges(lo, hi, n):
                return pd.Float64Index(np.linspace(lo, hi, n))
            """,
            """
            import numpy as np
            import pandas as pd


            def edges(lo, hi, n):
                return pd.Index(np.linspace(lo, hi, n), dtype="float64")
            """,
            """
            import solution

            idx = solution.edges(0.0, 1.0, 3)
            assert list(idx) == [0.0, 0.5, 1.0]
            assert str(idx.dtype) == "float64"
            """,
            "Evenly spaced float bucket edges as an index.",
        ),
        _p(
            "id_registry",
            """
            import pandas as pd


            class Registry:
                def __init__(self, ids):
                    self.ids = pd.Int64Index(sorted(set(ids)))

                def contains(self, value):
                    return value in self.ids

                def position(self, value):
                    return int(self.ids.get_loc(value))
            """,
            """
            import pandas as pd


            class Registry:
                def __init__(self, ids):
                    self.ids = pd.Index(sorted(set(ids)), dtype="int64")

                def contains(self, value):
                    return value in self.ids

                def position(self, value):
                    return int(self.ids.get_loc(value))
            """,
            """
            import solution

            reg = solution.Registry([30, 10, 20, 10])
            assert reg.contains(20) is True
            assert reg.contains(25) is False
            assert reg.position(30) == 2
            """,
            "Sorted integer id registry with membership and position.",
        ),
        _p(
            "integer_index_check",
            """
            import pandas as pd


            def has_integer_index(df):
                idx = df.index
                return isinstance(idx, pd.Int64Index)
            """,
            """
            import pandas as pd


            def has_integer_index(df):
                idx = df.index
                return bool(pd.api.types.is_integer_dtype(idx))
            """,
            """
            import pandas as pd
            import solution

            ints = pd.DataFrame({"v": [1, 2]}, index=pd.Index([5, 6], dtype="int64"))
            labels = pd.DataFrame({"v": [1, 2]}, index=["a", "b"])
            assert solution.has_integer_index(ints) is True
            assert solution.has_integer_index(labels) is False
            """,
            "Detect whether a frame has an integer-typed index.",
        ),
        _p(
            "float_keyed_frame",
            """
            import pandas as pd


            def keyed(rows):
                keys = pd.Float64Index([r[0] for r in rows])
                return pd.DataFrame({"v": [r[1] for r in rows]}, index=keys)
            """,
            """
            import pandas as pd


            def keyed(rows):
                keys = pd.Index([r[0] for r in rows], dtype="float64")
                return pd.DataFrame({"v": [r[1] for r in rows]}, index=keys)
            """,
            """
            import solution

            df = solution.keyed([(0.5, "a"), (1.5, "b")])
            assert df.loc[1.5, "v"] == "b"
            assert str(df.index.dtype) == "float64"
            """,
            "Frame keyed by float values.",
        ),
        _p(
            "union_ids",
            """
            import pandas as pd


            def union_ids(a, b):
                merged = pd.Int64Index(a).union(pd.Int64Index(b))
                return merged.tolist()
            """,
            """
            import pandas as pd


            def union_ids(a, b):
                merged = pd.Index(a, dtype="int64").union(pd.Index(b, dtype="int64"))
                return merged.tolist()
            """,
            """
            import solution

            assert solution.union_ids([3, 1], [2, 3]) == [1, 2, 3]
            """,
            "Sorted union of two integer id lists.",
        ),
    ],
    [
        _s(
            "range_index",
            """
            import pandas as pd


            def positions(n, step):
                return pd.RangeIndex(0, n, step).tolist()
            """,
            """
            import solution

            assert solution.positions(10, 3) == [0, 3, 6, 9]
            """,
            "pd.RangeIndex is unchanged.",
        ),
        _s(
            "multi_index",
            """
            import pandas as pd


            def pairs(a, b):
                return pd.MultiIndex.from_product([a, b]).tolist()
            """,
            """
            import solution

            assert solution.pairs([1, 2], ["x"]) == [(1, "x"), (2, "x")]
            """,
            "pd.MultiIndex.from_product is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: read_csv(error_bad_lines=, warn_bad_lines=) removed (2.0) -> on_bad_lines
# ---------------------------------------------------------------------------

ERROR_BAD_LINES = _pandas_change(
    "pandas.read_csv.error_bad_lines",
    "read_csv(error_bad_lines=False)",
    'read_csv(on_bad_lines="skip")',
    "read_csv's error_bad_lines and warn_bad_lines arguments were removed in pandas 2.0; use on_bad_lines.",
    [
        _p(
            "parse_lenient",
            """
            import io
            import pandas as pd


            def parse_lenient(text):
                return pd.read_csv(io.StringIO(text), error_bad_lines=False)
            """,
            """
            import io
            import pandas as pd


            def parse_lenient(text):
                return pd.read_csv(io.StringIO(text), on_bad_lines="skip")
            """,
            """
            import solution

            text = "a,b\\n1,2\\n3,4,5\\n6,7\\n"
            df = solution.parse_lenient(text)
            assert list(df.columns) == ["a", "b"]
            assert df["a"].tolist() == [1, 6]
            assert df["b"].tolist() == [2, 7]
            """,
            "Parse CSV text, skipping rows with too many fields.",
        ),
        _p(
            "quiet_loader",
            """
            import io
            import pandas as pd


            def load_records(text):
                df = pd.read_csv(io.StringIO(text), error_bad_lines=False, warn_bad_lines=False)
                return df.to_dict("records")
            """,
            """
            import io
            import pandas as pd


            def load_records(text):
                df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
                return df.to_dict("records")
            """,
            """
            import solution

            text = "name,score\\nann,1\\nbob,2,extra\\ncid,3\\n"
            assert solution.load_records(text) == [{"name": "ann", "score": 1}, {"name": "cid", "score": 3}]
            """,
            "Load CSV records, silently dropping malformed lines.",
        ),
        _p(
            "strict_or_lenient",
            """
            import io
            import pandas as pd


            class CsvLoader:
                def __init__(self, strict):
                    self.strict = strict

                def load(self, text):
                    return pd.read_csv(io.StringIO(text), error_bad_lines=self.strict)

                def row_count(self, text):
                    return int(len(self.load(text)))
            """,
            """
            import io
            import pandas as pd


            class CsvLoader:
                def __init__(self, strict):
                    self.strict = strict

                def load(self, text):
                    mode = "error" if self.strict else "skip"
                    return pd.read_csv(io.StringIO(text), on_bad_lines=mode)

                def row_count(self, text):
                    return int(len(self.load(text)))
            """,
            """
            import solution

            good = "x,y\\n1,2\\n3,4\\n"
            bad = "x,y\\n1,2\\n3,4,5\\n"
            assert solution.CsvLoader(False).row_count(bad) == 1
            assert solution.CsvLoader(True).row_count(good) == 2
            try:
                solution.CsvLoader(True).load(bad)
                raise AssertionError("expected a parser error")
            except AssertionError:
                raise
            except Exception:
                pass
            """,
            "Loader that is either strict or skips bad lines.",
        ),
        _p(
            "count_valid",
            """
            import io
            import pandas as pd


            def count_valid_rows(text):
                df = pd.read_csv(io.StringIO(text), error_bad_lines=False)
                return int(df.dropna().shape[0])
            """,
            """
            import io
            import pandas as pd


            def count_valid_rows(text):
                df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
                return int(df.dropna().shape[0])
            """,
            """
            import solution

            text = "a,b\\n1,2\\n3\\n4,5,6\\n7,8\\n"
            assert solution.count_valid_rows(text) == 2
            """,
            "Count fully populated rows after skipping overlong lines.",
        ),
        _p(
            "sum_column",
            """
            import io
            import pandas as pd


            def sum_column(text, column):
                df = pd.read_csv(io.StringIO(text), error_bad_lines=False)
                return float(df[column].sum())
            """,
            """
            import io
            import pandas as pd


            def sum_column(text, column):
                df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
                return float(df[column].sum())
            """,
            """
            import solution

            text = "k,amount\\np,1.5\\nq,2.5,oops\\nr,4.0\\n"
            assert solution.sum_column(text, "amount") == 5.5
            """,
            "Sum a column of a CSV with malformed lines.",
        ),
        _p(
            "tsv_reader",
            """
            import io
            import pandas as pd


            def read_tsv(text):
                return pd.read_csv(io.StringIO(text), sep="\\t", error_bad_lines=False)
            """,
            """
            import io
            import pandas as pd


            def read_tsv(text):
                return pd.read_csv(io.StringIO(text), sep="\\t", on_bad_lines="skip")
            """,
            """
            import solution

            text = "id\\tval\\n1\\ta\\n2\\tb\\tc\\n3\\td\\n"
            df = solution.read_tsv(text)
            assert df["id"].tolist() == [1, 3]
            assert df["val"].tolist() == ["a", "d"]
            """,
            "Read tab-separated text, skipping bad lines.",
        ),
    ],
    [
        _s(
            "skip_rows",
            """
            import io
            import pandas as pd


            def read_after_preamble(text, n):
                return pd.read_csv(io.StringIO(text), skiprows=n)
            """,
            """
            import solution

            text = "junk\\nmore junk\\na,b\\n1,2\\n"
            df = solution.read_after_preamble(text, 2)
            assert list(df.columns) == ["a", "b"]
            assert df["b"].tolist() == [2]
            """,
            "read_csv(skiprows=) is unchanged.",
        ),
        _s(
            "use_cols",
            """
            import io
            import pandas as pd


            def read_subset(text, cols):
                return pd.read_csv(io.StringIO(text), usecols=cols)
            """,
            """
            import solution

            df = solution.read_subset("a,b,c\\n1,2,3\\n", ["a", "c"])
            assert list(df.columns) == ["a", "c"]
            assert df.iloc[0].tolist() == [1, 3]
            """,
            "read_csv(usecols=) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: read_csv(squeeze=True) removed (2.0) -> .squeeze("columns")
# ---------------------------------------------------------------------------

READ_SQUEEZE = _pandas_change(
    "pandas.read_csv.squeeze",
    "read_csv(squeeze=True)",
    'read_csv(...).squeeze("columns")',
    "read_csv's squeeze argument was removed in pandas 2.0; call .squeeze('columns') on the result.",
    [
        _p(
            "load_series",
            """
            import io
            import pandas as pd


            def load_series(text):
                return pd.read_csv(io.StringIO(text), squeeze=True)
            """,
            """
            import io
            import pandas as pd


            def load_series(text):
                return pd.read_csv(io.StringIO(text)).squeeze("columns")
            """,
            """
            import pandas as pd
            import solution

            s = solution.load_series("temp\\n1.5\\n2.5\\n")
            assert isinstance(s, pd.Series)
            assert s.name == "temp"
            assert s.tolist() == [1.5, 2.5]
            """,
            "Load a one-column CSV as a Series.",
        ),
        _p(
            "indexed_series",
            """
            import io
            import pandas as pd


            def load_indexed(text):
                return pd.read_csv(io.StringIO(text), index_col=0, squeeze=True)
            """,
            """
            import io
            import pandas as pd


            def load_indexed(text):
                return pd.read_csv(io.StringIO(text), index_col=0).squeeze("columns")
            """,
            """
            import pandas as pd
            import solution

            s = solution.load_indexed("key,value\\na,1\\nb,2\\n")
            assert isinstance(s, pd.Series)
            assert s.to_dict() == {"a": 1, "b": 2}
            assert s.name == "value"
            """,
            "Load a key/value CSV into a labelled Series.",
        ),
        _p(
            "headerless_values",
            """
            import io
            import pandas as pd


            def read_values(text):
                s = pd.read_csv(io.StringIO(text), header=None, squeeze=True)
                return s.tolist()
            """,
            """
            import io
            import pandas as pd


            def read_values(text):
                s = pd.read_csv(io.StringIO(text), header=None).squeeze("columns")
                return s.tolist()
            """,
            """
            import solution

            assert solution.read_values("4\\n5\\n6\\n") == [4, 5, 6]
            """,
            "Read a header-less single column into a list.",
        ),
        _p(
            "column_reader",
            """
            import io
            import pandas as pd


            class ColumnReader:
                def __init__(self, sep):
                    self.sep = sep

                def column(self, text):
                    return pd.read_csv(io.StringIO(text), sep=self.sep, squeeze=True)

                def maximum(self, text):
                    return float(self.column(text).max())
            """,
            """
            import io
            import pandas as pd


            class ColumnReader:
                def __init__(self, sep):
                    self.sep = sep

                def column(self, text):
                    return pd.read_csv(io.StringIO(text), sep=self.sep).squeeze("columns")

                def maximum(self, text):
                    return float(self.column(text).max())
            """,
            """
            import pandas as pd
            import solution

            reader = solution.ColumnReader(";")
            assert reader.maximum("score\\n3\\n9\\n4\\n") == 9.0
            assert isinstance(reader.column("score\\n3\\n"), pd.Series)
            """,
            "Reader class that squeezes a one-column file.",
        ),
        _p(
            "top_value",
            """
            import io
            import pandas as pd


            def top_label(text):
                s = pd.read_csv(io.StringIO(text), index_col=0, squeeze=True)
                return s.idxmax()
            """,
            """
            import io
            import pandas as pd


            def top_label(text):
                s = pd.read_csv(io.StringIO(text), index_col=0).squeeze("columns")
                return s.idxmax()
            """,
            """
            import solution

            assert solution.top_label("city,pop\\nx,10\\ny,30\\nz,20\\n") == "y"
            """,
            "Label of the largest value in a key/value CSV.",
        ),
        _p(
            "series_stats",
            """
            import io
            import pandas as pd


            def stats(text):
                s = pd.read_csv(io.StringIO(text), squeeze=True)
                return {"n": int(s.size), "mean": float(s.mean()), "name": s.name}
            """,
            """
            import io
            import pandas as pd


            def stats(text):
                s = pd.read_csv(io.StringIO(text)).squeeze("columns")
                return {"n": int(s.size), "mean": float(s.mean()), "name": s.name}
            """,
            """
            import solution

            assert solution.stats("w\\n2\\n4\\n") == {"n": 2, "mean": 3.0, "name": "w"}
            """,
            "Summary statistics of a single-column CSV.",
        ),
    ],
    [
        _s(
            "select_column",
            """
            import io
            import pandas as pd


            def column_values(text, name):
                return pd.read_csv(io.StringIO(text))[name].tolist()
            """,
            """
            import solution

            assert solution.column_values("a,b\\n1,2\\n3,4\\n", "b") == [2, 4]
            """,
            "Selecting a column from read_csv output is unchanged.",
        ),
        _s(
            "frame_squeeze",
            """
            import pandas as pd


            def as_series(df):
                return df.squeeze(axis=1)
            """,
            """
            import pandas as pd
            import solution

            s = solution.as_series(pd.DataFrame({"only": [1, 2]}))
            assert isinstance(s, pd.Series)
            assert s.tolist() == [1, 2]
            """,
            "DataFrame.squeeze(axis=1) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: pd.datetime removed (2.0) -> datetime.datetime
# ---------------------------------------------------------------------------

PD_DATETIME = _pandas_change(
    "pandas.datetime",
    "pd.datetime",
    "datetime.datetime",
    "The pandas.datetime alias was removed in pandas 2.0; import datetime from the standard library.",
    [
        _p(
            "make_stamp",
            """
            import pandas as pd


            def make_stamp(year, month, day):
                stamp = pd.datetime(year, month, day)
                return stamp.isoformat()
            """,
            """
            import datetime
            import pandas as pd


            def make_stamp(year, month, day):
                stamp = datetime.datetime(year, month, day)
                return stamp.isoformat()
            """,
            """
            import solution

            assert solution.make_stamp(2021, 3, 4) == "2021-03-04T00:00:00"
            """,
            "ISO timestamp from year/month/day.",
        ),
        _p(
            "parse_date",
            """
            import pandas as pd


            def parse_date(text):
                d = pd.datetime.strptime(text, "%d/%m/%Y")
                return (d.year, d.month, d.day)
            """,
            """
            import datetime
            import pandas as pd


            def parse_date(text):
                d = datetime.datetime.strptime(text, "%d/%m/%Y")
                return (d.year, d.month, d.day)
            """,
            """
            import solution

            assert solution.parse_date("05/11/2019") == (2019, 11, 5)
            """,
            "Parse a day/month/year string.",
        ),
        _p(
            "days_between",
            """
            import pandas as pd


            def days_between(a, b):
                start = pd.datetime(*a)
                end = pd.datetime(*b)
                return (end - start).days
            """,
            """
            import datetime
            import pandas as pd


            def days_between(a, b):
                start = datetime.datetime(*a)
                end = datetime.datetime(*b)
                return (end - start).days
            """,
            """
            import solution

            assert solution.days_between((2020, 1, 1), (2020, 3, 1)) == 60
            """,
            "Whole days between two calendar dates.",
        ),
        _p(
            "deadline",
            """
            import pandas as pd


            class Deadline:
                def __init__(self, year, month, day):
                    self.when = pd.datetime(year, month, day)

                def overdue_on(self, year, month, day):
                    return pd.datetime(year, month, day) > self.when

                def label(self):
                    return self.when.strftime("%Y%m%d")
            """,
            """
            import datetime
            import pandas as pd


            class Deadline:
                def __init__(self, year, month, day):
                    self.when = datetime.datetime(year, month, day)

                def overdue_on(self, year, month, day):
                    return datetime.datetime(year, month, day) > self.when

                def label(self):
                    return self.when.strftime("%Y%m%d")
            """,
            """
            import solution

            d = solution.Deadline(2022, 6, 30)
            assert d.overdue_on(2022, 7, 1) is True
            assert d.overdue_on(2022, 6, 30) is False
            assert d.label() == "20220630"
            """,
            "Deadline object with overdue check and label.",
        ),
        _p(
            "weekend_check",
            """
            import pandas as pd


            def is_weekend(year, month, day):
                day_of_week = pd.datetime(year, month, day).weekday()
                return day_of_week >= 5
            """,
            """
            import datetime
            import pandas as pd


            def is_weekend(year, month, day):
                day_of_week = datetime.datetime(year, month, day).weekday()
                return day_of_week >= 5
            """,
            """
            import solution

            assert solution.is_weekend(2023, 1, 7) is True
            assert solution.is_weekend(2023, 1, 9) is False
            """,
            "Whether a date falls on a weekend.",
        ),
        _p(
            "series_of_dates",
            """
            import pandas as pd


            def month_starts(year, months):
                dates = [pd.datetime(year, m, 1) for m in months]
                return pd.Series(dates).dt.strftime("%b").tolist()
            """,
            """
            import datetime
            import pandas as pd


            def month_starts(year, months):
                dates = [datetime.datetime(year, m, 1) for m in months]
                return pd.Series(dates).dt.strftime("%b").tolist()
            """,
            """
            import solution

            assert solution.month_starts(2020, [1, 4, 12]) == ["Jan", "Apr", "Dec"]
            """,
            "Month abbreviations for the first of given months.",
        ),
    ],
    [
        _s(
            "timestamp_build",
            """
            import pandas as pd


            def stamp(year, month, day):
                return pd.Timestamp(year=year, month=month, day=day).isoformat()
            """,
            """
            import solution

            assert solution.stamp(2021, 3, 4) == "2021-03-04T00:00:00"
            """,
            "pd.Timestamp is unchanged.",
        ),
        _s(
            "to_datetime_parse",
            """
            import pandas as pd


            def parse(text):
                d = pd.to_datetime(text, format="%d/%m/%Y")
                return (d.year, d.month, d.day)
            """,
            """
            import solution

            assert solution.parse("05/11/2019") == (2019, 11, 5)
            """,
            "pd.to_datetime with an explicit format is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: Series.between(inclusive=<bool>) removed (2.0) -> "both"/"neither"
# ---------------------------------------------------------------------------

BETWEEN = _pandas_change(
    "pandas.Series.between.inclusive",
    "Series.between(lo, hi, inclusive=True/False)",
    'Series.between(lo, hi, inclusive="both"/"neither")',
    "Boolean values for Series.between's inclusive argument were removed in pandas 2.0; use 'both', 'neither', 'left' or 'right'.",
    [
        _p(
            "in_range",
            """
            import pandas as pd


            def in_range(values, lo, hi):
                mask = pd.Series(values).between(lo, hi, inclusive=True)
                return mask.tolist()
            """,
            """
            import pandas as pd


            def in_range(values, lo, hi):
                mask = pd.Series(values).between(lo, hi, inclusive="both")
                return mask.tolist()
            """,
            """
            import solution

            assert solution.in_range([1, 2, 3, 4], 2, 3) == [False, True, True, False]
            """,
            "Inclusive range mask over a list.",
        ),
        _p(
            "strictly_inside",
            """
            import pandas as pd


            def strictly_inside(values, lo, hi):
                s = pd.Series(values)
                return int(s.between(lo, hi, inclusive=False).sum())
            """,
            """
            import pandas as pd


            def strictly_inside(values, lo, hi):
                s = pd.Series(values)
                return int(s.between(lo, hi, inclusive="neither").sum())
            """,
            """
            import solution

            assert solution.strictly_inside([1, 2, 3, 4, 5], 1, 5) == 3
            assert solution.strictly_inside([1, 5], 1, 5) == 0
            """,
            "Count values strictly between two bounds.",
        ),
        _p(
            "band_count",
            """
            import pandas as pd


            def count_in_band(df, column, lo, hi):
                mask = df[column].between(lo, hi, inclusive=True)
                return int(mask.sum())
            """,
            """
            import pandas as pd


            def count_in_band(df, column, lo, hi):
                mask = df[column].between(lo, hi, inclusive="both")
                return int(mask.sum())
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"t": [10.0, 20.0, 30.0, 40.0]})
            assert solution.count_in_band(df, "t", 20.0, 30.0) == 2
            """,
            "Count rows whose column lies in a closed band.",
        ),
        _p(
            "range_filter",
            """
            import pandas as pd


            class RangeFilter:
                def __init__(self, column, lo, hi, closed):
                    self.column = column
                    self.lo = lo
                    self.hi = hi
                    self.closed = closed

                def apply(self, df):
                    mask = df[self.column].between(self.lo, self.hi, inclusive=self.closed)
                    return df[mask].reset_index(drop=True)
            """,
            """
            import pandas as pd


            class RangeFilter:
                def __init__(self, column, lo, hi, closed):
                    self.column = column
                    self.lo = lo
                    self.hi = hi
                    self.closed = closed

                def apply(self, df):
                    mode = "both" if self.closed else "neither"
                    mask = df[self.column].between(self.lo, self.hi, inclusive=mode)
                    return df[mask].reset_index(drop=True)
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"x": [1, 2, 3, 4], "tag": ["a", "b", "c", "d"]})
            assert solution.RangeFilter("x", 2, 4, True).apply(df)["tag"].tolist() == ["b", "c", "d"]
            assert solution.RangeFilter("x", 2, 4, False).apply(df)["tag"].tolist() == ["c"]
            """,
            "Configurable closed/open range filter on a frame.",
        ),
        _p(
            "flag_outliers",
            """
            import pandas as pd


            def outlier_positions(values, lo, hi):
                s = pd.Series(values)
                mask = ~s.between(lo, hi, inclusive=True)
                return s[mask].index.tolist()
            """,
            """
            import pandas as pd


            def outlier_positions(values, lo, hi):
                s = pd.Series(values)
                mask = ~s.between(lo, hi, inclusive="both")
                return s[mask].index.tolist()
            """,
            """
            import solution

            assert solution.outlier_positions([5, 50, 7, -1, 10], 0, 10) == [1, 3]
            """,
            "Positions of values outside a closed interval.",
        ),
        _p(
            "open_interval_rows",
            """
            import pandas as pd


            def rows_strictly_between(df, column, lo, hi):
                mask = df[column].between(lo, hi, inclusive=False)
                return df.loc[mask, column].tolist()
            """,
            """
            import pandas as pd


            def rows_strictly_between(df, column, lo, hi):
                mask = df[column].between(lo, hi, inclusive="neither")
                return df.loc[mask, column].tolist()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"p": [0.1, 0.5, 0.9, 1.0]})
            assert solution.rows_strictly_between(df, "p", 0.1, 1.0) == [0.5, 0.9]
            """,
            "Values strictly inside an open interval.",
        ),
    ],
    [
        _s(
            "clip_values",
            """
            import pandas as pd


            def clipped(values, lo, hi):
                return pd.Series(values).clip(lo, hi).tolist()
            """,
            """
            import solution

            assert solution.clipped([-5, 0, 5, 50], 0, 10) == [0, 0, 5, 10]
            """,
            "Series.clip is unchanged.",
        ),
        _s(
            "isin_mask",
            """
            import pandas as pd


            def members(values, allowed):
                return pd.Series(values).isin(allowed).tolist()
            """,
            """
            import solution

            assert solution.members([1, 2, 3], [2]) == [False, True, False]
            """,
            "Series.isin is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# pandas: DatetimeIndex.weekofyear / .week and Series.dt.week removed (2.0)
# ---------------------------------------------------------------------------

WEEKOFYEAR = _pandas_change(
    "pandas.DatetimeIndex.weekofyear",
    "DatetimeIndex.weekofyear / .week / Series.dt.week",
    "isocalendar().week",
    "DatetimeIndex.week, DatetimeIndex.weekofyear, Series.dt.week and Series.dt.weekofyear were removed in pandas 2.0; use isocalendar().week.",
    [
        _p(
            "week_numbers",
            """
            import pandas as pd


            def week_numbers(dates):
                idx = pd.to_datetime(dates)
                return idx.weekofyear.tolist()
            """,
            """
            import pandas as pd


            def week_numbers(dates):
                idx = pd.to_datetime(dates)
                return [int(w) for w in idx.isocalendar().week]
            """,
            """
            import solution

            assert solution.week_numbers(["2021-01-04", "2021-01-11", "2021-12-31"]) == [1, 2, 52]
            """,
            "ISO week number for each date string.",
        ),
        _p(
            "add_week_column",
            """
            import pandas as pd


            def add_week(df):
                out = df.copy()
                out["week"] = out["date"].dt.week
                return out
            """,
            """
            import pandas as pd


            def add_week(df):
                out = df.copy()
                out["week"] = out["date"].dt.isocalendar().week
                return out
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"date": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-12"])})
            out = solution.add_week(df)
            assert [int(w) for w in out["week"]] == [1, 2, 2]
            assert list(out.columns) == ["date", "week"]
            """,
            "Append an ISO week column to a frame.",
        ),
        _p(
            "weekly_totals",
            """
            import pandas as pd


            def weekly_totals(df):
                weeks = df["date"].dt.weekofyear
                totals = df.groupby(weeks)["amount"].sum()
                return {int(k): float(v) for k, v in totals.items()}
            """,
            """
            import pandas as pd


            def weekly_totals(df):
                weeks = df["date"].dt.isocalendar().week
                totals = df.groupby(weeks)["amount"].sum()
                return {int(k): float(v) for k, v in totals.items()}
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({
                "date": pd.to_datetime(["2022-01-03", "2022-01-05", "2022-01-10"]),
                "amount": [1.0, 2.0, 5.0],
            })
            assert solution.weekly_totals(df) == {1: 3.0, 2: 5.0}
            """,
            "Sum amounts per ISO week.",
        ),
        _p(
            "week_calendar",
            """
            import pandas as pd


            class WeekCalendar:
                def __init__(self, start, periods):
                    self.days = pd.date_range(start, periods=periods, freq="D")

                def weeks(self):
                    return sorted(set(int(w) for w in self.days.week))
            """,
            """
            import pandas as pd


            class WeekCalendar:
                def __init__(self, start, periods):
                    self.days = pd.date_range(start, periods=periods, freq="D")

                def weeks(self):
                    return sorted(set(int(w) for w in self.days.isocalendar().week))
            """,
            """
            import solution

            cal = solution.WeekCalendar("2021-01-04", 10)
            assert cal.weeks() == [1, 2]
            """,
            "Distinct ISO weeks covered by a daily calendar.",
        ),
        _p(
            "same_week",
            """
            import pandas as pd


            def same_week(a, b):
                idx = pd.DatetimeIndex([a, b])
                weeks = idx.weekofyear
                return bool(weeks[0] == weeks[1]) and bool(idx.year[0] == idx.year[1])
            """,
            """
            import pandas as pd


            def same_week(a, b):
                idx = pd.DatetimeIndex([a, b])
                weeks = idx.isocalendar().week.tolist()
                return bool(weeks[0] == weeks[1]) and bool(idx.year[0] == idx.year[1])
            """,
            """
            import solution

            assert solution.same_week("2022-03-07", "2022-03-13") is True
            assert solution.same_week("2022-03-07", "2022-03-14") is False
            """,
            "Whether two dates fall in the same ISO week of the same year.",
        ),
        _p(
            "first_in_week",
            """
            import pandas as pd


            def first_dates_per_week(df):
                weeks = df["day"].dt.week
                firsts = df.groupby(weeks)["day"].min()
                return [d.strftime("%Y-%m-%d") for d in firsts]
            """,
            """
            import pandas as pd


            def first_dates_per_week(df):
                weeks = df["day"].dt.isocalendar().week
                firsts = df.groupby(weeks)["day"].min()
                return [d.strftime("%Y-%m-%d") for d in firsts]
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"day": pd.to_datetime(["2022-01-05", "2022-01-04", "2022-01-12", "2022-01-11"])})
            assert solution.first_dates_per_week(df) == ["2022-01-04", "2022-01-11"]
            """,
            "Earliest date in each ISO week.",
        ),
    ],
    [
        _s(
            "day_of_week",
            """
            import pandas as pd


            def weekdays(dates):
                return pd.to_datetime(dates).dayofweek.tolist()
            """,
            """
            import solution

            assert solution.weekdays(["2022-01-03", "2022-01-09"]) == [0, 6]
            """,
            "DatetimeIndex.dayofweek is unchanged.",
        ),
        _s(
            "quarter_of_series",
            """
            import pandas as pd


            def quarters(df):
                return df["date"].dt.quarter.tolist()
            """,
            """
            import pandas as pd
            import solution

            df = pd.DataFrame({"date": pd.to_datetime(["2022-02-01", "2022-11-01"])})
            assert solution.quarters(df) == [1, 4]
            """,
            "Series.dt.quarter is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: OneHotEncoder(sparse=) removed (1.4) -> sparse_output=
# ---------------------------------------------------------------------------

ONEHOT_SPARSE = _sklearn_change(
    "sklearn.preprocessing.OneHotEncoder.sparse",
    "OneHotEncoder(sparse=False)",
    "OneHotEncoder(sparse_output=False)",
    SKLEARN_DATE_14,
    SKLEARN_URL_14,
    "OneHotEncoder's sparse parameter was renamed to sparse_output in 1.2 and removed in 1.4.",
    [
        _p(
            "encode_colors",
            """
            from sklearn.preprocessing import OneHotEncoder


            def encode_colors(colors):
                enc = OneHotEncoder(sparse=False)
                matrix = enc.fit_transform([[c] for c in colors])
                return matrix.astype(int).tolist()
            """,
            """
            from sklearn.preprocessing import OneHotEncoder


            def encode_colors(colors):
                enc = OneHotEncoder(sparse_output=False)
                matrix = enc.fit_transform([[c] for c in colors])
                return matrix.astype(int).tolist()
            """,
            """
            import solution

            assert solution.encode_colors(["red", "blue", "red"]) == [[0, 1], [1, 0], [0, 1]]
            """,
            "One-hot encode a list of colour names into a dense matrix.",
        ),
        _p(
            "category_encoder",
            """
            from sklearn.preprocessing import OneHotEncoder


            class CategoryEncoder:
                def __init__(self):
                    self.enc = OneHotEncoder(sparse=False, handle_unknown="ignore")

                def fit(self, labels):
                    self.enc.fit([[x] for x in labels])
                    return self

                def transform(self, labels):
                    return self.enc.transform([[x] for x in labels]).astype(int).tolist()
            """,
            """
            from sklearn.preprocessing import OneHotEncoder


            class CategoryEncoder:
                def __init__(self):
                    self.enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

                def fit(self, labels):
                    self.enc.fit([[x] for x in labels])
                    return self

                def transform(self, labels):
                    return self.enc.transform([[x] for x in labels]).astype(int).tolist()
            """,
            """
            import solution

            enc = solution.CategoryEncoder().fit(["a", "b", "c"])
            assert enc.transform(["b", "zzz"]) == [[0, 1, 0], [0, 0, 0]]
            """,
            "Encoder wrapper that ignores unknown categories.",
        ),
        _p(
            "two_columns",
            """
            from sklearn.preprocessing import OneHotEncoder


            def encode_rows(rows):
                enc = OneHotEncoder(sparse=False, dtype=int)
                return enc.fit_transform(rows).tolist()
            """,
            """
            from sklearn.preprocessing import OneHotEncoder


            def encode_rows(rows):
                enc = OneHotEncoder(sparse_output=False, dtype=int)
                return enc.fit_transform(rows).tolist()
            """,
            """
            import solution

            rows = [["x", "big"], ["y", "small"], ["x", "small"]]
            assert solution.encode_rows(rows) == [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]]
            """,
            "Dense integer one-hot encoding of two categorical columns.",
        ),
        _p(
            "feature_names",
            """
            from sklearn.preprocessing import OneHotEncoder


            def encoded_feature_names(rows, names):
                enc = OneHotEncoder(sparse=False)
                enc.fit(rows)
                return enc.get_feature_names_out(names).tolist()
            """,
            """
            from sklearn.preprocessing import OneHotEncoder


            def encoded_feature_names(rows, names):
                enc = OneHotEncoder(sparse_output=False)
                enc.fit(rows)
                return enc.get_feature_names_out(names).tolist()
            """,
            """
            import solution

            rows = [["cat"], ["dog"]]
            assert solution.encoded_feature_names(rows, ["pet"]) == ["pet_cat", "pet_dog"]
            """,
            "Names of the generated one-hot columns.",
        ),
        _p(
            "drop_first",
            """
            from sklearn.preprocessing import OneHotEncoder


            def dummies_drop_first(values):
                enc = OneHotEncoder(sparse=False, drop="first")
                out = enc.fit_transform([[v] for v in values])
                return out.astype(int).tolist()
            """,
            """
            from sklearn.preprocessing import OneHotEncoder


            def dummies_drop_first(values):
                enc = OneHotEncoder(sparse_output=False, drop="first")
                out = enc.fit_transform([[v] for v in values])
                return out.astype(int).tolist()
            """,
            """
            import solution

            assert solution.dummies_drop_first(["a", "b", "c", "a"]) == [[0, 0], [1, 0], [0, 1], [0, 0]]
            """,
            "Dummy coding with the first level dropped.",
        ),
        _p(
            "onehot_pipeline",
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OneHotEncoder


            def fit_and_predict(rows, labels, new_rows):
                model = make_pipeline(OneHotEncoder(sparse=False), LogisticRegression())
                model.fit(rows, labels)
                return model.predict(new_rows).tolist()
            """,
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OneHotEncoder


            def fit_and_predict(rows, labels, new_rows):
                model = make_pipeline(OneHotEncoder(sparse_output=False), LogisticRegression())
                model.fit(rows, labels)
                return model.predict(new_rows).tolist()
            """,
            """
            import solution

            rows = [["sun"], ["sun"], ["rain"], ["rain"]]
            labels = [1, 1, 0, 0]
            assert solution.fit_and_predict(rows, labels, [["rain"], ["sun"]]) == [0, 1]
            """,
            "Pipeline of dense one-hot encoding and logistic regression.",
        ),
    ],
    [
        _s(
            "ordinal_codes",
            """
            from sklearn.preprocessing import OrdinalEncoder


            def ordinal_codes(values):
                enc = OrdinalEncoder()
                return enc.fit_transform([[v] for v in values]).astype(int).ravel().tolist()
            """,
            """
            import solution

            assert solution.ordinal_codes(["b", "a", "c", "a"]) == [1, 0, 2, 0]
            """,
            "OrdinalEncoder is unchanged.",
        ),
        _s(
            "label_binarizer",
            """
            from sklearn.preprocessing import LabelBinarizer


            def binarize(labels):
                return LabelBinarizer().fit_transform(labels).tolist()
            """,
            """
            import solution

            assert solution.binarize(["x", "y", "z", "x"]) == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
            """,
            "LabelBinarizer is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: SGDClassifier(loss="log") removed (1.3) -> "log_loss"
# ---------------------------------------------------------------------------

SGD_LOG = _sklearn_change(
    "sklearn.linear_model.SGDClassifier.loss_log",
    'SGDClassifier(loss="log")',
    'SGDClassifier(loss="log_loss")',
    SKLEARN_DATE_13,
    SKLEARN_URL_13,
    "The SGDClassifier loss 'log' was renamed to 'log_loss' in 1.1 and removed in 1.3.",
    [
        _p(
            "train_log_sgd",
            """
            from sklearn.linear_model import SGDClassifier


            def train_and_predict(X, y, X_new):
                clf = SGDClassifier(loss="log", random_state=0, max_iter=200, tol=1e-3)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            from sklearn.linear_model import SGDClassifier


            def train_and_predict(X, y, X_new):
                clf = SGDClassifier(loss="log_loss", random_state=0, max_iter=200, tol=1e-3)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.train_and_predict(X, y, [[0.5, 0.5], [10.5, 10.5]]) == [0, 1]
            """,
            "Logistic-loss SGD classifier predictions.",
        ),
        _p(
            "probability_model",
            """
            from sklearn.linear_model import SGDClassifier


            class ProbabilityModel:
                def __init__(self):
                    self.clf = SGDClassifier(loss="log", random_state=1, max_iter=300)

                def fit(self, X, y):
                    self.clf.fit(X, y)
                    return self

                def positive_probability(self, X):
                    return self.clf.predict_proba(X)[:, 1].tolist()
            """,
            """
            from sklearn.linear_model import SGDClassifier


            class ProbabilityModel:
                def __init__(self):
                    self.clf = SGDClassifier(loss="log_loss", random_state=1, max_iter=300)

                def fit(self, X, y):
                    self.clf.fit(X, y)
                    return self

                def positive_probability(self, X):
                    return self.clf.predict_proba(X)[:, 1].tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            model = solution.ProbabilityModel().fit(X, y)
            probs = model.positive_probability([[0, 0], [11, 11]])
            assert len(probs) == 2
            assert probs[0] < 0.5 < probs[1]
            assert all(0.0 <= p <= 1.0 for p in probs)
            """,
            "Model wrapper exposing positive-class probabilities.",
        ),
        _p(
            "loss_comparison",
            """
            from sklearn.linear_model import SGDClassifier


            def training_accuracies(X, y):
                out = []
                for loss in ("hinge", "log"):
                    clf = SGDClassifier(loss=loss, random_state=0, max_iter=200)
                    clf.fit(X, y)
                    out.append(float(clf.score(X, y)))
                return out
            """,
            """
            from sklearn.linear_model import SGDClassifier


            def training_accuracies(X, y):
                out = []
                for loss in ("hinge", "log_loss"):
                    clf = SGDClassifier(loss=loss, random_state=0, max_iter=200)
                    clf.fit(X, y)
                    out.append(float(clf.score(X, y)))
                return out
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.training_accuracies(X, y) == [1.0, 1.0]
            """,
            "Training accuracy for hinge and logistic losses.",
        ),
        _p(
            "scaled_sgd_pipeline",
            """
            from sklearn.linear_model import SGDClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def build_and_predict(X, y, X_new):
                model = make_pipeline(StandardScaler(), SGDClassifier(loss="log", random_state=0))
                model.fit(X, y)
                return model.predict(X_new).tolist()
            """,
            """
            from sklearn.linear_model import SGDClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def build_and_predict(X, y, X_new):
                model = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", random_state=0))
                model.fit(X, y)
                return model.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.build_and_predict(X, y, [[11, 10], [1, 0], [0, 1]]) == [1, 0, 0]
            """,
            "Scaled logistic SGD pipeline predictions.",
        ),
        _p(
            "rank_by_confidence",
            """
            from sklearn.linear_model import SGDClassifier


            def rank_candidates(X, y, candidates):
                clf = SGDClassifier(loss="log", random_state=0, max_iter=300).fit(X, y)
                scores = clf.predict_proba(candidates)[:, 1]
                order = sorted(range(len(candidates)), key=lambda i: -scores[i])
                return order
            """,
            """
            from sklearn.linear_model import SGDClassifier


            def rank_candidates(X, y, candidates):
                clf = SGDClassifier(loss="log_loss", random_state=0, max_iter=300).fit(X, y)
                scores = clf.predict_proba(candidates)[:, 1]
                order = sorted(range(len(candidates)), key=lambda i: -scores[i])
                return order
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.rank_candidates(X, y, [[0, 0], [11, 11], [5, 5]]) == [1, 2, 0]
            """,
            "Rank candidate points by positive-class probability.",
        ),
        _p(
            "incremental_fit",
            """
            import numpy as np
            from sklearn.linear_model import SGDClassifier


            def incremental_predict(batches, X_new):
                clf = SGDClassifier(loss="log", random_state=0)
                classes = np.array([0, 1])
                for _ in range(20):
                    for X, y in batches:
                        clf.partial_fit(X, y, classes=classes)
                return clf.predict(X_new).tolist()
            """,
            """
            import numpy as np
            from sklearn.linear_model import SGDClassifier


            def incremental_predict(batches, X_new):
                clf = SGDClassifier(loss="log_loss", random_state=0)
                classes = np.array([0, 1])
                for _ in range(20):
                    for X, y in batches:
                        clf.partial_fit(X, y, classes=classes)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            batches = [
                ([[0, 0], [10, 10]], [0, 1]),
                ([[1, 1], [11, 11]], [0, 1]),
            ]
            assert solution.incremental_predict(batches, [[0, 1], [10, 11]]) == [0, 1]
            """,
            "Logistic SGD trained with partial_fit over batches.",
        ),
    ],
    [
        _s(
            "hinge_sgd",
            """
            from sklearn.linear_model import SGDClassifier


            def hinge_predict(X, y, X_new):
                clf = SGDClassifier(loss="hinge", random_state=0, max_iter=200).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.hinge_predict(X, y, [[0, 0], [11, 11]]) == [0, 1]
            """,
            "SGDClassifier(loss='hinge') is unchanged.",
        ),
        _s(
            "modified_huber_sgd",
            """
            from sklearn.linear_model import SGDClassifier


            def huber_proba_shape(X, y):
                clf = SGDClassifier(loss="modified_huber", random_state=0, max_iter=200).fit(X, y)
                return list(clf.predict_proba(X).shape)
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.huber_proba_shape(X, y) == [8, 2]
            """,
            "SGDClassifier(loss='modified_huber') is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: LinearRegression(normalize=) removed (1.2)
# ---------------------------------------------------------------------------

LINREG_NORMALIZE = _sklearn_change(
    "sklearn.linear_model.LinearRegression.normalize",
    "LinearRegression(normalize=True)",
    "make_pipeline(StandardScaler(), LinearRegression())",
    SKLEARN_DATE_12,
    SKLEARN_URL_12,
    "The normalize parameter of LinearRegression (and other linear models) was deprecated in 1.0 and removed in 1.2; scale with StandardScaler in a Pipeline instead.",
    [
        _p(
            "predict_line",
            """
            from sklearn.linear_model import LinearRegression


            def predict_line(xs, ys, new_xs):
                model = LinearRegression(normalize=True)
                model.fit([[x] for x in xs], ys)
                return [round(float(v), 6) for v in model.predict([[x] for x in new_xs])]
            """,
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def predict_line(xs, ys, new_xs):
                model = make_pipeline(StandardScaler(), LinearRegression())
                model.fit([[x] for x in xs], ys)
                return [round(float(v), 6) for v in model.predict([[x] for x in new_xs])]
            """,
            """
            import solution

            xs = [0.0, 1.0, 2.0, 3.0]
            ys = [1.0, 3.0, 5.0, 7.0]
            assert solution.predict_line(xs, ys, [4.0, 10.0]) == [9.0, 21.0]
            """,
            "Fit a line and predict new points.",
        ),
        _p(
            "slope_intercept",
            """
            from sklearn.linear_model import LinearRegression


            def slope_intercept(xs, ys):
                model = LinearRegression(normalize=True).fit([[x] for x in xs], ys)
                return round(float(model.coef_[0]), 6), round(float(model.intercept_), 6)
            """,
            """
            from sklearn.linear_model import LinearRegression


            def slope_intercept(xs, ys):
                model = LinearRegression().fit([[x] for x in xs], ys)
                return round(float(model.coef_[0]), 6), round(float(model.intercept_), 6)
            """,
            """
            import solution

            assert solution.slope_intercept([1.0, 2.0, 3.0], [4.0, 6.0, 8.0]) == (2.0, 2.0)
            """,
            "Slope and intercept of a fitted line in original units.",
        ),
        _p(
            "trend_model",
            """
            from sklearn.linear_model import LinearRegression


            class TrendModel:
                def __init__(self):
                    self.model = LinearRegression(normalize=True)

                def fit(self, series):
                    X = [[i] for i in range(len(series))]
                    self.model.fit(X, series)
                    return self

                def forecast(self, steps_ahead, n_known):
                    X = [[n_known + k] for k in range(steps_ahead)]
                    return [round(float(v), 4) for v in self.model.predict(X)]
            """,
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            class TrendModel:
                def __init__(self):
                    self.model = make_pipeline(StandardScaler(), LinearRegression())

                def fit(self, series):
                    X = [[i] for i in range(len(series))]
                    self.model.fit(X, series)
                    return self

                def forecast(self, steps_ahead, n_known):
                    X = [[n_known + k] for k in range(steps_ahead)]
                    return [round(float(v), 4) for v in self.model.predict(X)]
            """,
            """
            import solution

            series = [10.0, 12.0, 14.0, 16.0]
            model = solution.TrendModel().fit(series)
            assert model.forecast(2, 4) == [18.0, 20.0]
            """,
            "Linear trend forecaster.",
        ),
        _p(
            "multi_feature",
            """
            from sklearn.linear_model import LinearRegression


            def fit_predict(X, y, X_new):
                model = LinearRegression(normalize=True).fit(X, y)
                return [round(float(v), 5) for v in model.predict(X_new)]
            """,
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def fit_predict(X, y, X_new):
                model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
                return [round(float(v), 5) for v in model.predict(X_new)]
            """,
            """
            import solution

            X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 5.0]]
            y = [2.0 * a + 3.0 * b + 1.0 for a, b in X]
            assert solution.fit_predict(X, y, [[1.0, 1.0], [0.0, 0.0]]) == [6.0, 1.0]
            """,
            "Two-feature linear fit and predictions.",
        ),
        _p(
            "fit_quality",
            """
            from sklearn.linear_model import LinearRegression


            def r_squared(X, y):
                model = LinearRegression(normalize=True).fit(X, y)
                return round(float(model.score(X, y)), 6)
            """,
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def r_squared(X, y):
                model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
                return round(float(model.score(X, y)), 6)
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            assert solution.r_squared(X, [1.0, 2.0, 3.0, 4.0]) == 1.0
            assert solution.r_squared(X, [1.0, 3.0, 1.0, 3.0]) == 0.2
            """,
            "R-squared of a linear fit on the training data.",
        ),
        _p(
            "residuals",
            """
            from sklearn.linear_model import LinearRegression


            def residuals(xs, ys):
                X = [[x] for x in xs]
                model = LinearRegression(normalize=True).fit(X, ys)
                return [round(float(a - b), 6) for a, b in zip(ys, model.predict(X))]
            """,
            """
            from sklearn.linear_model import LinearRegression


            def residuals(xs, ys):
                X = [[x] for x in xs]
                model = LinearRegression().fit(X, ys)
                return [round(float(a - b), 6) for a, b in zip(ys, model.predict(X))]
            """,
            """
            import solution

            assert solution.residuals([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]) == [0.0, 0.0, 0.0]
            assert solution.residuals([0.0, 1.0, 2.0], [0.0, 3.0, 0.0]) == [-1.0, 2.0, -1.0]
            """,
            "Residuals of an ordinary least squares fit.",
        ),
    ],
    [
        _s(
            "no_intercept",
            """
            from sklearn.linear_model import LinearRegression


            def slope_through_origin(xs, ys):
                model = LinearRegression(fit_intercept=False).fit([[x] for x in xs], ys)
                return round(float(model.coef_[0]), 6)
            """,
            """
            import solution

            assert solution.slope_through_origin([1.0, 2.0], [3.0, 6.0]) == 3.0
            """,
            "LinearRegression(fit_intercept=False) is unchanged.",
        ),
        _s(
            "positive_coefs",
            """
            from sklearn.linear_model import LinearRegression


            def positive_fit(X, y):
                model = LinearRegression(positive=True).fit(X, y)
                return [round(float(c), 6) for c in model.coef_]
            """,
            """
            import solution

            X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]]
            y = [2.0 * a + 5.0 * b for a, b in X]
            assert solution.positive_fit(X, y) == [2.0, 5.0]
            """,
            "LinearRegression(positive=True) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: mean_squared_error(squared=False) removed (1.6)
# ---------------------------------------------------------------------------

MSE_SQUARED = _sklearn_change(
    "sklearn.metrics.mean_squared_error.squared",
    "mean_squared_error(y, p, squared=False)",
    "root_mean_squared_error(y, p)",
    SKLEARN_DATE_16,
    SKLEARN_URL_16,
    "The squared parameter of mean_squared_error / mean_squared_log_error was deprecated in 1.4 and removed in 1.6; use root_mean_squared_error.",
    [
        _p(
            "rmse",
            """
            from sklearn.metrics import mean_squared_error


            def rmse(y_true, y_pred):
                value = mean_squared_error(y_true, y_pred, squared=False)
                return round(float(value), 6)
            """,
            """
            from sklearn.metrics import root_mean_squared_error


            def rmse(y_true, y_pred):
                value = root_mean_squared_error(y_true, y_pred)
                return round(float(value), 6)
            """,
            """
            import solution

            assert solution.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
            assert solution.rmse([0.0, 0.0], [3.0, 4.0]) == 3.535534
            """,
            "Root mean squared error of two vectors.",
        ),
        _p(
            "error_report",
            """
            from sklearn.metrics import mean_squared_error


            class Evaluator:
                def __init__(self, y_true):
                    self.y_true = list(y_true)

                def report(self, y_pred):
                    mse = mean_squared_error(self.y_true, y_pred)
                    rmse = mean_squared_error(self.y_true, y_pred, squared=False)
                    return {"mse": round(float(mse), 6), "rmse": round(float(rmse), 6)}
            """,
            """
            from sklearn.metrics import mean_squared_error, root_mean_squared_error


            class Evaluator:
                def __init__(self, y_true):
                    self.y_true = list(y_true)

                def report(self, y_pred):
                    mse = mean_squared_error(self.y_true, y_pred)
                    rmse = root_mean_squared_error(self.y_true, y_pred)
                    return {"mse": round(float(mse), 6), "rmse": round(float(rmse), 6)}
            """,
            """
            import solution

            ev = solution.Evaluator([2.0, 4.0])
            assert ev.report([4.0, 6.0]) == {"mse": 4.0, "rmse": 2.0}
            """,
            "Report MSE and RMSE for predictions.",
        ),
        _p(
            "rmse_per_output",
            """
            from sklearn.metrics import mean_squared_error


            def rmse_per_output(Y_true, Y_pred):
                values = mean_squared_error(Y_true, Y_pred, squared=False, multioutput="raw_values")
                return [round(float(v), 6) for v in values]
            """,
            """
            from sklearn.metrics import root_mean_squared_error


            def rmse_per_output(Y_true, Y_pred):
                values = root_mean_squared_error(Y_true, Y_pred, multioutput="raw_values")
                return [round(float(v), 6) for v in values]
            """,
            """
            import solution

            Y_true = [[0.0, 0.0], [0.0, 0.0]]
            Y_pred = [[1.0, 3.0], [1.0, 3.0]]
            assert solution.rmse_per_output(Y_true, Y_pred) == [1.0, 3.0]
            """,
            "RMSE for each output column.",
        ),
        _p(
            "weighted_rmse",
            """
            from sklearn.metrics import mean_squared_error


            def weighted_rmse(y_true, y_pred, weights):
                value = mean_squared_error(y_true, y_pred, sample_weight=weights, squared=False)
                return round(float(value), 6)
            """,
            """
            from sklearn.metrics import root_mean_squared_error


            def weighted_rmse(y_true, y_pred, weights):
                value = root_mean_squared_error(y_true, y_pred, sample_weight=weights)
                return round(float(value), 6)
            """,
            """
            import solution

            assert solution.weighted_rmse([0.0, 0.0], [1.0, 3.0], [1.0, 0.0]) == 1.0
            assert solution.weighted_rmse([0.0, 0.0], [1.0, 3.0], [0.0, 1.0]) == 3.0
            """,
            "Sample-weighted RMSE.",
        ),
        _p(
            "model_rmse",
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error


            def training_rmse(X, y):
                model = LinearRegression().fit(X, y)
                return round(float(mean_squared_error(y, model.predict(X), squared=False)), 4)
            """,
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import root_mean_squared_error


            def training_rmse(X, y):
                model = LinearRegression().fit(X, y)
                return round(float(root_mean_squared_error(y, model.predict(X))), 4)
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            assert solution.training_rmse(X, [1.0, 2.0, 3.0, 4.0]) == 0.0
            assert solution.training_rmse(X, [1.0, 3.0, 1.0, 3.0]) == 0.8944
            """,
            "RMSE of a linear model on its training data.",
        ),
        _p(
            "best_by_rmse",
            """
            from sklearn.metrics import mean_squared_error


            def best_candidate(y_true, candidates):
                scores = {}
                for name, preds in candidates.items():
                    scores[name] = mean_squared_error(y_true, preds, squared=False)
                return min(scores, key=scores.get)
            """,
            """
            from sklearn.metrics import root_mean_squared_error


            def best_candidate(y_true, candidates):
                scores = {}
                for name, preds in candidates.items():
                    scores[name] = root_mean_squared_error(y_true, preds)
                return min(scores, key=scores.get)
            """,
            """
            import solution

            y = [1.0, 2.0, 3.0]
            cands = {"a": [1.0, 2.0, 5.0], "b": [1.1, 2.1, 3.1], "c": [0.0, 0.0, 0.0]}
            assert solution.best_candidate(y, cands) == "b"
            """,
            "Pick the candidate prediction set with the lowest RMSE.",
        ),
    ],
    [
        _s(
            "mae",
            """
            from sklearn.metrics import mean_absolute_error


            def mae(y_true, y_pred):
                return round(float(mean_absolute_error(y_true, y_pred)), 6)
            """,
            """
            import solution

            assert solution.mae([0.0, 0.0], [3.0, 4.0]) == 3.5
            """,
            "mean_absolute_error is unchanged.",
        ),
        _s(
            "plain_mse",
            """
            from sklearn.metrics import mean_squared_error


            def mse(y_true, y_pred):
                return round(float(mean_squared_error(y_true, y_pred)), 6)
            """,
            """
            import solution

            assert solution.mse([0.0, 0.0], [3.0, 4.0]) == 12.5
            """,
            "mean_squared_error without squared= is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: make_scorer(needs_proba=) removed (1.6) -> response_method=
# ---------------------------------------------------------------------------

SCORER_NEEDS_PROBA = _sklearn_change(
    "sklearn.metrics.make_scorer.needs_proba",
    "make_scorer(metric, needs_proba=True)",
    'make_scorer(metric, response_method="predict_proba")',
    SKLEARN_DATE_16,
    SKLEARN_URL_16,
    "make_scorer's needs_proba and needs_threshold parameters were deprecated in 1.4 and removed in 1.6; use response_method.",
    [
        _p(
            "auc_scorer",
            """
            from sklearn.metrics import make_scorer, roc_auc_score
            from sklearn.tree import DecisionTreeClassifier


            def training_auc(X, y):
                scorer = make_scorer(roc_auc_score, needs_proba=True)
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            from sklearn.metrics import make_scorer, roc_auc_score
            from sklearn.tree import DecisionTreeClassifier


            def training_auc(X, y):
                scorer = make_scorer(roc_auc_score, response_method="predict_proba")
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            import solution

            X = [[0], [1], [2], [3], [10], [11], [12], [13]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.training_auc(X, y) == 1.0
            """,
            "ROC AUC scorer built from predicted probabilities.",
        ),
        _p(
            "cv_log_loss",
            """
            from sklearn.metrics import log_loss, make_scorer
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier


            def cv_neg_log_loss(X, y):
                scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
                scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=2, scoring=scorer)
                return [round(abs(float(s)), 6) for s in scores]
            """,
            """
            from sklearn.metrics import log_loss, make_scorer
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier


            def cv_neg_log_loss(X, y):
                scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba")
                scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=2, scoring=scorer)
                return [round(abs(float(s)), 6) for s in scores]
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.cv_neg_log_loss(X, y) == [0.0, 0.0]
            """,
            "Cross-validated log loss via a probability scorer.",
        ),
        _p(
            "brier_evaluator",
            """
            from sklearn.metrics import brier_score_loss, make_scorer
            from sklearn.neighbors import KNeighborsClassifier


            class ProbaEvaluator:
                def __init__(self):
                    self.scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)

                def brier(self, X, y):
                    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
                    return round(abs(float(self.scorer(clf, X, y))), 6)
            """,
            """
            from sklearn.metrics import brier_score_loss, make_scorer
            from sklearn.neighbors import KNeighborsClassifier


            class ProbaEvaluator:
                def __init__(self):
                    self.scorer = make_scorer(brier_score_loss, greater_is_better=False, response_method="predict_proba")

                def brier(self, X, y):
                    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
                    return round(abs(float(self.scorer(clf, X, y))), 6)
            """,
            """
            import solution

            X = [[0], [1], [5], [6]]
            y = [0, 0, 1, 1]
            assert solution.ProbaEvaluator().brier(X, y) == 0.0
            """,
            "Brier score of a 1-NN classifier on its training data.",
        ),
        _p(
            "average_precision",
            """
            from sklearn.metrics import average_precision_score, make_scorer
            from sklearn.tree import DecisionTreeClassifier


            def train_ap(X, y, max_depth):
                scorer = make_scorer(average_precision_score, needs_proba=True)
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            from sklearn.metrics import average_precision_score, make_scorer
            from sklearn.tree import DecisionTreeClassifier


            def train_ap(X, y, max_depth):
                scorer = make_scorer(average_precision_score, response_method="predict_proba")
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            import solution

            X = [[0], [1], [2], [3], [10], [11], [12], [13]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.train_ap(X, y, 1) == 1.0
            """,
            "Average precision scorer on predicted probabilities.",
        ),
        _p(
            "grid_by_logloss",
            """
            from sklearn.metrics import log_loss, make_scorer
            from sklearn.model_selection import GridSearchCV
            from sklearn.tree import DecisionTreeClassifier


            def best_depth(X, y, splits):
                scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
                grid = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [1, 3]}, scoring=scorer, cv=splits)
                grid.fit(X, y)
                return int(grid.best_params_["max_depth"])
            """,
            """
            from sklearn.metrics import log_loss, make_scorer
            from sklearn.model_selection import GridSearchCV
            from sklearn.tree import DecisionTreeClassifier


            def best_depth(X, y, splits):
                scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba")
                grid = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [1, 3]}, scoring=scorer, cv=splits)
                grid.fit(X, y)
                return int(grid.best_params_["max_depth"])
            """,
            """
            import solution

            # XOR-like data: depth 1 cannot separate, depth 3 can.
            X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 4
            y = [0, 1, 1, 0] * 4
            splits = [(list(range(8)), list(range(8, 16))), (list(range(8, 16)), list(range(8)))]
            assert solution.best_depth(X, y, splits) == 3
            """,
            "Grid search over tree depth using a log-loss probability scorer.",
        ),
        _p(
            "custom_confidence",
            """
            from sklearn.metrics import make_scorer
            from sklearn.neighbors import KNeighborsClassifier


            def mean_positive_confidence(y_true, proba):
                return float(sum(proba) / len(proba))


            def confidence_score(X, y, k):
                scorer = make_scorer(mean_positive_confidence, needs_proba=True)
                clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            from sklearn.metrics import make_scorer
            from sklearn.neighbors import KNeighborsClassifier


            def mean_positive_confidence(y_true, proba):
                return float(sum(proba) / len(proba))


            def confidence_score(X, y, k):
                scorer = make_scorer(mean_positive_confidence, response_method="predict_proba")
                clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
                return round(float(scorer(clf, X, y)), 6)
            """,
            """
            import solution

            X = [[0], [1], [10], [11]]
            y = [0, 0, 1, 1]
            assert solution.confidence_score(X, y, 1) == 0.5
            assert solution.confidence_score(X, y, 4) == 0.5
            """,
            "Custom probability-based scorer (mean positive-class probability).",
        ),
    ],
    [
        _s(
            "accuracy_scorer",
            """
            from sklearn.metrics import accuracy_score, make_scorer
            from sklearn.tree import DecisionTreeClassifier


            def training_accuracy(X, y):
                scorer = make_scorer(accuracy_score)
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return float(scorer(clf, X, y))
            """,
            """
            import solution

            assert solution.training_accuracy([[0], [1], [5], [6]], [0, 0, 1, 1]) == 1.0
            """,
            "make_scorer with a plain prediction metric is unchanged.",
        ),
        _s(
            "neg_mae_scorer",
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import make_scorer, mean_absolute_error


            def neg_mae(X, y):
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                model = LinearRegression().fit(X, y)
                return round(float(scorer(model, X, y)), 6)
            """,
            """
            import solution

            assert solution.neg_mae([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0]) == 0.0
            """,
            "make_scorer(greater_is_better=False) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: base_estimator= removed from ensembles (1.4) -> estimator=
# ---------------------------------------------------------------------------

BASE_ESTIMATOR = _sklearn_change(
    "sklearn.ensemble.base_estimator",
    "AdaBoostClassifier(base_estimator=...) / BaggingClassifier(base_estimator=...)",
    "estimator=",
    SKLEARN_DATE_14,
    SKLEARN_URL_14,
    "The base_estimator parameter of AdaBoost*, Bagging* (and the base_estimator_ attribute) was renamed to estimator in 1.2 and removed in 1.4.",
    [
        _p(
            "boosted_stumps",
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier


            def boosted_predict(X, y, X_new):
                stump = DecisionTreeClassifier(max_depth=1)
                clf = AdaBoostClassifier(base_estimator=stump, n_estimators=5, random_state=0)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier


            def boosted_predict(X, y, X_new):
                stump = DecisionTreeClassifier(max_depth=1)
                clf = AdaBoostClassifier(estimator=stump, n_estimators=5, random_state=0)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0], [1], [2], [3], [10], [11], [12], [13]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.boosted_predict(X, y, [[1.5], [12.5]]) == [0, 1]
            """,
            "AdaBoost of decision stumps.",
        ),
        _p(
            "bagged_trees",
            """
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier


            def bagged_predict(X, y, X_new):
                clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=3, bootstrap=False, random_state=0)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier


            def bagged_predict(X, y, X_new):
                clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=3, bootstrap=False, random_state=0)
                clf.fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.bagged_predict(X, y, [[10, 10], [0, 0], [11, 11]]) == [1, 0, 1]
            """,
            "Bagged decision trees without bootstrap sampling.",
        ),
        _p(
            "bagged_regressor",
            """
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor


            class TreeEnsemble:
                def __init__(self, n):
                    self.model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=n, bootstrap=False, random_state=0)

                def fit(self, X, y):
                    self.model.fit(X, y)
                    return self

                def predict(self, X):
                    return [round(float(v), 6) for v in self.model.predict(X)]

                def size(self):
                    return len(self.model.estimators_)
            """,
            """
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor


            class TreeEnsemble:
                def __init__(self, n):
                    self.model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=n, bootstrap=False, random_state=0)

                def fit(self, X, y):
                    self.model.fit(X, y)
                    return self

                def predict(self, X):
                    return [round(float(v), 6) for v in self.model.predict(X)]

                def size(self):
                    return len(self.model.estimators_)
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            y = [5.0, 7.0, 2.0, 9.0]
            ens = solution.TreeEnsemble(4).fit(X, y)
            assert ens.predict(X) == [5.0, 7.0, 2.0, 9.0]
            assert ens.size() == 4
            """,
            "Bagging regressor of full trees fitted without bootstrap.",
        ),
        _p(
            "early_stop_boost",
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.linear_model import LogisticRegression


            def fitted_rounds(X, y):
                clf = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=10, random_state=0)
                clf.fit(X, y)
                return len(clf.estimators_), clf.predict(X).tolist()
            """,
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.linear_model import LogisticRegression


            def fitted_rounds(X, y):
                clf = AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=0)
                clf.fit(X, y)
                return len(clf.estimators_), clf.predict(X).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            rounds, preds = solution.fitted_rounds(X, y)
            assert rounds == 1
            assert preds == y
            """,
            "AdaBoost over logistic regression stops after a perfect first round.",
        ),
        _p(
            "bagged_knn",
            """
            from sklearn.ensemble import BaggingClassifier
            from sklearn.neighbors import KNeighborsClassifier


            def bagged_knn_predict(X, y):
                base = KNeighborsClassifier(n_neighbors=1)
                clf = BaggingClassifier(base_estimator=base, n_estimators=3, bootstrap=False, random_state=0)
                return clf.fit(X, y).predict(X).tolist()
            """,
            """
            from sklearn.ensemble import BaggingClassifier
            from sklearn.neighbors import KNeighborsClassifier


            def bagged_knn_predict(X, y):
                base = KNeighborsClassifier(n_neighbors=1)
                clf = BaggingClassifier(estimator=base, n_estimators=3, bootstrap=False, random_state=0)
                return clf.fit(X, y).predict(X).tolist()
            """,
            """
            import solution

            X = [[0], [3], [6], [9], [12]]
            y = [0, 1, 0, 1, 0]
            assert solution.bagged_knn_predict(X, y) == [0, 1, 0, 1, 0]
            """,
            "Bagged 1-NN classifiers reproduce the training labels.",
        ),
        _p(
            "staged_scores",
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier


            def staged_training_scores(X, y):
                clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=4, random_state=0)
                clf.fit(X, y)
                return [round(float(s), 6) for s in clf.staged_score(X, y)]
            """,
            """
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier


            def staged_training_scores(X, y):
                clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=4, random_state=0)
                clf.fit(X, y)
                return [round(float(s), 6) for s in clf.staged_score(X, y)]
            """,
            """
            import solution

            X = [[0], [1], [2], [3], [10], [11], [12], [13]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.staged_training_scores(X, y) == [1.0]
            """,
            "Staged accuracy of a boosted stump ensemble on separable data.",
        ),
    ],
    [
        _s(
            "random_forest",
            """
            from sklearn.ensemble import RandomForestClassifier


            def forest_predict(X, y, X_new):
                clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.forest_predict(X, y, [[0, 0], [11, 11]]) == [0, 1]
            """,
            "RandomForestClassifier(n_estimators=) is unchanged.",
        ),
        _s(
            "voting_ensemble",
            """
            from sklearn.ensemble import VotingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier


            def vote_predict(X, y, X_new):
                clf = VotingClassifier(estimators=[("knn", KNeighborsClassifier(1)), ("tree", DecisionTreeClassifier(random_state=0))])
                return clf.fit(X, y).predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0], [1], [10], [11]]
            y = [0, 0, 1, 1]
            assert solution.vote_predict(X, y, [[0.5], [10.5]]) == [0, 1]
            """,
            "VotingClassifier(estimators=) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: LogisticRegression(penalty="none") removed (1.4) -> penalty=None
# ---------------------------------------------------------------------------

PENALTY_NONE = _sklearn_change(
    "sklearn.linear_model.LogisticRegression.penalty_none",
    'LogisticRegression(penalty="none")',
    "LogisticRegression(penalty=None)",
    SKLEARN_DATE_14,
    SKLEARN_URL_14,
    "The string 'none' for LogisticRegression's penalty was deprecated in 1.2 and removed in 1.4; pass penalty=None.",
    [
        _p(
            "unpenalized_fit",
            """
            from sklearn.linear_model import LogisticRegression


            def unpenalized_predict(X, y, X_new):
                clf = LogisticRegression(penalty="none", max_iter=500).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            from sklearn.linear_model import LogisticRegression


            def unpenalized_predict(X, y, X_new):
                clf = LogisticRegression(penalty=None, max_iter=500).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.unpenalized_predict(X, y, [[0.5, 0.5], [10.5, 10.5]]) == [0, 1]
            """,
            "Unpenalised logistic regression predictions.",
        ),
        _p(
            "plain_logit_class",
            """
            from sklearn.linear_model import LogisticRegression


            class PlainLogit:
                def __init__(self, max_iter=500):
                    self.clf = LogisticRegression(penalty="none", max_iter=max_iter)

                def fit(self, X, y):
                    self.clf.fit(X, y)
                    return self

                def classes(self):
                    return self.clf.classes_.tolist()

                def predict(self, X):
                    return self.clf.predict(X).tolist()
            """,
            """
            from sklearn.linear_model import LogisticRegression


            class PlainLogit:
                def __init__(self, max_iter=500):
                    self.clf = LogisticRegression(penalty=None, max_iter=max_iter)

                def fit(self, X, y):
                    self.clf.fit(X, y)
                    return self

                def classes(self):
                    return self.clf.classes_.tolist()

                def predict(self, X):
                    return self.clf.predict(X).tolist()
            """,
            """
            import solution

            X = [[0], [1], [2], [10], [11], [12]]
            y = ["low", "low", "low", "high", "high", "high"]
            model = solution.PlainLogit().fit(X, y)
            assert model.classes() == ["high", "low"]
            assert model.predict([[1.5], [11.5]]) == ["low", "high"]
            """,
            "Wrapper class around an unpenalised logistic model.",
        ),
        _p(
            "multiclass_no_penalty",
            """
            from sklearn.linear_model import LogisticRegression


            def three_class_predict(X, y, X_new):
                clf = LogisticRegression(penalty="none", max_iter=1000).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            from sklearn.linear_model import LogisticRegression


            def three_class_predict(X, y, X_new):
                clf = LogisticRegression(penalty=None, max_iter=1000).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 0], [1, 0], [0, 1], [10, 0], [11, 0], [10, 1], [0, 10], [1, 10], [0, 11]]
            y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            assert solution.three_class_predict(X, y, [[0.5, 0.5], [10.5, 0.5], [0.5, 10.5]]) == [0, 1, 2]
            """,
            "Three-class unpenalised logistic regression.",
        ),
        _p(
            "penalty_comparison",
            """
            from sklearn.linear_model import LogisticRegression


            def accuracies(X, y):
                out = []
                for penalty in ("l2", "none"):
                    clf = LogisticRegression(penalty=penalty, max_iter=500).fit(X, y)
                    out.append(float(clf.score(X, y)))
                return out
            """,
            """
            from sklearn.linear_model import LogisticRegression


            def accuracies(X, y):
                out = []
                for penalty in ("l2", None):
                    clf = LogisticRegression(penalty=penalty, max_iter=500).fit(X, y)
                    out.append(float(clf.score(X, y)))
                return out
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.accuracies(X, y) == [1.0, 1.0]
            """,
            "Training accuracy with and without an L2 penalty.",
        ),
        _p(
            "scaled_logit_pipeline",
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def pipeline_predict(X, y, X_new):
                model = make_pipeline(StandardScaler(), LogisticRegression(penalty="none", max_iter=500))
                return model.fit(X, y).predict(X_new).tolist()
            """,
            """
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler


            def pipeline_predict(X, y, X_new):
                model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=500))
                return model.fit(X, y).predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0, 100], [1, 110], [0, 120], [10, 100], [11, 110], [10, 120]]
            y = [0, 0, 0, 1, 1, 1]
            assert solution.pipeline_predict(X, y, [[0.5, 105], [10.5, 115]]) == [0, 1]
            """,
            "Scaled unpenalised logistic regression pipeline.",
        ),
        _p(
            "newton_solver",
            """
            from sklearn.linear_model import LogisticRegression


            def newton_predict_proba_sign(X, y, X_new):
                clf = LogisticRegression(penalty="none", solver="newton-cg", max_iter=200).fit(X, y)
                proba = clf.predict_proba(X_new)[:, 1]
                return [int(p > 0.5) for p in proba]
            """,
            """
            from sklearn.linear_model import LogisticRegression


            def newton_predict_proba_sign(X, y, X_new):
                clf = LogisticRegression(penalty=None, solver="newton-cg", max_iter=200).fit(X, y)
                proba = clf.predict_proba(X_new)[:, 1]
                return [int(p > 0.5) for p in proba]
            """,
            """
            import solution

            X = [[0], [1], [2], [10], [11], [12]]
            y = [0, 0, 0, 1, 1, 1]
            assert solution.newton_predict_proba_sign(X, y, [[0], [12], [1]]) == [0, 1, 0]
            """,
            "Unpenalised logistic regression with the newton-cg solver.",
        ),
    ],
    [
        _s(
            "l2_logit",
            """
            from sklearn.linear_model import LogisticRegression


            def l2_predict(X, y, X_new):
                clf = LogisticRegression(penalty="l2", C=1.0, max_iter=500).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0], [1], [2], [10], [11], [12]]
            y = [0, 0, 0, 1, 1, 1]
            assert solution.l2_predict(X, y, [[0.5], [11.5]]) == [0, 1]
            """,
            "LogisticRegression(penalty='l2') is unchanged.",
        ),
        _s(
            "l1_liblinear",
            """
            from sklearn.linear_model import LogisticRegression


            def l1_predict(X, y, X_new):
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=10.0).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            X = [[0], [1], [2], [10], [11], [12]]
            y = [0, 0, 0, 1, 1, 1]
            assert solution.l1_predict(X, y, [[0.5], [11.5]]) == [0, 1]
            """,
            "LogisticRegression(penalty='l1', solver='liblinear') is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: tree criterion "mse"/"mae" removed (1.2) -> "squared_error"/"absolute_error"
# ---------------------------------------------------------------------------

TREE_CRITERION = _sklearn_change(
    "sklearn.tree.criterion_mse",
    'DecisionTreeRegressor(criterion="mse") / RandomForestRegressor(criterion="mae")',
    'criterion="squared_error" / criterion="absolute_error"',
    SKLEARN_DATE_12,
    SKLEARN_URL_12,
    "The regression criteria 'mse' and 'mae' were deprecated in 1.0 and removed in 1.2 in favour of 'squared_error' and 'absolute_error'.",
    [
        _p(
            "tree_mse",
            """
            from sklearn.tree import DecisionTreeRegressor


            def fit_tree(X, y):
                tree = DecisionTreeRegressor(criterion="mse", random_state=0).fit(X, y)
                return [round(float(v), 6) for v in tree.predict(X)]
            """,
            """
            from sklearn.tree import DecisionTreeRegressor


            def fit_tree(X, y):
                tree = DecisionTreeRegressor(criterion="squared_error", random_state=0).fit(X, y)
                return [round(float(v), 6) for v in tree.predict(X)]
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            y = [1.5, 0.5, 9.0, 4.0]
            assert solution.fit_tree(X, y) == [1.5, 0.5, 9.0, 4.0]
            """,
            "Squared-error regression tree fitted to completion.",
        ),
        _p(
            "forest_mae",
            """
            from sklearn.ensemble import RandomForestRegressor


            def forest_predict(X, y):
                forest = RandomForestRegressor(criterion="mae", n_estimators=3, bootstrap=False, random_state=0)
                forest.fit(X, y)
                return [round(float(v), 6) for v in forest.predict(X)]
            """,
            """
            from sklearn.ensemble import RandomForestRegressor


            def forest_predict(X, y):
                forest = RandomForestRegressor(criterion="absolute_error", n_estimators=3, bootstrap=False, random_state=0)
                forest.fit(X, y)
                return [round(float(v), 6) for v in forest.predict(X)]
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            y = [2.0, 8.0, 4.0, 6.0]
            assert solution.forest_predict(X, y) == [2.0, 8.0, 4.0, 6.0]
            """,
            "Absolute-error random forest without bootstrap.",
        ),
        _p(
            "extra_trees_model",
            """
            from sklearn.ensemble import ExtraTreesRegressor


            class ExtraTreesModel:
                def __init__(self):
                    self.model = ExtraTreesRegressor(criterion="mse", n_estimators=4, random_state=0)

                def fit(self, X, y):
                    self.model.fit(X, y)
                    return self

                def predict(self, X):
                    return [round(float(v), 6) for v in self.model.predict(X)]
            """,
            """
            from sklearn.ensemble import ExtraTreesRegressor


            class ExtraTreesModel:
                def __init__(self):
                    self.model = ExtraTreesRegressor(criterion="squared_error", n_estimators=4, random_state=0)

                def fit(self, X, y):
                    self.model.fit(X, y)
                    return self

                def predict(self, X):
                    return [round(float(v), 6) for v in self.model.predict(X)]
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
            y = [1.0, 3.0, 2.0, 5.0, 4.0]
            assert solution.ExtraTreesModel().fit(X, y).predict(X) == [1.0, 3.0, 2.0, 5.0, 4.0]
            """,
            "Extra-trees regressor fitted to completion.",
        ),
        _p(
            "leaf_count",
            """
            from sklearn.tree import DecisionTreeRegressor


            def leaves_and_depth(X, y):
                tree = DecisionTreeRegressor(criterion="mae", random_state=0).fit(X, y)
                return int(tree.get_n_leaves()), int(tree.get_depth())
            """,
            """
            from sklearn.tree import DecisionTreeRegressor


            def leaves_and_depth(X, y):
                tree = DecisionTreeRegressor(criterion="absolute_error", random_state=0).fit(X, y)
                return int(tree.get_n_leaves()), int(tree.get_depth())
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0]]
            y = [1.0, 2.0, 3.0, 4.0]
            leaves, depth = solution.leaves_and_depth(X, y)
            assert leaves == 4
            assert depth in (2, 3)
            """,
            "Leaf count of an absolute-error tree on distinct targets.",
        ),
        _p(
            "criterion_grid",
            """
            import numpy as np
            from sklearn.model_selection import GridSearchCV
            from sklearn.tree import DecisionTreeRegressor


            def search(X, y, splits):
                grid = GridSearchCV(DecisionTreeRegressor(random_state=0), {"criterion": ["mse", "friedman_mse"]}, cv=splits, error_score="raise")
                grid.fit(X, y)
                finite = bool(np.isfinite(grid.cv_results_["mean_test_score"]).all())
                return len(grid.cv_results_["params"]), finite, [round(float(v), 6) for v in grid.best_estimator_.predict(X)]
            """,
            """
            import numpy as np
            from sklearn.model_selection import GridSearchCV
            from sklearn.tree import DecisionTreeRegressor


            def search(X, y, splits):
                grid = GridSearchCV(DecisionTreeRegressor(random_state=0), {"criterion": ["squared_error", "friedman_mse"]}, cv=splits, error_score="raise")
                grid.fit(X, y)
                finite = bool(np.isfinite(grid.cv_results_["mean_test_score"]).all())
                return len(grid.cv_results_["params"]), finite, [round(float(v), 6) for v in grid.best_estimator_.predict(X)]
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
            y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            splits = [([0, 2, 4], [1, 3, 5]), ([1, 3, 5], [0, 2, 4])]
            n, finite, preds = solution.search(X, y, splits)
            assert n == 2
            assert finite is True
            assert preds == y
            """,
            "Grid search over regression criteria.",
        ),
        _p(
            "bagged_mse_trees",
            """
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor


            def bagged_predict(X, y):
                base = DecisionTreeRegressor(criterion="mse", random_state=0)
                model = BaggingRegressor(base, n_estimators=3, bootstrap=False, random_state=0).fit(X, y)
                return [round(float(v), 6) for v in model.predict(X)]
            """,
            """
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor


            def bagged_predict(X, y):
                base = DecisionTreeRegressor(criterion="squared_error", random_state=0)
                model = BaggingRegressor(base, n_estimators=3, bootstrap=False, random_state=0).fit(X, y)
                return [round(float(v), 6) for v in model.predict(X)]
            """,
            """
            import solution

            X = [[0.0], [1.0], [2.0]]
            y = [10.0, 20.0, 15.0]
            assert solution.bagged_predict(X, y) == [10.0, 20.0, 15.0]
            """,
            "Bagging of squared-error trees without bootstrap.",
        ),
    ],
    [
        _s(
            "entropy_tree",
            """
            from sklearn.tree import DecisionTreeClassifier


            def entropy_predict(X, y, X_new):
                clf = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
                return clf.predict(X_new).tolist()
            """,
            """
            import solution

            assert solution.entropy_predict([[0], [1], [10], [11]], [0, 0, 1, 1], [[0.5], [10.5]]) == [0, 1]
            """,
            "DecisionTreeClassifier(criterion='entropy') is unchanged.",
        ),
        _s(
            "friedman_forest",
            """
            from sklearn.ensemble import RandomForestRegressor


            def friedman_predict(X, y):
                forest = RandomForestRegressor(criterion="friedman_mse", n_estimators=2, bootstrap=False, random_state=0)
                return [round(float(v), 6) for v in forest.fit(X, y).predict(X)]
            """,
            """
            import solution

            assert solution.friedman_predict([[0.0], [1.0], [2.0]], [3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]
            """,
            "RandomForestRegressor(criterion='friedman_mse') is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: cross_val_score(fit_params=) removed (1.6) -> params=
# ---------------------------------------------------------------------------

CV_FIT_PARAMS = _sklearn_change(
    "sklearn.model_selection.cross_val_score.fit_params",
    "cross_val_score(..., fit_params={...})",
    "cross_val_score(..., params={...})",
    SKLEARN_DATE_16,
    SKLEARN_URL_16,
    "The fit_params argument of cross_val_score, cross_validate and cross_val_predict was deprecated in 1.4 and removed in 1.6; use params.",
    [
        _p(
            "weighted_cv_scores",
            """
            import numpy as np
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier


            def weighted_scores(X, y, weights):
                clf = DecisionTreeClassifier(random_state=0)
                scores = cross_val_score(clf, X, y, cv=2, fit_params={"sample_weight": np.asarray(weights)})
                return [float(s) for s in scores]
            """,
            """
            import numpy as np
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier


            def weighted_scores(X, y, weights):
                clf = DecisionTreeClassifier(random_state=0)
                scores = cross_val_score(clf, X, y, cv=2, params={"sample_weight": np.asarray(weights)})
                return [float(s) for s in scores]
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.weighted_scores(X, y, [1.0] * 8) == [1.0, 1.0]
            """,
            "Cross-validated accuracy with sample weights passed to fit.",
        ),
        _p(
            "cross_validate_weighted",
            """
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_validate


            def test_scores(X, y, weights):
                res = cross_validate(LogisticRegression(), X, y, cv=2, fit_params={"sample_weight": np.asarray(weights)})
                return [round(float(s), 6) for s in res["test_score"]]
            """,
            """
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_validate


            def test_scores(X, y, weights):
                res = cross_validate(LogisticRegression(), X, y, cv=2, params={"sample_weight": np.asarray(weights)})
                return [round(float(s), 6) for s in res["test_score"]]
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.test_scores(X, y, [2.0] * 8) == [1.0, 1.0]
            """,
            "cross_validate with weighted fitting.",
        ),
        _p(
            "oof_predictions",
            """
            import numpy as np
            from sklearn.model_selection import cross_val_predict
            from sklearn.tree import DecisionTreeClassifier


            def out_of_fold(X, y, weights):
                clf = DecisionTreeClassifier(random_state=0)
                preds = cross_val_predict(clf, X, y, cv=2, fit_params={"sample_weight": np.asarray(weights)})
                return preds.tolist()
            """,
            """
            import numpy as np
            from sklearn.model_selection import cross_val_predict
            from sklearn.tree import DecisionTreeClassifier


            def out_of_fold(X, y, weights):
                clf = DecisionTreeClassifier(random_state=0)
                preds = cross_val_predict(clf, X, y, cv=2, params={"sample_weight": np.asarray(weights)})
                return preds.tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.out_of_fold(X, y, [1.0] * 8) == y
            """,
            "Out-of-fold predictions with sample weights.",
        ),
        _p(
            "cv_runner",
            """
            import numpy as np
            from sklearn.model_selection import cross_val_score
            from sklearn.naive_bayes import GaussianNB


            class CVRunner:
                def __init__(self, folds):
                    self.folds = folds

                def mean_score(self, X, y, weights):
                    clf = GaussianNB()
                    scores = cross_val_score(clf, X, y, cv=self.folds, fit_params={"sample_weight": np.asarray(weights)})
                    return round(float(np.mean(scores)), 6)
            """,
            """
            import numpy as np
            from sklearn.model_selection import cross_val_score
            from sklearn.naive_bayes import GaussianNB


            class CVRunner:
                def __init__(self, folds):
                    self.folds = folds

                def mean_score(self, X, y, weights):
                    clf = GaussianNB()
                    scores = cross_val_score(clf, X, y, cv=self.folds, params={"sample_weight": np.asarray(weights)})
                    return round(float(np.mean(scores)), 6)
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.CVRunner(2).mean_score(X, y, [1.0] * 8) == 1.0
            """,
            "Mean cross-validated score of a weighted Gaussian naive Bayes classifier.",
        ),
        _p(
            "zero_weight_rows",
            """
            import numpy as np
            from sklearn.model_selection import cross_val_predict
            from sklearn.tree import DecisionTreeClassifier


            def predict_ignoring(X, y, ignore):
                weights = np.ones(len(y))
                for i in ignore:
                    weights[i] = 0.0
                clf = DecisionTreeClassifier(random_state=0)
                return cross_val_predict(clf, X, y, cv=2, fit_params={"sample_weight": weights}).tolist()
            """,
            """
            import numpy as np
            from sklearn.model_selection import cross_val_predict
            from sklearn.tree import DecisionTreeClassifier


            def predict_ignoring(X, y, ignore):
                weights = np.ones(len(y))
                for i in ignore:
                    weights[i] = 0.0
                clf = DecisionTreeClassifier(random_state=0)
                return cross_val_predict(clf, X, y, cv=2, params={"sample_weight": weights}).tolist()
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.predict_ignoring(X, y, [0, 4]) == y
            """,
            "Out-of-fold predictions with some rows zero-weighted.",
        ),
        _p(
            "train_and_test_scores",
            """
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_validate


            def train_test_scores(X, y, weights):
                forest = RandomForestClassifier(n_estimators=3, random_state=0)
                res = cross_validate(forest, X, y, cv=2, return_train_score=True,
                                     fit_params={"sample_weight": np.asarray(weights)})
                return [float(v) for v in res["train_score"]], [float(v) for v in res["test_score"]]
            """,
            """
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_validate


            def train_test_scores(X, y, weights):
                forest = RandomForestClassifier(n_estimators=3, random_state=0)
                res = cross_validate(forest, X, y, cv=2, return_train_score=True,
                                     params={"sample_weight": np.asarray(weights)})
                return [float(v) for v in res["train_score"]], [float(v) for v in res["test_score"]]
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            train, test = solution.train_test_scores(X, y, [1.0] * 8)
            assert train == [1.0, 1.0]
            assert test == [1.0, 1.0]
            """,
            "Weighted cross-validation returning train and test scores.",
        ),
    ],
    [
        _s(
            "scoring_argument",
            """
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier


            def balanced_scores(X, y):
                scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=2, scoring="balanced_accuracy")
                return [float(s) for s in scores]
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.balanced_scores(X, y) == [1.0, 1.0]
            """,
            "cross_val_score(scoring=) is unchanged.",
        ),
        _s(
            "proba_predictions",
            """
            from sklearn.model_selection import cross_val_predict
            from sklearn.tree import DecisionTreeClassifier


            def oof_proba_shape(X, y):
                proba = cross_val_predict(DecisionTreeClassifier(random_state=0), X, y, cv=2, method="predict_proba")
                return list(proba.shape)
            """,
            """
            import solution

            X = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
            y = [0, 0, 0, 0, 1, 1, 1, 1]
            assert solution.oof_proba_shape(X, y) == [8, 2]
            """,
            "cross_val_predict(method=) is unchanged.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# scikit-learn: tree/forest n_features_ attribute removed (1.2) -> n_features_in_
# ---------------------------------------------------------------------------

N_FEATURES = _sklearn_change(
    "sklearn.tree.n_features_",
    "estimator.n_features_",
    "estimator.n_features_in_",
    SKLEARN_DATE_12,
    SKLEARN_URL_12,
    "The n_features_ attribute of tree, forest and gradient boosting estimators was deprecated in 1.0 and removed in 1.2; use n_features_in_.",
    [
        _p(
            "feature_count",
            """
            from sklearn.tree import DecisionTreeClassifier


            def fitted_feature_count(X, y):
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return int(clf.n_features_)
            """,
            """
            from sklearn.tree import DecisionTreeClassifier


            def fitted_feature_count(X, y):
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return int(clf.n_features_in_)
            """,
            """
            import solution

            assert solution.fitted_feature_count([[0, 1, 2], [3, 4, 5]], [0, 1]) == 3
            """,
            "Number of features seen by a fitted tree.",
        ),
        _p(
            "forest_summary",
            """
            from sklearn.ensemble import RandomForestClassifier


            def summarize(X, y):
                clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
                return {"features": int(clf.n_features_), "classes": int(clf.n_classes_), "trees": len(clf.estimators_)}
            """,
            """
            from sklearn.ensemble import RandomForestClassifier


            def summarize(X, y):
                clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
                return {"features": int(clf.n_features_in_), "classes": int(clf.n_classes_), "trees": len(clf.estimators_)}
            """,
            """
            import solution

            X = [[0, 0], [1, 1], [10, 10], [11, 11]]
            y = [0, 0, 1, 1]
            assert solution.summarize(X, y) == {"features": 2, "classes": 2, "trees": 3}
            """,
            "Summary dictionary of a fitted random forest.",
        ),
        _p(
            "model_info",
            """
            from sklearn.tree import DecisionTreeRegressor


            class ModelInfo:
                def __init__(self, X, y):
                    self.model = DecisionTreeRegressor(random_state=0).fit(X, y)

                def describe(self):
                    return "tree with " + str(self.model.n_features_) + " features and " + str(self.model.get_n_leaves()) + " leaves"
            """,
            """
            from sklearn.tree import DecisionTreeRegressor


            class ModelInfo:
                def __init__(self, X, y):
                    self.model = DecisionTreeRegressor(random_state=0).fit(X, y)

                def describe(self):
                    return "tree with " + str(self.model.n_features_in_) + " features and " + str(self.model.get_n_leaves()) + " leaves"
            """,
            """
            import solution

            info = solution.ModelInfo([[0.0, 1.0], [1.0, 0.0], [2.0, 5.0]], [1.0, 2.0, 3.0])
            assert info.describe() == "tree with 2 features and 3 leaves"
            """,
            "Human-readable description of a fitted regression tree.",
        ),
        _p(
            "boosting_features",
            """
            from sklearn.ensemble import GradientBoostingRegressor


            def boosted_feature_count(X, y):
                model = GradientBoostingRegressor(n_estimators=2, random_state=0).fit(X, y)
                return int(model.n_features_)
            """,
            """
            from sklearn.ensemble import GradientBoostingRegressor


            def boosted_feature_count(X, y):
                model = GradientBoostingRegressor(n_estimators=2, random_state=0).fit(X, y)
                return int(model.n_features_in_)
            """,
            """
            import solution

            X = [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 2.0, 3.0], [2.0, 5.0, 1.0, 0.0]]
            assert solution.boosted_feature_count(X, [1.0, 2.0, 3.0]) == 4
            """,
            "Feature count recorded by a gradient boosting model.",
        ),
        _p(
            "shape_check",
            """
            from sklearn.ensemble import ExtraTreesClassifier


            def accepts(X, y, candidate):
                clf = ExtraTreesClassifier(n_estimators=2, random_state=0).fit(X, y)
                return len(candidate) == clf.n_features_
            """,
            """
            from sklearn.ensemble import ExtraTreesClassifier


            def accepts(X, y, candidate):
                clf = ExtraTreesClassifier(n_estimators=2, random_state=0).fit(X, y)
                return len(candidate) == clf.n_features_in_
            """,
            """
            import solution

            X = [[0, 0], [1, 1], [10, 10], [11, 11]]
            y = [0, 0, 1, 1]
            assert solution.accepts(X, y, [5, 5]) is True
            assert solution.accepts(X, y, [5]) is False
            """,
            "Check that a candidate row matches the fitted feature count.",
        ),
        _p(
            "importance_table",
            """
            from sklearn.tree import DecisionTreeClassifier


            def importance_by_index(X, y):
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return {i: round(float(clf.feature_importances_[i]), 6) for i in range(clf.n_features_)}
            """,
            """
            from sklearn.tree import DecisionTreeClassifier


            def importance_by_index(X, y):
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return {i: round(float(clf.feature_importances_[i]), 6) for i in range(clf.n_features_in_)}
            """,
            """
            import solution

            X = [[0, 7], [1, 7], [10, 7], [11, 7]]
            y = [0, 0, 1, 1]
            assert solution.importance_by_index(X, y) == {0: 1.0, 1: 0.0}
            """,
            "Feature importances keyed by feature index.",
        ),
    ],
    [
        _s(
            "class_count",
            """
            from sklearn.tree import DecisionTreeClassifier


            def class_and_output_counts(X, y):
                clf = DecisionTreeClassifier(random_state=0).fit(X, y)
                return int(clf.n_classes_), int(clf.n_outputs_)
            """,
            """
            import solution

            assert solution.class_and_output_counts([[0], [1], [2]], ["a", "b", "c"]) == (3, 1)
            """,
            "n_classes_ and n_outputs_ are unchanged.",
        ),
        _s(
            "node_count",
            """
            from sklearn.tree import DecisionTreeRegressor


            def node_count(X, y):
                tree = DecisionTreeRegressor(random_state=0).fit(X, y)
                return int(tree.tree_.node_count)
            """,
            """
            import solution

            assert solution.node_count([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0]) == 5
            """,
            "tree_.node_count is unchanged.",
        ),
    ],
)


CHANGES: list[ApiChange] = [
    DF_APPEND,
    SERIES_APPEND,
    ITERITEMS,
    LOOKUP,
    IS_MONOTONIC,
    MAD,
    NUMERIC_INDEX,
    ERROR_BAD_LINES,
    READ_SQUEEZE,
    PD_DATETIME,
    BETWEEN,
    WEEKOFYEAR,
    ONEHOT_SPARSE,
    SGD_LOG,
    LINREG_NORMALIZE,
    MSE_SQUARED,
    SCORER_NEEDS_PROBA,
    BASE_ESTIMATOR,
    PENALTY_NONE,
    TREE_CRITERION,
    CV_FIT_PARAMS,
    N_FEATURES,
]
