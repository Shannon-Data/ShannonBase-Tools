# Copyright (c) 2023, 2024, 2025 Shannon Data AI and/or its affiliates.
#
# Training-data generator for the Rapid selective-offload classifier
# (ShannonBase::ML::Query_arbitrator).
#
# Why this exists
# ---------------
# The classifier does not see SQL text.  It sees exactly the 18-float vector
# built by Query_arbitrator::features_to_vector(), which is filled in by
# Query_arbitrator::extract_features() in
# storage/rapid_engine/ml/query_arbitrator.cpp.  Any training row that could
# not have been produced by that function is a row the model will never meet in
# production, and every such row spends model capacity on an impossible region
# of the feature space.
#
# So this generator does NOT draw the 18 columns independently.  It draws a
# *query shape* (tables and their cardinalities, how the primary plan reaches
# them, clauses, subqueries) and then derives the 18 features with the same
# arithmetic extract_features() uses, honouring its invariants:
#
#   * has_join            == (table_count > 1)                       [hard]
#   * has_group_by        implies has_aggregation                    [hard]
#   * has_rollup          implies has_group_by                       [hard]
#   * count_ref_index_ts  <= count_all_base_tables                   [hard]
#   * count_all_base_tables <= table_count (derived tables add to
#     table_count only)                                              [hard]
#   * are_all_ts_index_ref -> f_mysql_total_ts_nrows == 0 before the
#     subquery walk adds inner-block scan rows                       [plan path]
#   * estimated_rows = base_table_sum_nrows, *0.1 if WHERE, *0.01 if
#     GROUP BY, then clamped by the LIMIT value                      [hard]
#   * fallback path (no cached primary plan):
#         mysql_cost = base_table_sum_nrows * 1.1 * max(table_count, 1)
#
# The label is not a tautology such as "has_group_by => OLAP".  It comes from a
# physical time model: the row is labelled 1 (offload to Rapid) when Rapid's
# estimated response time beats InnoDB's.  That makes the interesting cases
# fall out on their own -- an index-reached lookup into a 400M-row table is
# labelled 0 even though every "size" feature is huge, and a GROUP BY over a
# 4k-row table is labelled 0 because the offload overhead alone exceeds the
# whole InnoDB execution.  Those two families are where a naive generator
# teaches the model the wrong rule.
#
# Usage
#   python generate_offload_dataset.py                     # 5000 balanced rows
#   python generate_offload_dataset.py -n 20000 --seed 7
#   python generate_offload_dataset.py --stats             # per-archetype report

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Column order.  MUST match Query_arbitrator::features_to_vector() index 0..17.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "f_mysql_total_ts_nrows",   # 0
    "f_MySQLCost",              # 1
    "f_count_all_base_tables",  # 2
    "f_count_ref_index_ts",     # 3
    "f_BaseTableSumNrows",      # 4
    "f_are_all_ts_index_ref",   # 5
    "table_count",              # 6
    "has_having",               # 7
    "has_group_by",             # 8
    "has_rollup",               # 9
    "has_order_by",             # 10
    "has_limit",                # 11
    "has_join",                 # 12
    "has_subquery",             # 13
    "has_aggregation",          # 14
    "select_list_size",         # 15
    "where_condition_count",    # 16
    "estimated_rows",           # 17
]
LABEL_COLUMN = "IS_OLAP"

# ---------------------------------------------------------------------------
# Physical cost model used only to derive the label.
# Units are seconds.  The constants are order-of-magnitude figures for a warm
# buffer pool and an in-memory Rapid population; the classifier only needs the
# *ordering* of the two estimates to be right, not their absolute values.
# ---------------------------------------------------------------------------
INNODB_SCAN_ROW = 2.5e-7      # sequential row-store row, warm pages
INNODB_INDEX_LOOKUP = 1.8e-6  # one B-tree descent + row fetch
INNODB_SORT_ROW = 1.6e-7      # filesort per row (x log2 n)
INNODB_TMP_AGG_ROW = 3.0e-7   # temp-table GROUP BY, per input row
INNODB_FIXED = 6.0e-5         # parse/optimize/latch overhead

RAPID_FIXED = 1.2e-3          # offload decision, plan ship, secondary prepare
RAPID_DOP = 8                 # scan degree of parallelism
RAPID_SCAN_CELL = 1.2e-9      # one projected column value, one thread
RAPID_HASH_AGG_ROW = 2.0e-9   # per input row, one thread
RAPID_HASH_JOIN_ROW = 4.0e-9  # build+probe per inner row, one thread
RAPID_SHIP_ROW = 8.0e-7       # materialise a result row back to the server

# MySQL optimizer cost constants (used to synthesise f_MySQLCost on the plan
# path, mirroring what Secondary_engine_statement_context caches).
MYSQL_ROW_EVAL = 0.1
MYSQL_BLOCK_READ = 1.0
MYSQL_ROWS_PER_BLOCK = 30.0


@dataclass
class TableRef:
    """One entry of Query_block::get_table_list()."""
    rows: float                  # handler row count (0 for derived tables)
    is_derived: bool = False     # view or derived table
    index_reached: bool = False  # the primary plan reaches it via ref/eq_ref
    has_key: bool = True         # TABLE_SHARE::keys > 0


@dataclass
class SubBlock:
    """An inner Query_block reached by walk_item_for_subqueries()."""
    tables: List[TableRef]
    has_where: bool = True


@dataclass
class QueryShape:
    kind: str
    tables: List[TableRef]
    select_list_size: int
    where_condition_count: int = 0
    has_group_by: bool = False
    has_having: bool = False
    has_rollup: bool = False
    has_order_by: bool = False
    limit_value: Optional[float] = None
    has_agg_func: bool = False        # SUM/AVG/COUNT without GROUP BY
    sub_blocks: List[SubBlock] = field(default_factory=list)
    semijoin_or_scalar: bool = False  # sets has_subquery without a derived table

    # Physical facts, used only by the labeller -- never exposed as a feature.
    innodb_scan_rows: float = 0.0
    innodb_index_lookups: float = 0.0
    out_rows: float = 1.0
    group_cardinality: float = 1.0
    rapid_prune: float = 1.0          # zone-map pruning factor in (0, 1]
    const_plan: bool = False          # plan folded to const tables (cost << 1)
    const_cost: float = 0.1

    @property
    def has_where(self) -> bool:
        return self.where_condition_count > 0

    @property
    def base_tables(self) -> List[TableRef]:
        return [t for t in self.tables if not t.is_derived]


# ---------------------------------------------------------------------------
# Feature derivation -- a faithful port of extract_features().
# ---------------------------------------------------------------------------
def _accumulate_subquery_cost(blocks: List[SubBlock]):
    """Port of accumulate_subquery_cost() / walk_item_for_subqueries()."""
    scan_rows = 0.0
    cost = 0.0
    for blk in blocks:
        for t in blk.tables:
            if t.is_derived:
                continue
            if blk.has_where and t.has_key:
                scan_rows += t.rows * 0.1
                cost += t.rows * 0.1 * 1.1
            else:
                scan_rows += t.rows
                cost += t.rows * 1.1
    return scan_rows, cost


def _synthesise_primary_cost(shape: QueryShape) -> float:
    """What Secondary_engine_statement_context::get_primary_cost() would hold."""
    if shape.const_plan:
        # A plan folded to const/eq_ref tables costs a fraction of a unit;
        # tpch_sf1 measured 0.099 for a two-table PK-equality join.
        return shape.const_cost
    cost = shape.innodb_scan_rows * (
        MYSQL_ROW_EVAL + MYSQL_BLOCK_READ / MYSQL_ROWS_PER_BLOCK
    )
    cost += shape.innodb_index_lookups * (MYSQL_BLOCK_READ + MYSQL_ROW_EVAL)
    if shape.has_group_by or shape.has_agg_func:
        cost += shape.innodb_scan_rows * MYSQL_ROW_EVAL * 0.5
    if shape.has_order_by:
        cost += shape.out_rows * MYSQL_ROW_EVAL
    return max(cost, 0.05)


def derive_features(shape: QueryShape, have_primary_plan: bool) -> dict:
    f = {}

    table_count = len(shape.tables)
    base = shape.base_tables
    base_table_count = len(base)
    base_table_sum_nrows = sum(t.rows for t in base)

    f["table_count"] = table_count
    f["f_count_all_base_tables"] = base_table_count
    f["f_BaseTableSumNrows"] = base_table_sum_nrows

    # --- heuristic path, always computed first in the C++ ------------------
    total_ts_nrows = 0.0
    count_ref_index_ts = 0
    are_all_ts_index_ref = True
    for t in base:
        if shape.has_where and t.has_key:
            count_ref_index_ts += 1
        else:
            total_ts_nrows += t.rows
            are_all_ts_index_ref = False
    mysql_cost = base_table_sum_nrows * 1.1 * (table_count if table_count > 1 else 1)

    # --- plan path overrides four features ---------------------------------
    if have_primary_plan and base_table_count > 0:
        mysql_cost = _synthesise_primary_cost(shape)
        count_ref_index_ts = sum(1 for t in base if t.index_reached)
        are_all_ts_index_ref = all(t.index_reached for t in base)
        total_ts_nrows = 0.0 if are_all_ts_index_ref else base_table_sum_nrows

    f["f_mysql_total_ts_nrows"] = total_ts_nrows
    f["f_MySQLCost"] = mysql_cost
    f["f_count_ref_index_ts"] = count_ref_index_ts
    f["f_are_all_ts_index_ref"] = 1 if are_all_ts_index_ref else 0

    # --- estimated_rows ----------------------------------------------------
    est = base_table_sum_nrows
    if shape.has_where:
        est *= 0.1
    if shape.has_group_by:
        est *= 0.01
    if shape.limit_value is not None and shape.limit_value < est:
        est = float(shape.limit_value)
    f["estimated_rows"] = est

    # --- clause flags ------------------------------------------------------
    has_subquery = bool(shape.sub_blocks) or shape.semijoin_or_scalar
    f["has_having"] = 1 if shape.has_having else 0
    f["has_group_by"] = 1 if shape.has_group_by else 0
    f["has_rollup"] = 1 if shape.has_rollup else 0
    f["has_order_by"] = 1 if shape.has_order_by else 0
    f["has_limit"] = 1 if shape.limit_value is not None else 0
    f["has_join"] = 1 if table_count > 1 else 0
    f["has_subquery"] = 1 if has_subquery else 0
    # extract_features(): has_aggregation = has_group_by, then OR'd with a scan
    # of the select list for SUM_FUNC_ITEM.
    f["has_aggregation"] = 1 if (shape.has_group_by or shape.has_agg_func) else 0

    f["select_list_size"] = shape.select_list_size
    f["where_condition_count"] = shape.where_condition_count

    # --- subquery walk, applied last, in both paths ------------------------
    if has_subquery:
        sub_scan, sub_cost = _accumulate_subquery_cost(shape.sub_blocks)
        f["f_mysql_total_ts_nrows"] += sub_scan
        f["f_MySQLCost"] += sub_cost

    return f


# ---------------------------------------------------------------------------
# Labeller: estimated response time, InnoDB vs Rapid.
# ---------------------------------------------------------------------------
def estimate_innodb_seconds(shape: QueryShape) -> float:
    t = INNODB_FIXED
    t += shape.innodb_scan_rows * INNODB_SCAN_ROW
    t += shape.innodb_index_lookups * INNODB_INDEX_LOOKUP
    if shape.has_group_by or shape.has_agg_func:
        # No columnar hash aggregation on the primary: temp table or sort.
        t += (shape.innodb_scan_rows + shape.innodb_index_lookups) * INNODB_TMP_AGG_ROW
    if shape.has_order_by:
        n = max(shape.out_rows, 1.0)
        t += n * INNODB_SORT_ROW * max(math.log2(n), 1.0)
    return t


def estimate_rapid_seconds(shape: QueryShape) -> float:
    # Rapid has no secondary indexes: it scans the base tables, minus whatever
    # zone maps prune.
    scanned = sum(t.rows for t in shape.base_tables) * shape.rapid_prune
    scanned += sum(t.rows for b in shape.sub_blocks for t in b.tables if not t.is_derived)

    cols = max(shape.select_list_size, 1) + shape.where_condition_count
    t = RAPID_FIXED
    t += scanned * cols * RAPID_SCAN_CELL / RAPID_DOP
    if shape.has_group_by or shape.has_agg_func:
        t += scanned * RAPID_HASH_AGG_ROW / RAPID_DOP
    if len(shape.base_tables) > 1:
        inner = sum(sorted((t_.rows for t_ in shape.base_tables), reverse=True)[1:])
        t += inner * RAPID_HASH_JOIN_ROW / RAPID_DOP
    t += shape.out_rows * RAPID_SHIP_ROW
    return t


def label_shape(shape: QueryShape, rng: random.Random) -> int:
    """1 = offload to Rapid (analytical), 0 = keep on InnoDB (transactional)."""
    jitter = lambda: math.exp(rng.gauss(0.0, 0.18))
    return 1 if estimate_rapid_seconds(shape) * jitter() < \
                estimate_innodb_seconds(shape) * jitter() else 0


# ---------------------------------------------------------------------------
# Archetypes.  Each returns a QueryShape.
# ---------------------------------------------------------------------------
def _logu(rng, lo, hi) -> float:
    return float(int(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


# ---- transactional shapes -------------------------------------------------
def a_point_lookup(rng):
    n = _logu(rng, 1e3, 4e8)
    where = rng.choice([1, 1, 1, 2])
    return QueryShape(
        kind="point_lookup",
        tables=[TableRef(rows=n, index_reached=True)],
        select_list_size=rng.randint(1, 12),
        where_condition_count=where,
        limit_value=rng.choice([None, 1.0, 1.0]),
        innodb_index_lookups=1.0, innodb_scan_rows=0.0, out_rows=1.0,
        rapid_prune=rng.uniform(0.05, 0.6),
    )


def a_index_range(rng):
    n = _logu(rng, 1e4, 2e8)
    hits = min(n, _logu(rng, 1, 5e3))
    lim = rng.choice([None, 10.0, 20.0, 50.0, 100.0])
    return QueryShape(
        kind="index_range",
        tables=[TableRef(rows=n, index_reached=True)],
        select_list_size=rng.randint(2, 15),
        where_condition_count=rng.randint(1, 4),
        has_order_by=rng.random() < 0.7,
        limit_value=lim,
        innodb_index_lookups=hits,
        out_rows=min(hits, lim if lim else hits),
        rapid_prune=rng.uniform(0.1, 0.8),
    )


def a_pk_join(rng):
    k = rng.randint(2, 5)
    tables = [TableRef(rows=_logu(rng, 1e3, 2e8), index_reached=True) for _ in range(k)]
    drive = _logu(rng, 1, 500)
    lim = rng.choice([None, 20.0, 100.0])
    return QueryShape(
        kind="pk_join",
        tables=tables,
        select_list_size=rng.randint(3, 20),
        where_condition_count=rng.randint(1, 5),
        has_order_by=rng.random() < 0.5,
        limit_value=lim,
        innodb_index_lookups=drive * k,
        out_rows=min(drive, lim if lim else drive),
        rapid_prune=rng.uniform(0.1, 0.9),
    )


def a_indexed_count(rng):
    """COUNT(*)/SUM() behind an equality on an indexed column: aggregation, but
    the plan never scans.  The family the model must NOT call analytical."""
    n = _logu(rng, 1e4, 4e8)
    hits = min(n, _logu(rng, 1, 2e4))
    return QueryShape(
        kind="indexed_agg",
        tables=[TableRef(rows=n, index_reached=True)],
        select_list_size=rng.randint(1, 4),
        where_condition_count=rng.randint(1, 3),
        has_agg_func=True,
        has_group_by=rng.random() < 0.25,
        innodb_index_lookups=hits,
        out_rows=1.0 if rng.random() < 0.7 else min(hits, 50.0),
        group_cardinality=1.0,
        rapid_prune=rng.uniform(0.05, 0.5),
    )


def a_exists_oltp(rng):
    outer = TableRef(rows=_logu(rng, 1e4, 5e7), index_reached=True)
    inner = TableRef(rows=_logu(rng, 1e3, 5e6), index_reached=True)
    return QueryShape(
        kind="exists_oltp",
        tables=[outer],
        select_list_size=rng.randint(2, 10),
        where_condition_count=rng.randint(1, 3),
        sub_blocks=[SubBlock(tables=[inner], has_where=True)],
        semijoin_or_scalar=True,
        limit_value=rng.choice([None, 10.0, 100.0]),
        innodb_index_lookups=_logu(rng, 1, 2e3),
        out_rows=_logu(rng, 1, 200),
        rapid_prune=rng.uniform(0.1, 0.7),
    )


# ---- analytical shapes ----------------------------------------------------
def a_scan_agg(rng):
    n = _logu(rng, 5e5, 1e9)
    gc = _logu(rng, 1, 5e4)
    return QueryShape(
        kind="scan_agg",
        tables=[TableRef(rows=n, index_reached=False)],
        select_list_size=rng.randint(3, 25),
        where_condition_count=rng.randint(0, 5),
        has_group_by=True,
        has_having=rng.random() < 0.35,
        has_rollup=rng.random() < 0.12,
        has_order_by=rng.random() < 0.6,
        limit_value=rng.choice([None, None, 10.0, 100.0]),
        innodb_scan_rows=n,
        out_rows=gc, group_cardinality=gc,
        rapid_prune=rng.uniform(0.4, 1.0),
    )


def a_star_join(rng):
    fact = TableRef(rows=_logu(rng, 1e6, 6e8), index_reached=False)
    dims = [TableRef(rows=_logu(rng, 5, 5e5), index_reached=rng.random() < 0.35)
            for _ in range(rng.randint(2, 7))]
    gc = _logu(rng, 5, 2e5)
    lim = rng.choice([None, None, 10.0, 20.0, 100.0])
    return QueryShape(
        kind="star_join_agg",
        tables=[fact] + dims,
        select_list_size=rng.randint(4, 30),
        where_condition_count=rng.randint(1, 8),
        has_group_by=True,
        has_having=rng.random() < 0.4,
        has_order_by=rng.random() < 0.85,
        limit_value=lim,
        innodb_scan_rows=fact.rows + sum(d.rows for d in dims),
        out_rows=min(gc, lim if lim else gc), group_cardinality=gc,
        rapid_prune=rng.uniform(0.3, 1.0),
    )


def a_etl_scan(rng):
    """Wide unaggregated export: no GROUP BY at all, still belongs on Rapid."""
    n = _logu(rng, 2e6, 8e8)
    sel = rng.uniform(0.02, 0.6)
    return QueryShape(
        kind="etl_scan",
        tables=[TableRef(rows=n, index_reached=False)],
        select_list_size=rng.randint(8, 40),
        where_condition_count=rng.randint(0, 4),
        has_order_by=rng.random() < 0.3,
        innodb_scan_rows=n,
        out_rows=n * sel,
        rapid_prune=rng.uniform(0.5, 1.0),
    )


def a_nested_analytic(rng):
    outer = TableRef(rows=_logu(rng, 5e5, 3e8), index_reached=False)
    inner = [TableRef(rows=_logu(rng, 1e5, 1e8), index_reached=False)]
    derived = rng.random() < 0.5
    tables = [outer]
    if derived:
        tables.append(TableRef(rows=0.0, is_derived=True))
    gc = _logu(rng, 10, 1e5)
    return QueryShape(
        kind="nested_analytic",
        tables=tables,
        select_list_size=rng.randint(3, 20),
        where_condition_count=rng.randint(0, 5),
        has_group_by=rng.random() < 0.8,
        has_having=rng.random() < 0.4,
        has_order_by=rng.random() < 0.7,
        limit_value=rng.choice([None, None, 100.0]),
        sub_blocks=[SubBlock(tables=inner, has_where=rng.random() < 0.6)],
        semijoin_or_scalar=not derived,
        innodb_scan_rows=outer.rows + sum(t.rows for t in inner),
        out_rows=gc, group_cardinality=gc,
        rapid_prune=rng.uniform(0.4, 1.0),
    )


# ---- the ambiguous band ---------------------------------------------------
def a_huge_but_indexed(rng):
    """Every size feature is enormous, yet the plan touches a handful of rows.
    Must stay on InnoDB."""
    k = rng.randint(1, 4)
    tables = [TableRef(rows=_logu(rng, 5e7, 1e9), index_reached=True) for _ in range(k)]
    hits = _logu(rng, 1, 3e3)
    return QueryShape(
        kind="huge_but_indexed",
        tables=tables,
        select_list_size=rng.randint(2, 18),
        where_condition_count=rng.randint(1, 6),
        has_group_by=rng.random() < 0.3,
        has_having=rng.random() < 0.1,
        has_order_by=rng.random() < 0.6,
        limit_value=rng.choice([None, 10.0, 50.0, 200.0]),
        innodb_index_lookups=hits * k,
        out_rows=min(hits, 500.0),
        rapid_prune=rng.uniform(0.2, 1.0),
    )


def a_small_table_report(rng):
    """GROUP BY + HAVING + ORDER BY over a few thousand rows: analytical in
    shape, but the offload overhead alone loses to InnoDB."""
    k = rng.randint(1, 4)
    tables = [TableRef(rows=_logu(rng, 50, 3e4), index_reached=rng.random() < 0.4)
              for _ in range(k)]
    total = sum(t.rows for t in tables)
    gc = max(1.0, min(total, _logu(rng, 1, 500)))
    return QueryShape(
        kind="small_table_report",
        tables=tables,
        select_list_size=rng.randint(3, 15),
        where_condition_count=rng.randint(0, 4),
        has_group_by=True,
        has_having=rng.random() < 0.5,
        has_rollup=rng.random() < 0.15,
        has_order_by=rng.random() < 0.8,
        limit_value=rng.choice([None, 10.0, 100.0]),
        innodb_scan_rows=total,
        out_rows=gc, group_cardinality=gc,
        rapid_prune=rng.uniform(0.6, 1.0),
    )


def a_medium_scan(rng):
    """The genuine coin-flip zone: 10^5..10^7 rows, no index, no aggregation."""
    n = _logu(rng, 5e4, 2e7)
    return QueryShape(
        kind="medium_scan",
        tables=[TableRef(rows=n, index_reached=False)],
        select_list_size=rng.randint(1, 20),
        where_condition_count=rng.randint(0, 5),
        has_group_by=rng.random() < 0.4,
        has_agg_func=rng.random() < 0.3,
        has_order_by=rng.random() < 0.5,
        limit_value=rng.choice([None, 10.0, 100.0, 1000.0]),
        innodb_scan_rows=n,
        out_rows=_logu(rng, 1, max(2.0, n * 0.1)),
        rapid_prune=rng.uniform(0.3, 1.0),
    )


def a_mixed_join(rng):
    """Some tables index-reached, some scanned -- are_all_ts_index_ref is 0 but
    only part of the input is really read."""
    k = rng.randint(2, 6)
    tables = []
    for i in range(k):
        idx = rng.random() < 0.6
        tables.append(TableRef(rows=_logu(rng, 1e3, 3e8), index_reached=idx))
    if all(t.index_reached for t in tables):
        tables[0].index_reached = False
    scanned = sum(t.rows for t in tables if not t.index_reached)
    lookups = sum(1 for t in tables if t.index_reached) * _logu(rng, 1, 1e4)
    gc = _logu(rng, 1, 1e5)
    grp = rng.random() < 0.5
    return QueryShape(
        kind="mixed_join",
        tables=tables,
        select_list_size=rng.randint(2, 25),
        where_condition_count=rng.randint(1, 7),
        has_group_by=grp,
        has_having=grp and rng.random() < 0.4,
        has_order_by=rng.random() < 0.6,
        limit_value=rng.choice([None, 10.0, 100.0, 1000.0]),
        innodb_scan_rows=scanned,
        innodb_index_lookups=lookups,
        out_rows=gc if grp else _logu(rng, 1, 1e4),
        rapid_prune=rng.uniform(0.2, 1.0),
    )


def a_const_join(rng):
    """PK-equality lookup that the optimizer folds into const tables: the cost
    the primary reports is a fraction of a unit, far below anything the cost
    formula above produces.  Observed on tpch_sf1 as f_MySQLCost=0.099."""
    k = rng.randint(1, 3)
    tables = [TableRef(rows=_logu(rng, 1e3, 5e8), index_reached=rng.random() < 0.5)
              for _ in range(k)]
    return QueryShape(
        kind="const_join",
        tables=tables,
        select_list_size=rng.randint(1, 20),
        where_condition_count=rng.randint(1, 4),
        has_order_by=rng.random() < 0.3,
        limit_value=rng.choice([None, 1.0, 10.0]),
        innodb_index_lookups=rng.randint(1, 4),
        out_rows=rng.randint(1, 5),
        const_plan=True, const_cost=rng.uniform(0.05, 2.5),
        rapid_prune=rng.uniform(0.05, 0.5),
    )


def a_tiny_dim_join(rng):
    """A join of lookup/dimension tables of a few dozen rows -- tpch NATION and
    REGION are 25 and 5.  Everything is scanned, so are_all_ts_index_ref is 0,
    but the whole query is cheaper than the offload handshake."""
    k = rng.randint(1, 4)
    tables = [TableRef(rows=_logu(rng, 2, 2e3), index_reached=rng.random() < 0.3)
              for _ in range(k)]
    total = sum(t.rows for t in tables)
    grp = rng.random() < 0.4
    return QueryShape(
        kind="tiny_dim_join",
        tables=tables,
        select_list_size=rng.randint(1, 12),
        where_condition_count=rng.randint(0, 3),
        has_group_by=grp,
        has_having=grp and rng.random() < 0.3,
        has_order_by=rng.random() < 0.6,
        limit_value=rng.choice([None, 10.0, 100.0]),
        innodb_scan_rows=total,
        out_rows=max(1.0, total * rng.uniform(0.05, 1.0)),
        rapid_prune=rng.uniform(0.7, 1.0),
    )


ARCHETYPES = [
    # (sampler, weight)
    (a_point_lookup, 8),
    (a_index_range, 10),
    (a_pk_join, 8),
    (a_indexed_count, 7),
    (a_exists_oltp, 5),
    (a_scan_agg, 10),
    (a_star_join, 10),
    (a_etl_scan, 6),
    (a_nested_analytic, 7),
    (a_huge_but_indexed, 9),
    (a_small_table_report, 8),
    (a_medium_scan, 9),
    (a_mixed_join, 10),
    (a_const_join, 7),
    (a_tiny_dim_join, 6),
]


# ---------------------------------------------------------------------------
# Sampling with class balancing.
# ---------------------------------------------------------------------------
def sample_rows(n_rows, seed, olap_ratio, plan_ratio, collect_kinds=False):
    rng = random.Random(seed)
    samplers = [s for s, w in ARCHETYPES for _ in range(w)]

    want1 = int(round(n_rows * olap_ratio))
    want0 = n_rows - want1
    got1 = got0 = 0
    rows, kinds = [], []

    guard = 0
    while (got0 < want0 or got1 < want1) and guard < n_rows * 400:
        guard += 1
        shape = rng.choice(samplers)(rng)
        label = label_shape(shape, rng)
        if label == 1 and got1 >= want1:
            continue
        if label == 0 and got0 >= want0:
            continue
        feats = derive_features(shape, have_primary_plan=rng.random() < plan_ratio)
        rows.append((feats, label))
        kinds.append(shape.kind)
        got1 += label
        got0 += 1 - label

    rng.shuffle(rows)
    return (rows, kinds) if collect_kinds else (rows, None)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FEATURE_COLUMNS + [LABEL_COLUMN])
        for feats, label in rows:
            out = []
            for c in FEATURE_COLUMNS:
                v = feats[c]
                out.append(int(v) if isinstance(v, int) else round(float(v), 6))
            w.writerow(out + [label])


# ---------------------------------------------------------------------------
# Golden set: hand-encoded real workloads (TPC-H SF1 and sysbench OLTP).
# Never used for training -- it is the external check that the synthetic
# distribution actually transfers.
# ---------------------------------------------------------------------------
TPCH_SF1 = dict(lineitem=6001215, orders=1500000, partsupp=800000, part=200000,
                customer=150000, supplier=10000, nation=25, region=5)
SB = 10000000  # sysbench sbtest1 rows


def _tpch(kind, tabs, sel, wcnt, group=True, having=False, order=True,
          limit=None, subq=None, agg=True):
    tables = [TableRef(rows=TPCH_SF1[t], index_reached=False) for t in tabs]
    total = sum(t.rows for t in tables)
    sub_blocks = [SubBlock(tables=[TableRef(rows=TPCH_SF1[t]) for t in subq],
                           has_where=True)] if subq else []
    return QueryShape(
        kind=kind, tables=tables, select_list_size=sel, where_condition_count=wcnt,
        has_group_by=group, has_having=having, has_order_by=order,
        limit_value=limit, has_agg_func=agg, sub_blocks=sub_blocks,
        innodb_scan_rows=total, out_rows=100.0, rapid_prune=0.8,
    )


def golden_shapes():
    g = []
    # --- TPC-H SF1, expected label 1 -------------------------------------
    g += [
        (_tpch("tpch_q1", ["lineitem"], 10, 1, having=False, limit=None), 1),
        (_tpch("tpch_q3", ["customer", "orders", "lineitem"], 4, 5, limit=10), 1),
        (_tpch("tpch_q5", ["customer", "orders", "lineitem", "supplier", "nation",
                           "region"], 2, 6), 1),
        (_tpch("tpch_q6", ["lineitem"], 1, 4, group=False, order=False), 1),
        (_tpch("tpch_q9", ["part", "supplier", "lineitem", "partsupp", "orders",
                           "nation"], 3, 5), 1),
        (_tpch("tpch_q10", ["customer", "orders", "lineitem", "nation"], 8, 4,
               limit=20), 1),
        (_tpch("tpch_q12", ["orders", "lineitem"], 3, 6), 1),
        (_tpch("tpch_q14", ["lineitem", "part"], 1, 3, group=False, order=False), 1),
        (_tpch("tpch_q18", ["customer", "orders", "lineitem"], 6, 1, limit=100,
               subq=["lineitem"]), 1),
        (_tpch("tpch_q19", ["lineitem", "part"], 1, 9, group=False, order=False), 1),
        (_tpch("tpch_q21", ["supplier", "lineitem", "orders", "nation"], 2, 5,
               limit=100, subq=["lineitem"]), 1),
        (_tpch("tpch_q22", ["customer", "orders"], 3, 3, subq=["customer"]), 1),
    ]
    # --- sysbench OLTP, expected label 0 ---------------------------------
    g += [
        (QueryShape(kind="sb_point_select", tables=[TableRef(rows=SB, index_reached=True)],
                    select_list_size=1, where_condition_count=1,
                    innodb_index_lookups=1, out_rows=1, rapid_prune=0.1), 0),
        (QueryShape(kind="sb_simple_range", tables=[TableRef(rows=SB, index_reached=True)],
                    select_list_size=1, where_condition_count=2,
                    innodb_index_lookups=100, out_rows=100, rapid_prune=0.2), 0),
        (QueryShape(kind="sb_sum_range", tables=[TableRef(rows=SB, index_reached=True)],
                    select_list_size=1, where_condition_count=2, has_agg_func=True,
                    innodb_index_lookups=100, out_rows=1, rapid_prune=0.2), 0),
        (QueryShape(kind="sb_order_range", tables=[TableRef(rows=SB, index_reached=True)],
                    select_list_size=1, where_condition_count=2, has_order_by=True,
                    innodb_index_lookups=100, out_rows=100, rapid_prune=0.2), 0),
        (QueryShape(kind="sb_distinct_range", tables=[TableRef(rows=SB, index_reached=True)],
                    select_list_size=1, where_condition_count=2, has_order_by=True,
                    has_group_by=True, innodb_index_lookups=100, out_rows=100,
                    rapid_prune=0.2), 0),
        # Typical web-app reads.
        (QueryShape(kind="app_user_by_email",
                    tables=[TableRef(rows=4.2e6, index_reached=True)],
                    select_list_size=9, where_condition_count=1, limit_value=1,
                    innodb_index_lookups=1, out_rows=1, rapid_prune=0.05), 0),
        (QueryShape(kind="app_recent_orders",
                    tables=[TableRef(rows=8.0e7, index_reached=True),
                            TableRef(rows=4.2e6, index_reached=True)],
                    select_list_size=12, where_condition_count=2, has_order_by=True,
                    limit_value=25, innodb_index_lookups=50, out_rows=25,
                    rapid_prune=0.1), 0),
        (QueryShape(kind="app_cart_count",
                    tables=[TableRef(rows=1.1e7, index_reached=True)],
                    select_list_size=1, where_condition_count=1, has_agg_func=True,
                    innodb_index_lookups=7, out_rows=1, rapid_prune=0.05), 0),
        # Dashboard aggregates over a real warehouse table -> Rapid.
        (QueryShape(kind="app_daily_revenue",
                    tables=[TableRef(rows=8.0e7, index_reached=False),
                            TableRef(rows=1.2e5, index_reached=False)],
                    select_list_size=5, where_condition_count=2, has_group_by=True,
                    has_order_by=True, innodb_scan_rows=8.012e7, out_rows=365,
                    rapid_prune=0.9), 1),
        (QueryShape(kind="app_cohort_rollup",
                    tables=[TableRef(rows=3.0e8, index_reached=False)],
                    select_list_size=7, where_condition_count=3, has_group_by=True,
                    has_rollup=True, has_having=True, has_order_by=True,
                    innodb_scan_rows=3.0e8, out_rows=5000, rapid_prune=1.0), 1),
    ]
    return g


def write_golden(path, plan_path=True):
    rows = []
    for shape, expected in golden_shapes():
        rows.append((derive_features(shape, have_primary_plan=plan_path), expected))
    write_csv(path, rows)
    return len(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--olap-ratio", type=float, default=0.5)
    ap.add_argument("--plan-ratio", type=float, default=0.88,
                    help="fraction of rows taken from the cached-primary-plan "
                         "path of extract_features(); the rest use the "
                         "heuristic fallback")
    ap.add_argument("-o", "--out", default="mysql_offload_balanced_5000_IS_OLAP.csv")
    ap.add_argument("--golden-out", default="golden_tpch_sysbench.csv")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    rows, kinds = sample_rows(args.n_rows, args.seed, args.olap_ratio,
                              args.plan_ratio, collect_kinds=args.stats)
    write_csv(args.out, rows)
    pos = sum(l for _, l in rows)
    print(f"wrote {args.out}: {len(rows)} rows, OLAP={pos} "
          f"({pos / max(len(rows), 1):.1%}), OLTP={len(rows) - pos}")

    if not args.no_golden:
        n = write_golden(args.golden_out)
        print(f"wrote {args.golden_out}: {n} hand-encoded TPC-H / sysbench rows")

    if args.stats:
        from collections import Counter
        c = Counter(kinds)
        print("\nper-archetype counts:")
        for k, v in c.most_common():
            print(f"  {k:22s} {v:6d}")


if __name__ == "__main__":
    main()
