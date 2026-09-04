using this tools to generate the model file, `shannon_rapid_classifier.onnx`, which is classifier to determine
the type of the query statement. Analytical workload or transactional workload.

There're 18 features used, listed below.

```
    double mysql_total_ts_nrows;  // f_mysql_total_ts_nrows: Total table scan rows
    double mysql_cost;            // f_MySQLCost: MySQL optimizer cost
    int count_all_base_tables;    // f_count_all_base_tables: Number of base tables
    int count_ref_index_ts;       // f_count_ref_index_ts: Number of index ref accesses
    double base_table_sum_nrows;  // f_BaseTableSumNrows: Sum of all base table rows
    bool are_all_ts_index_ref;    // f_are_all_ts_index_ref: All tables use index?

    // ===== Additional OLAP/OLTP detection features =====
    int table_count;            // Total table count in query
    bool has_having;            // Has HAVING clause
    bool has_group_by;          // Has GROUP BY
    bool has_rollup;            // Has ROLLUP
    bool has_order_by;          // Has ORDER BY
    bool has_limit;             // Has LIMIT
    bool has_join;              // Has JOIN operations
    bool has_subquery;          // Has subqueries
    bool has_aggregation;       // Has aggregation functions (SUM, AVG, COUNT, etc.)
    int select_list_size;       // Number of items in SELECT list
    int where_condition_count;  // Number of WHERE conditions
    double estimated_rows;      // Estimated result rows

```

If your traninng your own model, pls given the data ordered as described above.
```
+---------------------------------------------------------------------------------------
+ total_ts_nrows | cost | n_all_base_tables | n_ref_index | nrows_all_base_tables | ...
+---------------------------------------------------------------------------------------
+  
+
```



## Generating the training data

`generate_offload_dataset.py` produces the labelled dataset the training script
consumes. Run it from this directory:

```
python generate_offload_dataset.py                 # 5000 balanced rows + golden set
python generate_offload_dataset.py -n 20000 --seed 7 --stats
```

It writes two files:

| file | rows | purpose |
| --- | --- | --- |
| `mysql_offload_balanced_5000_IS_OLAP.csv` | 5000 (50/50) | training data |
| `golden_tpch_sysbench.csv` | 22 | hand-encoded TPC-H SF1 + sysbench shapes, **never train on this** |

### Why it is not a plain random sampler

The classifier never sees SQL text. It sees exactly the 18-float vector built by
`Query_arbitrator::features_to_vector()` and filled in by
`Query_arbitrator::extract_features()`
(`storage/rapid_engine/ml/query_arbitrator.cpp`). Drawing the 18 columns
independently produces rows that function can never emit, and the model then
spends capacity on a region of feature space it will never meet in production.

So the generator draws a *query shape* — tables and cardinalities, how the
primary plan reaches each one, clauses, subqueries — and derives the 18 features
with the same arithmetic as the C++ extractor, preserving its invariants:

* `has_join == (table_count > 1)`
* `has_group_by` implies `has_aggregation`; `has_rollup` implies `has_group_by`
* `f_count_ref_index_ts <= f_count_all_base_tables <= table_count`
  (derived tables raise `table_count` only)
* `f_are_all_ts_index_ref` ⇒ `f_mysql_total_ts_nrows == 0`, before the subquery
  walk adds inner-block scan rows
* `estimated_rows = f_BaseTableSumNrows`, `*0.1` with a WHERE, `*0.01` with a
  GROUP BY, then clamped by the LIMIT value
* fallback path (no cached primary plan):
  `f_MySQLCost = f_BaseTableSumNrows * 1.1 * max(table_count, 1)`

`--plan-ratio` controls how many rows come from the cached-primary-plan branch
of `extract_features()` versus the heuristic fallback (default 0.88).

### Labels are a cost model, not a tautology

`IS_OLAP` is **not** "has GROUP BY". Each shape is costed twice — an InnoDB
response time (index descents, row-store scan, filesort, temp-table GROUP BY)
and a Rapid response time (fixed offload overhead, parallel columnar scan, hash
aggregation, hash join, result shipping) — and labelled 1 when Rapid wins. Both
estimates get lognormal jitter, so rows near the crossover carry realistic label
noise instead of a knife edge.

That is what makes the two adversarial families come out right:

* **`huge_but_indexed`** — every size feature is enormous, but the plan reaches
  a handful of rows through an index. Labelled 0. A naive generator teaches the
  model "big tables ⇒ offload" and gets these wrong.
* **`small_table_report`** — GROUP BY + HAVING + ORDER BY over a few thousand
  rows. Labelled 0, because the offload overhead alone exceeds the whole InnoDB
  execution.

13 archetypes are sampled in total, spanning transactional shapes (point lookup,
index range, PK join, indexed aggregate, EXISTS), analytical shapes (scan+agg,
star join, ETL scan, nested analytic) and the genuinely ambiguous band
(mixed join, medium scan, plus the two above).

### Measured quality

Training `train_offload_classifier.py` on the generated data:

* held-out synthetic test (1000 rows): **accuracy 0.94, AUC 0.988**
* golden TPC-H / sysbench set (22 rows, never trained on): **22/22**

The residual 6% on the synthetic test is the deliberately ambiguous band — a
model scoring near 100% there would mean the dataset had no hard cases left.
Top-importance features are `f_BaseTableSumNrows`, `f_MySQLCost` and
`f_are_all_ts_index_ref` rather than `has_group_by`, which is the intended
outcome: the model learned the access path, not the clause list.

### Column order

The CSV column order is authoritative and must stay identical to
`features_to_vector()` indices 0..17. `train_offload_classifier.py` preserves it
via `df.drop(columns=["IS_OLAP"])`, and the ONNX model consumes a bare tensor
with no column names, so a reordering here silently corrupts every prediction.
