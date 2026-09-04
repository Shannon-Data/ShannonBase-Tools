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

## Building shannon_rapid_classifier.onnx

`build_classifier_model.py` does train -> export -> verify in one step:

```
python build_classifier_model.py
cp shannon_rapid_classifier.onnx <shannonbase>/extra/llm-models/
```

### The output contract (important)

`Query_arbitrator::load_model()` binds **output index 0 and nothing else**:

```cpp
auto output_name_ptr = m_session->GetOutputNameAllocated(0, allocator);
```

and `predict_with_features()` reads that tensor as `float`:

```cpp
float *output_data = output_tensors[0].GetTensorMutableData<float>();
if (shape.size() >= 2 && shape[1] == 2) prediction_score = output_data[1];
else                                    prediction_score = output_data[0];
```

A stock `convert_lightgbm()` export puts `label` (int64, shape `[1]`) at output
index 0 and hides the probabilities behind a `ZipMap`, which is a sequence of
maps rather than a tensor. Loaded that way the int64 label is reinterpreted as
a float: label 0 reads as `0.0` and label 1 reads as the denormal `1.4e-45`, so
`prediction_score > threshold` is **never** true and the arbitrator returns
`TO_PRIMARY` for every query no matter what the model predicts.

`build_classifier_model.py` therefore exports with `zipmap=False` and reorders
the graph outputs so the float `[N, 2]` probability tensor sits at index 0. That
lands on the `shape[1] == 2` branch, where `output_data[1]` is
P(offload to Rapid). No C++ change is required, and the script asserts the
contract rather than trusting it:

* exactly one input, `[None, 18]`
* `output[0]` is `tensor(float)` with shape `[None, 2]`
* ONNX probabilities match `LGBMClassifier.predict_proba` to < 1e-5 with zero
  decision disagreements over the whole dataset
* a batch-of-1 run returns shape `(1, 2)`, the shape the server actually uses

It finishes by scoring the golden set through the server's *effective* threshold,
including the `OLAP_FACTOR` reduction applied when
`has_group_by + has_having + has_aggregation + has_order_by + has_subquery >= 4`.

Keep the export at opset `ai.onnx 9` / `ai.onnx.ml 1` so the model loads on the
ONNX Runtime the server links (`ONNXRUNTIME_VERSION` in `cmake/FindONNXRuntime.cmake`).

## Real-workload validation

Synthetic accuracy is not evidence on its own — it only counts if the server
really produces those feature vectors. The model was validated against a live
ShannonBase instance (`8.4.8-debug`, `tpch_sf1`) by setting
`log_error_verbosity=3`, which un-gates the `[Note]` output of
`Query_arbitrator::log_decision()`, and running `EXPLAIN` on each query so the
pre-prepare hook fires without executing anything.

Result: **11 / 11 correct** on the decisions the model was actually asked to
make.

| query | f_MySQLCost | f_BaseTableSumNrows | all_idx_ref | score | decision |
| --- | ---: | ---: | :---: | ---: | --- |
| TPC-H Q1 (scan + GROUP BY) | 2.65e6 | 5.99e6 | no | 0.9999 | TO_SECONDARY |
| TPC-H Q6 (scan + SUM) | 6.55e5 | 5.99e6 | no | 0.9975 | TO_SECONDARY |
| TPC-H Q3 (3-way + GROUP BY) | 2.01e6 | 7.64e6 | no | 1.0000 | TO_SECONDARY |
| TPC-H Q5 (6-way + GROUP BY) | 3.19e6 | 7.65e6 | no | 0.9999 | TO_SECONDARY |
| TPC-H Q10 (4-way top-N) | 4.45e6 | 7.64e6 | no | 0.9999 | TO_SECONDARY |
| wide ETL scan, no aggregation | 6.55e5 | 5.99e6 | no | 0.9179 | TO_SECONDARY |
| GROUP BY … WITH ROLLUP | 6.65e6 | 5.99e6 | no | 0.9995 | TO_SECONDARY |
| indexed range + ORDER BY + LIMIT | 5.84 | 1.50e6 | yes | 0.0001 | TO_PRIMARY |
| PK-equality 2-way join | 0.099 | 1.50e5 | no | 0.0002 | TO_PRIMARY |
| NATION ⋈ REGION (30 rows) | 26.0 | 30 | no | 0.0000 | TO_PRIMARY |

Three further point lookups (`WHERE pk = const`) never reach the model at all:
`SecondaryEnginePrePrepareHook()` routes them to
`standard_cost_threshold_classifier()` via the `is_very_fast_query()`
short-circuit, and the optimizer trace confirms
`"cost: 1.000000, threshold: 100000.000000" -> primary`. That is by design — the
model is only consulted where the answer is not already obvious.

### What the live run changed in this dataset

Checking each real feature vector against the training ranges found two regions
the generator was not covering, both at the small end:

* `f_MySQLCost = 0.099` for the PK-equality join — **below** the old minimum of
  1.1. The optimizer folds such plans to const tables and reports a fraction of
  a cost unit, which the cost formula could never produce.
* `f_BaseTableSumNrows = 30` for NATION ⋈ REGION — **below** the old minimum of
  56.

Two archetypes were added to cover them (`const_join`, `tiny_dim_join`), the
small-table floor was lowered, and `_synthesise_primary_cost()` gained a
const-plan branch. Effect:

| | before | after |
| --- | --- | --- |
| held-out accuracy | 0.940 | **0.958** |
| held-out AUC | 0.988 | **0.994** |
| golden set | 22/22 | 22/22 |
| real workload | 11/11 | 11/11 |
| wide ETL scan margin | 0.533 (borderline) | **0.918** |

This is the loop worth repeating when the model is retrained: run the workload,
diff the logged feature vectors against the training ranges, and add coverage
wherever production lands outside them.
