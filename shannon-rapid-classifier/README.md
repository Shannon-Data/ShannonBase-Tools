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


