# Copyright (c) 2023, 2024, 2025 Shannon Data AI and/or its affiliates.
#
# Build shannon_rapid_classifier.onnx end to end: train LightGBM on the
# generated dataset, export ONNX, then verify the graph against the contract
# ShannonBase::ML::Query_arbitrator actually implements.
#
# THE OUTPUT CONTRACT -- read before changing anything here
# --------------------------------------------------------
# Query_arbitrator::load_model() binds output index 0 and nothing else:
#
#     auto output_name_ptr = m_session->GetOutputNameAllocated(0, allocator);
#
# and predict_with_features() then reads that tensor as float:
#
#     float *output_data = output_tensors[0].GetTensorMutableData<float>();
#     if (shape.size() >= 2 && shape[1] == 2) prediction_score = output_data[1];
#     else                                    prediction_score = output_data[0];
#
# A stock onnxmltools/skl2onnx classifier export puts `label` (int64) at output
# index 0 and the probabilities behind a ZipMap (a sequence of maps, not a
# tensor).  Loaded that way, the int64 label is reinterpreted as float: label 0
# gives 0.0 and label 1 gives the denormal 1.4e-45, so `score > threshold` is
# never true and the arbitrator answers TO_PRIMARY for every query regardless of
# what the model predicts.
#
# So this script exports with zipmap=False and then reorders the graph outputs
# to put the float [N, 2] probability tensor at index 0.  That lands on the
# `shape[1] == 2` branch above, where output_data[1] is P(offload to Rapid) --
# which is what the C++ was written for.  No C++ change is needed.
#
# Usage
#   python build_classifier_model.py
#   python build_classifier_model.py --out /path/to/shannon_rapid_classifier.onnx

import argparse
import os

import joblib
import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import save_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

INPUT_NAME = "float_input"   # keep the name the previously shipped model used
N_FEATURES = 18
TO_RAPID_THRESHOLD = 0.5     # Query_arbitrator::TO_RAPID_THRESHOLD
OLAP_FEATURE_THRESHOLD = 4   # Query_arbitrator::OLAP_FEATURE_THRESHOLD
OLAP_FACTOR = 0.6            # Query_arbitrator::OLAP_FACTOR


def train(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    X, y = df.drop(columns=["IS_OLAP"]), df["IS_OLAP"]
    assert X.shape[1] == N_FEATURES, f"expected {N_FEATURES} features, got {X.shape[1]}"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05, max_depth=8,
        num_leaves=64, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob > TO_RAPID_THRESHOLD).astype(int)
    print("=== held-out test ===")
    print("samples :", len(X_te))
    print("accuracy:", round(accuracy_score(y_te, pred), 4))
    print("auc     :", round(roc_auc_score(y_te, prob), 4))
    print("confusion matrix (rows=true 0/1):\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["TO_PRIMARY", "TO_SECONDARY"]))
    return model, X


def export_onnx(model, out_path):
    booster = model.booster_ if hasattr(model, "booster_") else model
    onnx_model = convert_lightgbm(
        booster,
        initial_types=[(INPUT_NAME, FloatTensorType([None, N_FEATURES]))],
        zipmap=False,
    )

    graph = onnx_model.graph
    assert not any(n.op_type == "ZipMap" for n in graph.node), \
        "ZipMap survived zipmap=False; probabilities would not be a tensor"

    # Put the float [N, 2] probability tensor at output index 0.
    def is_prob(vi):
        t = vi.type.tensor_type
        return t.elem_type == onnx.TensorProto.FLOAT and len(t.shape.dim) == 2

    outputs = list(graph.output)
    probs = [o for o in outputs if is_prob(o)]
    assert probs, "no float [N, 2] output found to place at index 0: " + \
                  str([(o.name, o.type.tensor_type.elem_type) for o in outputs])
    ordered = probs[:1] + [o for o in outputs if o is not probs[0]]
    del graph.output[:]
    graph.output.extend(ordered)

    onnx.checker.check_model(onnx_model)
    save_model(onnx_model, out_path)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def verify(out_path, X, model):
    """Reproduce exactly what load_model()/predict_with_features() will do."""
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()

    print("\n=== graph contract ===")
    for i in ins:
        print(f"  input  : {i.name} {i.type} {i.shape}")
    for n, o in enumerate(outs):
        print(f"  output[{n}]: {o.name} {o.type} {o.shape}")

    assert len(ins) == 1, "load_model() rejects a model with != 1 input"
    assert ins[0].shape[1] == N_FEATURES
    o0 = outs[0]
    assert o0.type == "tensor(float)", \
        f"output[0] is {o0.type}; the C++ reads it as float* and would misread it"
    assert len(o0.shape) == 2 and o0.shape[1] == 2, \
        f"output[0] shape {o0.shape} misses the shape[1]==2 branch"
    print("  OK  output[0] is a float [N,2] tensor -> output_data[1] = P(TO_SECONDARY)")

    # Parity: ONNX output[0][:,1] must match LightGBM's predict_proba.
    feats = X.to_numpy(dtype=np.float32)
    onnx_p = sess.run([o0.name], {ins[0].name: feats})[0][:, 1]
    lgb_p = model.predict_proba(X)[:, 1]
    max_dev = float(np.max(np.abs(onnx_p - lgb_p)))
    disagree = int(np.sum((onnx_p > TO_RAPID_THRESHOLD) != (lgb_p > TO_RAPID_THRESHOLD)))
    print(f"\n=== ONNX vs LightGBM parity over {len(X)} rows ===")
    print(f"  max |prob deviation| : {max_dev:.3e}")
    print(f"  decision disagreements: {disagree}")
    assert max_dev < 1e-5 and disagree == 0, "ONNX export does not match the trained model"

    # Single-row path, as the server calls it (batch of 1).
    one = sess.run([o0.name], {ins[0].name: feats[:1]})[0]
    assert one.shape == (1, 2), one.shape
    print(f"  single-row run -> shape {one.shape}, score={one[0][1]:.4f}")
    return sess


def check_golden(sess, golden_csv):
    if not os.path.exists(golden_csv):
        return
    g = pd.read_csv(golden_csv)
    y = g["IS_OLAP"].to_numpy()
    X = g.drop(columns=["IS_OLAP"])
    name_in = sess.get_inputs()[0].name
    name_out = sess.get_outputs()[0].name
    p = sess.run([name_out], {name_in: X.to_numpy(dtype=np.float32)})[0][:, 1]

    # Apply the arbitrator's OLAP threshold adjustment so this mirrors the
    # decision the server really makes, not just the raw score.
    olap_score = (X.has_group_by + X.has_having + X.has_aggregation
                  + X.has_order_by + X.has_subquery).to_numpy()
    thr = np.where(olap_score >= OLAP_FEATURE_THRESHOLD,
                   TO_RAPID_THRESHOLD * OLAP_FACTOR, TO_RAPID_THRESHOLD)
    decision = (p > thr).astype(int)
    print(f"\n=== golden set ({len(g)} real TPC-H / sysbench shapes, not trained on) ===")
    print(f"  accuracy with the server's effective threshold: "
          f"{accuracy_score(y, decision):.4f} ({int((decision == y).sum())}/{len(y)})")
    for i in range(len(y)):
        if decision[i] != y[i]:
            print(f"  MISS row {i}: expected {y[i]}, score {p[i]:.3f}, thr {thr[i]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mysql_offload_balanced_5000_IS_OLAP.csv")
    ap.add_argument("--golden", default="golden_tpch_sysbench.csv")
    ap.add_argument("--out", default="shannon_rapid_classifier.onnx")
    ap.add_argument("--pkl", default="rapid_offload_classifier_model.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, X = train(args.data, args.seed)
    joblib.dump(model, args.pkl)
    export_onnx(model, args.out)
    sess = verify(args.out, X, model)
    check_golden(sess, args.golden)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
