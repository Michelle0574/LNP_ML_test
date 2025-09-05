import os
import argparse
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def process_one_dir(cv_dir: str, cv: int, threshold: float):
    pva_path = os.path.join(cv_dir, "predicted_vs_actual.csv")
    clf_score_path = os.path.join(cv_dir, "test_scores_clf.csv")
    out_proba_path = os.path.join(cv_dir, "predicted_vs_actual_proba.csv")
    out_labels_path = os.path.join(cv_dir, "predicted_vs_actual_labels.csv")

    if not os.path.exists(pva_path):
        print(f"[skip] {pva_path} not found.")
        return

    df = pd.read_csv(pva_path)

    # Collect classification tasks from scores file
    clf_tasks = []
    if os.path.exists(clf_score_path):
        tdf = pd.read_csv(clf_score_path)
        if "Task" in tdf.columns:
            clf_tasks = [str(x) for x in tdf["Task"].dropna().tolist()]

    clf_pred_cols = [f"cv_{cv}_pred_{t}" for t in clf_tasks if f"cv_{cv}_pred_{t}" in df.columns]

    # 1) Classification: logits -> probabilities with Sigmoid
    for c in clf_pred_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = sigmoid(df[c])

    # 2) Regression post-processing
    #    quantified_delivery: allow negative (no clip)
    qd_col = f"cv_{cv}_pred_quantified_delivery"
    if qd_col in df.columns:
        df[qd_col] = pd.to_numeric(df[qd_col], errors="coerce")

    #    Biodistribution_*: percentage in [0,1]
    biodist_cols = [col for col in df.columns if col.startswith(f"cv_{cv}_pred_Biodistribution_")]
    for c in biodist_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = np.clip(df[c], 0.0, 1.0)

    #    quantified_total_luminescence: percentage in [0,1]
    qtl_col = f"cv_{cv}_pred_quantified_total_luminescence"
    if qtl_col in df.columns:
        df[qtl_col] = pd.to_numeric(df[qtl_col], errors="coerce")
        df[qtl_col] = np.clip(df[qtl_col], 0.0, 1.0)

    # 3) Save probability file
    df.to_csv(out_proba_path, index=False)

    # 4) Create hard 0/1 labels for classification columns
    df_labels = df.copy()
    for c in clf_pred_cols:
        df_labels[c] = (pd.to_numeric(df_labels[c], errors="coerce") > threshold).astype(int)

    df_labels.to_csv(out_labels_path, index=False)

    print(f"Wrote: {out_proba_path}")
    print(f"Wrote: {out_labels_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--cv", type=int, required=True)
    parser.add_argument("--root", type=str, default="/home/gongruyi/drug_delivery/LNP_ML_test/results/crossval_splits")
    parser.add_argument("--threshold", type=float, default=0.5)  # threshold for hard labels
    parser.add_argument("--model", type=str, choices=["base", "attn", "both"], default="base",
                        help="Process base (cv_X), attention (cv_X_attn), or both.")
    args = parser.parse_args()

    base_dir = os.path.join(args.root, args.split, f"cv_{args.cv}")
    attn_dir = os.path.join(args.root, args.split, f"cv_{args.cv}_attn")

    if args.model in ("base", "both"):
        process_one_dir(base_dir, args.cv, args.threshold)

    if args.model in ("attn", "both"):
        process_one_dir(attn_dir, args.cv, args.threshold)

if __name__ == "__main__":
    main()