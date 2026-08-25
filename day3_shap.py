"""
Day 3: SHAP explainability for the LightGBM Rossmann model.

Outputs:
- shap_feature_importance.csv
- shap_feature_importance_bar.png
- shap_summary_beeswarm.png

NOTE: the model predicts log1p(Sales), so SHAP values here are on the
log-sales scale (standard practice; contributions are still directly
comparable/rankable across features).
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from day2_model import FEATURE_COLS


def compute_shap_importance(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance_abs_mean": feature_importance}
    ).sort_values("Importance_abs_mean", ascending=False).reset_index(drop=True)

    bar_path = "shap_feature_importance_bar.png"
    top15 = importance_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top15["Feature"], top15["Importance_abs_mean"], color="#1f77b4")
    plt.xlabel("Mean |SHAP value| (log-sales scale)")
    plt.title("Top 15 Feature Importances (SHAP)")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()

    beeswarm_path = "shap_summary_beeswarm.png"
    top10_features = importance_df.head(10)["Feature"].tolist()
    idx = [feature_names.index(f) for f in top10_features]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values[:, idx],
        X_test.iloc[:, idx],
        feature_names=top10_features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()

    return {"importance_df": importance_df, "saved_files": [bar_path, beeswarm_path]}


if __name__ == "__main__":
    print("Loading model and test predictions...")
    with open("models/lightgbm_model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/test_predictions.csv", low_memory=False)

    sample_n = min(8000, len(df))
    X_test = df[FEATURE_COLS].sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Computing SHAP on a sample of {sample_n} rows...")

    result = compute_shap_importance(model, X_test, FEATURE_COLS)
    result["importance_df"].to_csv("shap_feature_importance.csv", index=False)

    print(result["importance_df"].head(10))
    print(f"Saved: {result['saved_files']} and shap_feature_importance.csv")
    print("Day 3 complete.")
