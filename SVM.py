
# -*- coding: utf-8 -*-
"""
Support Vector Regression (SVR) with an RBF kernel:
five-fold cross-validation and out-of-fold (OOF) predictions.

Input workbook: ETII.xlsx
Worksheet: MICE_All_Exploratory
Input variables (13): Excel columns G-I and K-T
Target variables (10): Excel columns U-AD
Model: MultiOutputRegressor(SVR)

Outputs:
- Five-fold OOF predictions
- R2, RMSE, MAE, and MSE metrics
- Per-fold performance metrics
- Excel results workbook
- OOF scatter plots in PNG and PDF formats

The same random five-fold splitting configuration is used as
in the ResNet-MLP model (KFold, shuffle=True, random_state=42).
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


# ==================================================
# 1. Configuration
# ==================================================

SEED = 42
K_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)


# ==================================================
# 2. Input and Output Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ETII.xlsx"
SHEET_NAME = "MICE_All_Exploratory"
OUTPUT_DIR = BASE_DIR / "SVR_ETII_Results"

OUTPUT_EXCEL = OUTPUT_DIR / "SVR_ETII_5Fold_OOF_Results.xlsx"
PLOT_PNG = OUTPUT_DIR / "SVR_ETII_5Fold_OOF_Scatter.png"
PLOT_PDF = OUTPUT_DIR / "SVR_ETII_5Fold_OOF_Scatter.pdf"


# ==================================================
# 3. SVR Hyperparameters
# ==================================================

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 100,
    "gamma": "scale",
    "epsilon": 1,
    "verbose": False,
}


# ==================================================
# 4. Excel Column Indices (Zero-Based)
# ==================================================

# Inputs: G, H, I, K, L, M, N, O, P, Q, R, S, T
INPUT_INDEX = [6, 7, 8] + list(range(10, 20))

# Targets: U, V, W, X, Y, Z, AA, AB, AC, AD
OUTPUT_INDEX = list(range(20, 30))


# ==================================================
# 5. Input and Target Variables
# ==================================================

feature_cols = [
    "FC_dry",
    "VM_dry",
    "Ash_dry",
    "Feedstock_C",
    "Feedstock_H",
    "Feedstock_O",
    "Feedstock_N",
    "Cellulose",
    "Hemicellulose",
    "Lignin",
    "Residence_time",
    "Temperature",
    "Heating_rate",
]

target_cols = [
    "Biochar_yield",
    "TPV",
    "Biochar_ash",
    "Zeta_potential",
    "pH",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O",
    "O_C_ratio",
    "H_C_ratio",
]

assert len(INPUT_INDEX) == len(feature_cols) == 13
assert len(OUTPUT_INDEX) == len(target_cols) == 10


# ==================================================
# 6. Plot Settings
# ==================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
})


# ==================================================
# 7. Load Excel Data
# ==================================================

def load_data():
    print("\n--- 1. Loading ETII dataset... ---")

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place the Excel workbook and the Python script in the same folder."
        )

    # Do not interpret the first row as column names.
    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    if raw.shape[1] < 30:
        raise ValueError("The Excel workbook has fewer than 30 columns.")

    # Skip the first four header rows.
    body = raw.iloc[4:, :].copy()

    # Column A contains the sample identifiers.
    ids = pd.to_numeric(body.iloc[:, 0], errors="coerce")

    # Retain rows with valid numeric identifiers.
    body = body.loc[ids.notna()].reset_index(drop=True)

    # Extract the 13 input variables.
    X_df = body.iloc[:, INPUT_INDEX].copy()
    X_df.columns = feature_cols
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # Extract the 10 target variables.
    y_df = body.iloc[:, OUTPUT_INDEX].copy()
    y_df.columns = target_cols
    y_df = y_df.apply(pd.to_numeric, errors="coerce")

    # scikit-learn uses float64 in this workflow.
    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64)

    sample_ids = body.iloc[:, 0].to_numpy()

    # Read reference identifiers if column AE is present.
    if body.shape[1] > 30:
        references = (
            body.iloc[:, 30]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .to_numpy()
        )
    else:
        references = np.array(["Unknown"] * len(body))

    # Validate numerical completeness.
    print("Data loaded successfully.")
    print("Number of samples:", len(body))
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("Missing input values (NaN):", int(np.isnan(X).sum()))
    print("Missing target values (NaN):", int(np.isnan(y).sum()))

    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        bad_x = np.argwhere(~np.isfinite(X))
        bad_y = np.argwhere(~np.isfinite(y))
        raise ValueError(
            "The selected 23 variables still contain NaN or Inf values.\n"
            f"Invalid input positions: {bad_x[:5].tolist()}\n"
            f"Invalid target positions: {bad_y[:5].tolist()}"
        )

    print("Data validation passed.")
    return X, y, sample_ids, references


# ==================================================
# 8. Evaluation Metrics
# ==================================================

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MSE": float(mse),
    }


# ==================================================
# 9. OOF Scatter Plots
# ==================================================

def draw_scatter(y, predictions, metrics_df):
    print("\n--- Generating OOF scatter plots... ---")

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.flatten()

    for j, name in enumerate(target_cols):
        ax = axes[j]
        true_values = y[:, j]
        pred_values = predictions[:, j]

        # Scatter points.
        ax.scatter(
            true_values,
            pred_values,
            s=22,
            alpha=0.60,
            color="#4C78A8",
            edgecolors="none",
        )

        # Ideal 1:1 prediction line.
        minimum = float(min(true_values.min(), pred_values.min()))
        maximum = float(max(true_values.max(), pred_values.max()))
        margin = max(0.04 * (maximum - minimum), 1e-8)
        limits = [minimum - margin, maximum + margin]

        ax.plot(
            limits,
            limits,
            "--",
            linewidth=1.4,
            color="#D97757",
            label="1:1 line",
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("ETII value")
        ax.set_ylabel("SVR OOF prediction")

        r2_value = metrics_df.loc[j, "OOF_R2"]
        ax.set_title(f"{name}\nR² = {r2_value:.4f}")
        ax.grid(alpha=0.20, linestyle="--")
        ax.legend(frameon=False, fontsize=9)

    # Hide unused subplot panels.
    for ax in axes[len(target_cols):]:
        ax.set_visible(False)

    fig.suptitle("SVR (RBF): 5-fold OOF predictions", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Export both PNG and PDF figures.
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Scatter plots saved.")


# ==================================================
# 10. Main Program
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the dataset.
    X, y, sample_ids, references = load_data()
    print(f"\n--- 2. Starting {K_FOLDS}-fold SVR cross-validation ---")

    # Step 2: Use the same five-fold split as ResNet-MLP.
    kf = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    oof_predictions = np.full(y.shape, np.nan, dtype=np.float64)
    fold_labels = np.zeros(len(y), dtype=int)
    fold_rows = []

    # Step 3: Train and evaluate each fold.
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\nFold {fold}/{K_FOLDS}")
        print("Training samples:", len(train_idx))
        print("Test samples:", len(test_idx))

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit input scaler using training data only.
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        # Fit target scaler using training data only.
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)

        # Train one SVR model per target.
        base_svr = SVR(**SVM_PARAMS)
        model = MultiOutputRegressor(base_svr, n_jobs=1)
        model.fit(X_train_scaled, y_train_scaled)

        # Predict held-out samples in the original target units.
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)

        # Store OOF predictions.
        oof_predictions[test_idx] = y_pred_orig
        fold_labels[test_idx] = fold

        current_r2 = []
        for j, name in enumerate(target_cols):
            score = evaluate(y_test[:, j], y_pred_orig[:, j])
            fold_rows.append({
                "Fold": fold,
                "Target": name,
                "N_test": len(test_idx),
                **score,
            })
            current_r2.append(score["R2"])

        mean_r2 = np.mean(current_r2)
        print(f"Fold {fold} mean R2 = {mean_r2:.4f}")

    # Step 4: Check the completeness of OOF predictions.
    if not np.isfinite(oof_predictions).all():
        raise RuntimeError("Some samples have no OOF predictions.")

    # Step 5: Calculate final OOF metrics.
    print("\n--- 3. Calculating final OOF metrics... ---")
    final_rows = []

    for j, name in enumerate(target_cols):
        score = evaluate(y[:, j], oof_predictions[:, j])
        final_rows.append({
            "Target": name,
            "N": len(y),
            "OOF_R2": score["R2"],
            "OOF_RMSE": score["RMSE"],
            "OOF_MAE": score["MAE"],
            "OOF_MSE": score["MSE"],
        })

    final_df = pd.DataFrame(final_rows)
    fold_df = pd.DataFrame(fold_rows)

    # Step 6: Summarize per-fold R2 metrics.
    cv_detail = fold_df.pivot(
        index="Fold", columns="Target", values="R2"
    )
    cv_detail = cv_detail.reindex(columns=target_cols)
    cv_detail.index = [f"Fold {i}" for i in cv_detail.index]
    cv_detail.loc["Average"] = cv_detail.mean()

    # Step 7: Assemble per-sample predictions.
    predictions_df = pd.DataFrame({
        "Index_No": sample_ids,
        "Reference": references,
        "Fold": fold_labels,
    })

    for j, name in enumerate(target_cols):
        predictions_df[f"{name}_ETII_Value"] = y[:, j]
        predictions_df[f"{name}_SVR_OOF_Pred"] = oof_predictions[:, j]
        predictions_df[f"{name}_Residual"] = y[:, j] - oof_predictions[:, j]

    # Step 8: Calculate overall model summary.
    summary_df = pd.DataFrame([{
        "Model": "SVR (RBF)",
        "N_samples": len(y),
        "N_features": len(feature_cols),
        "N_targets": len(target_cols),
        "CV": "Random 5-fold OOF",
        "Mean_OOF_R2_10_targets": final_df["OOF_R2"].mean(),
        "Mean_OOF_RMSE_10_targets": final_df["OOF_RMSE"].mean(),
        "Mean_OOF_MAE_10_targets": final_df["OOF_MAE"].mean(),
    }])

    # Step 9: Record the model settings.
    settings_df = pd.DataFrame([
        ("Input_file", INPUT_FILE.name),
        ("Sheet", SHEET_NAME),
        ("CV_seed", SEED),
        ("K_FOLDS", K_FOLDS),
        ("Kernel", SVM_PARAMS["kernel"]),
        ("C", SVM_PARAMS["C"]),
        ("gamma", SVM_PARAMS["gamma"]),
        ("epsilon", SVM_PARAMS["epsilon"]),
        (
            "Scaling",
            "X and y StandardScaler fitted within each training fold",
        ),
        ("Label_source", "ETII-completed targets"),
    ], columns=["Parameter", "Value"])

    # Step 10: Export the Excel workbook.
    print("\n--- 4. Saving Excel results... ---")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        final_df.to_excel(writer, sheet_name="OOF_Metrics", index=False)
        fold_df.to_excel(writer, sheet_name="Fold_Metrics", index=False)
        cv_detail.to_excel(writer, sheet_name="CV_Detail", index=True)
        predictions_df.to_excel(writer, sheet_name="OOF_Predictions", index=False)
        settings_df.to_excel(writer, sheet_name="Settings", index=False)

    # Step 11: Draw and save the scatter plots.
    draw_scatter(y, oof_predictions, final_df)

    # Step 12: Print the final results.
    print("\n========== Final Five-Fold OOF Results ==========")
    print(final_df.round(4).to_string(index=False))
    print(
        "\nMean OOF R2 across 10 targets:",
        round(final_df["OOF_R2"].mean(), 4),
    )
    print("\nExcel workbook:", OUTPUT_EXCEL)
    print("\nFigure:", PLOT_PNG)
    print("\nAll steps completed.")


# ==================================================
# Program Entry Point
# ==================================================

if __name__ == "__main__":
    main()


