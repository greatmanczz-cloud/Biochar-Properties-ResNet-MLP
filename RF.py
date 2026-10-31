
# -*- coding: utf-8 -*-
"""
Random Forest (RF): five-fold out-of-fold (OOF) prediction.

Input workbook: ETII.xlsx
Worksheet: MICE_All_Exploratory
Input variables (13): Excel columns G-I and K-T
Target variables (10): Excel columns U-AD
Output folder: RF_ETII_Results

The targets may include values previously imputed using ETII.
OOF evaluation on these targets is not independent validation
against experimentally measured values alone.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ==================================================
# 1. Basic Configuration
# ==================================================

SEED = 42
K_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ETII.xlsx"
SHEET_NAME = "MICE_All_Exploratory"
OUTPUT_DIR = BASE_DIR / "RF_ETII_Results"

OUTPUT_EXCEL = OUTPUT_DIR / "RF_ETII_5Fold_OOF_Results.xlsx"
PLOT_PNG = OUTPUT_DIR / "RF_ETII_5Fold_OOF_Scatter.png"
PLOT_PDF = OUTPUT_DIR / "RF_ETII_5Fold_OOF_Scatter.pdf"
DPI = 300


# ==================================================
# 2. Random Forest Hyperparameters
# ==================================================

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 16,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": SEED,
}


# ==================================================
# 3. Excel Column Indices (Zero-Based)
# ==================================================

INPUT_INDEX = [6, 7, 8] + list(range(10, 20))
OUTPUT_INDEX = list(range(20, 30))


# ==================================================
# 4. Input and Target Names
# ==================================================

FEATURES = [
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

TARGETS = [
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


# ==================================================
# 5. Matplotlib Settings
# ==================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ==================================================
# 6. Load ETII Dataset
# ==================================================

def load_data():

    print("\n--- 1. Loading ETII data ---")

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input workbook not found: {INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
    )

    if raw.shape[1] < 30:
        raise ValueError(
            "The Excel workbook must contain at least 30 columns."
        )

    # Skip the first four header rows.
    body = raw.iloc[4:, :].copy()

    # Keep valid records based on numeric sample IDs in column A.
    sample_no = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    body = body.loc[sample_no.notna()].reset_index(drop=True)

    # Extract the 13 input variables.
    X_df = body.iloc[:, INPUT_INDEX].copy()
    X_df.columns = FEATURES
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # Extract the 10 target variables.
    y_df = body.iloc[:, OUTPUT_INDEX].copy()
    y_df.columns = TARGETS
    y_df = y_df.apply(pd.to_numeric, errors="coerce")

    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64)

    # Validate the completed dataset.
    if not np.isfinite(X).all():
        raise ValueError(
            "NaN or Inf values found among the 13 input variables."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "NaN or Inf values found among the 10 target variables."
        )

    sample_ids = body.iloc[:, 0].to_numpy()

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

    print("Valid samples:", len(body))
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    return X, y, sample_ids, references


# ==================================================
# 7. Evaluation Metrics
# ==================================================

def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
    }


# ==================================================
# 8. Plot OOF Predictions
# ==================================================

def plot_oof(y, predictions, final_df):

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    axes = axes.ravel()

    for j, target in enumerate(TARGETS):

        ax = axes[j]
        actual = y[:, j]
        pred = predictions[:, j]

        ax.scatter(
            actual,
            pred,
            s=18,
            alpha=0.58,
            color="#4C78A8",
            edgecolors="none",
        )

        low = float(min(actual.min(), pred.min()))
        high = float(max(actual.max(), pred.max()))
        margin = max(0.04 * (high - low), 1e-8)
        limits = [low - margin, high + margin]

        # Reference line: predicted value equals observed value.
        ax.plot(
            limits,
            limits,
            "--",
            color="#D97757",
            linewidth=1.5,
            label="1:1 line",
        )

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("ETII value")
        ax.set_ylabel("RF OOF prediction")

        r2_value = final_df.loc[j, "OOF_R2"]
        ax.set_title(f"{target}\nR2 = {r2_value:.4f}")
        ax.grid(linestyle="--", alpha=0.18)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[len(TARGETS):]:
        ax.set_visible(False)

    fig.suptitle(
        "Random Forest: 5-fold OOF predictions",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(PLOT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# 9. Main Workflow
# ==================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, sample_ids, references = load_data()

    print(f"\n--- 2. Starting RF {K_FOLDS}-fold cross-validation ---")

    # Use the same random split configuration as ResNet-MLP and SVR.
    kf = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    # Initialize out-of-fold prediction arrays.
    oof = np.full(y.shape, np.nan, dtype=np.float64)
    fold_labels = np.zeros(len(X), dtype=int)
    fold_rows = []
    importance_rows = []

    # Train and evaluate the five folds.
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        print(f"\nTraining fold {fold}/{K_FOLDS}")

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit StandardScaler only on the training fold.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # RandomForestRegressor supports multi-output regression directly.
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_scaled, y_train)

        # Predict the held-out fold and save OOF predictions.
        y_pred = model.predict(X_test_scaled)
        oof[test_idx] = y_pred
        fold_labels[test_idx] = fold

        current_r2 = []

        # Compute fold-level metrics for each of the ten targets.
        for j, name in enumerate(TARGETS):

            scores = calculate_metrics(y_test[:, j], y_pred[:, j])

            fold_rows.append({
                "Fold": fold,
                "Target": name,
                "N_test": len(test_idx),
                **scores,
            })

            current_r2.append(scores["R2"])

        # Record feature importance for this fold.
        for name, value in zip(FEATURES, model.feature_importances_):

            importance_rows.append({
                "Fold": fold,
                "Feature": name,
                "Importance": float(value),
            })

        print(f"Fold {fold} mean R2 = {np.mean(current_r2):.4f}")

    # Verify that every OOF prediction is available.
    if not np.isfinite(oof).all():
        raise RuntimeError("Some OOF predictions are missing.")

    # Calculate final OOF metrics for all targets.
    final_rows = []

    for j, name in enumerate(TARGETS):

        score = calculate_metrics(y[:, j], oof[:, j])

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

    # Summary table of R2 scores for each fold.
    cv_detail = fold_df.pivot(
        index="Fold",
        columns="Target",
        values="R2",
    )

    cv_detail = cv_detail.reindex(columns=TARGETS)
    cv_detail.index = [f"Fold {i}" for i in cv_detail.index]
    cv_detail.loc["Average"] = cv_detail.mean(axis=0)

    # Per-sample OOF predictions.
    predictions_df = pd.DataFrame({
        "Index_No": sample_ids,
        "Reference": references,
        "Fold": fold_labels,
    })

    for j, name in enumerate(TARGETS):

        predictions_df[name + "_ETII_Value"] = y[:, j]
        predictions_df[name + "_RF_OOF"] = oof[:, j]
        predictions_df[name + "_Residual"] = y[:, j] - oof[:, j]

    # Feature importance across folds.
    importance_df = (
        pd.DataFrame(importance_rows)
        .pivot(index="Feature", columns="Fold", values="Importance")
        .reindex(FEATURES)
    )

    importance_df["Mean_Importance"] = importance_df.mean(axis=1)

    importance_df = (
        importance_df
        .sort_values("Mean_Importance", ascending=False)
        .reset_index()
    )

    # Overall model summary.
    summary = pd.DataFrame([{
        "Model": "Random Forest",
        "N_samples": len(y),
        "N_inputs": len(FEATURES),
        "N_outputs": len(TARGETS),
        "CV": "Random 5-fold OOF",
        "Mean_OOF_R2_10_targets": final_df["OOF_R2"].mean(),
        "Mean_OOF_RMSE": final_df["OOF_RMSE"].mean(),
        "Mean_OOF_MAE": final_df["OOF_MAE"].mean(),
    }])

    # Record the actual training hyperparameters.
    settings = pd.DataFrame([
        ("Input_file", INPUT_FILE.name),
        ("Sheet", SHEET_NAME),
        ("SEED", SEED),
        ("K_FOLDS", K_FOLDS),
        ("n_estimators", RF_PARAMS["n_estimators"]),
        ("max_depth", RF_PARAMS["max_depth"]),
        ("min_samples_split", RF_PARAMS["min_samples_split"]),
        ("min_samples_leaf", RF_PARAMS["min_samples_leaf"]),
        ("n_jobs", RF_PARAMS["n_jobs"]),
        ("X_scaling", "StandardScaler fitted on training fold"),
        ("Y_scaling", "None"),
        ("Label_source", "ETII-completed outputs"),
    ], columns=["Parameter", "Value"])

    # Export results to Excel.
    print("\n--- 3. Exporting results ---")

    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

            summary.to_excel(
                writer, sheet_name="Summary", index=False
            )

            final_df.to_excel(
                writer, sheet_name="OOF_Metrics", index=False
            )

            fold_df.to_excel(
                writer, sheet_name="Fold_Metrics", index=False
            )

            cv_detail.to_excel(
                writer, sheet_name="CV_Detail", index=True
            )

            predictions_df.to_excel(
                writer, sheet_name="OOF_Predictions", index=False
            )

            importance_df.to_excel(
                writer, sheet_name="RF_Feature_Importance", index=False
            )

            settings.to_excel(
                writer, sheet_name="Settings", index=False
            )

    except PermissionError as exc:

        raise PermissionError(
            "Unable to save the Excel workbook. "
            "Close any open file with the same name."
        ) from exc

    # Draw and save the OOF scatter plots.
    plot_oof(y, oof, final_df)

    # Print the final model evaluation.
    print("\n========== Final RF OOF Results ==========")

    print(
        final_df[["Target", "OOF_R2", "OOF_RMSE", "OOF_MAE"]]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nMean OOF R2 across 10 targets:",
        round(final_df["OOF_R2"].mean(), 4),
    )

    print("\nExcel workbook:", OUTPUT_EXCEL)
    print("\nFigure:", PLOT_PNG)
    print("\nRF training completed.")


# ==================================================
# Program Entry Point
# ==================================================

if __name__ == "__main__":
    main()
