
# -*- coding: utf-8 -*-
"""
Biochar dataset: optimized KNN imputation and R2 validation.

Procedure:
1. Scale variables using RobustScaler.
2. Apply KNNImputer with three neighbors and distance weighting.
3. Perform five-fold artificial-masking validation.
4. Calculate R2, RMSE, and MAE.
5. Preserve all original measured values.
6. Apply physical bounds only to originally missing entries.
7. Calculate missing O/C and H/C atomic ratios from elemental contents.

Input: dataset.xlsx
Output: Biochar_KNN_Optimized_Imputation_R2.xlsx
PyTorch and CUDA are not required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =====================================================
# 1. Configuration
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "dataset.xlsx"
OUTPUT_FILE = BASE_DIR / "Biochar_KNN_Optimized_Imputation_R2.xlsx"
SHEET_NAME = "Raw dataset"
HEADER_ROWS = 4
RANDOM_SEED = 2024
N_FOLDS = 5
N_NEIGHBORS = 3
WEIGHTS = "distance"


# =====================================================
# 2. Dataset Columns
# =====================================================
# Excel column indices are zero-based.
INPUT_COLS = [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
TARGET_COLS = list(range(20, 30))
ALL_COLS = INPUT_COLS + TARGET_COLS

VARIABLE_NAMES = [
    # 13 input variables
    "FC_dry", "VM_dry", "Ash_dry", "Feedstock_C", "Feedstock_H",
    "Feedstock_O", "Feedstock_N", "Cellulose", "Hemicellulose",
    "Lignin", "Residence_time", "Pyrolysis_temperature", "Heating_rate",
    # 10 target variables
    "Biochar_yield", "Total_pore_volume", "Biochar_ash",
    "Zeta_potential", "pH_or_pHPZC", "Biochar_C", "Biochar_H",
    "Biochar_O", "O_C_atomic_ratio", "H_C_atomic_ratio"
]

# Indices of percentage variables in the 23-column matrix.
PERCENTAGE_COLS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 15, 18, 19, 20}


# =====================================================
# 3. KNN Imputation
# =====================================================
def knn_fill(matrix, improved=True):
    if improved:
        # Scale input variables using RobustScaler.
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(matrix)

        # Optimized KNN imputation.
        imputer = KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS,
            metric="nan_euclidean",
            keep_empty_features=True
        )
        filled_scaled = imputer.fit_transform(scaled_data)
        filled = scaler.inverse_transform(filled_scaled)
    else:
        # Baseline KNN without feature scaling.
        imputer = KNNImputer(
            n_neighbors=5,
            weights="distance",
            metric="nan_euclidean",
            keep_empty_features=True
        )
        filled = imputer.fit_transform(matrix)

    if filled.shape != matrix.shape:
        raise ValueError("The matrix shape changed after imputation.")

    # Preserve all originally observed values.
    filled = np.where(np.isfinite(matrix), matrix, filled)
    return filled


# =====================================================
# 4. Physical-Range Constraints and Atomic Ratios
# =====================================================
def postprocess_only_missing(original, completed):
    result = completed.copy()
    missing = ~np.isfinite(original)

    for j in range(original.shape[1]):
        mask = missing[:, j]

        if j in PERCENTAGE_COLS:
            # Restrict percentage variables to [0, 100].
            result[mask, j] = np.clip(result[mask, j], 0, 100)
        elif j == 17:
            # Restrict pH or pHPZC to [0, 14].
            result[mask, j] = np.clip(result[mask, j], 0, 14)
        elif j != 16:
            # Zeta potential (index 16) may be negative.
            result[mask, j] = np.maximum(result[mask, j], 0)

    # Calculate atomic ratios only for the complete 23-variable dataset.
    if original.shape[1] == 23:
        M_C = 12.011
        M_H = 1.008
        M_O = 15.999

        # Column indices: Biochar C=18, H=19, O=20, O/C=21, H/C=22.
        ratio_settings = [(21, 20, M_O), (22, 19, M_H)]

        for ratio_col, element_col, mass in ratio_settings:
            mask = (
                missing[:, ratio_col]
                & np.isfinite(result[:, element_col])
                & np.isfinite(result[:, 18])
                & (result[:, 18] > 0)
            )
            result[mask, ratio_col] = (
                (result[mask, element_col] / mass)
                / (result[mask, 18] / M_C)
            )

    return result


# =====================================================
# 5. Five-Fold Artificial-Masking Validation
# =====================================================
def evaluate_5fold(original):
    print("\nStarting five-fold artificial-masking validation...")
    rng = np.random.RandomState(RANDOM_SEED)
    n_features = original.shape[1]
    folds = []

    # Create independent five-fold partitions for every variable.
    for j in range(n_features):
        observed_indices = np.flatnonzero(np.isfinite(original[:, j]))
        shuffled = rng.permutation(observed_indices)
        folds.append(np.array_split(shuffled, N_FOLDS))

    old_predictions = np.full_like(original, np.nan)
    new_predictions = np.full_like(original, np.nan)

    for fold in range(N_FOLDS):
        train_data = original.copy()

        # Mask observed values belonging to the current fold.
        for j in range(n_features):
            indices = folds[j][fold]
            train_data[indices, j] = np.nan

        # Baseline KNN and optimized KNN.
        old_filled = knn_fill(train_data, improved=False)
        new_filled = knn_fill(train_data, improved=True)
        new_filled = postprocess_only_missing(train_data, new_filled)

        # Store predictions at masked positions only.
        for j in range(n_features):
            indices = folds[j][fold]
            old_predictions[indices, j] = old_filled[indices, j]
            new_predictions[indices, j] = new_filled[indices, j]

        print(f"Fold {fold + 1}/{N_FOLDS} completed")

    # Calculate per-variable predictive performance.
    results = []
    for j in range(n_features):
        mask = np.isfinite(original[:, j])
        y_true = original[mask, j]
        y_old = old_predictions[mask, j]
        y_new = new_predictions[mask, j]

        results.append({
            "Variable": VARIABLE_NAMES[j],
            "Observed": int(mask.sum()),
            "Original_KNN_R2": r2_score(y_true, y_old),
            "Optimized_KNN_R2": r2_score(y_true, y_new),
            "Optimized_KNN_RMSE": np.sqrt(mean_squared_error(y_true, y_new)),
            "Optimized_KNN_MAE": mean_absolute_error(y_true, y_new)
        })

    return pd.DataFrame(results)


# =====================================================
# 6. Read Original Excel Data
# =====================================================
def read_original_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        keep_default_na=False
    )

    if raw.shape[1] < 30:
        raise ValueError("Insufficient columns; please check the Excel structure.")

    header = raw.iloc[:HEADER_ROWS].copy()
    body = raw.iloc[HEADER_ROWS:].copy()

    # Exclude rows with empty record identifiers.
    valid_rows = body.iloc[:, 0].astype(str).str.strip().ne("")
    body = body.loc[valid_rows].copy()

    # Extract the 23 numerical variables.
    numeric = np.column_stack([
        pd.to_numeric(body.iloc[:, col], errors="coerce").to_numpy(dtype=float)
        for col in ALL_COLS
    ])

    return raw, header, body, numeric


# =====================================================
# 7. Reconstruct Excel Tables
# =====================================================
def reconstruct_excel(header, body, original, filled):
    output_body = body.copy()

    for j, excel_col in enumerate(ALL_COLS):
        missing = ~np.isfinite(original[:, j])
        if missing.any():
            output_body.iloc[np.flatnonzero(missing), excel_col] = filled[missing, j]

    return pd.concat([header, output_body], ignore_index=True)


# =====================================================
# 8. Main Program
# =====================================================
def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Optimized KNN imputation and R2 validation")
    print("=" * 60)
    print("Computing device: CPU")
    print("PyTorch is not required.")

    raw, header, body, numeric = read_original_data()
    print(f"\nNumber of samples: {len(body)}")
    print(f"Number of variables: {numeric.shape[1]}")
    print(f"Original missing values: {np.isnan(numeric).sum()}")

    # Require at least 10 observed values per variable.
    observed_counts = np.isfinite(numeric).sum(axis=0)
    if (observed_counts < 10).any():
        raise ValueError(
            "Some variables have fewer than 10 observed values; "
            "validation cannot proceed."
        )

    # Step 1: Perform R2 validation.
    results = evaluate_5fold(numeric)
    old_r2 = results["Original_KNN_R2"].mean()
    optimized_r2 = results["Optimized_KNN_R2"].mean()

    print("\n" + "=" * 60)
    print("R2 validation results")
    print("=" * 60)
    print(f"Mean R2 (baseline KNN): {old_r2:.4f}")
    print(f"Mean R2 (optimized KNN): {optimized_r2:.4f}")
    print(
        "\nNote: The reported mean R2 is the average "
        "of the 23 per-variable R2 scores."
    )
    print("\nPer-variable R2 scores:")
    print(results[["Variable", "Observed", "Optimized_KNN_R2"]].to_string(index=False))

    # Step 2: Impute the 13 input variables only.
    input_data = numeric[:, :13]
    input_filled = knn_fill(input_data, improved=True)
    input_filled = postprocess_only_missing(input_data, input_filled)
    input_only = numeric.copy()
    input_only[:, :13] = input_filled

    # Step 3: Impute all 23 variables.
    all_filled = knn_fill(numeric, improved=True)
    all_filled = postprocess_only_missing(numeric, all_filled)

    # Step 4: Reconstruct Excel worksheets.
    input_excel = reconstruct_excel(header, body, numeric, input_only)
    complete_excel = reconstruct_excel(header, body, numeric, all_filled)

    # Step 5: Summarize missing values.
    missing_summary = pd.DataFrame({
        "Variable": VARIABLE_NAMES,
        "Type": ["Input"] * 13 + ["Target"] * 10,
        "Observed": np.isfinite(numeric).sum(axis=0),
        "Missing": np.isnan(numeric).sum(axis=0),
        "Missing_Ratio (%)": np.isnan(numeric).mean(axis=0) * 100
    })

    flags = pd.DataFrame(
        (~np.isfinite(numeric)).astype(int),
        columns=VARIABLE_NAMES
    )
    flags.insert(0, "Index No.", body.iloc[:, 0].to_numpy())

    summary = pd.DataFrame({
        "Metric": [
            "Original KNN mean R2",
            "Optimized KNN mean R2",
            "Number of samples",
            "Number of features",
            "KNN neighbors",
            "Scaling",
            "Cross-validation"
        ],
        "Value": [
            old_r2,
            optimized_r2,
            len(body),
            numeric.shape[1],
            N_NEIGHBORS,
            "RobustScaler",
            "5-fold masked validation"
        ]
    })

    # Step 6: Save all results.
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="Original", index=False, header=False)
        input_excel.to_excel(writer, sheet_name="KNN_InputOnly", index=False, header=False)
        complete_excel.to_excel(writer, sheet_name="KNN_All_Exploratory", index=False, header=False)
        results.to_excel(writer, sheet_name="R2_5fold", index=False)
        summary.to_excel(writer, sheet_name="R2_Summary", index=False)
        missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)
        flags.to_excel(writer, sheet_name="Imputation_Flags", index=False)

    print("\n" + "=" * 60)
    print("KNN imputation and R2 validation completed")
    print("=" * 60)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("\nPlease review the following worksheets:")
    print("KNN_InputOnly: input-variable imputation")
    print("KNN_All_Exploratory: imputation of all variables")
    print("R2_5fold: per-variable validation R2 scores")
    print("R2_Summary: mean R2 scores")


# =====================================================
# 9. Entry Point
# =====================================================
if __name__ == "__main__":
    main()
