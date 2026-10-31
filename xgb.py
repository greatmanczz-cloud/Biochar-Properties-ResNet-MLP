
# -*- coding: utf-8 -*-
"""
Pure XGBoost imputation with five-fold R2 validation.

The workflow does not use MICE, IterativeImputer, KNN, ExtraTrees,
or PyTorch.

Input workbook: dataset.xlsx
Worksheet: Raw dataset
Input variables (13): Excel columns G-I and K-T
Target variables (10): Excel columns U-AD

The label in Excel column Y is standardized to pH.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================================================
# 1. Configuration
# =====================================================
ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "dataset.xlsx"
OUTPUT_FILE = ROOT / "Biochar_Pure_XGBoost_Imputation_R2.xlsx"
SHEET_NAME = "Raw dataset"
HEADER_ROWS = 4
SEED = 2024
FOLDS = 5

# Enable or disable five-fold R2 validation.
RUN_R2_VALIDATION = True

# Apply physical bounds only to imputed predictions.
# Set False to retain the raw XGBoost predictions.
APPLY_BOUNDS = True


# =====================================================
# 2. XGBoost Hyperparameters
# =====================================================
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "missing": np.nan,
    "n_jobs": 4,
    "random_state": SEED,
}


# =====================================================
# 3. Excel Variable Mapping
# =====================================================
INPUT_MAP = {
    6: "FC_dry",
    7: "VM_dry",
    8: "Ash_dry",
    10: "Feedstock_C",
    11: "Feedstock_H",
    12: "Feedstock_O",
    13: "Feedstock_N",
    14: "Cellulose",
    15: "Hemicellulose",
    16: "Lignin",
    17: "Residence_time",
    18: "Temperature",
    19: "Heating_rate",
}

OUTPUT_MAP = {
    20: "Biochar_yield",
    21: "Total_pore_volume",
    22: "Biochar_ash",
    23: "Zeta_potential",
    24: "pH",
    25: "Biochar_C",
    26: "Biochar_H",
    27: "Biochar_O",
    28: "O_C_ratio",
    29: "H_C_ratio",
}

COLUMN_MAP = {**INPUT_MAP, **OUTPUT_MAP}
NAMES = list(COLUMN_MAP.values())
N_INPUTS = len(INPUT_MAP)

PERCENTAGE_NAMES = {
    "FC_dry", "VM_dry", "Ash_dry", "Feedstock_C", "Feedstock_H",
    "Feedstock_O", "Feedstock_N", "Cellulose", "Hemicellulose",
    "Lignin", "Biochar_yield", "Biochar_ash", "Biochar_C",
    "Biochar_H", "Biochar_O",
}

MISSING_TEXT = {
    "", "nan", "na", "n/a", "none", "null", "nr", "-", "—"
}


# =====================================================
# 4. Read Original Data
# =====================================================
def read_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input data file not found: {INPUT_FILE}")

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        dtype=object,
        keep_default_na=False,
    )

    if raw.shape[1] < 30:
        raise ValueError("The Excel workbook must have at least 30 columns.")

    # Preserve the first four header rows.
    header = raw.iloc[:HEADER_ROWS, :].copy()

    # Rename the column Y label from PZC to pH.
    header.iat[HEADER_ROWS - 1, 24] = "pH"

    body = raw.iloc[HEADER_ROWS:, :].copy()

    # Retain only rows with valid numeric sample identifiers.
    record_ids = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    body = body.loc[record_ids.notna()].reset_index(drop=True)

    if body.empty:
        raise ValueError("No valid sample records were found.")

    values = pd.DataFrame(index=body.index)
    invalid_rows = []

    # Extract 23 variables, preserving invalid-cell information.
    for excel_col, name in COLUMN_MAP.items():
        source = body.iloc[:, excel_col]
        text = source.astype(str).str.strip()
        numeric = pd.to_numeric(
            text.str.replace(",", "", regex=False),
            errors="coerce",
        )
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

        bad_mask = (
            numeric.isna()
            & ~text.str.lower().isin(MISSING_TEXT)
        )

        for index in np.flatnonzero(bad_mask.to_numpy()):
            invalid_rows.append({
                "Excel_row": int(index) + HEADER_ROWS + 1,
                "Variable": name,
                "Original_text": str(source.iloc[index]),
                "Note": "Non-numeric or below-detection-limit entry; manual review required",
            })

        values[name] = numeric.astype(float)

    # Ensure each variable has enough measured values to train.
    for name in NAMES:
        count = values[name].notna().sum()
        if count < FOLDS + 2:
            raise ValueError(
                f"Too few measured values for {name}: {count}"
            )

    invalid = pd.DataFrame(
        invalid_rows,
        columns=["Excel_row", "Variable", "Original_text", "Note"],
    )

    return raw, header, body, values, invalid


# =====================================================
# 5. Select Predictors for Each Target Variable
# =====================================================
def predictor_columns(j):
    # For an input variable, use the other 12 input variables.
    if j < N_INPUTS:
        return [k for k in range(N_INPUTS) if k != j]

    # For a target variable, use only the 13 input variables.
    return list(range(N_INPUTS))


# =====================================================
# 6. Apply Physical Bounds to Predicted Values
# =====================================================
def bound_prediction(pred, name):
    pred = np.asarray(pred, dtype=float)

    if not APPLY_BOUNDS:
        return pred

    # Zeta potential may be negative.
    if name == "Zeta_potential":
        return pred

    # Percentage variables must remain within 0-100.
    if name in PERCENTAGE_NAMES:
        return np.clip(pred, 0, 100)

    # pH must remain within 0-14.
    if name == "pH":
        return np.clip(pred, 0, 14)

    # Other quantities must be nonnegative.
    return np.maximum(pred, 0)


# =====================================================
# 7. Train an Independent XGBoost Model
# =====================================================
def train_predict(data_matrix, j, train_indices, pred_indices):
    features = predictor_columns(j)
    model = XGBRegressor(**XGB_PARAMS)

    # Train using measured values of the current target only.
    X_train = data_matrix[np.ix_(train_indices, features)]
    y_train = data_matrix[train_indices, j]

    # XGBoost can handle NaN values among input features natively.
    model.fit(X_train, y_train)

    # Predict entries whose values are missing or held out.
    X_pred = data_matrix[np.ix_(pred_indices, features)]
    prediction = model.predict(X_pred)
    prediction = bound_prediction(prediction, NAMES[j])

    return prediction


# =====================================================
# 8. Impute All 23 Variables With Pure XGBoost
# =====================================================
def impute_all(values):
    matrix = values.to_numpy(dtype=float)
    completed = matrix.copy()
    rows = np.arange(len(matrix))
    missing_flags = ~np.isfinite(matrix)

    for j, name in enumerate(NAMES):
        known = rows[~missing_flags[:, j]]
        missing = rows[missing_flags[:, j]]

        if len(missing) > 0:
            if len(known) < 8:
                raise ValueError(f"Insufficient measured values for {name}.")

            # Train a separate XGBoost model for this variable.
            predictions = train_predict(matrix, j, known, missing)
            completed[missing, j] = predictions

        print(
            f"Imputing {name:24s} "
            f"Measured={len(known):3d} "
            f"Missing={len(missing):3d}"
        )

    # Confirm that original measured values were preserved.
    if not np.array_equal(
        completed[~missing_flags], matrix[~missing_flags]
    ):
        raise AssertionError("Original measured values were modified.")

    # Verify that imputation is complete.
    if not np.isfinite(completed).all():
        raise AssertionError("Missing values remain after imputation.")

    return pd.DataFrame(completed, columns=NAMES, index=values.index)


# =====================================================
# 9. Five-Fold R2 Validation on Measured Values
# =====================================================
def fivefold_r2(values):
    matrix = values.to_numpy(dtype=float)
    results = []

    print("\nStarting five-fold R2 validation...")

    for j, name in enumerate(NAMES):
        # Validate only originally measured values.
        measured = np.flatnonzero(np.isfinite(matrix[:, j]))
        oof = np.full(len(matrix), np.nan)

        cv = KFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED,
        )

        for train_loc, test_loc in cv.split(measured):
            train_rows = measured[train_loc]
            test_rows = measured[test_loc]

            # Train a new model for the current fold.
            predictions = train_predict(
                matrix, j, train_rows, test_rows
            )
            oof[test_rows] = predictions

        if np.isnan(oof[measured]).any():
            raise AssertionError(
                f"Some validation values were not predicted for {name}."
            )

        y_true = matrix[measured, j]
        y_pred = oof[measured]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            "Variable": name,
            "Measured_count": len(measured),
            "R2_5fold_OOF": r2,
            "RMSE_5fold_OOF": rmse,
            "MAE_5fold_OOF": mae,
        })

        print(f"Validated {name:24s} R2={r2:.4f}")

    return pd.DataFrame(results)


# =====================================================
# 10. Restore the Original Excel Layout
# =====================================================
def restore_sheet(header, body, raw_numeric, completed, input_only=False):
    out = body.copy()
    allowed = INPUT_MAP if input_only else COLUMN_MAP

    for excel_col, name in allowed.items():
        missing = raw_numeric[name].isna().to_numpy()

        if missing.any():
            out.iloc[
                np.flatnonzero(missing), excel_col
            ] = completed.loc[missing, name].to_numpy()

    return pd.concat([header, out], ignore_index=True)


# =====================================================
# 11. Main Program
# =====================================================
def main():
    print("=" * 60)
    print("Pure XGBoost Missing-Value Imputation")
    print("=" * 60)
    print("MICE, KNN, and PyTorch are not used.")

    # Step 1: Read the original data.
    raw, header, body, numeric, invalid = read_data()
    print(f"\nNumber of samples: {len(body)}")
    print(f"Number of variables: {len(NAMES)}")
    print(f"Original missing values: {int(numeric.isna().sum().sum())}")

    # Step 2: Impute missing values with XGBoost.
    completed = impute_all(numeric)

    # Step 3: Perform five-fold R2 validation if enabled.
    scores = None
    if RUN_R2_VALIDATION:
        scores = fivefold_r2(numeric)

    # Step 4: Record the original missing-value locations.
    flags = numeric.isna().astype(int)
    flags.insert(0, "Index_No", body.iloc[:, 0].to_numpy())

    # Step 5: Calculate missing-value statistics.
    missing_summary = pd.DataFrame({
        "Variable": NAMES,
        "Role": ["Input"] * N_INPUTS + ["Target"] * len(OUTPUT_MAP),
        "Measured": numeric.notna().sum().to_numpy(),
        "Missing": numeric.isna().sum().to_numpy(),
        "Missing_pct": (
            numeric.isna().mean().mul(100).round(2).to_numpy()
        ),
    })

    # Step 6: Build the original, input-only, and full datasets.
    original_out = pd.concat([header, body], ignore_index=True)
    input_out = restore_sheet(
        header, body, numeric, completed, input_only=True
    )
    full_out = restore_sheet(
        header, body, numeric, completed, input_only=False
    )

    # Step 7: Export Excel results.
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        original_out.to_excel(
            writer, sheet_name="Original", index=False, header=False
        )
        input_out.to_excel(
            writer, sheet_name="XGB_InputOnly", index=False, header=False
        )
        full_out.to_excel(
            writer, sheet_name="XGB_All_Exploratory", index=False, header=False
        )
        missing_summary.to_excel(
            writer, sheet_name="Missing_Summary", index=False
        )
        flags.to_excel(
            writer, sheet_name="Imputation_Flags", index=False
        )
        invalid.to_excel(
            writer, sheet_name="Invalid_Cells", index=False
        )

        if scores is not None:
            scores.to_excel(
                writer, sheet_name="R2_5fold", index=False
            )

            summary = pd.DataFrame({
                "Metric": [
                    "Mean_R2_All_23_Variables",
                    "Mean_R2_13_Input_Variables",
                    "Mean_R2_10_Target_Variables",
                ],
                "Value": [
                    scores["R2_5fold_OOF"].mean(),
                    scores.iloc[:N_INPUTS]["R2_5fold_OOF"].mean(),
                    scores.iloc[N_INPUTS:]["R2_5fold_OOF"].mean(),
                ],
            })

            summary.to_excel(
                writer, sheet_name="R2_Summary", index=False
            )

            print("\n====== R2 Validation Results ======")
            for row in summary.itertuples(index=False):
                print(f"{row.Metric}: {row.Value:.4f}")

    print("\n" + "=" * 60)
    print("Pure XGBoost imputation completed.")
    print("=" * 60)
    print(f"\nOutput workbook: {OUTPUT_FILE}")
    print("All original measured values were preserved.")
    print("The label in column Y was renamed from PZC to pH.")


# =====================================================
# 12. Program Entry Point
# =====================================================
if __name__ == "__main__":
    main()
