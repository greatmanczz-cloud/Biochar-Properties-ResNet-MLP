
from pathlib import Path
from collections import Counter
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "dataset.xlsx"
OUTPUT_FILE = BASE_DIR / "ETII.xlsx"

SHEET_NAME = "Raw dataset"

HEADER_ROWS = 4
SEED = 2024
FOLDS = 5

TOP_FEEDSTOCK_TYPES = 15

N_TREES = 125
MIN_SAMPLES_LEAF = 1
MAX_ITER = 4
N_JOBS = 4

# Enable five-fold cross-reference validation
RUN_GROUP_CV = True

# Parameters for the GHI composition constraint
SUM_TARGET = 100.0
SUM_TOL = 1e-8


# ============================================================
# 2. Excel Input and Target Variables
# ============================================================

# 13 input variables: columns G-I and K-T
# 10 target variables: columns U-AD

EXCEL_COLS = [
    6, 7, 8,
    10, 11, 12, 13,
    14, 15, 16, 17, 18, 19
] + list(range(20, 30))

NAMES = [
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
    "Biochar_yield",
    "Total_pore_volume",
    "Biochar_ash",
    "Zeta_potential",
    "pH",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O",
    "O_C_ratio",
    "H_C_ratio"
]

N_INPUTS = 13

PERCENT_NAMES = {
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
    "Biochar_yield",
    "Biochar_ash",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O"
}

MISSING_TEXT = {
    "",
    "nan",
    "na",
    "n/a",
    "nr",
    "null",
    "none",
    "-",
    "—"
}


# ============================================================
# 3. Load Data
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Excel file not found: {INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        dtype=object,
        keep_default_na=False
    )

    if raw.shape[1] < 31:
        raise ValueError(
            "The source spreadsheet must contain at least 31 columns."
        )

    headers = raw.iloc[:HEADER_ROWS].copy()

    # Preserve the original column naming convention.
    # Verify the source if this property actually represents PZC.
    headers.iat[3, 24] = "pH"

    body = raw.iloc[HEADER_ROWS:].copy()

    record_numbers = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    body = body.loc[
        record_numbers.notna()
    ].reset_index(drop=True)

    if body.empty:
        raise ValueError(
            "No valid samples were found."
        )

    cols = []
    invalid = []

    for pos, name in zip(EXCEL_COLS, NAMES):

        raw_col = body.iloc[:, pos]

        txt = (
            raw_col.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
        )

        numeric = pd.to_numeric(
            txt,
            errors="coerce"
        )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan
        )

        bad = (
            numeric.isna()
            & ~txt.str.lower().isin(MISSING_TEXT)
        )

        for i in np.flatnonzero(bad.to_numpy()):

            invalid.append({
                "Excel_row": i + HEADER_ROWS + 1,
                "Variable": name,
                "Original_text": str(raw_col.iloc[i]),
                "Note": (
                    "Non-numeric text treated as missing; "
                    "please verify."
                )
            })

        cols.append(
            numeric.to_numpy(dtype=float)
        )

    data = np.column_stack(cols)

    if np.any(
        np.isfinite(data).sum(axis=0) < 10
    ):
        raise ValueError(
            "Some variables have fewer than 10 observed values."
        )

    # Auxiliary feedstock-type features
    types = (
        body.iloc[:, 1]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    top_types = [
        k for k, _ in Counter(types).most_common(
            TOP_FEEDSTOCK_TYPES
        )
    ]

    feedstock_aux = np.column_stack([
        np.array(
            [float(t == kind) for t in types],
            dtype=float
        )
        for kind in top_types
    ])

    # Reference identifiers
    sources = (
        body.iloc[:, 30]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    sources = np.array([
        " ".join(s.split())
        if s else f"unknown_ref_{i}"
        for i, s in enumerate(sources)
    ])

    return (
        raw,
        headers,
        body,
        data,
        feedstock_aux,
        sources,
        pd.DataFrame(invalid)
    )


# ============================================================
# 4. Create the ETII Imputer
# ============================================================

def make_imputer(seed=SEED):

    regressor = ExtraTreesRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=0.90,
        random_state=SEED,
        n_jobs=N_JOBS
    )

    return IterativeImputer(
        estimator=regressor,
        initial_strategy="median",
        max_iter=MAX_ITER,
        imputation_order="ascending",
        skip_complete=True,
        sample_posterior=False,
        keep_empty_features=True,
        random_state=seed,
        tol=1e-3
    )


# ============================================================
# 5. Apply Numerical Bounds
# ============================================================

def apply_bounds(original, completed, names):

    output = completed.copy()
    missing = ~np.isfinite(original)

    for j, name in enumerate(names):

        mask = missing[:, j]

        if name == "Zeta_potential":
            continue

        if name == "pH":

            output[mask, j] = np.clip(
                output[mask, j],
                0,
                14
            )

        elif name in PERCENT_NAMES:

            output[mask, j] = np.clip(
                output[mask, j],
                0,
                100
            )

        else:

            output[mask, j] = np.maximum(
                output[mask, j],
                0
            )

    # Preserve originally observed values
    output[~missing] = original[~missing]

    return output


# ============================================================
# 6. Nonnegative Simplex Projection
# ============================================================

def project_nonnegative_simplex(
    prediction,
    target_total
):

    """
    Project predicted values onto the feasible set:

        x >= 0
        sum(x) = target_total

    Applicable when two or three composition
    variables are simultaneously missing.
    """

    vector = np.asarray(
        prediction,
        dtype=float
    )

    if not np.isfinite(vector).all():
        raise ValueError(
            "GHI predictions contain non-finite values."
        )

    if target_total < -SUM_TOL:
        raise ValueError(
            "The sum of observed components exceeds 100%."
        )

    target_total = max(
        0.0,
        float(target_total)
    )

    if len(vector) == 1:

        return np.array(
            [target_total],
            dtype=float
        )

    if target_total == 0:
        return np.zeros_like(vector)

    ordered = np.sort(vector)[::-1]

    cumulative = (
        np.cumsum(ordered)
        - target_total
    )

    candidates = np.flatnonzero(
        ordered
        - cumulative
        / np.arange(1, len(vector) + 1)
        > 0
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "Simplex projection failed."
        )

    rho = int(candidates[-1])

    theta = (
        cumulative[rho]
        / (rho + 1)
    )

    projected = np.maximum(
        vector - theta,
        0.0
    )

    # Correct floating-point rounding errors
    index = int(np.argmax(projected))

    projected[index] += (
        target_total
        - float(projected.sum())
    )

    return projected


# ============================================================
# 7. Enforce the GHI Mass-Balance Constraint
# ============================================================

def enforce_ghi_sum(observed, completed):

    """
    Column G: FC_dry
    Column H: VM_dry
    Column I: Ash_dry

    FC_dry + VM_dry + Ash_dry = 100

    Only originally missing entries are modified.
    All observed values remain unchanged.
    """

    observed = np.asarray(
        observed,
        dtype=float
    )

    adjusted = np.asarray(
        completed,
        dtype=float
    ).copy()

    if observed.shape != adjusted.shape:
        raise ValueError(
            "Input matrices have inconsistent shapes."
        )

    for i in range(len(adjusted)):

        missing = ~np.isfinite(
            observed[i, :3]
        )

        # Keep fully observed records unchanged
        if not missing.any():
            continue

        known = observed[i, :3][~missing]

        if np.any(known < -SUM_TOL):
            raise ValueError(
                f"Record {i + 1} contains "
                "a negative observed component."
            )

        known_sum = float(known.sum())

        remaining = SUM_TARGET - known_sum

        if remaining < -SUM_TOL:
            raise ValueError(
                f"Excel row {i + HEADER_ROWS + 1}: "
                "The observed components sum to more than 100%; "
                "please check the source data."
            )

        predicted_missing = (
            adjusted[i, :3][missing]
        )

        corrected = project_nonnegative_simplex(
            predicted_missing,
            remaining
        )

        # Update missing components only
        adjusted[i, :3][missing] = corrected

        final_sum = float(
            adjusted[i, :3].sum()
        )

        if abs(final_sum - 100) > SUM_TOL:
            raise AssertionError(
                f"Record {i + 1} failed the GHI constraint."
            )

    # Verify preservation of observed values
    measured = np.isfinite(observed)

    if not np.array_equal(
        adjusted[measured],
        observed[measured]
    ):
        raise AssertionError(
            "Originally observed values were modified."
        )

    if not np.isfinite(adjusted).all():
        raise AssertionError(
            "Missing values remain after applying constraints."
        )

    return adjusted


# ============================================================
# 8. GHI Composition Validation Report
# ============================================================

def composition_report(observed, final):

    sums = final[:, :3].sum(axis=1)

    missing_count = (
        np.isnan(observed[:, :3]).sum(axis=1)
    )

    report = pd.DataFrame({

        "Excel_row":
            np.arange(len(final)) + HEADER_ROWS + 1,

        "FC_dry":
            final[:, 0],

        "VM_dry":
            final[:, 1],

        "Ash_dry":
            final[:, 2],

        "GHI_sum":
            sums,

        "Deviation_from_100":
            sums - 100,

        "Original_GHI_missing_count":
            missing_count,

        "Record_type":
            np.where(
                missing_count == 0,
                "All measured",
                "Contains imputation"
            )
    })

    mask = missing_count > 0

    if np.any(
        np.abs(sums[mask] - 100) > SUM_TOL
    ):
        raise AssertionError(
            "Some imputed records violate the GHI constraint."
        )

    return report


# ============================================================
# 9. ETII Imputation
# ============================================================

def fit_impute(
    matrix,
    feedstock_aux,
    names,
    seed=SEED
):

    model_input = np.column_stack([
        matrix,
        feedstock_aux
    ])

    model = make_imputer(seed)

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message=(
                ".*Early stopping criterion "
                "not reached.*"
            )
        )

        filled_all = model.fit_transform(
            model_input
        )

    completed = filled_all[
        :, :matrix.shape[1]
    ]

    measured = np.isfinite(matrix)

    completed[measured] = matrix[measured]

    if not np.isfinite(completed).all():
        raise RuntimeError(
            "NaN values remain after ETII imputation."
        )

    return completed


# ============================================================
# 10. Imputation Performance Metrics
# ============================================================

def metrics(
    truth,
    raw_predictions,
    bounded_predictions,
    constrained_predictions,
    names
):

    records = []

    for j, name in enumerate(names):

        mask = (
            np.isfinite(truth[:, j])
            & np.isfinite(raw_predictions[:, j])
        )

        y = truth[mask, j]

        a = raw_predictions[mask, j]
        b = bounded_predictions[mask, j]
        c = constrained_predictions[mask, j]

        records.append({

            "Variable": name,

            "Observed":
                int(mask.sum()),

            "Originally_missing":
                int(np.isnan(truth[:, j]).sum()),

            "R2_model":
                r2_score(y, a),

            "R2_after_bounds":
                r2_score(y, b),

            "RMSE_after_bounds":
                np.sqrt(mean_squared_error(y, b)),

            "MAE_after_bounds":
                mean_absolute_error(y, b),

            "R2_after_constraints":
                r2_score(y, c),

            "RMSE_after_constraints":
                np.sqrt(mean_squared_error(y, c)),

            "MAE_after_constraints":
                mean_absolute_error(y, c)
        })

    return pd.DataFrame(records)


# ============================================================
# 11. Five-Fold Random-Masking Validation
# ============================================================

def random_mask_fivefold(data, aux, names):

    n, p = data.shape

    rng = np.random.default_rng(SEED)

    folds = []

    for j in range(p):

        known = np.flatnonzero(
            np.isfinite(data[:, j])
        )

        folds.append(
            np.array_split(
                rng.permutation(known),
                FOLDS
            )
        )

    raw_pred = np.full(
        (n, p),
        np.nan
    )

    bounded_pred = np.full(
        (n, p),
        np.nan
    )

    constrained_pred = np.full(
        (n, p),
        np.nan
    )

    for fold in range(FOLDS):

        masked = data.copy()

        # Mask observed values assigned to the current fold
        for j in range(p):

            masked[
                folds[j][fold],
                j
            ] = np.nan

        # Original ETII imputation
        filled = fit_impute(
            masked,
            aux,
            names,
            SEED
        )

        # Apply numerical bounds
        bounded = apply_bounds(
            masked,
            filled,
            names
        )

        # Apply the GHI mass-balance constraint
        constrained = enforce_ghi_sum(
            masked,
            bounded
        )

        # Store predictions for held-out observations
        for j in range(p):

            idx = folds[j][fold]

            raw_pred[idx, j] = filled[idx, j]

            bounded_pred[idx, j] = bounded[idx, j]

            constrained_pred[idx, j] = (
                constrained[idx, j]
            )

        print(
            f"Random-masking validation "
            f"Fold {fold + 1}/{FOLDS} completed",
            flush=True
        )

    return metrics(
        data,
        raw_pred,
        bounded_pred,
        constrained_pred,
        names
    )


# ============================================================
# 12. Five-Fold Cross-Reference Validation
# ============================================================

def grouped_target_cv(
    data,
    aux,
    references
):

    if len(set(references)) < FOLDS:

        print(
            "Too few references; skipping grouped validation."
        )

        return None

    pred = np.full(
        (len(data), 10),
        np.nan
    )

    cv = GroupKFold(
        n_splits=FOLDS
    )

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            np.arange(len(data)),
            groups=references
        ),
        start=1
    ):

        model = make_imputer(SEED)

        train = np.column_stack([
            data[train_idx],
            aux[train_idx]
        ])

        test = data[test_idx].copy()

        # Mask all target variables in the test set
        test[:, N_INPUTS:] = np.nan

        test = np.column_stack([
            test,
            aux[test_idx]
        ])

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    ".*Early stopping criterion "
                    "not reached.*"
                )
            )

            model.fit(train)

            predictions = model.transform(test)

        targets_pred = predictions[
            :, N_INPUTS:23
        ]

        # Apply physical bounds to target predictions only
        fully_masked_targets = np.full_like(
            targets_pred,
            np.nan
        )

        targets_bounded = apply_bounds(
            fully_masked_targets,
            targets_pred,
            NAMES[N_INPUTS:]
        )

        pred[test_idx] = targets_bounded

        print(
            f"Cross-reference validation "
            f"Fold {fold}/{FOLDS} completed",
            flush=True
        )

    scores = []

    for j, name in enumerate(
        NAMES[N_INPUTS:]
    ):

        mask = (
            np.isfinite(data[:, N_INPUTS + j])
            & np.isfinite(pred[:, j])
        )

        if mask.sum() < 2:
            continue

        y = data[
            mask,
            N_INPUTS + j
        ]

        p = pred[mask, j]

        scores.append({

            "Variable": name,

            "Observed":
                int(mask.sum()),

            "R2_grouped":
                r2_score(y, p),

            "RMSE_grouped":
                np.sqrt(mean_squared_error(y, p)),

            "MAE_grouped":
                mean_absolute_error(y, p)
        })

    return pd.DataFrame(scores)


# ============================================================
# 13. Restore the Original Excel Layout
# ============================================================

def rebuild_sheet(
    header,
    body,
    observed,
    filled,
    columns_to_fill
):

    output = body.copy()

    for j in columns_to_fill:

        missing_rows = np.flatnonzero(
            ~np.isfinite(observed[:, j])
        )

        if len(missing_rows):

            output.iloc[
                missing_rows,
                EXCEL_COLS[j]
            ] = filled[
                missing_rows,
                j
            ]

    return pd.concat(
        [header, output],
        ignore_index=True
    )


# ============================================================
# 14. Main Program
# ============================================================

def main():

    print("=" * 65)
    print("ETII with the GHI mass-balance constraint")
    print("=" * 65)

    (
        raw,
        header,
        body,
        data,
        aux,
        sources,
        invalid
    ) = load_data()

    print(
        f"Number of samples: {len(body)}"
    )

    print(
        f"Number of variables: {data.shape[1]}"
    )

    print(
        f"Original missing values: {np.isnan(data).sum()}"
    )

    # --------------------------------------------------------
    # Step 1: Five-fold random-masking validation
    # --------------------------------------------------------

    print(
        "\nStarting five-fold random-masking validation..."
    )

    scores = random_mask_fivefold(
        data,
        aux,
        NAMES
    )

    # --------------------------------------------------------
    # Step 2: Cross-reference validation
    # --------------------------------------------------------

    if RUN_GROUP_CV:

        group_scores = grouped_target_cv(
            data,
            aux,
            sources
        )

    else:

        group_scores = None

    # --------------------------------------------------------
    # Step 3: Impute the 13 input variables
    # --------------------------------------------------------

    print(
        "\nStarting imputation of the 13 input variables..."
    )

    input_filled = fit_impute(
        data[:, :N_INPUTS],
        aux,
        NAMES[:N_INPUTS]
    )

    input_filled = apply_bounds(
        data[:, :N_INPUTS],
        input_filled,
        NAMES[:N_INPUTS]
    )

    # Apply the GHI constraint
    input_filled = enforce_ghi_sum(
        data[:, :N_INPUTS],
        input_filled
    )

    input_only = data.copy()

    input_only[:, :N_INPUTS] = input_filled

    # --------------------------------------------------------
    # Step 4: Impute all 23 variables
    # --------------------------------------------------------

    print(
        "\nStarting complete imputation of all 23 variables..."
    )

    completed = fit_impute(
        data,
        aux,
        NAMES
    )

    completed = apply_bounds(
        data,
        completed,
        NAMES
    )

    # Apply the GHI constraint
    completed = enforce_ghi_sum(
        data,
        completed
    )

    # --------------------------------------------------------
    # Step 5: Validation checks
    # --------------------------------------------------------

    measured = np.isfinite(data)

    assert np.array_equal(
        completed[measured],
        data[measured]
    ), "Observed values were modified."

    assert np.isfinite(
        completed
    ).all(), "Missing values remain."

    check = composition_report(
        data,
        completed
    )

    imputed_rows = (
        check["Original_GHI_missing_count"] > 0
    )

    print(
        "Records requiring GHI imputation:",
        int(imputed_rows.sum())
    )

    print(
        "Maximum GHI sum deviation among imputed records:",
        float(
            check.loc[
                imputed_rows,
                "Deviation_from_100"
            ].abs().max()
        )
    )

    # --------------------------------------------------------
    # Step 6: Summarize R2 metrics
    # --------------------------------------------------------

    eligible = (
        scores["Originally_missing"] > 0
    )

    summary = pd.DataFrame([

        {
            "Metric":
                "Mean_R2_23Variables_OriginalETII",

            "Value":
                scores["R2_model"].mean()
        },

        {
            "Metric":
                "Mean_R2_22IncompleteVariables_OriginalETII",

            "Value":
                scores.loc[
                    eligible,
                    "R2_model"
                ].mean()
        },

        {
            "Metric":
                "Mean_R2_22IncompleteVariables_AfterBounds",

            "Value":
                scores.loc[
                    eligible,
                    "R2_after_bounds"
                ].mean()
        },

        {
            "Metric":
                "Mean_R2_22IncompleteVariables_AfterGHIConstraints",

            "Value":
                scores.loc[
                    eligible,
                    "R2_after_constraints"
                ].mean()
        },

        {
            "Metric":
                "Mean_R2_13Inputs_AfterConstraints",

            "Value":
                scores.iloc[
                    :N_INPUTS
                ]["R2_after_constraints"].mean()
        },

        {
            "Metric":
                "Mean_R2_10Targets_RandomMasking",

            "Value":
                scores.iloc[
                    N_INPUTS:
                ]["R2_after_constraints"].mean()
        },

        {
            "Metric":
                "Mean_R2_10Targets_CrossReference",

            "Value":
                (
                    group_scores[
                        "R2_grouped"
                    ].mean()
                    if group_scores is not None
                    else np.nan
                )
        }
    ])

    # --------------------------------------------------------
    # Step 7: Summarize missing values
    # --------------------------------------------------------

    missing_stats = pd.DataFrame({

        "Variable":
            NAMES,

        "Role":
            ["Input"] * N_INPUTS
            + ["Target"] * (
                len(NAMES) - N_INPUTS
            ),

        "Observed":
            np.isfinite(data).sum(axis=0),

        "Missing":
            np.isnan(data).sum(axis=0),

        "Missing_percent":
            np.isnan(data).mean(axis=0) * 100
    })

    flags = pd.DataFrame(
        np.isnan(data).astype(int),
        columns=NAMES
    )

    flags.insert(
        0,
        "Index_No",
        body.iloc[:, 0].to_numpy()
    )

    # --------------------------------------------------------
    # Step 8: Export results to Excel
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        # Original dataset
        rebuild_sheet(
            header,
            body,
            data,
            data,
            []
        ).to_excel(
            writer,
            sheet_name="Original",
            header=False,
            index=False
        )

        # Input-only imputation
        rebuild_sheet(
            header,
            body,
            data,
            input_only,
            list(range(N_INPUTS))
        ).to_excel(
            writer,
            sheet_name="MICE_InputOnly",
            header=False,
            index=False
        )

        # Complete imputation
        rebuild_sheet(
            header,
            body,
            data,
            completed,
            list(range(len(NAMES)))
        ).to_excel(
            writer,
            sheet_name="MICE_All_Exploratory",
            header=False,
            index=False
        )

        # Random-masking validation
        scores.to_excel(
            writer,
            sheet_name="R2_Random5fold",
            index=False
        )

        # Summary statistics
        summary.to_excel(
            writer,
            sheet_name="R2_Summary",
            index=False
        )

        # Cross-reference validation
        if group_scores is not None:

            group_scores.to_excel(
                writer,
                sheet_name="R2_Group5fold",
                index=False
            )

        # Missing-value statistics
        missing_stats.to_excel(
            writer,
            sheet_name="Missing_Stats",
            index=False
        )

        # Missing-value indicators
        flags.to_excel(
            writer,
            sheet_name="Missing_Flags",
            index=False
        )

        # Invalid input cells
        invalid.to_excel(
            writer,
            sheet_name="Invalid_Cells",
            index=False
        )

        # GHI constraint validation
        check.to_excel(
            writer,
            sheet_name="GHI_Constraint_Check",
            index=False
        )

    print("\n========== R2 Summary ==========")

    print(
        summary.to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUTPUT_FILE
    )

    print(
        "\nAll steps completed."
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
