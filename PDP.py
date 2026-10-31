
# -*- coding: utf-8 -*-
"""
ResNet-MLP five-fold PDP recalculation and response heatmap generation.

Previously saved PDP Excel files are not required.
Model retraining is not required.
SHAP installation is not required.

Requirements:
1. The original Excel workbook used for model training.
2. Checkpoint files for the five ResNet-MLP folds.

Outputs:
1. Complete five-fold PDP data workbook.
2. Origin-ready PDP response heatmap data workbook.
3. Ten individual target response heatmaps.
4. One overview heatmap covering all ten targets.
5. PNG and PDF figures.

Colors:
Blue: predicted response below the midpoint baseline.
Red: predicted response above the midpoint baseline.

The 0%-100% horizontal axis represents the relative position
within each feature's P5-P95 grid-value range.
It does not represent empirical sample percentiles.

PDP describes model predictions, not causal effects.
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import torch
from torch import nn


# =====================================================
# 1. Configuration
# =====================================================

EXCEL_NAME = "ETII.xlsx"
SHEET_NAME = "MICE_All_Exploratory"

K_FOLDS = 5
HIDDEN_SIZE = 128
NUM_BLOCKS = 3
DROPOUT_P = 0.4

GRID_POINTS = 31
GRID_PERCENTILE_LOW = 5
GRID_PERCENTILE_HIGH = 95

PRED_BATCH_SIZE = 512
HEATMAP_POINTS = 31
DPI = 300

DRAW_10_TARGET_OVERVIEW = True


# =====================================================
# 2. File Paths
# =====================================================

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / EXCEL_NAME

CKPT_DIR = (
    BASE
    / "ResNet_MLP_OOF_SHAP_Results"
    / "FiveFold_Checkpoints"
)

OUT_DIR = BASE / "ResNet_MLP_OOF_PDP_Results"
FIG_DIR = OUT_DIR / "PDP_Response_Heatmaps"

PDP_EXCEL = (
    OUT_DIR / "ResNet_MLP_5Fold_PDP_Origin_Data.xlsx"
)

HEATMAP_EXCEL = (
    OUT_DIR / "PDP_Response_Heatmap_Origin_Data.xlsx"
)


# =====================================================
# 3. Input and Target Variables
# =====================================================

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
    "Heating_rate"
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
    "H_C_ratio"
]

INPUT_INDEX = [6, 7, 8] + list(range(10, 20))
OUTPUT_INDEX = list(range(20, 30))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =====================================================
# 4. ResNet-MLP Architecture
# =====================================================

class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, dropout_p):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)

        return torch.relu(out + residual)


class HybridResNetMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.entry_layer = nn.Linear(13, HIDDEN_SIZE)
        self.bn_entry = nn.BatchNorm1d(HIDDEN_SIZE)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(HIDDEN_SIZE, DROPOUT_P)
            for _ in range(NUM_BLOCKS)
        ])

        self.output_layer = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        x = self.entry_layer(x)
        x = self.bn_entry(x)
        x = torch.relu(x)
        x = self.res_blocks(x)

        return self.output_layer(x)


# =====================================================
# 5. Load and Validate Original Dataset
# =====================================================

def load_data():

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Original Excel file not found: {DATA_PATH}"
        )

    raw = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl"
    )

    if raw.shape[1] < 30:
        raise ValueError(
            "The Excel workbook contains fewer than 30 columns."
        )

    body = raw.iloc[4:, :].copy()

    ids = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    body = body.loc[ids.notna()].reset_index(drop=True)

    X = (
        body.iloc[:, INPUT_INDEX]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    y = (
        body.iloc[:, OUTPUT_INDEX]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    if X.shape[1] != 13 or y.shape[1] != 10:
        raise ValueError(
            f"Invalid array dimensions: X={X.shape}, y={y.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise ValueError(
            "The input or target data contain NaN or Inf."
        )

    data_sha = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    print(f"Original data: {DATA_PATH.name}")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"Data SHA256: {data_sha}")

    return X, y, data_sha


# =====================================================
# 6. Tensor Conversion
# =====================================================

def to_numpy(value):

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


# =====================================================
# 7. Load a Fold Checkpoint
# =====================================================

def load_fold_model(fold, data_sha, n_samples):

    path = CKPT_DIR / f"ResNet_MLP_Fold{fold}.pt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            "Confirm that all five model checkpoints were saved."
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if ckpt.get("data_sha256") != data_sha:
        raise ValueError(
            f"Fold {fold} checkpoint does not match "
            "the current Excel data.\n"
            "Use the exact original Excel file used for training.\n"
            "Renaming the file cannot bypass this validation."
        )

    expected_arch = (
        13, 10, HIDDEN_SIZE, NUM_BLOCKS, DROPOUT_P
    )

    if tuple(ckpt["architecture"]) != expected_arch:
        raise ValueError(
            f"Fold {fold} model architecture does not match."
        )

    train_idx = to_numpy(
        ckpt["train_idx"]
    ).astype(int)

    test_idx = to_numpy(
        ckpt["test_idx"]
    ).astype(int)

    if (
        len(test_idx) == 0
        or np.any(test_idx < 0)
        or np.any(test_idx >= n_samples)
        or len(np.unique(test_idx)) != len(test_idx)
        or np.intersect1d(train_idx, test_idx).size > 0
    ):
        raise ValueError(
            f"Fold {fold} has invalid training or test indices."
        )

    model = HybridResNetMLP().to(DEVICE)

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()

    return model, ckpt, test_idx


# =====================================================
# 8. Prediction in Original Units
# =====================================================

@torch.no_grad()
def predict_original_units(model, ckpt, X_original):

    x_mean = to_numpy(
        ckpt["x_mean"]
    ).reshape(1, -1)

    x_scale = to_numpy(
        ckpt["x_scale"]
    ).reshape(1, -1)

    y_mean = to_numpy(
        ckpt["y_mean"]
    ).reshape(1, -1)

    y_scale = to_numpy(
        ckpt["y_scale"]
    ).reshape(1, -1)

    if (
        np.any(x_scale <= 0)
        or np.any(y_scale <= 0)
    ):
        raise ValueError(
            "Invalid model scaling parameters."
        )

    X_standard = (
        (X_original - x_mean) / x_scale
    ).astype(np.float32)

    pred_batches = []

    for start in range(
        0, len(X_standard), PRED_BATCH_SIZE
    ):

        xb = torch.from_numpy(
            X_standard[start:start + PRED_BATCH_SIZE]
        ).to(DEVICE)

        pred = model(xb)

        pred_batches.append(
            pred.cpu().numpy()
        )

    y_standard = np.concatenate(
        pred_batches,
        axis=0
    )

    y_original = (
        y_standard * y_scale + y_mean
    )

    return y_original


# =====================================================
# 9. Generate PDP Grid
# =====================================================

def make_feature_grid(x):

    unique_values = np.unique(x)

    low, high = np.percentile(
        x,
        [GRID_PERCENTILE_LOW, GRID_PERCENTILE_HIGH]
    )

    within = unique_values[
        (unique_values >= low)
        & (unique_values <= high)
    ]

    if len(unique_values) <= 12 and len(within) > 0:
        return within.astype(float)

    if high > low:
        return np.linspace(
            low,
            high,
            GRID_POINTS
        )

    return np.array([float(low)])


# =====================================================
# 10. Recompute Five-Fold PDP
# =====================================================

def recompute_five_fold_pdp(X, data_sha):

    grids = {
        feature: make_feature_grid(X[:, j])
        for j, feature in enumerate(FEATURES)
    }

    fold_records = []

    seen = np.zeros(
        len(X),
        dtype=int
    )

    for fold in range(1, K_FOLDS + 1):

        model, ckpt, test_idx = load_fold_model(
            fold,
            data_sha,
            len(X)
        )

        seen[test_idx] += 1

        X_test = X[test_idx].copy()
        n_test = len(X_test)

        print(
            f"\nFold {fold}/{K_FOLDS}, "
            f"held-out samples: {n_test}",
            flush=True
        )

        for j, feature in enumerate(FEATURES):

            grid = grids[feature]

            X_changed = np.tile(
                X_test,
                (len(grid), 1)
            )

            X_changed[:, j] = np.repeat(
                grid,
                n_test
            )

            pred = predict_original_units(
                model,
                ckpt,
                X_changed
            )

            pred = pred.reshape(
                len(grid),
                n_test,
                len(TARGETS)
            )

            means = pred.mean(axis=1)

            for k, x_value in enumerate(grid):

                for t, target in enumerate(TARGETS):

                    fold_records.append({
                        "Fold": fold,
                        "Target": target,
                        "Feature": feature,
                        "X_Value": float(x_value),
                        "PDP_Fold_Mean": float(means[k, t]),
                        "N_HeldOut": n_test
                    })

            print(
                f"Completed {j + 1}/{len(FEATURES)}: "
                f"{feature}",
                flush=True
            )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not np.all(seen == 1):
        raise RuntimeError(
            "The five held-out folds do not cover "
            "every sample exactly once."
        )

    fold_df = pd.DataFrame(fold_records)
    records = []

    grouped = fold_df.groupby(
        ["Target", "Feature", "X_Value"],
        sort=False
    )

    for (target, feature, x_value), sub in grouped:

        values = sub[
            "PDP_Fold_Mean"
        ].to_numpy(dtype=float)

        weights = sub[
            "N_HeldOut"
        ].to_numpy(dtype=float)

        pdp_mean = np.average(
            values,
            weights=weights
        )

        pdp_sd = (
            np.std(values, ddof=1)
            if len(values) > 1
            else 0.0
        )

        records.append({
            "Target": target,
            "Feature": feature,
            "X_Value": float(x_value),
            "PDP_Mean": float(pdp_mean),
            "PDP_Fold_SD": float(pdp_sd),
            "N_Folds": len(values),
            "N_Total_HeldOut": int(weights.sum())
        })

    pdp_df = pd.DataFrame(records)

    pdp_df["Target"] = pd.Categorical(
        pdp_df["Target"],
        categories=TARGETS,
        ordered=True
    )

    pdp_df["Feature"] = pd.Categorical(
        pdp_df["Feature"],
        categories=FEATURES,
        ordered=True
    )

    pdp_df = pdp_df.sort_values(
        ["Target", "Feature", "X_Value"]
    ).reset_index(drop=True)

    pdp_df["Target"] = pdp_df["Target"].astype(str)
    pdp_df["Feature"] = pdp_df["Feature"].astype(str)

    return pdp_df, fold_df


# =====================================================
# 11. Create PDP Response Matrices
# =====================================================

def make_response_matrix(pdp_df, target):

    target_df = pdp_df[
        pdp_df["Target"] == target
    ]

    positions = np.linspace(
        0,
        100,
        HEATMAP_POINTS
    )

    matrix_rows = []
    raw_rows = []
    long_records = []
    range_records = []

    for feature in FEATURES:

        d = target_df[
            target_df["Feature"] == feature
        ].sort_values("X_Value")

        if d.empty:
            raise ValueError(
                f"Missing PDP data for {target}/{feature}"
            )

        x = d["X_Value"].to_numpy(dtype=float)
        y = d["PDP_Mean"].to_numpy(dtype=float)

        xu, ix = np.unique(
            x,
            return_index=True
        )

        yu = y[ix]

        if len(xu) == 1:

            raw = np.full_like(
                positions,
                yu[0],
                dtype=float
            )

            actual_x = np.full_like(
                positions,
                xu[0],
                dtype=float
            )

            baseline = float(yu[0])

        else:

            actual_x = (
                xu[0]
                + (xu[-1] - xu[0])
                * positions / 100.0
            )

            raw = np.interp(
                actual_x,
                xu,
                yu
            )

            midpoint = (
                xu[0] + xu[-1]
            ) / 2.0

            baseline = float(
                np.interp(midpoint, xu, yu)
            )

        delta = raw - baseline

        matrix_rows.append(delta)
        raw_rows.append(raw)

        range_records.append({
            "Target": target,
            "Feature": feature,
            "P5_Grid_Value": float(xu[0]),
            "P95_Grid_Value": float(xu[-1]),
            "Baseline_At_Range_Midpoint": baseline,
            "PDP_Change_LowToHigh": float(
                raw[-1] - raw[0]
            ),
            "PDP_MaxMinusMin": float(
                raw.max() - raw.min()
            )
        })

        for p, xv, rv, dv in zip(
            positions,
            actual_x,
            raw,
            delta
        ):

            long_records.append({
                "Target": target,
                "Feature": feature,
                "Range_Position_pct": float(p),
                "X_Actual_Value": float(xv),
                "PDP_Mean_Original_Units": float(rv),
                "Delta_PDP_vs_Midpoint": float(dv)
            })

    matrix = np.vstack(matrix_rows)
    raw_matrix = np.vstack(raw_rows)

    return (
        matrix,
        raw_matrix,
        pd.DataFrame(long_records),
        pd.DataFrame(range_records)
    )


# =====================================================
# 12. Draw Individual PDP Response Heatmap
# =====================================================

def draw_one_heatmap(target, matrix):

    scale = float(
        np.abs(matrix).max()
    )

    if scale == 0:
        scale = 1.0

    norm = TwoSlopeNorm(
        vmin=-scale,
        vcenter=0,
        vmax=scale
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 6.6)
    )

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_yticks(
        np.arange(len(FEATURES))
    )

    ax.set_yticklabels(
        FEATURES,
        fontsize=10
    )

    tick_idx = [
        0,
        (HEATMAP_POINTS - 1) / 4,
        (HEATMAP_POINTS - 1) / 2,
        3 * (HEATMAP_POINTS - 1) / 4,
        HEATMAP_POINTS - 1
    ]

    ax.set_xticks(tick_idx)

    ax.set_xticklabels(
        ["0", "25", "50", "75", "100"]
    )

    ax.set_xlabel(
        "Position within each feature's "
        "P5-P95 value range (%)"
    )

    ax.set_ylabel("Input feature")

    ax.set_title(
        "ResNet-MLP five-fold PDP response: "
        + target,
        pad=12
    )

    ax.axvline(
        (HEATMAP_POINTS - 1) / 2,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        alpha=0.75
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(FEATURES),
            1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        axis="y",
        color="white",
        alpha=0.55,
        linewidth=0.5
    )

    ax.tick_params(
        which="minor",
        left=False
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.040,
        pad=0.025
    )

    cbar.set_label(
        "Delta PDP relative to range midpoint "
        "(target units)"
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):

        fig.savefig(
            FIG_DIR / f"PDP_Response_Heatmap_{target}.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# =====================================================
# 13. Draw Ten-Target Overview Heatmap
# =====================================================

def draw_overview(matrices):

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(22, 10),
        sharex=True,
        sharey=True
    )

    norm = TwoSlopeNorm(
        vmin=-1,
        vcenter=0,
        vmax=1
    )

    image = None

    for i, target in enumerate(TARGETS):

        ax = axes.flat[i]
        matrix = matrices[target]

        scale = max(
            float(np.abs(matrix).max()),
            1e-12
        )

        normalized_matrix = matrix / scale

        image = ax.imshow(
            normalized_matrix,
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
            interpolation="nearest"
        )

        ax.set_title(
            target,
            fontsize=12
        )

        ax.set_xticks([
            0,
            (HEATMAP_POINTS - 1) / 2,
            HEATMAP_POINTS - 1
        ])

        ax.set_xticklabels(
            ["0", "50", "100"],
            fontsize=8
        )

        ax.set_yticks(
            np.arange(len(FEATURES))
        )

        if i % 5 == 0:
            ax.set_yticklabels(
                FEATURES,
                fontsize=8
            )

        ax.axvline(
            (HEATMAP_POINTS - 1) / 2,
            color="0.5",
            linestyle="--",
            linewidth=0.6,
            alpha=0.7
        )

    fig.suptitle(
        "ResNet-MLP five-fold PDP response "
        "patterns (10 targets)",
        fontsize=16
    )

    fig.supxlabel(
        "Position within each feature's "
        "P5-P95 value range (%)"
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.90,
        top=0.92,
        bottom=0.10,
        wspace=0.12,
        hspace=0.20
    )

    cax = fig.add_axes([
        0.915,
        0.20,
        0.012,
        0.60
    ])

    cbar = fig.colorbar(
        image,
        cax=cax
    )

    cbar.set_label(
        "Within-target normalized Delta PDP (unitless)"
    )

    for ext in ("png", "pdf"):

        fig.savefig(
            FIG_DIR / f"PDP_Response_Heatmap_10Targets.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# =====================================================
# 14. Export PDP and Heatmap Excel Workbooks
# =====================================================

def save_all_excels(
    pdp_df,
    fold_df,
    data_sha,
    matrices,
    raw_matrices,
    response_long,
    ranges
):

    settings = pd.DataFrame([
        (
            "Original_data_file",
            EXCEL_NAME
        ),
        (
            "Data_SHA256",
            data_sha
        ),
        (
            "Samples",
            str(int(pdp_df["N_Total_HeldOut"].max()))
        ),
        (
            "PDP_method",
            "1D brute-force PDP on each fold's "
            "held-out test samples"
        ),
        (
            "Aggregation",
            "Five folds weighted by held-out sample counts"
        ),
        (
            "Fold_SD",
            "SD across fold-level PDP estimates; NOT 95% CI"
        ),
        (
            "Grid",
            f"P{GRID_PERCENTILE_LOW}-P{GRID_PERCENTILE_HIGH}"
        ),
        (
            "Heatmap_x_axis",
            "Linear position from P5 to P95 grid endpoints, "
            "not empirical percentile"
        ),
        (
            "Heatmap_response",
            "PDP(x) minus PDP at (P5+P95)/2 value"
        ),
        (
            "Heatmap_color_scale",
            "Symmetric around 0, independently scaled per output"
        ),
        (
            "Overview",
            "Each output divided by its own maximum "
            "absolute Delta PDP"
        ),
        (
            "Limitations",
            "Predictive, not causal; correlated/compositional "
            "inputs can make unrealistic combinations"
        )
    ], columns=["Setting", "Value"])

    with pd.ExcelWriter(
        PDP_EXCEL,
        engine="openpyxl"
    ) as writer:

        pdp_df.to_excel(
            writer,
            sheet_name="PDP_Origin_Long",
            index=False
        )

        fold_df.to_excel(
            writer,
            sheet_name="PDP_FiveFold_Detail",
            index=False
        )

        for target in TARGETS:

            sub = pdp_df[
                pdp_df["Target"] == target
            ]

            sheet_name = (
                "PDP_" + target
            )[:31]

            sub.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

        settings.to_excel(
            writer,
            sheet_name="Settings",
            index=False
        )

    positions = np.linspace(
        0,
        100,
        HEATMAP_POINTS
    )

    pos_cols = [
        f"Pos_{p:.1f}"
        for p in positions
    ]

    with pd.ExcelWriter(
        HEATMAP_EXCEL,
        engine="openpyxl"
    ) as writer:

        response_long.to_excel(
            writer,
            sheet_name="Heatmap_Long",
            index=False
        )

        ranges.to_excel(
            writer,
            sheet_name="Feature_Ranges",
            index=False
        )

        for target in TARGETS:

            delta_df = pd.DataFrame(
                matrices[target],
                columns=pos_cols
            )

            delta_df.insert(
                0,
                "Feature",
                FEATURES
            )

            delta_df.to_excel(
                writer,
                sheet_name=(target + "_Delta")[:31],
                index=False
            )

            raw_df = pd.DataFrame(
                raw_matrices[target],
                columns=pos_cols
            )

            raw_df.insert(
                0,
                "Feature",
                FEATURES
            )

            raw_df.to_excel(
                writer,
                sheet_name=(target + "_Raw")[:31],
                index=False
            )

        settings.to_excel(
            writer,
            sheet_name="Settings",
            index=False
        )


# =====================================================
# 15. Main Program
# =====================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 64)
    print(
        "ResNet-MLP Five-Fold PDP Recalculation "
        "and Response Heatmaps"
    )
    print("Computing device:", DEVICE)
    print("Loading original data and five trained models directly.")
    print("No previously saved PDP Excel file is required.")
    print("=" * 64, flush=True)

    # Step 1: Load original data
    X, y, data_sha = load_data()

    # Step 2: Recompute five-fold PDP
    print("\n--- Recomputing five-fold PDP ---")

    pdp_df, fold_df = recompute_five_fold_pdp(
        X,
        data_sha
    )

    # Step 3: Generate response heatmaps
    matrices = {}
    raw_matrices = {}
    all_long = []
    all_ranges = []

    print("\n--- Drawing PDP response heatmaps ---")

    for target in TARGETS:

        (
            matrix,
            raw_matrix,
            response_long,
            ranges
        ) = make_response_matrix(
            pdp_df,
            target
        )

        matrices[target] = matrix
        raw_matrices[target] = raw_matrix

        all_long.append(response_long)
        all_ranges.append(ranges)

        draw_one_heatmap(
            target,
            matrix
        )

        print(
            "Heatmap generated:",
            target,
            flush=True
        )

    # Step 4: Generate ten-target overview heatmap
    if DRAW_10_TARGET_OVERVIEW:

        print(
            "\n--- Drawing the ten-target overview heatmap ---"
        )

        draw_overview(matrices)

    # Step 5: Export Excel data
    print("\n--- Exporting Origin-ready data ---")

    response_long_df = pd.concat(
        all_long,
        ignore_index=True
    )

    ranges_df = pd.concat(
        all_ranges,
        ignore_index=True
    )

    save_all_excels(
        pdp_df,
        fold_df,
        data_sha,
        matrices,
        raw_matrices,
        response_long_df,
        ranges_df
    )

    # Step 6: Completion
    print("\n" + "=" * 64)
    print("All steps completed!")
    print("Complete five-fold PDP Excel:", PDP_EXCEL)
    print("Origin-ready heatmap Excel:", HEATMAP_EXCEL)
    print("Heatmap folder:", FIG_DIR)
    print("Each heatmap contains 13 input features.")
    print(
        "Blue indicates a response below the midpoint baseline; "
        "red indicates a response above the midpoint baseline."
    )
    print("=" * 64)


# =====================================================
# 16. Program Entry Point
# =====================================================

if __name__ == "__main__":
    main()
