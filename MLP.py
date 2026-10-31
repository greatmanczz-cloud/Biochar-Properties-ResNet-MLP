
# -*- coding: utf-8 -*-
"""
ResNet-MLP vs. MLP residual-connection ablation study.

Run all five folds for ResNet-MLP, followed by all five folds for MLP.
Set the random seed once per model, without resetting it between folds.
Use the original DataLoader shuffling behavior. The residual shortcut
is the only architectural difference between the models.

Input: ETII.xlsx, worksheet MICE_All_Exploratory
Inputs: 13 variables in Excel columns G-I and K-T
Targets: 10 variables in Excel columns U-AD
Output folder: ResNet_MLP_Ablation_Fixed

Outputs include an ablation workbook, OOF and per-fold metrics,
per-sample predictions, R2 comparison plots, delta-R2 plots,
OOF scatter plots, and a comparison with previous ResNet results.

The ETII-completed dataset may contain imputed target labels.
This experiment assesses the contribution of residual connections;
it does not replace independent validation on measured target labels.
"""

from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =====================================================
# 1. Configuration
# =====================================================
SEED = 42
K_FOLDS = 5
EPOCHS = 2600
LEARNING_RATE = 5e-4
BATCH_SIZE = 48
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 128
NUM_BLOCKS = 3
DROPOUT_P = 0.4

# "random": random five-fold CV; "reference": grouped five-fold CV.
CV_TYPE = "random"
DPI = 300

# =====================================================
# 2. File Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_NAME = "ETII.xlsx"
SHEET_NAME = "MICE_All_Exploratory"
INPUT_FILE = BASE_DIR / EXCEL_NAME
OUTPUT_DIR = BASE_DIR / "ResNet_MLP_Ablation_Fixed"

# Compare R2 scores with previous ResNet results if available.
OLD_RESNET_EXCEL = (
    BASE_DIR / "ResNet_ETII_Results" / "ResNet_ETII_OOF_Results.xlsx"
)

# =====================================================
# 3. Excel Columns and Variable Names
# =====================================================
# Zero-based Python column indices.
INPUT_INDEX = [6, 7, 8] + list(range(10, 20))
OUTPUT_INDEX = list(range(20, 30))

FEATURES = [
    "FC_dry", "VM_dry", "Ash_dry", "Feedstock_C", "Feedstock_H",
    "Feedstock_O", "Feedstock_N", "Cellulose", "Hemicellulose",
    "Lignin", "Residence_time", "Temperature", "Heating_rate"
]
TARGETS = [
    "Biochar_yield", "TPV", "Biochar_ash", "Zeta_potential", "pH",
    "Biochar_C", "Biochar_H", "Biochar_O", "O_C_ratio", "H_C_ratio"
]
MODELS = ("ResNet-MLP", "MLP")
assert len(FEATURES) == 13
assert len(TARGETS) == 10

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =====================================================
# 4. Reproducibility
# =====================================================
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================
# 5. Load ETII Data
# =====================================================
def load_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place the Excel workbook in the same folder as this script."
        )
    raw = pd.read_excel(
        INPUT_FILE, sheet_name=SHEET_NAME, header=None, engine="openpyxl"
    )
    if raw.shape[1] < 30:
        raise ValueError("The Excel workbook must contain at least 30 columns.")

    # The first four rows contain headers. Keep valid sample IDs only.
    df = raw.iloc[4:, :].copy()
    sample_number = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df = df.loc[sample_number.notna()].reset_index(drop=True)
    if len(df) < K_FOLDS * 3:
        raise ValueError("Insufficient samples for five-fold validation.")

    X_df = df.iloc[:, INPUT_INDEX].apply(pd.to_numeric, errors="coerce")
    y_df = df.iloc[:, OUTPUT_INDEX].apply(pd.to_numeric, errors="coerce")
    x_valid = np.isfinite(X_df.to_numpy(dtype=float)).all()
    y_valid = np.isfinite(y_df.to_numpy(dtype=float)).all()
    if not (x_valid and y_valid):
        raise ValueError(
            "NaN/Inf found among the 13 inputs or 10 targets.\n"
            "NaN values in columns C-F do not affect this model."
        )

    X = X_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)
    sample_ids = df.iloc[:, 0].to_numpy()
    if df.shape[1] > 30:
        references = (
            df.iloc[:, 30].fillna("Unknown").astype(str)
            .str.strip().to_numpy()
        )
    else:
        references = np.array(["Unknown"] * len(df))

    checksum = hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest()
    print("\n========== Data Loading ==========")
    print("Input file:", INPUT_FILE)
    print("Data SHA256:", checksum)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print(
        "Note: ETII-completed targets include imputed labels; "
        "OOF results are not independent measured-label validation."
    )
    return X, y, sample_ids, references, checksum

# =====================================================
# 6. Residual and Non-Residual Network Blocks
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_p, use_residual):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        # The shortcut is the only structural difference between models.
        if self.use_residual:
            out = out + residual
        out = torch.relu(out)
        return out


class HybridResNetMLP(nn.Module):
    def __init__(
        self, input_size=13, output_size=10, hidden_size=128,
        num_blocks=3, dropout_p=0.4, use_residual=True
    ):
        super().__init__()
        self.entry_layer = nn.Linear(input_size, hidden_size)
        self.bn_entry = nn.BatchNorm1d(hidden_size)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_p, use_residual)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.entry_layer(x)
        x = self.bn_entry(x)
        x = torch.relu(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x

# =====================================================
# 7. Five-Fold Split and Metrics
# =====================================================
def make_folds(X, references):
    if CV_TYPE == "random":
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        return list(kf.split(X))
    elif CV_TYPE == "reference":
        if len(set(references)) < K_FOLDS:
            raise ValueError("Fewer than five independent reference groups.")
        gkf = GroupKFold(n_splits=K_FOLDS)
        return list(gkf.split(X, groups=references))
    else:
        raise ValueError("CV_TYPE must be either random or reference.")


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)}

# =====================================================
# 8. Complete Five-Fold Training for One Model
# =====================================================
def train_full_fivefold(model_name, X, y, folds, device):
    """Run all five folds, setting the random seed once per model."""
    seed_all(SEED)
    use_residual = model_name == "ResNet-MLP"
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    fold_rows = []
    history = []
    fold_labels = np.zeros(len(X), dtype=int)
    param_count = None

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        print(f"\n[{model_name}] Fold {fold}/{K_FOLDS}")
        print("Training samples:", len(train_idx))
        print("Test samples:", len(test_idx))

        # Fit scalers only on the training data of this fold.
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_test = x_scaler.transform(X[test_idx]).astype(np.float32)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y[train_idx]).astype(np.float32)

        # Do not reset the random seed between folds.
        model = HybridResNetMLP(
            input_size=13, output_size=10, hidden_size=HIDDEN_SIZE,
            num_blocks=NUM_BLOCKS, dropout_p=DROPOUT_P,
            use_residual=use_residual
        ).to(device)
        this_count = sum(p.numel() for p in model.parameters())
        if param_count is None:
            param_count = this_count

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.MSELoss()
        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        # Prevent BatchNorm from receiving a single-sample batch.
        drop_last = len(train_dataset) % BATCH_SIZE == 1
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, drop_last=drop_last
        )
        if len(train_loader) == 0:
            raise RuntimeError("DataLoader is empty.")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch == 1 or epoch % 200 == 0 or epoch == EPOCHS:
                mean_loss = epoch_loss / len(train_loader)
                print(
                    f"{model_name} | Fold {fold} | "
                    f"Epoch {epoch}/{EPOCHS} | Loss={mean_loss:.6f}",
                    flush=True
                )
                history.append({
                    "Model": model_name, "Fold": fold,
                    "Epoch": epoch, "Train_MSE_scaled": mean_loss
                })

        # Evaluate this fold and inverse-transform target predictions.
        model.eval()
        with torch.no_grad():
            test_tensor = torch.from_numpy(X_test).to(device)
            pred_scaled = model(test_tensor).cpu().numpy()
        pred_original = y_scaler.inverse_transform(pred_scaled)
        oof[test_idx] = pred_original
        fold_labels[test_idx] = fold

        fold_r2s = []
        for j, target in enumerate(TARGETS):
            score = metrics(y[test_idx, j], pred_original[:, j])
            fold_r2s.append(score["R2"])
            fold_rows.append({
                "Model": model_name, "Fold": fold, "Target": target,
                "N_test": len(test_idx), **score
            })
        print(
            f"{model_name} Fold {fold} Mean R2={np.mean(fold_r2s):.4f}",
            flush=True
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name} has incomplete OOF predictions.")
    return (
        oof, fold_labels, pd.DataFrame(fold_rows),
        pd.DataFrame(history), param_count
    )

# =====================================================
# 9. Draw R2 and Delta-R2 Comparison Charts
# =====================================================
def draw_bar_charts(comparison):
    idx = np.arange(len(TARGETS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(
        idx - width / 2, comparison["ResNet_R2"], width,
        color="#4C78A8", label="ResNet-MLP"
    )
    ax.bar(
        idx + width / 2, comparison["MLP_R2"], width,
        color="#D97757", label="MLP (without shortcut)"
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(TARGETS, rotation=38, ha="right")
    ax.set_ylabel("5-fold OOF R²")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"Ablation_R2_Comparison.{ext}",
            dpi=DPI, bbox_inches="tight"
        )
    plt.close(fig)

    # Delta R2 = ResNet-MLP R2 minus MLP R2.
    fig, ax = plt.subplots(figsize=(12, 5))
    delta = comparison["Delta_R2_ResNet_minus_MLP"].to_numpy()
    colors = np.where(delta >= 0, "#4C78A8", "#D97757")
    ax.bar(idx, delta, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(idx)
    ax.set_xticklabels(TARGETS, rotation=38, ha="right")
    ax.set_ylabel("Delta R² = ResNet-MLP - MLP")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"Ablation_Delta_R2.{ext}",
            dpi=DPI, bbox_inches="tight"
        )
    plt.close(fig)

# =====================================================
# 10. Draw OOF Scatter Plots
# =====================================================
def draw_oof_scatter(y, predictions, model_name, result_rows):
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    axes = axes.ravel()
    for j, target in enumerate(TARGETS):
        ax = axes[j]
        true_values = y[:, j]
        pred_values = predictions[:, j]
        color = "#4C78A8" if model_name == "ResNet-MLP" else "#D97757"
        ax.scatter(
            true_values, pred_values, s=15, alpha=0.55,
            color=color, edgecolors="none"
        )
        low = float(min(true_values.min(), pred_values.min()))
        high = float(max(true_values.max(), pred_values.max()))
        pad = max(0.04 * (high - low), 1e-6)
        limits = [low - pad, high + pad]
        ax.plot(limits, limits, "--", color="black", linewidth=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("ETII value")
        ax.set_ylabel("OOF prediction")
        ax.set_title(f"{target} R²={result_rows[j]['R2']:.3f}")
        ax.grid(alpha=0.15)
    for ax in axes[len(TARGETS):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{model_name}: 5-fold OOF on ETII-completed targets", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    safe_name = "ResNet" if model_name == "ResNet-MLP" else "MLP"
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"OOF_Scatter_{safe_name}.{ext}",
            dpi=DPI, bbox_inches="tight"
        )
    plt.close(fig)

# =====================================================
# 11. Compare With Previous ResNet Results
# =====================================================
def compare_to_old_resnet(new_resnet_df):
    if not OLD_RESNET_EXCEL.is_file():
        return pd.DataFrame([{
            "Status": "Previous ResNet results workbook not found",
            "Note": str(OLD_RESNET_EXCEL)
        }])
    try:
        old = pd.read_excel(OLD_RESNET_EXCEL, sheet_name="OOF_Metrics")
        if not {"Target", "OOF_R2"}.issubset(old.columns):
            raise ValueError("Previous workbook lacks Target or OOF_R2.")
        joined = old[["Target", "OOF_R2"]].merge(
            new_resnet_df[["Target", "R2"]], on="Target", how="inner"
        )
        joined = joined.rename(columns={
            "OOF_R2": "Old_ResNet_R2", "R2": "New_ResNet_R2"
        })
        joined["Difference_New_minus_Old"] = (
            joined["New_ResNet_R2"] - joined["Old_ResNet_R2"]
        )
        return joined
    except Exception as exc:
        return pd.DataFrame([{
            "Status": "Unable to read previous workbook", "Note": str(exc)
        }])

# =====================================================
# 12. Main Program
# =====================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    print("=" * 65)
    print("ResNet-MLP vs MLP Residual-Connection Ablation Study")
    print("=" * 65)

    X, y, sample_ids, references, checksum = load_data()
    folds = make_folds(X, references)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nComputing device:", device)
    if device.type == "cuda":
        print("GPU model:", torch.cuda.get_device_name(0))
    print(
        f"\nTraining settings: Epoch={EPOCHS}, Hidden={HIDDEN_SIZE}, "
        f"Blocks={NUM_BLOCKS}, Batch={BATCH_SIZE}, Seed={SEED}"
    )

    oof_results = {}
    metric_rows = []
    fold_dfs = []
    history_dfs = []
    counts = {}
    fold_labels = None

    # Complete all ResNet-MLP folds before starting MLP.
    for model_name in MODELS:
        print("\n" + "=" * 65)
        print("Starting training:", model_name)
        print("=" * 65)
        oof, labels, fold_df, history_df, n_parameters = train_full_fivefold(
            model_name, X, y, folds, device
        )
        oof_results[model_name] = oof
        counts[model_name] = n_parameters
        fold_labels = labels
        fold_dfs.append(fold_df)
        history_dfs.append(history_df)

        model_metrics = []
        for j, target in enumerate(TARGETS):
            score = metrics(y[:, j], oof[:, j])
            model_metrics.append(score)
            metric_rows.append({
                "Model": model_name, "Target": target, "N": len(y), **score
            })
        draw_oof_scatter(y, oof, model_name, model_metrics)

    # Both architectures must have identical parameter counts.
    if counts["ResNet-MLP"] != counts["MLP"]:
        raise AssertionError("The two models have different parameter counts!")

    oof_df = pd.DataFrame(metric_rows)
    fold_df = pd.concat(fold_dfs, ignore_index=True)
    training_df = pd.concat(history_dfs, ignore_index=True)
    fold_means = (
        fold_df.groupby(["Model", "Fold"], as_index=False)["R2"].mean()
        .rename(columns={"R2": "Mean_R2_10_targets"})
    )

    resnet_df = oof_df[oof_df["Model"] == "ResNet-MLP"].copy()
    mlp_df = oof_df[oof_df["Model"] == "MLP"].copy()
    comparison = pd.DataFrame({
        "Target": TARGETS,
        "ResNet_R2": resnet_df["R2"].to_numpy(),
        "MLP_R2": mlp_df["R2"].to_numpy(),
        "ResNet_RMSE": resnet_df["RMSE"].to_numpy(),
        "MLP_RMSE": mlp_df["RMSE"].to_numpy(),
        "ResNet_MAE": resnet_df["MAE"].to_numpy(),
        "MLP_MAE": mlp_df["MAE"].to_numpy()
    })
    comparison["Delta_R2_ResNet_minus_MLP"] = (
        comparison["ResNet_R2"] - comparison["MLP_R2"]
    )
    draw_bar_charts(comparison)

    summary = pd.DataFrame([{
        "Model": name,
        "Parameters": counts[name],
        "Mean_OOF_R2_10_targets": float(
            oof_df.loc[oof_df["Model"] == name, "R2"].mean()
        ),
        "Mean_OOF_RMSE_unscaled": float(
            oof_df.loc[oof_df["Model"] == name, "RMSE"].mean()
        ),
        "Epochs": EPOCHS,
        "Dataset": INPUT_FILE.name,
        "CV_type": CV_TYPE
    } for name in MODELS])

    detail = pd.DataFrame({
        "Index_No": sample_ids, "Reference": references, "Fold": fold_labels
    })
    for j, target in enumerate(TARGETS):
        detail[f"{target}_ETII_Value"] = y[:, j]
        detail[f"{target}_ResNet_OOF"] = oof_results["ResNet-MLP"][:, j]
        detail[f"{target}_MLP_OOF"] = oof_results["MLP"][:, j]

    old_comparison = compare_to_old_resnet(resnet_df)
    settings = pd.DataFrame([
        ("SEED", SEED),
        ("K_FOLDS", K_FOLDS),
        ("EPOCHS", EPOCHS),
        ("LEARNING_RATE", LEARNING_RATE),
        ("BATCH_SIZE", BATCH_SIZE),
        ("WEIGHT_DECAY", WEIGHT_DECAY),
        ("HIDDEN_SIZE", HIDDEN_SIZE),
        ("NUM_BLOCKS", NUM_BLOCKS),
        ("DROPOUT_P", DROPOUT_P),
        ("CV_TYPE", CV_TYPE),
        ("Input_Excel", INPUT_FILE.name),
        ("Input_Excel_SHA256", checksum),
        ("Sheet", SHEET_NAME),
        ("Randomness", "Reset seed once before each complete five-fold run"),
        ("Labels", "ETII imputed targets; not independent measured-label validation")
    ], columns=["Parameter", "Value"])

    output_path = OUTPUT_DIR / "Ablation_ResNet_vs_MLP_Fixed.xlsx"
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            comparison.to_excel(writer, sheet_name="R2_Comparison", index=False)
            oof_df.to_excel(writer, sheet_name="OOF_PerTarget", index=False)
            fold_df.to_excel(writer, sheet_name="Fold_PerTarget", index=False)
            fold_means.to_excel(writer, sheet_name="Fold_Mean_R2", index=False)
            detail.to_excel(writer, sheet_name="OOF_Predictions", index=False)
            training_df.to_excel(writer, sheet_name="Training_Log", index=False)
            old_comparison.to_excel(writer, sheet_name="Vs_Old_ResNet", index=False)
            settings.to_excel(writer, sheet_name="Settings", index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to: {output_path}\n"
            "Close the workbook in Excel before running the script."
        ) from exc

    print("\n========== Residual Ablation Results ==========")
    print(comparison[[
        "Target", "ResNet_R2", "MLP_R2", "Delta_R2_ResNet_minus_MLP"
    ]].round(4).to_string(index=False))
    print("\nMean OOF R2 Across 10 Targets:")
    print(summary[[
        "Model", "Parameters", "Mean_OOF_R2_10_targets"
    ]].round(4).to_string(index=False))
    if "Difference_New_minus_Old" in old_comparison.columns:
        print("\nR2 Differences Between Revised and Previous ResNet Results:")
        print(old_comparison.round(6).to_string(index=False))
    print("\nResults saved to:", output_path)
    print(
        "\nNote: Changing the ETII file, PyTorch version, "
        "GPU driver, or computing device may still change the results."
    )


# =====================================================
# 13. Program Entry Point
# =====================================================
if __name__ == "__main__":
    main()
