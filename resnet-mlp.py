
# -*- coding: utf-8 -*-
# ResNet-MLP five-fold OOF evaluation with optional SHAP analysis.
# Translated from the supplied script without changing model logic.

from pathlib import Path
import random
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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

# Cross-validation strategy: "random" or "reference"
CV_TYPE = 'random'

# SHAP configuration
RUN_SHAP = True
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SIZE = 80
DPI = 300

# =====================================================
# 2. Input and Output Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
EXCEL_NAME = 'ETII.xlsx'
SHEET_NAME = 'MICE_All_Exploratory'
INPUT_FILE = BASE_DIR / EXCEL_NAME
OUTPUT_DIR = BASE_DIR / 'ResNet_ETII_Results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 3. Input and Target Variables
# =====================================================

INPUT_INDEX = [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
OUTPUT_INDEX = list(range(20, 30))

feature_cols = [
    'FC_dry', 'VM_dry', 'Ash_dry',
    'Feedstock_C', 'Feedstock_H', 'Feedstock_O',
    'Feedstock_N', 'Cellulose', 'Hemicellulose',
    'Lignin', 'Residence_time', 'Temperature',
    'Heating_rate'
]

target_cols = [
    'Biochar_yield', 'TPV', 'Biochar_ash',
    'Zeta_potential', 'pH', 'Biochar_C',
    'Biochar_H', 'Biochar_O', 'O_C_ratio',
    'H_C_ratio'
]

assert len(feature_cols) == 13
assert len(target_cols) == 10

# =====================================================
# 4. Random Seeds and GPU Configuration
# =====================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print('=' * 65)
print('ResNet-MLP: ETII dataset')
print('=' * 65)
print('Computing device:', device)

if torch.cuda.is_available():
    print('GPU model:', torch.cuda.get_device_name(0))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# =====================================================
# 5. Load Excel Data
# =====================================================

def load_data():

    print('\n--- 1. Loading Excel data ---')

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f'File not found: {INPUT_FILE}\n'
            'Place the Excel workbook in the same folder as the Python script.'
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine='openpyxl'
    )

    if raw.shape[1] < 30:
        raise ValueError(
            'The Excel workbook has fewer than 30 columns; verify its structure.'
        )

    # Skip the first four header rows.
    df = raw.iloc[4:, :].copy()

    sample_number = pd.to_numeric(
        df.iloc[:, 0],
        errors='coerce'
    )

    df = df.loc[
        sample_number.notna()
    ].reset_index(drop=True)

    X_df = df.iloc[:, INPUT_INDEX].copy()
    X_df.columns = feature_cols

    y_df = df.iloc[:, OUTPUT_INDEX].copy()
    y_df.columns = target_cols

    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    y_df = y_df.apply(pd.to_numeric, errors='coerce')

    x_missing = X_df.isna().sum().sum()
    y_missing = y_df.isna().sum().sum()

    print('Number of samples:', len(df))
    print('Number of input variables:', X_df.shape[1])
    print('Number of target variables:', y_df.shape[1])
    print('Missing input values:', int(x_missing))
    print('Missing target values:', int(y_missing))

    if x_missing > 0 or y_missing > 0:
        raise ValueError(
            'The selected dataset has not been fully imputed. '
            'Check the MICE_All_Exploratory worksheet.'
        )

    X = X_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise ValueError(
            'The data contain NaN or infinite values.'
        )

    sample_ids = df.iloc[:, 0].tolist()

    if df.shape[1] > 30:
        references = (
            df.iloc[:, 30]
            .fillna('Unknown')
            .astype(str)
            .str.strip()
            .to_numpy()
        )
    else:
        references = np.array(['Unknown'] * len(df))

    print('\nX.shape =', X.shape)
    print('y.shape =', y.shape)
    print('\n13 input variables:')
    print(feature_cols)
    print('\n10 target variables:')
    print(target_cols)

    return X, y, sample_ids, references


# =====================================================
# 6. Residual Block
# =====================================================

class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, dropout_p):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):

        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)

        # Avoid in-place operations to support SHAP gradients.
        out = out + residual
        out = torch.relu(out)

        return out


# =====================================================
# 7. ResNet-MLP Architecture
# =====================================================

class HybridResNetMLP(nn.Module):

    def __init__(
        self, input_size, output_size,
        hidden_size, num_blocks, dropout_p
    ):
        super().__init__()

        self.entry_layer = nn.Linear(input_size, hidden_size)
        self.bn_entry = nn.BatchNorm1d(hidden_size)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_p)
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
# 8. Restore Original Output Units for SHAP
# =====================================================

class OriginalUnitModel(nn.Module):

    def __init__(self, model, y_scaler):
        super().__init__()

        self.model = model

        self.register_buffer(
            'y_mean',
            torch.tensor(y_scaler.mean_, dtype=torch.float32)
        )

        self.register_buffer(
            'y_scale',
            torch.tensor(y_scaler.scale_, dtype=torch.float32)
        )

    def forward(self, x):

        output = self.model(x)

        return output * self.y_scale + self.y_mean


# =====================================================
# 9. Cross-Validation Splits
# =====================================================

def get_folds(X, references):

    if CV_TYPE == 'random':

        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=SEED
        )
        folds = list(kf.split(X))

    elif CV_TYPE == 'reference':

        if len(set(references)) < K_FOLDS:
            raise ValueError(
                'Insufficient independent references '
                'for five-fold grouped validation.'
            )

        gkf = GroupKFold(n_splits=K_FOLDS)
        folds = list(gkf.split(X, groups=references))

    else:
        raise ValueError(
            'CV_TYPE must be either random or reference.'
        )

    return folds


# =====================================================
# 10. Five-Fold Training and OOF Evaluation
# =====================================================

def train_cross_validation(X, y, sample_ids, references):

    print('\n--- 2. Starting five-fold cross-validation ---')

    folds = get_folds(X, references)
    n_samples = len(X)
    n_targets = len(target_cols)

    oof_predictions = np.full(
        y.shape, np.nan, dtype=np.float32
    )

    fold_labels = np.zeros(n_samples, dtype=int)
    fold_results = []
    saved_models = []

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):

        print('\n' + '=' * 65)
        print(f'Training fold {fold}/{K_FOLDS}')
        print(f'Training samples: {len(train_idx)}')
        print(f'Test samples: {len(test_idx)}')
        print('=' * 65)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit scalers using training data only.
        x_scaler = StandardScaler()

        X_train_scaled = (
            x_scaler.fit_transform(X_train).astype(np.float32)
        )

        X_test_scaled = (
            x_scaler.transform(X_test).astype(np.float32)
        )

        y_scaler = StandardScaler()

        y_train_scaled = (
            y_scaler.fit_transform(y_train).astype(np.float32)
        )

        # Create the training DataLoader.
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train_scaled)
        )

        # Avoid single-sample batches in BatchNorm.
        drop_last = len(train_dataset) % BATCH_SIZE == 1

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=drop_last
        )

        model = HybridResNetMLP(
            input_size=13,
            output_size=10,
            hidden_size=HIDDEN_SIZE,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P
        ).to(device)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Train the current fold.
        for epoch in range(1, EPOCHS + 1):

            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (
                epoch == 1
                or epoch % 200 == 0
                or epoch == EPOCHS
            ):

                mean_loss = (
                    epoch_loss / max(len(train_loader), 1)
                )

                print(
                    f'Fold {fold} | '
                    f'Epoch {epoch}/{EPOCHS} | '
                    f'Loss = {mean_loss:.6f}'
                )

        # Generate held-out predictions.
        model.eval()

        with torch.no_grad():
            test_tensor = torch.from_numpy(
                X_test_scaled
            ).to(device)

            pred_scaled = model(test_tensor)
            pred_scaled = pred_scaled.cpu().numpy()

        y_pred = y_scaler.inverse_transform(pred_scaled)

        oof_predictions[test_idx] = y_pred
        fold_labels[test_idx] = fold

        current_r2 = []

        for j, name in enumerate(target_cols):

            true_values = y_test[:, j]
            pred_values = y_pred[:, j]

            r2 = r2_score(true_values, pred_values)

            rmse = np.sqrt(
                mean_squared_error(true_values, pred_values)
            )

            mae = mean_absolute_error(
                true_values, pred_values
            )

            current_r2.append(r2)

            fold_results.append({
                'Fold': fold,
                'Target': name,
                'N_test': len(test_idx),
                'R2': float(r2),
                'RMSE': float(rmse),
                'MAE': float(mae)
            })

        mean_r2 = float(np.mean(current_r2))

        print(f'\nFold {fold} mean R2 = {mean_r2:.4f}')

        # Store fold parameters for subsequent SHAP analysis.
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        saved_models.append({
            'fold': fold,
            'state': model_state,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'mean_r2': mean_r2
        })

        del model

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Calculate overall OOF performance.
    print('\n--- 3. Calculating final OOF performance ---')

    if not np.isfinite(oof_predictions).all():
        raise RuntimeError(
            'Some OOF predictions have not been generated.'
        )

    final_results = []

    for j, name in enumerate(target_cols):

        true_values = y[:, j]
        pred_values = oof_predictions[:, j]

        r2 = r2_score(true_values, pred_values)

        rmse = np.sqrt(
            mean_squared_error(true_values, pred_values)
        )

        mae = mean_absolute_error(
            true_values, pred_values
        )

        final_results.append({
            'Target': name,
            'N': len(true_values),
            'OOF_R2': float(r2),
            'OOF_RMSE': float(rmse),
            'OOF_MAE': float(mae)
        })

    final_df = pd.DataFrame(final_results)
    fold_df = pd.DataFrame(fold_results)

    print('\nFinal OOF results:')
    print(final_df.round(4).to_string(index=False))

    print(
        '\nMean OOF R2 across 10 targets:',
        round(final_df['OOF_R2'].mean(), 4)
    )

    # Construct the per-sample OOF prediction table.
    result_data = pd.DataFrame({
        'Index_No': sample_ids,
        'Reference': references,
        'Fold': fold_labels
    })

    for j, name in enumerate(target_cols):

        result_data[name + '_ETII_Value'] = y[:, j]
        result_data[name + '_OOF_Pred'] = oof_predictions[:, j]
        result_data[name + '_Residual'] = (
            y[:, j] - oof_predictions[:, j]
        )

    excel_path = OUTPUT_DIR / 'ResNet_ETII_OOF_Results.xlsx'

    mapping = pd.DataFrame({
        'Variable': feature_cols + target_cols,
        'Role': ['Input'] * 13 + ['Output'] * 10,
        'Excel_Column': (
            ['G', 'H', 'I']
            + list('KLMNOPQRST')
            + [
                'U', 'V', 'W', 'X', 'Y', 'Z',
                'AA', 'AB', 'AC', 'AD'
            ]
        )
    })

    # Export OOF metrics and predictions.
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        final_df.to_excel(
            writer,
            sheet_name='OOF_Metrics',
            index=False
        )

        fold_df.to_excel(
            writer,
            sheet_name='Fold_Metrics',
            index=False
        )

        result_data.to_excel(
            writer,
            sheet_name='OOF_Predictions',
            index=False
        )

        mapping.to_excel(
            writer,
            sheet_name='Variable_Mapping',
            index=False
        )

    print('\nExcel workbook saved:', excel_path)

    draw_oof_scatter(y, oof_predictions, final_df)

    return saved_models, final_df


# =====================================================
# 11. OOF Scatter Plots
# =====================================================

def draw_oof_scatter(y, predictions, final_df):

    print('\n--- 4. Generating OOF scatter plots ---')

    fig, axes = plt.subplots(4, 3, figsize=(16, 19))
    axes = axes.flatten()

    for j, name in enumerate(target_cols):

        ax = axes[j]
        y_true = y[:, j]
        y_pred = predictions[:, j]

        ax.scatter(
            y_true,
            y_pred,
            s=20,
            alpha=0.6,
            color='#4C78A8',
            edgecolors='none'
        )

        minimum = min(
            float(np.min(y_true)),
            float(np.min(y_pred))
        )

        maximum = max(
            float(np.max(y_true)),
            float(np.max(y_pred))
        )

        margin = max(
            0.04 * (maximum - minimum),
            1e-6
        )

        limits = [
            minimum - margin,
            maximum + margin
        ]

        ax.plot(
            limits,
            limits,
            '--',
            color='#D97757',
            linewidth=1.5
        )

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel('ETII value', fontsize=10)
        ax.set_ylabel('OOF prediction', fontsize=10)

        r2 = final_df.iloc[j]['OOF_R2']

        ax.set_title(
            f'{name}\nR² = {r2:.4f}',
            fontsize=12
        )

        ax.grid(
            alpha=0.2,
            linestyle='--'
        )

    for k in range(len(target_cols), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        'ResNet-MLP: 5-fold OOF predictions',
        fontsize=17,
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    png_path = OUTPUT_DIR / 'ResNet_ETII_OOF_Scatter.png'
    pdf_path = OUTPUT_DIR / 'ResNet_ETII_OOF_Scatter.pdf'

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        pdf_path,
        bbox_inches='tight'
    )

    plt.close(fig)

    print('OOF scatter plots saved')


# =====================================================
# 12. Standardize SHAP Output Shapes
# =====================================================

def standardize_shap_values(
    shap_values,
    n_samples,
    n_features,
    n_outputs
):

    # Older versions may return a list of output arrays.
    if isinstance(shap_values, list):

        if len(shap_values) != n_outputs:
            raise ValueError(
                'Unexpected number of SHAP outputs.'
            )

        values = np.stack(
            [np.asarray(v) for v in shap_values],
            axis=2
        )

    else:

        values = np.asarray(shap_values)

        if values.shape == (
            n_outputs,
            n_samples,
            n_features
        ):
            values = np.moveaxis(values, 0, 2)

    expected_shape = (
        n_samples,
        n_features,
        n_outputs
    )

    if values.shape != expected_shape:
        raise ValueError(
            f'Invalid SHAP array shape: {values.shape}\n'
            f'Expected: {expected_shape}'
        )

    return values


# =====================================================
# 13. SHAP Analysis
# =====================================================

def run_shap_analysis(X, saved_models):

    if not RUN_SHAP:
        print('SHAP analysis is disabled.')
        return

    print('\n--- 5. Starting SHAP analysis ---')

    try:
        import shap

    except ImportError:
        print('SHAP is not installed. Run: pip install shap')
        return

    # Select the fold with the highest mean R2 for interpretation.
    best_model_info = max(
        saved_models,
        key=lambda item: item['mean_r2']
    )

    best_fold = best_model_info['fold']

    print(f'SHAP uses the model from fold {best_fold}')

    model = HybridResNetMLP(
        input_size=13,
        output_size=10,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        dropout_p=DROPOUT_P
    ).to(device)

    model.load_state_dict(best_model_info['state'])
    model.eval()

    x_scaler = best_model_info['x_scaler']
    y_scaler = best_model_info['y_scaler']

    shap_model = OriginalUnitModel(
        model, y_scaler
    ).to(device)

    shap_model.eval()

    train_idx = best_model_info['train_idx']
    test_idx = best_model_info['test_idx']

    rng = np.random.default_rng(SEED)

    # Select background and explanation samples.
    bg_idx = rng.choice(
        train_idx,
        size=min(SHAP_BACKGROUND_SIZE, len(train_idx)),
        replace=False
    )

    eval_idx = rng.choice(
        test_idx,
        size=min(SHAP_EXPLAIN_SIZE, len(test_idx)),
        replace=False
    )

    X_background = (
        x_scaler.transform(X[bg_idx]).astype(np.float32)
    )

    X_explain = (
        x_scaler.transform(X[eval_idx]).astype(np.float32)
    )

    background_tensor = torch.from_numpy(
        X_background
    ).to(device)

    explain_tensor = torch.from_numpy(
        X_explain
    ).to(device)

    print(
        'Number of SHAP background samples:',
        len(bg_idx)
    )

    print(
        'Number of SHAP explanation samples:',
        len(eval_idx)
    )

    # Attempt DeepExplainer first.
    try:

        explainer = shap.DeepExplainer(
            shap_model,
            background_tensor
        )

        raw_shap = explainer.shap_values(
            explain_tensor,
            check_additivity=False
        )

        method_name = 'DeepExplainer'

    except Exception as exc:

        warnings.warn(
            f'DeepExplainer failed: {exc}; '
            'trying GradientExplainer'
        )

        try:

            explainer = shap.GradientExplainer(
                shap_model,
                background_tensor
            )

            raw_shap = explainer.shap_values(
                explain_tensor
            )

            method_name = 'GradientExplainer'

        except Exception as exc2:

            print('SHAP analysis failed:', exc2)
            print(
                'Five-fold cross-validation and OOF '
                'results were saved successfully.'
            )
            return

    shap_cube = standardize_shap_values(
        raw_shap,
        n_samples=len(eval_idx),
        n_features=13,
        n_outputs=10
    )

    print('SHAP method:', method_name)
    print('SHAP dimensions:', shap_cube.shape)

    SHAP_DIR = OUTPUT_DIR / 'SHAP_Plots'

    SHAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_display = pd.DataFrame(
        X[eval_idx],
        columns=feature_cols
    )

    shap_importance = []
    shap_data = {}

    # Generate SHAP plots for all ten targets.
    for j, target_name in enumerate(target_cols):

        print(f'Generating SHAP plots: {target_name}')

        shap_values_j = shap_cube[:, :, j]

        mean_abs_shap = np.mean(
            np.abs(shap_values_j),
            axis=0
        )

        for i, feature_name in enumerate(feature_cols):

            shap_importance.append({
                'Target': target_name,
                'Feature': feature_name,
                'MeanAbsSHAP': float(mean_abs_shap[i])
            })

        shap_data[target_name] = pd.DataFrame(
            shap_values_j,
            columns=feature_cols
        )

        # SHAP beeswarm plot
        plt.figure(figsize=(10, 7))

        shap.summary_plot(
            shap_values_j,
            X_display,
            plot_type='dot',
            show=False,
            max_display=13
        )

        plt.title(
            f'{target_name} - SHAP Beeswarm',
            fontsize=14
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_DIR / f'SHAP_Beeswarm_{target_name}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        plt.close('all')

        # SHAP feature-importance bar chart
        order = np.argsort(mean_abs_shap)

        sorted_names = np.array(feature_cols)[order]
        sorted_values = mean_abs_shap[order]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.barh(
            sorted_names,
            sorted_values,
            color='#4C78A8'
        )

        ax.set_xlabel('Mean absolute SHAP value')
        ax.set_title(f'{target_name} - SHAP Importance')

        fig.tight_layout()

        fig.savefig(
            SHAP_DIR / f'SHAP_Bar_{target_name}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        plt.close(fig)

    # Export SHAP values and feature importances.
    shap_excel_path = OUTPUT_DIR / 'ResNet_ETII_SHAP_Data.xlsx'

    importance_df = pd.DataFrame(shap_importance)

    with pd.ExcelWriter(
        shap_excel_path,
        engine='openpyxl'
    ) as writer:

        importance_df.to_excel(
            writer,
            sheet_name='SHAP_Importance',
            index=False
        )

        X_display.assign(
            RowIndex=eval_idx
        ).to_excel(
            writer,
            sheet_name='SHAP_InputValues',
            index=False
        )

        for target_name, values_df in shap_data.items():

            sheet_name = 'SHAP_' + target_name[:24]

            values_df.assign(
                RowIndex=eval_idx
            ).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print('\nSHAP workbook saved:', shap_excel_path)
    print('SHAP figures saved to:', SHAP_DIR)


# =====================================================
# 14. Main Workflow
# =====================================================

def main():

    print('\nStarting the complete ResNet-MLP workflow')

    X, y, sample_ids, references = load_data()

    saved_models, final_df = train_cross_validation(
        X, y, sample_ids, references
    )

    run_shap_analysis(X, saved_models)

    print('\n' + '=' * 65)
    print('All steps completed.')
    print('=' * 65)
    print('Output directory:', OUTPUT_DIR)


# =====================================================
# 15. Program Entry Point
# =====================================================

if __name__ == '__main__':
    main()
