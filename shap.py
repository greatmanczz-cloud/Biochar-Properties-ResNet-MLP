
"""
ResNet-MLP (13 inputs -> 10 outputs)
Five-fold out-of-fold (OOF) Gradient SHAP

Usage:
1. Place this .py script and ETII.xlsx in the same folder.
2. For the first run, set MODE='train_and_shap'.
3. For subsequent SHAP-only runs, set MODE='shap_only'.

Notes:
The ETII dataset contains imputed target labels.
SHAP explains model predictions and does not establish causality.
"""

from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Training and interpretation settings.
SEED = 42
EXCEL_NAME = 'ETII.xlsx'
SHEET_NAME = 'MICE_All_Exploratory'
MODE = 'train_and_shap'
CV_TYPE = 'random'
K_FOLDS = 5
EPOCHS = 2600
LEARNING_RATE = 0.0005
BATCH_SIZE = 48
WEIGHT_DECAY = 1e-05
HIDDEN_SIZE = 128
NUM_BLOCKS = 3
DROPOUT_P = 0.4
SHAP_BACKGROUND_SIZE = 80
SHAP_NSAMPLES = 64
SHAP_CHUNK_SIZE = 20
DPI = 300

# Data and output folders.
BASE = Path(__file__).resolve().parent
INPUT = BASE / EXCEL_NAME
OUT = BASE / 'ResNet_MLP_OOF_SHAP_Results'
CKPT_DIR = OUT / 'FiveFold_Checkpoints'
PLOT_DIR = OUT / 'SHAP_Plots'

for directory in (OUT, CKPT_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Excel column indices are zero-based.
INPUT_INDEX = [6, 7, 8] + list(range(10, 20))
OUTPUT_INDEX = list(range(20, 30))

FEATURES = [
    'FC_dry', 'VM_dry', 'Ash_dry',
    'Feedstock_C', 'Feedstock_H', 'Feedstock_O',
    'Feedstock_N', 'Cellulose', 'Hemicellulose',
    'Lignin', 'Residence_time', 'Temperature',
    'Heating_rate'
]

TARGETS = [
    'Biochar_yield', 'TPV', 'Biochar_ash',
    'Zeta_potential', 'pH', 'Biochar_C',
    'Biochar_H', 'Biochar_O', 'O_C_ratio',
    'H_C_ratio'
]

assert len(FEATURES) == 13
assert len(TARGETS) == 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Figure settings.
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# Read the dataset and validate inputs, targets, and file hash.
def load_data():
    if not INPUT.is_file():
        raise FileNotFoundError(
            f'Dataset file not found: {INPUT}\n'
            'Place the Excel file and the Python script in the same folder.'
        )

    raw = pd.read_excel(
        INPUT,
        sheet_name=SHEET_NAME,
        header=None,
        engine='openpyxl'
    )

    if raw.shape[1] < 30:
        raise ValueError(
            'The Excel workbook must contain at least 30 columns.'
        )

    body = raw.iloc[4:, :].copy()

    ids_as_numbers = pd.to_numeric(
        body.iloc[:, 0],
        errors='coerce'
    )

    body = body.loc[
        ids_as_numbers.notna()
    ].reset_index(drop=True)

    X_df = body.iloc[:, INPUT_INDEX].apply(
        pd.to_numeric,
        errors='coerce'
    )

    y_df = body.iloc[:, OUTPUT_INDEX].apply(
        pd.to_numeric,
        errors='coerce'
    )

    X = X_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise ValueError(
            'NaN or Inf values found among the 13 inputs or 10 targets.'
        )

    ids = body.iloc[:, 0].to_numpy()

    if body.shape[1] > 30:
        refs = (
            body.iloc[:, 30]
            .fillna('Unknown')
            .astype(str)
            .str.strip()
            .to_numpy()
        )
    else:
        refs = np.array(['Unknown'] * len(body))

    sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()

    print(f'Dataset file: {INPUT.name}')
    print(f'X.shape = {X.shape}')
    print(f'y.shape = {y.shape}')
    print(f'SHA256 = {sha[:16]}...')

    return X, y, ids, refs, sha


# Residual block used in the ResNet-MLP architecture.
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
        out = out + residual
        return torch.relu(out)


# Complete ResNet-MLP network.
class HybridResNetMLP(nn.Module):

    def __init__(
        self,
        input_size=13,
        output_size=10,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        dropout_p=DROPOUT_P
    ):
        super().__init__()

        self.entry_layer = nn.Linear(
            input_size, hidden_size
        )
        self.bn_entry = nn.BatchNorm1d(hidden_size)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_p)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(
            hidden_size, output_size
        )

    def forward(self, x):
        x = self.entry_layer(x)
        x = self.bn_entry(x)
        x = torch.relu(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


# Restore model predictions to original target units for SHAP.
class OriginalUnitModel(nn.Module):

    def __init__(self, model, y_mean, y_scale):
        super().__init__()
        self.model = model
        self.register_buffer(
            'y_mean', y_mean.float().clone()
        )
        self.register_buffer(
            'y_scale', y_scale.float().clone()
        )

    def forward(self, x):
        return self.model(x) * self.y_scale + self.y_mean


# Generate random or reference-grouped five-fold splits.
def get_folds(X, refs):

    if CV_TYPE == 'random':
        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=SEED
        )
        return list(kf.split(X))

    if CV_TYPE == 'reference':
        gkf = GroupKFold(n_splits=K_FOLDS)
        return list(gkf.split(X, groups=refs))

    raise ValueError(
        'CV_TYPE must be either random or reference.'
    )


# Return the checkpoint path for a particular fold.
def checkpoint_path(fold):
    return CKPT_DIR / f'ResNet_MLP_Fold{fold}.pt'


# Train five folds and save checkpoints and OOF metrics.
def train_5_folds(X, y, refs, data_sha):

    folds = get_folds(X, refs)
    oof = np.full(y.shape, np.nan, dtype=np.float32)
    cv_rows = []

    for fold, (train_idx, test_idx) in enumerate(folds, 1):

        print(
            f'\n===== Fold {fold}/{K_FOLDS} =====',
            flush=True
        )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        xs = StandardScaler()
        ys = StandardScaler()

        X_train_scaled = xs.fit_transform(
            X_train
        ).astype(np.float32)

        X_test_scaled = xs.transform(
            X_test
        ).astype(np.float32)

        y_train_scaled = ys.fit_transform(
            y_train
        ).astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train_scaled)
        )

        drop_last = len(dataset) % BATCH_SIZE == 1

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=drop_last
        )

        model = HybridResNetMLP().to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        criterion = nn.MSELoss()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for bx, by in loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                optimizer.zero_grad()
                output = model(bx)
                loss = criterion(output, by)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())

            if (
                epoch == 1
                or epoch % 200 == 0
                or epoch == EPOCHS
            ):
                mean_loss = (
                    epoch_loss / max(len(loader), 1)
                )

                print(
                    f'Fold={fold} '
                    f'Epoch={epoch}/{EPOCHS} '
                    f'Loss={mean_loss:.6f}',
                    flush=True
                )

        model.eval()

        with torch.no_grad():
            test_tensor = torch.from_numpy(
                X_test_scaled
            ).to(DEVICE)

            y_test_scaled = (
                model(test_tensor)
                .cpu()
                .numpy()
            )

        y_pred = ys.inverse_transform(y_test_scaled)
        oof[test_idx] = y_pred

        for j, target in enumerate(TARGETS):
            cv_rows.append({
                'Fold': fold,
                'Target': target,
                'R2': r2_score(
                    y_test[:, j], y_pred[:, j]
                ),
                'RMSE': np.sqrt(
                    mean_squared_error(
                        y_test[:, j], y_pred[:, j]
                    )
                ),
                'MAE': mean_absolute_error(
                    y_test[:, j], y_pred[:, j]
                )
            })

        checkpoint = {
            'fold': fold,
            'data_sha256': data_sha,
            'cv_type': CV_TYPE,
            'architecture': (
                13, 10, HIDDEN_SIZE, NUM_BLOCKS, DROPOUT_P
            ),
            'state_dict': {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            'x_mean': torch.from_numpy(
                xs.mean_.astype(np.float32)
            ),
            'x_scale': torch.from_numpy(
                xs.scale_.astype(np.float32)
            ),
            'y_mean': torch.from_numpy(
                ys.mean_.astype(np.float32)
            ),
            'y_scale': torch.from_numpy(
                ys.scale_.astype(np.float32)
            ),
            'train_idx': torch.from_numpy(
                np.asarray(train_idx, dtype=np.int64)
            ),
            'test_idx': torch.from_numpy(
                np.asarray(test_idx, dtype=np.int64)
            )
        }

        torch.save(
            checkpoint,
            checkpoint_path(fold)
        )

        print(
            f'Fold {fold} checkpoint saved',
            flush=True
        )

        del model

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    if not np.isfinite(oof).all():
        raise RuntimeError('Some OOF predictions are missing.')

    metrics = []

    for j, name in enumerate(TARGETS):
        metrics.append({
            'Target': name,
            'OOF_R2': r2_score(y[:, j], oof[:, j]),
            'OOF_RMSE': np.sqrt(
                mean_squared_error(y[:, j], oof[:, j])
            ),
            'OOF_MAE': mean_absolute_error(
                y[:, j], oof[:, j]
            )
        })

    oof_excel = OUT / 'ResNet_MLP_OOF_Results.xlsx'

    with pd.ExcelWriter(
        oof_excel, engine='openpyxl'
    ) as writer:

        pd.DataFrame(metrics).to_excel(
            writer,
            sheet_name='OOF_Metrics',
            index=False
        )

        pd.DataFrame(cv_rows).to_excel(
            writer,
            sheet_name='Fold_Metrics',
            index=False
        )

        pred = pd.DataFrame({
            'RowIndex_0based': np.arange(len(y))
        })

        for j, name in enumerate(TARGETS):
            pred[f'{name}_true'] = y[:, j]
            pred[f'{name}_OOF_pred'] = oof[:, j]

        pred.to_excel(
            writer,
            sheet_name='OOF_Predictions',
            index=False
        )

    print('\nFinal OOF performance:')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(
        'Mean OOF R2:',
        pd.DataFrame(metrics)['OOF_R2'].mean()
    )


# Normalize SHAP outputs to (samples, features, targets).
def unify_shap(shap_values, n_rows):

    if isinstance(shap_values, list):
        arr = np.stack(
            [np.asarray(v) for v in shap_values],
            axis=-1
        )
    else:
        arr = np.asarray(shap_values)

    if arr.shape == (10, n_rows, 13):
        arr = np.moveaxis(arr, 0, 2)

    expected = (n_rows, 13, 10)

    if arr.shape != expected:
        raise ValueError(
            f'Incorrect SHAP array shape: {arr.shape}; '
            f'expected {expected}'
        )

    return arr.astype(np.float32)


# Compute OOF SHAP values for all five folds.
def run_5fold_shap(X, y, sample_ids, refs, data_sha):

    import shap

    print(
        '\n===== Starting Five-Fold OOF SHAP =====',
        flush=True
    )

    shap_oof = np.full(
        (len(X), 13, 10),
        np.nan,
        dtype=np.float32
    )

    fold_ids = np.zeros(len(X), dtype=int)
    method = 'GradientExplainer (expected gradients)'
    used = set()

    for fold in range(1, K_FOLDS + 1):

        path = checkpoint_path(fold)

        if not path.exists():
            raise FileNotFoundError(
                f'Missing checkpoint file: {path}\n'
                'Run train_and_shap first to create the checkpoints.'
            )

        ckpt = torch.load(
            path,
            map_location='cpu',
            weights_only=True
        )

        if ckpt['data_sha256'] != data_sha:
            raise ValueError(
                f'Fold {fold}: The current Excel file '
                'does not match the file used to train the checkpoint. '
                'Restore the original dataset or retrain the model.'
            )

        if ckpt['cv_type'] != CV_TYPE:
            raise ValueError(
                f'Fold {fold}: CV_TYPE does not match the checkpoint.'
            )

        arch = tuple(ckpt['architecture'])

        expected_arch = (
            13, 10, HIDDEN_SIZE, NUM_BLOCKS, DROPOUT_P
        )

        if arch != expected_arch:
            raise ValueError(
                f'Fold {fold}: '
                'Model hyperparameters do not match the checkpoint.'
            )

        train_idx = ckpt['train_idx'].numpy()
        test_idx = ckpt['test_idx'].numpy()

        if any(idx in used for idx in test_idx):
            raise ValueError(
                'Held-out test samples occur in more than one fold.'
            )

        used.update(test_idx.tolist())
        fold_ids[test_idx] = fold

        model = HybridResNetMLP().to(DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        unit_model = OriginalUnitModel(
            model,
            ckpt['y_mean'],
            ckpt['y_scale']
        ).to(DEVICE)

        unit_model.eval()

        xs_mean = ckpt['x_mean'].numpy()
        xs_scale = ckpt['x_scale'].numpy()

        rng = np.random.default_rng(SEED + fold)

        background_ids = rng.choice(
            train_idx,
            size=min(SHAP_BACKGROUND_SIZE, len(train_idx)),
            replace=False
        )

        X_bg = (
            (X[background_ids] - xs_mean) / xs_scale
        ).astype(np.float32)

        bg = torch.from_numpy(X_bg).to(DEVICE)

        explainer = shap.GradientExplainer(
            unit_model, bg
        )

        print(
            f'Fold {fold}: '
            f'background samples={len(background_ids)}, '
            f'explained samples={len(test_idx)}',
            flush=True
        )

        for start in range(
            0, len(test_idx), SHAP_CHUNK_SIZE
        ):

            batch_idx = test_idx[
                start:start + SHAP_CHUNK_SIZE
            ]

            X_ex = (
                (X[batch_idx] - xs_mean) / xs_scale
            ).astype(np.float32)

            ex_tensor = torch.from_numpy(
                X_ex
            ).to(DEVICE)

            raw = explainer.shap_values(
                ex_tensor,
                nsamples=SHAP_NSAMPLES,
                rseed=SEED + fold * 1000 + start
            )

            shap_oof[batch_idx] = unify_shap(
                raw, len(batch_idx)
            )

            print(
                f'  SHAP completed '
                f'{start + len(batch_idx)}/{len(test_idx)}',
                flush=True
            )

        del explainer
        del unit_model
        del model

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    if (
        len(used) != len(X)
        or not np.isfinite(shap_oof).all()
    ):
        raise RuntimeError(
            'The OOF SHAP array is incomplete or contains invalid values.'
        )

    importance_rows = []
    abs_matrix = np.zeros((13, 10), dtype=float)

    for j, target in enumerate(TARGETS):

        values = shap_oof[:, :, j]
        mean_abs = np.mean(np.abs(values), axis=0)
        abs_matrix[:, j] = mean_abs

        relative = (
            100 * mean_abs
            / max(float(np.sum(mean_abs)), 1e-12)
        )

        for k, feature in enumerate(FEATURES):
            importance_rows.append({
                'Target': target,
                'Feature': feature,
                'MeanAbsSHAP': float(mean_abs[k]),
                'Relative_Importance_pct': float(relative[k]),
                'OutputUnit': 'Original target unit',
                'Number_of_explained_samples': len(X)
            })

        X_display = pd.DataFrame(
            X, columns=FEATURES
        )

        shap.summary_plot(
            values,
            X_display,
            plot_type='dot',
            max_display=13,
            show=False
        )

        fig = plt.gcf()

        fig.suptitle(
            f'{target}: OOF SHAP beeswarm',
            fontsize=13,
            y=1.02
        )

        fig.savefig(
            PLOT_DIR / f'SHAP_Beeswarm_{target}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        fig.savefig(
            PLOT_DIR / f'SHAP_Beeswarm_{target}.pdf',
            bbox_inches='tight'
        )

        plt.close('all')

        order = np.argsort(mean_abs)
        sorted_features = np.array(FEATURES)[order]
        sorted_values = mean_abs[order]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.barh(
            sorted_features,
            sorted_values,
            color='#4C78A8'
        )

        ax.set_xlabel(
            'Mean |SHAP| (original output unit)'
        )
        ax.set_title(
            f'{target}: OOF SHAP importance'
        )

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR / f'SHAP_Bar_{target}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        fig.savefig(
            PLOT_DIR / f'SHAP_Bar_{target}.pdf',
            bbox_inches='tight'
        )

        plt.close(fig)

        print(
            f'Completed target: {target}',
            flush=True
        )

    wide = pd.DataFrame(
        abs_matrix,
        index=FEATURES,
        columns=TARGETS
    )

    rel_wide = (
        wide.div(
            wide.sum(axis=0).replace(0, np.nan),
            axis=1
        ).mul(100)
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    hm = ax.imshow(
        rel_wide.to_numpy(),
        aspect='auto',
        cmap='YlOrBr',
        vmin=0
    )

    ax.set_yticks(range(13), FEATURES)

    ax.set_xticks(
        range(10),
        TARGETS,
        rotation=35,
        ha='right'
    )

    ax.set_title(
        'ResNet-MLP: Relative OOF SHAP importance'
    )

    colorbar = fig.colorbar(hm, ax=ax)

    colorbar.set_label(
        'Within-target importance (%)'
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / 'SHAP_RelativeImportance_Heatmap.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        PLOT_DIR / 'SHAP_RelativeImportance_Heatmap.pdf',
        bbox_inches='tight'
    )

    plt.close(fig)

    meta = pd.DataFrame([
        ('Input file', INPUT.name),
        ('Input SHA256', data_sha),
        ('Model', 'ResNet-MLP (13-128-3-10)'),
        ('CV', f'{CV_TYPE}, {K_FOLDS} folds, seed={SEED}'),
        ('SHAP method', method),
        ('SHAP data', f'Held-out samples; total n={len(X)}'),
        (
            'Background',
            f'Up to {SHAP_BACKGROUND_SIZE} '
            f'training samples per fold'
        ),
        ('Explainer samples', SHAP_NSAMPLES),
        ('Output units', 'Original target units'),
        ('Caveat', 'ETII includes imputed labels')
    ], columns=['Setting', 'Value'])

    input_df = pd.DataFrame(X, columns=FEATURES)

    input_df.insert(0, 'Fold', fold_ids)
    input_df.insert(0, 'Reference', refs)
    input_df.insert(0, 'Index_No', sample_ids)

    excel = OUT / 'ResNet_MLP_5Fold_OOF_SHAP_Data.xlsx'

    with pd.ExcelWriter(excel, engine='openpyxl') as writer:

        pd.DataFrame(importance_rows).to_excel(
            writer,
            sheet_name='SHAP_Importance',
            index=False
        )

        wide.reset_index(names='Feature').to_excel(
            writer,
            sheet_name='MeanAbsSHAP_Wide',
            index=False
        )

        rel_wide.reset_index(names='Feature').to_excel(
            writer,
            sheet_name='Relative_Importance',
            index=False
        )

        input_df.to_excel(
            writer,
            sheet_name='OOF_InputValues',
            index=False
        )

        meta.to_excel(
            writer,
            sheet_name='Settings',
            index=False
        )

        for j, target in enumerate(TARGETS):
            values_df = pd.DataFrame(
                shap_oof[:, :, j],
                columns=FEATURES
            )

            values_df.insert(
                0, 'Fold', fold_ids
            )

            values_df.insert(
                0, 'Index_No', sample_ids
            )

            values_df.to_excel(
                writer,
                sheet_name=('SHAP_' + target)[:31],
                index=False
            )

    print('\nSHAP Excel:', excel, flush=True)
    print(
        'SHAP figure directory:',
        PLOT_DIR,
        flush=True
    )
    print('All SHAP calculations completed.')


# Run full training + SHAP, or SHAP-only using saved checkpoints.
def main():
    print('Computing device:', DEVICE)

    if DEVICE.type == 'cuda':
        print(
            'GPU model:',
            torch.cuda.get_device_name(0)
        )

    X, y, sample_ids, refs, sha = load_data()

    if MODE == 'train_and_shap':
        train_5_folds(X, y, refs, sha)
    elif MODE != 'shap_only':
        raise ValueError(
            'MODE must be either train_and_shap or shap_only.'
        )

    run_5fold_shap(
        X, y, sample_ids, refs, sha
    )


if __name__ == '__main__':
    main()
