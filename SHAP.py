
# -*- coding: utf-8 -*-
"""
ResNet-MLP (13 inputs -> 10 outputs)
Five-fold out-of-fold (OOF) Gradient SHAP

使用方法：
1. 将本.py文件与ETII.xlsx放在同一文件夹
2. 首次运行 MODE='train_and_shap'
3. 以后只重新计算SHAP，使用 MODE='shap_only'

注意：
ETII数据包含插补后的输出标签。
SHAP解释的是模型预测，不代表因果关系。
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

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# =====================================================
# 1. 参数设置
# =====================================================

SEED = 42

# 最新数据使用 ETII(6).xlsx
EXCEL_NAME = 'ETII.xlsx'

SHEET_NAME = 'MICE_All_Exploratory'

# 第一次运行：train_and_shap
# 已有五折模型：shap_only
MODE = 'train_and_shap'

# random：随机五折
# reference：按文献来源分组五折
CV_TYPE = 'random'

K_FOLDS = 5

EPOCHS = 2600

LEARNING_RATE = 5e-4

BATCH_SIZE = 48

WEIGHT_DECAY = 1e-5

HIDDEN_SIZE = 128

NUM_BLOCKS = 3

DROPOUT_P = 0.4


# =====================================================
# 2. SHAP参数
# =====================================================

# 每折从训练集中选80条作为背景
SHAP_BACKGROUND_SIZE = 80

# GradientExplainer抽样次数
SHAP_NSAMPLES = 64

# 每批解释20条测试样本
SHAP_CHUNK_SIZE = 20

DPI = 300


# =====================================================
# 3. 输入和输出路径
# =====================================================

BASE = Path(__file__).resolve().parent

INPUT = BASE / EXCEL_NAME

OUT = BASE / 'ResNet_MLP_OOF_SHAP_Results'

CKPT_DIR = OUT / 'FiveFold_Checkpoints'

PLOT_DIR = OUT / 'SHAP_Plots'

for directory in (
    OUT,
    CKPT_DIR,
    PLOT_DIR
):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =====================================================
# 4. Excel列位置
# =====================================================

# 输入：G、H、I、K-T
INPUT_INDEX = [
    6, 7, 8
] + list(range(10, 20))

# 输出：U-AD
OUTPUT_INDEX = list(range(20, 30))


# =====================================================
# 5. 输入变量
# =====================================================

FEATURES = [
    'FC_dry',
    'VM_dry',
    'Ash_dry',
    'Feedstock_C',
    'Feedstock_H',
    'Feedstock_O',
    'Feedstock_N',
    'Cellulose',
    'Hemicellulose',
    'Lignin',
    'Residence_time',
    'Temperature',
    'Heating_rate'
]


# =====================================================
# 6. 输出变量
# =====================================================

TARGETS = [
    'Biochar_yield',
    'TPV',
    'Biochar_ash',
    'Zeta_potential',
    'pH',
    'Biochar_C',
    'Biochar_H',
    'Biochar_O',
    'O_C_ratio',
    'H_C_ratio'
]

assert len(FEATURES) == 13
assert len(TARGETS) == 10


# =====================================================
# 7. 固定随机种子
# =====================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


# =====================================================
# 8. 绘图设置
# =====================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# =====================================================
# 9. 读取数据
# =====================================================

def load_data():

    if not INPUT.is_file():

        raise FileNotFoundError(
            f'找不到数据文件：{INPUT}\n'
            '请把Excel文件与Python脚本放在同一文件夹。'
        )

    raw = pd.read_excel(
        INPUT,
        sheet_name=SHEET_NAME,
        header=None,
        engine='openpyxl'
    )

    if raw.shape[1] < 30:

        raise ValueError(
            'Excel至少需要30列。'
        )

    # 跳过前4行表头
    body = raw.iloc[4:, :].copy()

    # 保留具有有效样本编号的行
    ids_as_numbers = pd.to_numeric(
        body.iloc[:, 0],
        errors='coerce'
    )

    body = body.loc[
        ids_as_numbers.notna()
    ].reset_index(drop=True)

    # 提取13个输入
    X_df = body.iloc[
        :, INPUT_INDEX
    ].apply(
        pd.to_numeric,
        errors='coerce'
    )

    # 提取10个输出
    y_df = body.iloc[
        :, OUTPUT_INDEX
    ].apply(
        pd.to_numeric,
        errors='coerce'
    )

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = y_df.to_numpy(
        dtype=np.float32
    )

    # 检查缺失值
    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):

        raise ValueError(
            '13个输入或10个输出存在NaN/Inf。'
        )

    # 样本编号
    ids = body.iloc[:, 0].to_numpy()

    # 文献来源
    if body.shape[1] > 30:

        refs = (
            body.iloc[:, 30]
            .fillna('Unknown')
            .astype(str)
            .str.strip()
            .to_numpy()
        )

    else:

        refs = np.array(
            ['Unknown'] * len(body)
        )

    # 记录输入文件哈希
    sha = hashlib.sha256(
        INPUT.read_bytes()
    ).hexdigest()

    print(
        f'数据文件：{INPUT.name}'
    )

    print(
        f'X.shape = {X.shape}'
    )

    print(
        f'y.shape = {y.shape}'
    )

    print(
        f'SHA256 = {sha[:16]}...'
    )

    return X, y, ids, refs, sha


# =====================================================
# 10. ResNet残差模块
# =====================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        dropout_p
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout_p
        )

    def forward(self, x):

        residual = x

        out = self.fc1(x)

        out = self.bn1(out)

        out = torch.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.bn2(out)

        # 残差连接
        out = out + residual

        return torch.relu(out)


# =====================================================
# 11. ResNet-MLP完整网络
# =====================================================

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
            input_size,
            hidden_size
        )

        self.bn_entry = nn.BatchNorm1d(
            hidden_size
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_size,
                    dropout_p
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        x = self.entry_layer(x)

        x = self.bn_entry(x)

        x = torch.relu(x)

        x = self.res_blocks(x)

        return self.output_layer(x)


# =====================================================
# 12. 将输出恢复到原始单位
# =====================================================

class OriginalUnitModel(nn.Module):

    def __init__(
        self,
        model,
        y_mean,
        y_scale
    ):

        super().__init__()

        self.model = model

        self.register_buffer(
            'y_mean',
            y_mean.float().clone()
        )

        self.register_buffer(
            'y_scale',
            y_scale.float().clone()
        )

    def forward(self, x):

        return (
            self.model(x)
            * self.y_scale
            + self.y_mean
        )


# =====================================================
# 13. 五折划分
# =====================================================

def get_folds(X, refs):

    if CV_TYPE == 'random':

        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=SEED
        )

        return list(kf.split(X))

    if CV_TYPE == 'reference':

        gkf = GroupKFold(
            n_splits=K_FOLDS
        )

        return list(
            gkf.split(
                X,
                groups=refs
            )
        )

    raise ValueError(
        'CV_TYPE必须是random或reference'
    )


# =====================================================
# 14. 模型保存路径
# =====================================================

def checkpoint_path(fold):

    return (
        CKPT_DIR
        / f'ResNet_MLP_Fold{fold}.pt'
    )


# =====================================================
# 15. 五折训练
# =====================================================

def train_5_folds(
    X,
    y,
    refs,
    data_sha
):

    folds = get_folds(
        X,
        refs
    )

    oof = np.full(
        y.shape,
        np.nan,
        dtype=np.float32
    )

    cv_rows = []

    # 五折依次训练
    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        folds,
        1
    ):

        print(
            f'\n===== Fold {fold}/{K_FOLDS} =====',
            flush=True
        )

        X_train = X[train_idx]

        X_test = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        # ==========================================
        # 训练折标准化
        # ==========================================

        xs = StandardScaler()

        ys = StandardScaler()

        X_train_scaled = (
            xs.fit_transform(
                X_train
            ).astype(np.float32)
        )

        X_test_scaled = (
            xs.transform(
                X_test
            ).astype(np.float32)
        )

        y_train_scaled = (
            ys.fit_transform(
                y_train
            ).astype(np.float32)
        )

        # ==========================================
        # DataLoader
        # ==========================================

        dataset = TensorDataset(
            torch.from_numpy(
                X_train_scaled
            ),
            torch.from_numpy(
                y_train_scaled
            )
        )

        # 避免BatchNorm出现单样本批次
        drop_last = (
            len(dataset) % BATCH_SIZE == 1
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=drop_last
        )

        # ==========================================
        # 创建模型
        # ==========================================

        model = HybridResNetMLP().to(
            DEVICE
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        criterion = nn.MSELoss()

        # ==========================================
        # 开始训练
        # ==========================================

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            model.train()

            epoch_loss = 0.0

            for bx, by in loader:

                bx = bx.to(DEVICE)

                by = by.to(DEVICE)

                optimizer.zero_grad()

                output = model(bx)

                loss = criterion(
                    output,
                    by
                )

                loss.backward()

                optimizer.step()

                epoch_loss += float(
                    loss.item()
                )

            if (
                epoch == 1
                or epoch % 200 == 0
                or epoch == EPOCHS
            ):

                mean_loss = (
                    epoch_loss
                    / max(len(loader), 1)
                )

                print(
                    f'Fold={fold} '
                    f'Epoch={epoch}/{EPOCHS} '
                    f'Loss={mean_loss:.6f}',
                    flush=True
                )

        # ==========================================
        # 当前折测试集预测
        # ==========================================

        model.eval()

        with torch.no_grad():

            test_tensor = (
                torch.from_numpy(
                    X_test_scaled
                ).to(DEVICE)
            )

            y_test_scaled = (
                model(test_tensor)
                .cpu()
                .numpy()
            )

        y_pred = (
            ys.inverse_transform(
                y_test_scaled
            )
        )

        # OOF预测
        oof[test_idx] = y_pred

        # ==========================================
        # 各目标性能
        # ==========================================

        for j, target in enumerate(
            TARGETS
        ):

            cv_rows.append({

                'Fold': fold,

                'Target': target,

                'R2': r2_score(
                    y_test[:, j],
                    y_pred[:, j]
                ),

                'RMSE': np.sqrt(
                    mean_squared_error(
                        y_test[:, j],
                        y_pred[:, j]
                    )
                ),

                'MAE': mean_absolute_error(
                    y_test[:, j],
                    y_pred[:, j]
                )
            })

        # ==========================================
        # 保存本折模型及标准化器
        # ==========================================

        checkpoint = {

            'fold': fold,

            'data_sha256': data_sha,

            'cv_type': CV_TYPE,

            'architecture': (
                13,
                10,
                HIDDEN_SIZE,
                NUM_BLOCKS,
                DROPOUT_P
            ),

            'state_dict': {
                k: v.detach().cpu().clone()
                for k, v in
                model.state_dict().items()
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
                np.asarray(
                    train_idx,
                    dtype=np.int64
                )
            ),

            'test_idx': torch.from_numpy(
                np.asarray(
                    test_idx,
                    dtype=np.int64
                )
            )
        }

        torch.save(
            checkpoint,
            checkpoint_path(fold)
        )

        print(
            f'Fold {fold} 已保存模型',
            flush=True
        )

        del model

        if DEVICE.type == 'cuda':

            torch.cuda.empty_cache()

    # ==============================================
    # 检查OOF
    # ==============================================

    if not np.isfinite(oof).all():

        raise RuntimeError(
            '部分OOF预测没有完成。'
        )

    # ==============================================
    # 计算全部样本OOF指标
    # ==============================================

    metrics = []

    for j, name in enumerate(
        TARGETS
    ):

        metrics.append({

            'Target': name,

            'OOF_R2': r2_score(
                y[:, j],
                oof[:, j]
            ),

            'OOF_RMSE': np.sqrt(
                mean_squared_error(
                    y[:, j],
                    oof[:, j]
                )
            ),

            'OOF_MAE': mean_absolute_error(
                y[:, j],
                oof[:, j]
            )
        })

    # ==============================================
    # 导出OOF Excel
    # ==============================================

    oof_excel = (
        OUT / 'ResNet_MLP_OOF_Results.xlsx'
    )

    with pd.ExcelWriter(
        oof_excel,
        engine='openpyxl'
    ) as writer:

        pd.DataFrame(
            metrics
        ).to_excel(
            writer,
            sheet_name='OOF_Metrics',
            index=False
        )

        pd.DataFrame(
            cv_rows
        ).to_excel(
            writer,
            sheet_name='Fold_Metrics',
            index=False
        )

        pred = pd.DataFrame({
            'RowIndex_0based':
                np.arange(len(y))
        })

        for j, name in enumerate(
            TARGETS
        ):

            pred[
                f'{name}_true'
            ] = y[:, j]

            pred[
                f'{name}_OOF_pred'
            ] = oof[:, j]

        pred.to_excel(
            writer,
            sheet_name='OOF_Predictions',
            index=False
        )

    print(
        '\n最终OOF性能：'
    )

    print(
        pd.DataFrame(
            metrics
        ).to_string(index=False)
    )

    print(
        '平均OOF R2：',
        pd.DataFrame(
            metrics
        )['OOF_R2'].mean()
    )


# =====================================================
# 16. 兼容不同SHAP返回格式
# =====================================================

def unify_shap(
    shap_values,
    n_rows
):

    if isinstance(
        shap_values,
        list
    ):

        arr = np.stack(
            [
                np.asarray(v)
                for v in shap_values
            ],
            axis=-1
        )

    else:

        arr = np.asarray(
            shap_values
        )

    # 兼容(outputs, samples, features)
    if arr.shape == (
        10,
        n_rows,
        13
    ):

        arr = np.moveaxis(
            arr,
            0,
            2
        )

    expected = (
        n_rows,
        13,
        10
    )

    if arr.shape != expected:

        raise ValueError(
            f'SHAP数组形状错误：{arr.shape}，'
            f'预期为{expected}'
        )

    return arr.astype(
        np.float32
    )


# =====================================================
# 17. 五折OOF SHAP
# =====================================================

def run_5fold_shap(
    X,
    y,
    sample_ids,
    refs,
    data_sha
):

    import shap

    print(
        '\n===== 开始五折OOF SHAP =====',
        flush=True
    )

    # 522 × 13 × 10
    shap_oof = np.full(
        (len(X), 13, 10),
        np.nan,
        dtype=np.float32
    )

    fold_ids = np.zeros(
        len(X),
        dtype=int
    )

    method = (
        'GradientExplainer '
        '(expected gradients)'
    )

    used = set()

    # ==============================================
    # 依次计算五折SHAP
    # ==============================================

    for fold in range(
        1,
        K_FOLDS + 1
    ):

        path = checkpoint_path(fold)

        if not path.exists():

            raise FileNotFoundError(
                f'缺少模型文件：{path}\n'
                '请先用train_and_shap训练。'
            )

        ckpt = torch.load(
            path,
            map_location='cpu',
            weights_only=True
        )

        # ==========================================
        # 检查数据一致性
        # ==========================================

        if ckpt['data_sha256'] != data_sha:

            raise ValueError(
                f'Fold {fold}：当前Excel文件'
                '与保存模型时使用的文件不同。'
                '请恢复原数据或重新训练。'
            )

        if ckpt['cv_type'] != CV_TYPE:

            raise ValueError(
                f'Fold {fold}：CV_TYPE不同。'
            )

        arch = tuple(
            ckpt['architecture']
        )

        expected_arch = (
            13,
            10,
            HIDDEN_SIZE,
            NUM_BLOCKS,
            DROPOUT_P
        )

        if arch != expected_arch:

            raise ValueError(
                f'Fold {fold}：'
                '模型超参数与检查点不一致。'
            )

        train_idx = (
            ckpt['train_idx'].numpy()
        )

        test_idx = (
            ckpt['test_idx'].numpy()
        )

        if any(
            idx in used
            for idx in test_idx
        ):

            raise ValueError(
                '测试样本在不同折中重复出现。'
            )

        used.update(
            test_idx.tolist()
        )

        fold_ids[test_idx] = fold

        # ==========================================
        # 重建模型
        # ==========================================

        model = (
            HybridResNetMLP()
            .to(DEVICE)
        )

        model.load_state_dict(
            ckpt['state_dict']
        )

        model.eval()

        # ==========================================
        # 输出恢复原始单位
        # ==========================================

        unit_model = OriginalUnitModel(
            model,
            ckpt['y_mean'],
            ckpt['y_scale']
        ).to(DEVICE)

        unit_model.eval()

        xs_mean = (
            ckpt['x_mean'].numpy()
        )

        xs_scale = (
            ckpt['x_scale'].numpy()
        )

        # ==========================================
        # 随机选背景样本
        # 仅选当前折训练样本
        # ==========================================

        rng = np.random.default_rng(
            SEED + fold
        )

        background_ids = rng.choice(
            train_idx,
            size=min(
                SHAP_BACKGROUND_SIZE,
                len(train_idx)
            ),
            replace=False
        )

        X_bg = (
            (
                X[background_ids]
                - xs_mean
            )
            / xs_scale
        ).astype(np.float32)

        bg = torch.from_numpy(
            X_bg
        ).to(DEVICE)

        # ==========================================
        # 创建GradientExplainer
        # ==========================================

        explainer = shap.GradientExplainer(
            unit_model,
            bg
        )

        print(
            f'Fold {fold}: '
            f'背景样本={len(background_ids)}, '
            f'解释样本={len(test_idx)}',
            flush=True
        )

        # ==========================================
        # 分批解释测试样本
        # ==========================================

        for start in range(
            0,
            len(test_idx),
            SHAP_CHUNK_SIZE
        ):

            batch_idx = test_idx[
                start:start + SHAP_CHUNK_SIZE
            ]

            X_ex = (
                (
                    X[batch_idx]
                    - xs_mean
                )
                / xs_scale
            ).astype(np.float32)

            ex_tensor = (
                torch.from_numpy(
                    X_ex
                ).to(DEVICE)
            )

            raw = explainer.shap_values(
                ex_tensor,
                nsamples=SHAP_NSAMPLES,
                rseed=(
                    SEED
                    + fold * 1000
                    + start
                )
            )

            shap_oof[batch_idx] = (
                unify_shap(
                    raw,
                    len(batch_idx)
                )
            )

            print(
                f'  SHAP完成 '
                f'{start + len(batch_idx)}'
                f'/{len(test_idx)}',
                flush=True
            )

        del explainer
        del unit_model
        del model

        if DEVICE.type == 'cuda':

            torch.cuda.empty_cache()

    # ==============================================
    # 检查全部OOF SHAP
    # ==============================================

    if (
        len(used) != len(X)
        or not np.isfinite(
            shap_oof
        ).all()
    ):

        raise RuntimeError(
            'SHAP OOF数组不完整或存在异常值。'
        )

    # ==============================================
    # 计算平均绝对SHAP
    # ==============================================

    importance_rows = []

    abs_matrix = np.zeros(
        (13, 10),
        dtype=float
    )

    for j, target in enumerate(
        TARGETS
    ):

        values = shap_oof[:, :, j]

        mean_abs = np.mean(
            np.abs(values),
            axis=0
        )

        abs_matrix[:, j] = mean_abs

        relative = (
            100
            * mean_abs
            / max(
                float(np.sum(mean_abs)),
                1e-12
            )
        )

        for k, feature in enumerate(
            FEATURES
        ):

            importance_rows.append({

                'Target': target,

                'Feature': feature,

                'MeanAbsSHAP': float(
                    mean_abs[k]
                ),

                'Relative_Importance_pct':
                    float(relative[k]),

                'OutputUnit':
                    'Original target unit',

                'Number_of_explained_samples':
                    len(X)
            })

        # ==========================================
        # 当前目标蜂群图
        # ==========================================

        X_display = pd.DataFrame(
            X,
            columns=FEATURES
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
            PLOT_DIR
            / f'SHAP_Beeswarm_{target}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        fig.savefig(
            PLOT_DIR
            / f'SHAP_Beeswarm_{target}.pdf',
            bbox_inches='tight'
        )

        plt.close('all')

        # ==========================================
        # 当前目标重要性柱状图
        # ==========================================

        order = np.argsort(
            mean_abs
        )

        sorted_features = (
            np.array(FEATURES)[order]
        )

        sorted_values = (
            mean_abs[order]
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

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
            PLOT_DIR
            / f'SHAP_Bar_{target}.png',
            dpi=DPI,
            bbox_inches='tight'
        )

        fig.savefig(
            PLOT_DIR
            / f'SHAP_Bar_{target}.pdf',
            bbox_inches='tight'
        )

        plt.close(fig)

        print(
            f'完成目标：{target}',
            flush=True
        )

    # ==============================================
    # 13×10平均重要性矩阵
    # ==============================================

    wide = pd.DataFrame(
        abs_matrix,
        index=FEATURES,
        columns=TARGETS
    )

    # 各目标内部归一化为100%
    rel_wide = (
        wide.div(
            wide.sum(axis=0).replace(
                0,
                np.nan
            ),
            axis=1
        )
        .mul(100)
    )

    # ==============================================
    # 重要性热图
    # ==============================================

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    hm = ax.imshow(
        rel_wide.to_numpy(),
        aspect='auto',
        cmap='YlOrBr',
        vmin=0
    )

    ax.set_yticks(
        range(13),
        FEATURES
    )

    ax.set_xticks(
        range(10),
        TARGETS,
        rotation=35,
        ha='right'
    )

    ax.set_title(
        'ResNet-MLP: Relative OOF SHAP importance'
    )

    colorbar = fig.colorbar(
        hm,
        ax=ax
    )

    colorbar.set_label(
        'Within-target importance (%)'
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / 'SHAP_RelativeImportance_Heatmap.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        PLOT_DIR
        / 'SHAP_RelativeImportance_Heatmap.pdf',
        bbox_inches='tight'
    )

    plt.close(fig)

    # ==============================================
    # 实验设置表
    # ==============================================

    meta = pd.DataFrame(
        [
            ('Input file', INPUT.name),

            ('Input SHA256', data_sha),

            (
                'Model',
                'ResNet-MLP (13-128-3-10)'
            ),

            (
                'CV',
                f'{CV_TYPE}, '
                f'{K_FOLDS} folds, '
                f'seed={SEED}'
            ),

            (
                'SHAP method',
                method
            ),

            (
                'SHAP data',
                'Held-out samples; '
                f'total n={len(X)}'
            ),

            (
                'Background',
                f'Up to '
                f'{SHAP_BACKGROUND_SIZE} '
                f'training samples per fold'
            ),

            (
                'Explainer samples',
                SHAP_NSAMPLES
            ),

            (
                'Output units',
                'Original target units'
            ),

            (
                'Caveat',
                'ETII includes imputed labels'
            )
        ],
        columns=[
            'Setting',
            'Value'
        ]
    )

    # ==============================================
    # 输入样本表
    # ==============================================

    input_df = pd.DataFrame(
        X,
        columns=FEATURES
    )

    input_df.insert(
        0,
        'Fold',
        fold_ids
    )

    input_df.insert(
        0,
        'Reference',
        refs
    )

    input_df.insert(
        0,
        'Index_No',
        sample_ids
    )

    # ==============================================
    # 保存SHAP Excel
    # ==============================================

    excel = (
        OUT
        / 'ResNet_MLP_5Fold_OOF_SHAP_Data.xlsx'
    )

    with pd.ExcelWriter(
        excel,
        engine='openpyxl'
    ) as writer:

        pd.DataFrame(
            importance_rows
        ).to_excel(
            writer,
            sheet_name='SHAP_Importance',
            index=False
        )

        wide.reset_index(
            names='Feature'
        ).to_excel(
            writer,
            sheet_name='MeanAbsSHAP_Wide',
            index=False
        )

        rel_wide.reset_index(
            names='Feature'
        ).to_excel(
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

        # 保存每个目标的逐样本SHAP值
        for j, target in enumerate(
            TARGETS
        ):

            values_df = pd.DataFrame(
                shap_oof[:, :, j],
                columns=FEATURES
            )

            values_df.insert(
                0,
                'Fold',
                fold_ids
            )

            values_df.insert(
                0,
                'Index_No',
                sample_ids
            )

            values_df.to_excel(
                writer,
                sheet_name=(
                    'SHAP_' + target
                )[:31],
                index=False
            )

    print(
        '\nSHAP Excel：',
        excel,
        flush=True
    )

    print(
        'SHAP图片目录：',
        PLOT_DIR,
        flush=True
    )

    print(
        'SHAP全部计算完成。'
    )


# =====================================================
# 18. 主函数
# =====================================================

def main():

    print(
        '计算设备：',
        DEVICE
    )

    if DEVICE.type == 'cuda':

        print(
            'GPU型号：',
            torch.cuda.get_device_name(0)
        )

    X, y, sample_ids, refs, sha = (
        load_data()
    )

    # 第一次训练并计算SHAP
    if MODE == 'train_and_shap':

        train_5_folds(
            X,
            y,
            refs,
            sha
        )

    elif MODE != 'shap_only':

        raise ValueError(
            'MODE必须是train_and_shap或shap_only'
        )

    # 计算五折OOF SHAP
    run_5fold_shap(
        X,
        y,
        sample_ids,
        refs,
        sha
    )


# =====================================================
# 19. 程序入口
# =====================================================

if __name__ == '__main__':

    main()
