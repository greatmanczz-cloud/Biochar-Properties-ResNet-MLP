
# -*- coding: utf-8 -*-
"""
ResNet-MLP 五折PDP重新计算与响应热图绘制

无需之前保存的PDP Excel。
无需重新训练模型。
无需安装SHAP。

需要：
1. 模型训练时使用的原始Excel。
2. 五折ResNet-MLP模型文件。

输出：
1. 完整五折PDP数据Excel
2. PDP响应热图Origin数据Excel
3. 10张单输出PDP响应热图
4. 1张10输出汇总热图
5. PNG及PDF格式

颜色：
蓝色：预测响应低于参考值
红色：预测响应高于参考值

说明：
横坐标0%-100%表示每个输入变量
从P5至P95取值范围内的相对位置。
不是实际样本的经验百分位数。

PDP解释模型预测响应，不代表因果效应。
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


# ==================================================
# 1. 基本参数
# ==================================================

EXCEL_NAME = "ETII.xlsx"

SHEET_NAME = "MICE_All_Exploratory"

K_FOLDS = 5

HIDDEN_SIZE = 128

NUM_BLOCKS = 3

DROPOUT_P = 0.4

# PDP网格点数
GRID_POINTS = 31

# PDP取值范围
GRID_PERCENTILE_LOW = 5
GRID_PERCENTILE_HIGH = 95

# 预测批量大小
PRED_BATCH_SIZE = 512

# 热图横向网格点数
HEATMAP_POINTS = 31

# 图片分辨率
DPI = 300

# 是否生成10输出汇总热图
DRAW_10_TARGET_OVERVIEW = True


# ==================================================
# 2. 文件路径
# ==================================================

BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / EXCEL_NAME

# 五折模型检查点目录
CKPT_DIR = (
    BASE
    / "ResNet_MLP_OOF_SHAP_Results"
    / "FiveFold_Checkpoints"
)

# PDP结果目录
OUT_DIR = (
    BASE
    / "ResNet_MLP_OOF_PDP_Results"
)

# 热图目录
FIG_DIR = (
    OUT_DIR
    / "PDP_Response_Heatmaps"
)

# 重新生成的完整PDP Excel
PDP_EXCEL = (
    OUT_DIR
    / "ResNet_MLP_5Fold_PDP_Origin_Data.xlsx"
)

# 响应热图Origin数据Excel
HEATMAP_EXCEL = (
    OUT_DIR
    / "PDP_Response_Heatmap_Origin_Data.xlsx"
)


# ==================================================
# 3. 输入变量
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

    "Heating_rate"

]


# ==================================================
# 4. 输出变量
# ==================================================

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


# Excel输入列：G、H、I、K-T
INPUT_INDEX = [
    6, 7, 8
] + list(range(10, 20))

# Excel输出列：U-AD
OUTPUT_INDEX = list(range(20, 30))


# ==================================================
# 5. 计算设备与绘图设置
# ==================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "axes.unicode_minus": False,

    "pdf.fonttype": 42,

    "ps.fonttype": 42

})


# ==================================================
# 6. ResNet残差模块
# ==================================================

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

        self.dropout = nn.Dropout(
            dropout_p
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_size
        )

    def forward(self, x):

        residual = x

        out = self.fc1(x)

        out = self.bn1(out)

        out = torch.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.bn2(out)

        return torch.relu(
            out + residual
        )


# ==================================================
# 7. ResNet-MLP模型
# ==================================================

class HybridResNetMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.entry_layer = nn.Linear(
            13,
            HIDDEN_SIZE
        )

        self.bn_entry = nn.BatchNorm1d(
            HIDDEN_SIZE
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    HIDDEN_SIZE,
                    DROPOUT_P
                )
                for _ in range(NUM_BLOCKS)
            ]
        )

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            10
        )

    def forward(self, x):

        x = self.entry_layer(x)

        x = self.bn_entry(x)

        x = torch.relu(x)

        x = self.res_blocks(x)

        return self.output_layer(x)


# ==================================================
# 8. 读取原始Excel数据
# ==================================================

def load_data():

    if not DATA_PATH.is_file():

        raise FileNotFoundError(
            f"找不到原始Excel：{DATA_PATH}"
        )

    raw = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl"
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "Excel列数不足30列。"
        )

    # 跳过前4行表头
    body = raw.iloc[4:, :].copy()

    ids = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    # 只保留有效样本
    body = body.loc[
        ids.notna()
    ].reset_index(drop=True)

    # 提取13个输入
    X = (
        body.iloc[:, INPUT_INDEX]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # 提取10个输出
    y = (
        body.iloc[:, OUTPUT_INDEX]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if X.shape[1] != 13 or y.shape[1] != 10:

        raise ValueError(
            f"维度不正确：X={X.shape}, y={y.shape}"
        )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):

        raise ValueError(
            "输入或输出数据包含NaN或Inf。"
        )

    # 计算原始文件SHA256
    data_sha = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    print(
        f"原始数据：{DATA_PATH.name}"
    )

    print(
        f"X.shape：{X.shape}"
    )

    print(
        f"y.shape：{y.shape}"
    )

    print(
        f"数据SHA256：{data_sha}"
    )

    return X, y, data_sha


# ==================================================
# 9. Tensor转NumPy
# ==================================================

def to_numpy(value):

    if isinstance(value, torch.Tensor):

        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(value)


# ==================================================
# 10. 读取指定折的模型
# ==================================================

def load_fold_model(
    fold,
    data_sha,
    n_samples
):

    path = (
        CKPT_DIR
        / f"ResNet_MLP_Fold{fold}.pt"
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"找不到模型：{path}\n"
            "请确认五折模型已经保存。"
        )

    # 只加载自己训练生成的可信模型
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # 校验原始数据版本
    if ckpt.get("data_sha256") != data_sha:

        raise ValueError(
            f"Fold {fold}模型与当前Excel数据不一致。\n"
            "必须使用训练模型时的原始Excel。\n"
            "不能通过修改文件名绕过校验。"
        )

    # 校验模型结构
    expected_arch = (
        13,
        10,
        HIDDEN_SIZE,
        NUM_BLOCKS,
        DROPOUT_P
    )

    if tuple(
        ckpt["architecture"]
    ) != expected_arch:

        raise ValueError(
            f"Fold {fold}模型结构不一致。"
        )

    train_idx = to_numpy(
        ckpt["train_idx"]
    ).astype(int)

    test_idx = to_numpy(
        ckpt["test_idx"]
    ).astype(int)

    # 检查测试索引
    if (
        len(test_idx) == 0
        or np.any(test_idx < 0)
        or np.any(test_idx >= n_samples)
        or len(np.unique(test_idx)) != len(test_idx)
        or np.intersect1d(
            train_idx,
            test_idx
        ).size > 0
    ):

        raise ValueError(
            f"Fold {fold}训练或测试索引异常。"
        )

    # 重建模型
    model = HybridResNetMLP().to(
        DEVICE
    )

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()

    return model, ckpt, test_idx


# ==================================================
# 11. 模型预测，恢复输出原始单位
# ==================================================

@torch.no_grad()
def predict_original_units(
    model,
    ckpt,
    X_original
):

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
            "模型标准化参数scale不合法。"
        )

    # 输入标准化
    X_standard = (
        (X_original - x_mean)
        / x_scale
    ).astype(np.float32)

    pred_batches = []

    # 批量预测
    for start in range(
        0,
        len(X_standard),
        PRED_BATCH_SIZE
    ):

        xb = torch.from_numpy(
            X_standard[
                start:start + PRED_BATCH_SIZE
            ]
        ).to(DEVICE)

        pred = model(xb)

        pred_batches.append(
            pred.cpu().numpy()
        )

    y_standard = np.concatenate(
        pred_batches,
        axis=0
    )

    # 恢复输出原始单位
    y_original = (
        y_standard * y_scale
        + y_mean
    )

    return y_original


# ==================================================
# 12. 生成PDP网格
# ==================================================

def make_feature_grid(x):

    unique_values = np.unique(x)

    low, high = np.percentile(
        x,
        [
            GRID_PERCENTILE_LOW,
            GRID_PERCENTILE_HIGH
        ]
    )

    within = unique_values[
        (unique_values >= low)
        & (unique_values <= high)
    ]

    # 少量离散值采用实际观测值
    if (
        len(unique_values) <= 12
        and len(within) > 0
    ):

        return within.astype(float)

    if high > low:

        return np.linspace(
            low,
            high,
            GRID_POINTS
        )

    return np.array([
        float(low)
    ])


# ==================================================
# 13. 重新计算五折PDP
# ==================================================

def recompute_five_fold_pdp(
    X,
    data_sha
):

    # 每个输入变量的PDP网格
    grids = {

        feature: make_feature_grid(
            X[:, j]
        )

        for j, feature in enumerate(
            FEATURES
        )

    }

    fold_records = []

    # 检查五折测试样本覆盖
    seen = np.zeros(
        len(X),
        dtype=int
    )

    # ----------------------------------------------
    # 遍历五折模型
    # ----------------------------------------------

    for fold in range(
        1,
        K_FOLDS + 1
    ):

        model, ckpt, test_idx = (
            load_fold_model(
                fold,
                data_sha,
                len(X)
            )
        )

        seen[test_idx] += 1

        # 当前折留出测试集
        X_test = X[
            test_idx
        ].copy()

        n_test = len(X_test)

        print(
            f"\nFold {fold}/{K_FOLDS}"
            f"，测试样本：{n_test}",
            flush=True
        )

        # ------------------------------------------
        # 遍历13个输入变量
        # ------------------------------------------

        for j, feature in enumerate(
            FEATURES
        ):

            grid = grids[
                feature
            ]

            # 构造所有网格对应的测试输入
            # 与逐个网格点替换变量完全等价
            X_changed = np.tile(
                X_test,
                (len(grid), 1)
            )

            # 每个网格值对应一批测试样本
            X_changed[:, j] = np.repeat(
                grid,
                n_test
            )

            # 一次计算当前变量所有网格的预测
            pred = predict_original_units(
                model,
                ckpt,
                X_changed
            )

            # 恢复维度：
            # 网格点数 × 测试样本数 × 10个输出
            pred = pred.reshape(
                len(grid),
                n_test,
                len(TARGETS)
            )

            # 对留出测试样本求平均
            means = pred.mean(
                axis=1
            )

            # --------------------------------------
            # 保存当前折的PDP
            # --------------------------------------

            for k, x_value in enumerate(
                grid
            ):

                for t, target in enumerate(
                    TARGETS
                ):

                    fold_records.append({

                        "Fold": fold,

                        "Target": target,

                        "Feature": feature,

                        "X_Value": float(
                            x_value
                        ),

                        "PDP_Fold_Mean": float(
                            means[k, t]
                        ),

                        "N_HeldOut": n_test

                    })

            print(
                f"已完成 {j+1}/"
                f"{len(FEATURES)}：{feature}",
                flush=True
            )

        del model

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

    # ----------------------------------------------
    # 检查五折覆盖完整性
    # ----------------------------------------------

    if not np.all(
        seen == 1
    ):

        raise RuntimeError(
            "五折测试样本没有恰好覆盖"
            "全部样本一次。"
        )

    # 五折明细
    fold_df = pd.DataFrame(
        fold_records
    )

    # ----------------------------------------------
    # 按测试样本数加权汇总
    # ----------------------------------------------

    records = []

    grouped = fold_df.groupby(
        [
            "Target",
            "Feature",
            "X_Value"
        ],
        sort=False
    )

    for (
        target,
        feature,
        x_value
    ), sub in grouped:

        values = sub[
            "PDP_Fold_Mean"
        ].to_numpy(dtype=float)

        weights = sub[
            "N_HeldOut"
        ].to_numpy(dtype=float)

        # 按留出测试样本数加权平均
        pdp_mean = np.average(
            values,
            weights=weights
        )

        # 折间标准差
        pdp_sd = (
            np.std(
                values,
                ddof=1
            )
            if len(values) > 1
            else 0.0
        )

        records.append({

            "Target": target,

            "Feature": feature,

            "X_Value": float(
                x_value
            ),

            "PDP_Mean": float(
                pdp_mean
            ),

            "PDP_Fold_SD": float(
                pdp_sd
            ),

            "N_Folds": len(
                values
            ),

            "N_Total_HeldOut": int(
                weights.sum()
            )

        })

    pdp_df = pd.DataFrame(
        records
    )

    # ----------------------------------------------
    # 按变量顺序排序
    # ----------------------------------------------

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
        [
            "Target",
            "Feature",
            "X_Value"
        ]
    ).reset_index(
        drop=True
    )

    pdp_df["Target"] = (
        pdp_df["Target"]
        .astype(str)
    )

    pdp_df["Feature"] = (
        pdp_df["Feature"]
        .astype(str)
    )

    return pdp_df, fold_df


# ==================================================
# 14. PDP曲线转换为响应热图数据
# ==================================================

def make_response_matrix(
    pdp_df,
    target
):

    target_df = pdp_df[
        pdp_df["Target"] == target
    ]

    # 每个变量的P5-P95范围相对位置
    positions = np.linspace(
        0,
        100,
        HEATMAP_POINTS
    )

    matrix_rows = []

    raw_rows = []

    long_records = []

    range_records = []

    # ----------------------------------------------
    # 遍历13个输入变量
    # ----------------------------------------------

    for feature in FEATURES:

        d = target_df[
            target_df["Feature"] == feature
        ].sort_values(
            "X_Value"
        )

        if d.empty:

            raise ValueError(
                f"缺少PDP：{target}/{feature}"
            )

        x = d[
            "X_Value"
        ].to_numpy(dtype=float)

        y = d[
            "PDP_Mean"
        ].to_numpy(dtype=float)

        # 去除重复X
        xu, ix = np.unique(
            x,
            return_index=True
        )

        yu = y[ix]

        # ------------------------------------------
        # 计算统一横坐标下的PDP
        # ------------------------------------------

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

            baseline = float(
                yu[0]
            )

        else:

            # 计算真实变量取值
            actual_x = (
                xu[0]
                + (xu[-1] - xu[0])
                * positions / 100.0
            )

            # PDP线性插值
            raw = np.interp(
                actual_x,
                xu,
                yu
            )

            # 取P5与P95网格端点的中间位置作为基准
            midpoint = (
                xu[0] + xu[-1]
            ) / 2.0

            baseline = float(
                np.interp(
                    midpoint,
                    xu,
                    yu
                )
            )

        # ------------------------------------------
        # 计算PDP响应变化量
        # ------------------------------------------

        delta = raw - baseline

        matrix_rows.append(
            delta
        )

        raw_rows.append(
            raw
        )

        range_records.append({

            "Target": target,

            "Feature": feature,

            "P5_Grid_Value": float(
                xu[0]
            ),

            "P95_Grid_Value": float(
                xu[-1]
            ),

            "Baseline_At_Range_Midpoint": baseline,

            "PDP_Change_LowToHigh": float(
                raw[-1] - raw[0]
            ),

            "PDP_MaxMinusMin": float(
                raw.max() - raw.min()
            )

        })

        # ------------------------------------------
        # 保存Origin长表数据
        # ------------------------------------------

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

    matrix = np.vstack(
        matrix_rows
    )

    raw_matrix = np.vstack(
        raw_rows
    )

    return (
        matrix,
        raw_matrix,
        pd.DataFrame(long_records),
        pd.DataFrame(range_records)
    )


# ==================================================
# 15. 绘制单输出PDP响应热图
# ==================================================

def draw_one_heatmap(
    target,
    matrix
):

    # 每个输出内部使用统一色标
    scale = float(
        np.abs(matrix).max()
    )

    if scale == 0:

        scale = 1.0

    # 以0为中心
    norm = TwoSlopeNorm(
        vmin=-scale,
        vcenter=0,
        vmax=scale
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 6.6)
    )

    # ----------------------------------------------
    # 绘制热图
    # ----------------------------------------------

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

    # 纵坐标：13个输入
    ax.set_yticks(
        np.arange(
            len(FEATURES)
        )
    )

    ax.set_yticklabels(
        FEATURES,
        fontsize=10
    )

    # 横坐标位置
    tick_idx = [

        0,

        (HEATMAP_POINTS - 1) / 4,

        (HEATMAP_POINTS - 1) / 2,

        3 * (HEATMAP_POINTS - 1) / 4,

        HEATMAP_POINTS - 1

    ]

    ax.set_xticks(
        tick_idx
    )

    ax.set_xticklabels(
        [
            "0",
            "25",
            "50",
            "75",
            "100"
        ]
    )

    ax.set_xlabel(
        "Position within each feature's "
        "P5-P95 value range (%)"
    )

    ax.set_ylabel(
        "Input feature"
    )

    ax.set_title(
        "ResNet-MLP five-fold PDP response: "
        + target,
        pad=12
    )

    # 50%基准位置
    ax.axvline(
        (HEATMAP_POINTS - 1) / 2,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        alpha=0.75
    )

    # 横向分隔线
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

    # ----------------------------------------------
    # 颜色条
    # ----------------------------------------------

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

    # ----------------------------------------------
    # 保存PNG和PDF
    # ----------------------------------------------

    for ext in (
        "png",
        "pdf"
    ):

        fig.savefig(
            FIG_DIR
            / f"PDP_Response_Heatmap_{target}.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# ==================================================
# 16. 绘制10输出汇总热图
# ==================================================

def draw_overview(
    matrices
):

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(22, 10),
        sharex=True,
        sharey=True
    )

    # 汇总图对每个输出分别归一化
    norm = TwoSlopeNorm(
        vmin=-1,
        vcenter=0,
        vmax=1
    )

    image = None

    for i, target in enumerate(
        TARGETS
    ):

        ax = axes.flat[i]

        matrix = matrices[
            target
        ]

        # 当前输出内部归一化
        scale = max(
            float(
                np.abs(matrix).max()
            ),
            1e-12
        )

        normalized_matrix = (
            matrix / scale
        )

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

        ax.set_xticks(
            [
                0,
                (HEATMAP_POINTS - 1) / 2,
                HEATMAP_POINTS - 1
            ]
        )

        ax.set_xticklabels(
            [
                "0",
                "50",
                "100"
            ],
            fontsize=8
        )

        ax.set_yticks(
            np.arange(
                len(FEATURES)
            )
        )

        if i % 5 == 0:

            ax.set_yticklabels(
                FEATURES,
                fontsize=8
            )

        # 中间基准位置
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

    # ----------------------------------------------
    # 汇总图统一颜色条
    # ----------------------------------------------

    cax = fig.add_axes(
        [
            0.915,
            0.20,
            0.012,
            0.60
        ]
    )

    cbar = fig.colorbar(
        image,
        cax=cax
    )

    cbar.set_label(
        "Within-target normalized Delta PDP "
        "(unitless)"
    )

    # ----------------------------------------------
    # 保存
    # ----------------------------------------------

    for ext in (
        "png",
        "pdf"
    ):

        fig.savefig(
            FIG_DIR
            / f"PDP_Response_Heatmap_10Targets.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# ==================================================
# 17. 导出完整PDP及热图Excel
# ==================================================

def save_all_excels(
    pdp_df,
    fold_df,
    data_sha,
    matrices,
    raw_matrices,
    response_long,
    ranges
):

    # ----------------------------------------------
    # 设置说明
    # ----------------------------------------------

    settings = pd.DataFrame(
        [

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
                str(
                    int(
                        pdp_df[
                            "N_Total_HeldOut"
                        ].max()
                    )
                )
            ),

            (
                "PDP_method",
                "1D brute-force PDP on each fold's "
                "held-out test samples"
            ),

            (
                "Aggregation",
                "Five folds weighted by held-out "
                "sample counts"
            ),

            (
                "Fold_SD",
                "SD across fold-level PDP estimates; "
                "NOT 95% CI"
            ),

            (
                "Grid",
                f"P{GRID_PERCENTILE_LOW}-"
                f"P{GRID_PERCENTILE_HIGH}"
            ),

            (
                "Heatmap_x_axis",
                "Linear position from P5 to P95 grid "
                "endpoints, not empirical percentile"
            ),

            (
                "Heatmap_response",
                "PDP(x) minus PDP at "
                "(P5+P95)/2 value"
            ),

            (
                "Heatmap_color_scale",
                "Symmetric around 0, independently "
                "scaled per output"
            ),

            (
                "Overview",
                "Each output divided by its own "
                "maximum absolute Delta PDP"
            ),

            (
                "Limitations",
                "Predictive, not causal; "
                "correlated/compositional inputs "
                "can make unrealistic combinations"
            )

        ],
        columns=[
            "Setting",
            "Value"
        ]
    )

    # ----------------------------------------------
    # 保存完整PDP数据
    # ----------------------------------------------

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

        # 每个输出一个工作表
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

    # ----------------------------------------------
    # 热图横坐标
    # ----------------------------------------------

    positions = np.linspace(
        0,
        100,
        HEATMAP_POINTS
    )

    pos_cols = [
        f"Pos_{p:.1f}"
        for p in positions
    ]

    # ----------------------------------------------
    # 保存Origin热图数据
    # ----------------------------------------------

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

        # 保存10个输出各自的矩阵
        for target in TARGETS:

            # --------------------------------------
            # Delta PDP矩阵
            # --------------------------------------

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
                sheet_name=(
                    target + "_Delta"
                )[:31],
                index=False
            )

            # --------------------------------------
            # 原始PDP均值矩阵
            # --------------------------------------

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
                sheet_name=(
                    target + "_Raw"
                )[:31],
                index=False
            )

        settings.to_excel(
            writer,
            sheet_name="Settings",
            index=False
        )


# ==================================================
# 18. 主程序
# ==================================================

def main():

    # 创建输出文件夹
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=" * 64
    )

    print(
        "ResNet-MLP 五折PDP重算与响应热图"
    )

    print(
        "计算设备：",
        DEVICE
    )

    print(
        "直接读取原始数据及5个模型。"
    )

    print(
        "不需要之前的PDP Excel。"
    )

    print(
        "=" * 64,
        flush=True
    )

    # ==============================================
    # Step 1：读取原始数据
    # ==============================================

    X, y, data_sha = load_data()

    # ==============================================
    # Step 2：重新计算五折PDP
    # ==============================================

    print(
        "\n--- 正在重新计算五折PDP ---"
    )

    pdp_df, fold_df = (
        recompute_five_fold_pdp(
            X,
            data_sha
        )
    )

    # ==============================================
    # Step 3：构建响应热图
    # ==============================================

    matrices = {}

    raw_matrices = {}

    all_long = []

    all_ranges = []

    print(
        "\n--- 正在绘制PDP响应热图 ---"
    )

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

        all_long.append(
            response_long
        )

        all_ranges.append(
            ranges
        )

        # 绘制当前输出的响应热图
        draw_one_heatmap(
            target,
            matrix
        )

        print(
            "已绘制：",
            target,
            flush=True
        )

    # ==============================================
    # Step 4：10输出汇总热图
    # ==============================================

    if DRAW_10_TARGET_OVERVIEW:

        print(
            "\n--- 正在绘制10输出汇总热图 ---"
        )

        draw_overview(
            matrices
        )

    # ==============================================
    # Step 5：导出Excel
    # ==============================================

    print(
        "\n--- 正在导出Origin数据 ---"
    )

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

    # ==============================================
    # Step 6：运行完成
    # ==============================================

    print(
        "\n" + "=" * 64
    )

    print(
        "全部完成！"
    )

    print(
        "完整五折PDP Excel：",
        PDP_EXCEL
    )

    print(
        "热图Origin Excel：",
        HEATMAP_EXCEL
    )

    print(
        "热图文件夹：",
        FIG_DIR
    )

    print(
        "每张热图包含13个输入变量。"
    )

    print(
        "蓝色表示低于中点基准，"
        "红色表示高于中点基准。"
    )

    print(
        "=" * 64
    )


# ==================================================
# 19. 程序入口
# ==================================================

if __name__ == "__main__":

    main()
