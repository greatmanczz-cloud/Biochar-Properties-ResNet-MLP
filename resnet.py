
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import copy
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import (
    KFold,
    GroupKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# =====================================================
# 1. 基本设置
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

# 交叉验证方式
# random：随机五折交叉验证
# reference：按照文献来源分组交叉验证
CV_TYPE = "random"

# 是否进行SHAP分析
RUN_SHAP = True

# SHAP背景样本数量
SHAP_BACKGROUND_SIZE = 100

# SHAP解释样本数量
SHAP_EXPLAIN_SIZE = 80

# 图片分辨率
DPI = 300


# =====================================================
# 2. 输入文件与输出路径
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_NAME = "ETII.xlsx"

SHEET_NAME = "MICE_All_Exploratory"

INPUT_FILE = BASE_DIR / EXCEL_NAME

OUTPUT_DIR = BASE_DIR / "ResNet_ETII_Results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 3. Excel输入输出列
# =====================================================

# Python从0开始计数
#
# 输入：
# G H I
# K L M N O P Q R S T
#
# 输出：
# U V W X Y Z AA AB AC AD

INPUT_INDEX = [
    6, 7, 8,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19
]

OUTPUT_INDEX = list(
    range(20, 30)
)


# =====================================================
# 4. 13个输入变量
# =====================================================

feature_cols = [
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


# =====================================================
# 5. 10个输出变量
# =====================================================

target_cols = [
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


assert len(feature_cols) == 13
assert len(target_cols) == 10


# =====================================================
# 6. 固定随机种子
# =====================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


# =====================================================
# 7. GPU配置
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 65)

print("ResNet-MLP：ETII数据集")

print("=" * 65)

print("计算设备：", device)

if torch.cuda.is_available():

    print(
        "GPU型号：",
        torch.cuda.get_device_name(0)
    )


# =====================================================
# 8. Matplotlib绘图设置
# =====================================================

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "axes.unicode_minus": False,

    "pdf.fonttype": 42,

    "ps.fonttype": 42

})


# =====================================================
# 9. 读取Excel
# =====================================================

def load_data():

    print("\n--- 1. 开始读取Excel数据 ---")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}\n"
            "请把Excel放在Python代码同一文件夹。"
        )

    # 不把前几行当作变量名
    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl"
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "Excel列数不足30列，请检查数据结构。"
        )

    # 跳过前4行表头
    df = raw.iloc[4:, :].copy()

    # 根据第一列样本编号识别有效行
    sample_number = pd.to_numeric(
        df.iloc[:, 0],
        errors="coerce"
    )

    df = df.loc[
        sample_number.notna()
    ].reset_index(drop=True)

    # 提取13个输入
    X_df = df.iloc[
        :, INPUT_INDEX
    ].copy()

    X_df.columns = feature_cols

    # 提取10个输出
    y_df = df.iloc[
        :, OUTPUT_INDEX
    ].copy()

    y_df.columns = target_cols

    # 转换为数值
    X_df = X_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    y_df = y_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 检查缺失值
    x_missing = X_df.isna().sum().sum()

    y_missing = y_df.isna().sum().sum()

    print("样本数量：", len(df))

    print("输入变量数量：", X_df.shape[1])

    print("输出变量数量：", y_df.shape[1])

    print("输入缺失值：", int(x_missing))

    print("输出缺失值：", int(y_missing))

    if x_missing > 0 or y_missing > 0:

        raise ValueError(
            "所选数据不是完整填补数据，"
            "请检查工作表MICE_All_Exploratory。"
        )

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = y_df.to_numpy(
        dtype=np.float32
    )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):

        raise ValueError(
            "数据中存在NaN或无穷大。"
        )

    # 获取样本编号
    sample_ids = df.iloc[:, 0].tolist()

    # 获取文献来源
    if df.shape[1] > 30:

        references = (
            df.iloc[:, 30]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .to_numpy()
        )

    else:

        references = np.array(
            ["Unknown"] * len(df)
        )

    print("\nX.shape =", X.shape)

    print("y.shape =", y.shape)

    print("\n13个输入变量：")

    print(feature_cols)

    print("\n10个输出变量：")

    print(target_cols)

    return X, y, sample_ids, references


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

        # 不使用原地+=，避免SHAP梯度异常
        out = out + residual

        out = torch.relu(out)

        return out


# =====================================================
# 11. ResNet-MLP模型
# =====================================================

class HybridResNetMLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_blocks,
        dropout_p
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

        x = self.output_layer(x)

        return x


# =====================================================
# 12. 用于SHAP：恢复输出原始单位
# =====================================================

class OriginalUnitModel(nn.Module):

    def __init__(
        self,
        model,
        y_scaler
    ):

        super().__init__()

        self.model = model

        self.register_buffer(
            "y_mean",
            torch.tensor(
                y_scaler.mean_,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "y_scale",
            torch.tensor(
                y_scaler.scale_,
                dtype=torch.float32
            )
        )

    def forward(self, x):

        output = self.model(x)

        return (
            output * self.y_scale
            + self.y_mean
        )


# =====================================================
# 13. 五折交叉验证划分
# =====================================================

def get_folds(X, references):

    if CV_TYPE == "random":

        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=SEED
        )

        folds = list(
            kf.split(X)
        )

    elif CV_TYPE == "reference":

        if len(set(references)) < K_FOLDS:

            raise ValueError(
                "独立文献数量不足以进行五折分组验证。"
            )

        gkf = GroupKFold(
            n_splits=K_FOLDS
        )

        folds = list(
            gkf.split(
                X,
                groups=references
            )
        )

    else:

        raise ValueError(
            "CV_TYPE必须是random或reference"
        )

    return folds


# =====================================================
# 14. 五折交叉验证训练
# =====================================================

def train_cross_validation(
    X,
    y,
    sample_ids,
    references
):

    print("\n--- 2. 开始五折交叉验证 ---")

    folds = get_folds(
        X,
        references
    )

    n_samples = len(X)

    n_targets = len(target_cols)

    # 初始化OOF预测矩阵
    oof_predictions = np.full(
        y.shape,
        np.nan,
        dtype=np.float32
    )

    fold_labels = np.zeros(
        n_samples,
        dtype=int
    )

    fold_results = []

    saved_models = []

    # =================================================
    # 逐折训练
    # =================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(folds, start=1):

        print("\n" + "=" * 65)

        print(
            f"正在训练 Fold {fold}/{K_FOLDS}"
        )

        print(
            f"训练样本：{len(train_idx)}"
        )

        print(
            f"测试样本：{len(test_idx)}"
        )

        print("=" * 65)

        X_train = X[train_idx]

        X_test = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        # ---------------------------------------------
        # X标准化
        # 只在训练集拟合StandardScaler
        # ---------------------------------------------

        x_scaler = StandardScaler()

        X_train_scaled = (
            x_scaler.fit_transform(
                X_train
            )
            .astype(np.float32)
        )

        X_test_scaled = (
            x_scaler.transform(
                X_test
            )
            .astype(np.float32)
        )

        # ---------------------------------------------
        # Y标准化
        # ---------------------------------------------

        y_scaler = StandardScaler()

        y_train_scaled = (
            y_scaler.fit_transform(
                y_train
            )
            .astype(np.float32)
        )

        # ---------------------------------------------
        # 训练数据加载器
        # ---------------------------------------------

        train_dataset = TensorDataset(
            torch.from_numpy(
                X_train_scaled
            ),
            torch.from_numpy(
                y_train_scaled
            )
        )

        # 避免BatchNorm遇到单样本批次
        drop_last = (
            len(train_dataset) % BATCH_SIZE == 1
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=drop_last
        )

        # ---------------------------------------------
        # 创建模型
        # ---------------------------------------------

        model = HybridResNetMLP(
            input_size=13,
            output_size=10,
            hidden_size=HIDDEN_SIZE,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P
        ).to(device)

        # 损失函数
        criterion = nn.MSELoss()

        # 优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # ---------------------------------------------
        # 开始训练
        # ---------------------------------------------

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            model.train()

            epoch_loss = 0.0

            for (
                batch_X,
                batch_y
            ) in train_loader:

                batch_X = batch_X.to(
                    device
                )

                batch_y = batch_y.to(
                    device
                )

                optimizer.zero_grad()

                outputs = model(
                    batch_X
                )

                loss = criterion(
                    outputs,
                    batch_y
                )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            # 每200轮打印一次
            if (
                epoch == 1
                or epoch % 200 == 0
                or epoch == EPOCHS
            ):

                mean_loss = (
                    epoch_loss
                    / max(len(train_loader), 1)
                )

                print(
                    f"Fold {fold} | "
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Loss = {mean_loss:.6f}"
                )

        # ---------------------------------------------
        # 测试集预测
        # ---------------------------------------------

        model.eval()

        with torch.no_grad():

            test_tensor = (
                torch.from_numpy(
                    X_test_scaled
                ).to(device)
            )

            pred_scaled = model(
                test_tensor
            )

            pred_scaled = (
                pred_scaled.cpu().numpy()
            )

        # 反标准化
        y_pred = (
            y_scaler.inverse_transform(
                pred_scaled
            )
        )

        # 保存OOF预测
        oof_predictions[test_idx] = (
            y_pred
        )

        fold_labels[test_idx] = fold

        # ---------------------------------------------
        # 当前折性能
        # ---------------------------------------------

        current_r2 = []

        for j, name in enumerate(
            target_cols
        ):

            true_values = y_test[:, j]

            pred_values = y_pred[:, j]

            r2 = r2_score(
                true_values,
                pred_values
            )

            rmse = np.sqrt(
                mean_squared_error(
                    true_values,
                    pred_values
                )
            )

            mae = mean_absolute_error(
                true_values,
                pred_values
            )

            current_r2.append(r2)

            fold_results.append({

                "Fold": fold,

                "Target": name,

                "N_test": len(test_idx),

                "R2": float(r2),

                "RMSE": float(rmse),

                "MAE": float(mae)

            })

        mean_r2 = float(
            np.mean(current_r2)
        )

        print(
            f"\nFold {fold} 平均R2 = "
            f"{mean_r2:.4f}"
        )

        # ---------------------------------------------
        # 保存本折模型及标准化器
        # ---------------------------------------------

        model_state = {
            key: value.detach().cpu().clone()
            for key, value in
            model.state_dict().items()
        }

        saved_models.append({

            "fold": fold,

            "state": model_state,

            "x_scaler": x_scaler,

            "y_scaler": y_scaler,

            "train_idx": train_idx,

            "test_idx": test_idx,

            "mean_r2": mean_r2

        })

        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # =================================================
    # 15. OOF性能
    # =================================================

    print("\n--- 3. 计算最终OOF性能 ---")

    if not np.isfinite(
        oof_predictions
    ).all():

        raise RuntimeError(
            "存在未生成的OOF预测值"
        )

    final_results = []

    for j, name in enumerate(
        target_cols
    ):

        true_values = y[:, j]

        pred_values = (
            oof_predictions[:, j]
        )

        r2 = r2_score(
            true_values,
            pred_values
        )

        rmse = np.sqrt(
            mean_squared_error(
                true_values,
                pred_values
            )
        )

        mae = mean_absolute_error(
            true_values,
            pred_values
        )

        final_results.append({

            "Target": name,

            "N": len(true_values),

            "OOF_R2": float(r2),

            "OOF_RMSE": float(rmse),

            "OOF_MAE": float(mae)

        })

    final_df = pd.DataFrame(
        final_results
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    print("\n最终OOF结果：")

    print(
        final_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\n10个输出平均OOF R2：",
        round(
            final_df["OOF_R2"].mean(),
            4
        )
    )

    # =================================================
    # 16. 逐样本OOF预测表
    # =================================================

    result_data = pd.DataFrame({

        "Index_No": sample_ids,

        "Reference": references,

        "Fold": fold_labels

    })

    for j, name in enumerate(
        target_cols
    ):

        result_data[
            name + "_ETII_Value"
        ] = y[:, j]

        result_data[
            name + "_OOF_Pred"
        ] = oof_predictions[:, j]

        result_data[
            name + "_Residual"
        ] = (
            y[:, j]
            - oof_predictions[:, j]
        )

    # =================================================
    # 17. 保存Excel
    # =================================================

    excel_path = (
        OUTPUT_DIR
        / "ResNet_ETII_OOF_Results.xlsx"
    )

    mapping = pd.DataFrame({

        "Variable": (
            feature_cols + target_cols
        ),

        "Role": (
            ["Input"] * 13
            + ["Output"] * 10
        ),

        "Excel_Column": (
            ["G", "H", "I"]
            + list("KLMNOPQRST")
            + [
                "U", "V", "W",
                "X", "Y", "Z",
                "AA", "AB",
                "AC", "AD"
            ]
        )

    })

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        final_df.to_excel(
            writer,
            sheet_name="OOF_Metrics",
            index=False
        )

        fold_df.to_excel(
            writer,
            sheet_name="Fold_Metrics",
            index=False
        )

        result_data.to_excel(
            writer,
            sheet_name="OOF_Predictions",
            index=False
        )

        mapping.to_excel(
            writer,
            sheet_name="Variable_Mapping",
            index=False
        )

    print(
        "\nExcel已保存：",
        excel_path
    )

    # =================================================
    # 18. 绘制OOF散点图
    # =================================================

    draw_oof_scatter(
        y,
        oof_predictions,
        final_df
    )

    return saved_models, final_df


# =====================================================
# 19. 绘制10个输出的OOF散点图
# =====================================================

def draw_oof_scatter(
    y,
    predictions,
    final_df
):

    print("\n--- 4. 正在绘制OOF散点图 ---")

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(16, 19)
    )

    axes = axes.flatten()

    for j, name in enumerate(
        target_cols
    ):

        ax = axes[j]

        y_true = y[:, j]

        y_pred = predictions[:, j]

        ax.scatter(
            y_true,
            y_pred,
            s=20,
            alpha=0.6,
            color="#4C78A8",
            edgecolors="none"
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
            "--",
            color="#D97757",
            linewidth=1.5
        )

        ax.set_xlim(limits)

        ax.set_ylim(limits)

        ax.set_xlabel(
            "ETII value",
            fontsize=10
        )

        ax.set_ylabel(
            "OOF prediction",
            fontsize=10
        )

        r2 = final_df.iloc[j][
            "OOF_R2"
        ]

        ax.set_title(
            f"{name}\n"
            f"R² = {r2:.4f}",
            fontsize=12
        )

        ax.grid(
            alpha=0.2,
            linestyle="--"
        )

    # 删除多余的子图
    for k in range(
        len(target_cols),
        len(axes)
    ):

        axes[k].set_visible(False)

    fig.suptitle(
        "ResNet-MLP: 5-fold OOF predictions",
        fontsize=17,
        y=0.995
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    png_path = (
        OUTPUT_DIR
        / "ResNet_ETII_OOF_Scatter.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "ResNet_ETII_OOF_Scatter.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("OOF散点图已保存")


# =====================================================
# 20. 兼容不同SHAP版本
# =====================================================

def standardize_shap_values(
    shap_values,
    n_samples,
    n_features,
    n_outputs
):

    # 旧版本：返回10个二维数组的列表
    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) != n_outputs:

            raise ValueError(
                "SHAP输出数量不正确"
            )

        values = np.stack(
            [
                np.asarray(v)
                for v in shap_values
            ],
            axis=2
        )

    else:

        values = np.asarray(
            shap_values
        )

        # 兼容(输出,样本,特征)
        if values.shape == (
            n_outputs,
            n_samples,
            n_features
        ):

            values = np.moveaxis(
                values,
                0,
                2
            )

    expected_shape = (
        n_samples,
        n_features,
        n_outputs
    )

    if values.shape != expected_shape:

        raise ValueError(
            f"SHAP形状错误：{values.shape}\n"
            f"预期：{expected_shape}"
        )

    return values


# =====================================================
# 21. SHAP分析
# =====================================================

def run_shap_analysis(
    X,
    saved_models
):

    if not RUN_SHAP:

        print(
            "已关闭SHAP分析"
        )

        return

    print("\n--- 5. 开始SHAP分析 ---")

    try:

        import shap

    except ImportError:

        print(
            "没有安装shap，请执行："
            "pip install shap"
        )

        return

    # ---------------------------------------------
    # 选择平均折R2最高的模型
    # 仅用于解释，不作为独立模型选择验证
    # ---------------------------------------------

    best_model_info = max(
        saved_models,
        key=lambda item: item[
            "mean_r2"
        ]
    )

    best_fold = best_model_info[
        "fold"
    ]

    print(
        f"SHAP使用第{best_fold}折模型"
    )

    # ---------------------------------------------
    # 重建网络
    # ---------------------------------------------

    model = HybridResNetMLP(
        input_size=13,
        output_size=10,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        dropout_p=DROPOUT_P
    ).to(device)

    model.load_state_dict(
        best_model_info["state"]
    )

    model.eval()

    x_scaler = best_model_info[
        "x_scaler"
    ]

    y_scaler = best_model_info[
        "y_scaler"
    ]

    # 恢复模型输出的原始单位
    shap_model = OriginalUnitModel(
        model,
        y_scaler
    ).to(device)

    shap_model.eval()

    train_idx = best_model_info[
        "train_idx"
    ]

    test_idx = best_model_info[
        "test_idx"
    ]

    rng = np.random.default_rng(
        SEED
    )

    # ---------------------------------------------
    # 选择背景样本
    # ---------------------------------------------

    bg_idx = rng.choice(
        train_idx,
        size=min(
            SHAP_BACKGROUND_SIZE,
            len(train_idx)
        ),
        replace=False
    )

    # ---------------------------------------------
    # 选择解释样本
    # ---------------------------------------------

    eval_idx = rng.choice(
        test_idx,
        size=min(
            SHAP_EXPLAIN_SIZE,
            len(test_idx)
        ),
        replace=False
    )

    X_background = (
        x_scaler.transform(
            X[bg_idx]
        ).astype(np.float32)
    )

    X_explain = (
        x_scaler.transform(
            X[eval_idx]
        ).astype(np.float32)
    )

    background_tensor = (
        torch.from_numpy(
            X_background
        ).to(device)
    )

    explain_tensor = (
        torch.from_numpy(
            X_explain
        ).to(device)
    )

    print(
        "SHAP背景样本数量：",
        len(bg_idx)
    )

    print(
        "SHAP解释样本数量：",
        len(eval_idx)
    )

    # =================================================
    # 22. 计算SHAP
    # =================================================

    try:

        explainer = shap.DeepExplainer(
            shap_model,
            background_tensor
        )

        raw_shap = explainer.shap_values(
            explain_tensor,
            check_additivity=False
        )

        method_name = "DeepExplainer"

    except Exception as exc:

        warnings.warn(
            f"DeepExplainer出现问题：{exc}，"
            "尝试GradientExplainer"
        )

        try:

            explainer = shap.GradientExplainer(
                shap_model,
                background_tensor
            )

            raw_shap = explainer.shap_values(
                explain_tensor
            )

            method_name = "GradientExplainer"

        except Exception as exc2:

            print(
                "SHAP分析失败：",
                exc2
            )

            print(
                "五折交叉验证和OOF结果已正常保存"
            )

            return

    # 统一SHAP格式
    shap_cube = standardize_shap_values(
        raw_shap,
        n_samples=len(eval_idx),
        n_features=13,
        n_outputs=10
    )

    print(
        "SHAP方法：",
        method_name
    )

    print(
        "SHAP维度：",
        shap_cube.shape
    )

    # =================================================
    # 23. 创建SHAP图片文件夹
    # =================================================

    SHAP_DIR = (
        OUTPUT_DIR / "SHAP_Plots"
    )

    SHAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # SHAP显示原始输入值
    X_display = pd.DataFrame(
        X[eval_idx],
        columns=feature_cols
    )

    shap_importance = []

    shap_data = {}

    # =================================================
    # 24. 循环绘制10个输出的SHAP图
    # =================================================

    for j, target_name in enumerate(
        target_cols
    ):

        print(
            f"正在生成SHAP图：{target_name}"
        )

        # 当前目标对应的SHAP
        shap_values_j = shap_cube[
            :, :, j
        ]

        # 平均绝对SHAP
        mean_abs_shap = np.mean(
            np.abs(shap_values_j),
            axis=0
        )

        # 保存特征重要性
        for i, feature_name in enumerate(
            feature_cols
        ):

            shap_importance.append({

                "Target": target_name,

                "Feature": feature_name,

                "MeanAbsSHAP": float(
                    mean_abs_shap[i]
                )

            })

        # 保存SHAP值表
        shap_data[
            target_name
        ] = pd.DataFrame(
            shap_values_j,
            columns=feature_cols
        )

        # =============================================
        # SHAP蜂群图
        # =============================================

        plt.figure(
            figsize=(10, 7)
        )

        shap.summary_plot(
            shap_values_j,
            X_display,
            plot_type="dot",
            show=False,
            max_display=13
        )

        plt.title(
            f"{target_name} - SHAP Beeswarm",
            fontsize=14
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_DIR
            / f"SHAP_Beeswarm_{target_name}.png",
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close("all")

        # =============================================
        # SHAP特征重要性柱状图
        # =============================================

        order = np.argsort(
            mean_abs_shap
        )

        sorted_names = (
            np.array(feature_cols)[order]
        )

        sorted_values = (
            mean_abs_shap[order]
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            sorted_names,
            sorted_values,
            color="#4C78A8"
        )

        ax.set_xlabel(
            "Mean absolute SHAP value"
        )

        ax.set_title(
            f"{target_name} - SHAP Importance"
        )

        fig.tight_layout()

        fig.savefig(
            SHAP_DIR
            / f"SHAP_Bar_{target_name}.png",
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close(fig)

    # =================================================
    # 25. 导出SHAP Excel
    # =================================================

    shap_excel_path = (
        OUTPUT_DIR
        / "ResNet_ETII_SHAP_Data.xlsx"
    )

    importance_df = pd.DataFrame(
        shap_importance
    )

    with pd.ExcelWriter(
        shap_excel_path,
        engine="openpyxl"
    ) as writer:

        importance_df.to_excel(
            writer,
            sheet_name="SHAP_Importance",
            index=False
        )

        X_display.assign(
            RowIndex=eval_idx
        ).to_excel(
            writer,
            sheet_name="SHAP_InputValues",
            index=False
        )

        for target_name, values_df in (
            shap_data.items()
        ):

            sheet_name = (
                "SHAP_" + target_name[:24]
            )

            values_df.assign(
                RowIndex=eval_idx
            ).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print(
        "\nSHAP Excel已保存：",
        shap_excel_path
    )

    print(
        "SHAP图片已保存到：",
        SHAP_DIR
    )


# =====================================================
# 26. 主函数
# =====================================================

def main():

    print("\n开始执行ResNet-MLP完整流程")

    # 读取数据
    X, y, sample_ids, references = (
        load_data()
    )

    # 五折交叉验证
    saved_models, final_df = (
        train_cross_validation(
            X,
            y,
            sample_ids,
            references
        )
    )

    # SHAP分析
    run_shap_analysis(
        X,
        saved_models
    )

    print("\n" + "=" * 65)

    print("全部运行完成")

    print("=" * 65)

    print(
        "结果输出路径：",
        OUTPUT_DIR
    )



# =====================================================
# 27. 程序运行入口
# =====================================================

if __name__ == "__main__":

    main()
