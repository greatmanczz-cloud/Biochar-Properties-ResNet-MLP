
# -*- coding: utf-8 -*-

"""
ResNet-MLP 与 MLP 残差连接消融实验

修正内容：
1. 先训练ResNet-MLP的完整五折
2. 再训练普通MLP的完整五折
3. 每种模型开始前仅设置一次随机种子
4. 不在不同Fold之间重置随机种子
5. 恢复原始DataLoader的随机洗牌方式
6. 两模型仅有残差连接的区别

输入：
ETII.xlsx
工作表：MICE_All_Exploratory

13个输入：
Excel G-I、K-T

10个输出：
Excel U-AD

输出：
ResNet_MLP_Ablation_Fixed文件夹

包括：
- 消融对比Excel
- 两模型OOF指标
- 五折指标
- 逐样本预测数据
- R2对比柱状图
- Delta R2图
- 两模型OOF散点图
- 旧版ResNet结果核对

注意：
当前ETII数据可能包含插补输出标签。
该实验用于比较残差连接的贡献，
不能直接替代原始实测标签的独立验证。
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
# 1. 参数设置
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

# random：随机五折
# reference：按文献来源分组五折
CV_TYPE = "random"

DPI = 300


# =====================================================
# 2. 数据文件与输出路径
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_NAME = "ETII.xlsx"

SHEET_NAME = "MICE_All_Exploratory"

INPUT_FILE = BASE_DIR / EXCEL_NAME

OUTPUT_DIR = (
    BASE_DIR / "ResNet_MLP_Ablation_Fixed"
)

# 原始ResNet-MLP结果文件
# 如果存在，程序自动进行R2对照
OLD_RESNET_EXCEL = (
    BASE_DIR
    / "ResNet_ETII_Results"
    / "ResNet_ETII_OOF_Results.xlsx"
)


# =====================================================
# 3. Excel输入输出列
# =====================================================

# Python列索引从0开始

# 13个输入：
# G H I
# K L M N O P Q R S T

INPUT_INDEX = [
    6, 7, 8
] + list(range(10, 20))

# 10个输出：
# U V W X Y Z AA AB AC AD

OUTPUT_INDEX = list(range(20, 30))


# =====================================================
# 4. 输入变量名称
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


# =====================================================
# 5. 输出变量名称
# =====================================================

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

MODELS = (
    "ResNet-MLP",
    "MLP"
)

assert len(FEATURES) == 13
assert len(TARGETS) == 10


# =====================================================
# 6. Matplotlib设置
# =====================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# =====================================================
# 7. 固定随机种子
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
# 8. 读取ETII数据
# =====================================================

def load_data():

    if not INPUT_FILE.is_file():

        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}\n"
            "请把Excel放到Python代码所在文件夹。"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl"
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "Excel至少需要30列。"
        )

    # 前4行为表头
    df = raw.iloc[4:, :].copy()

    # 根据第一列筛选有效样本
    sample_number = pd.to_numeric(
        df.iloc[:, 0],
        errors="coerce"
    )

    df = df.loc[
        sample_number.notna()
    ].reset_index(drop=True)

    if len(df) < K_FOLDS * 3:

        raise ValueError(
            "样本量不足，无法进行五折验证。"
        )

    # 提取13个输入
    X_df = df.iloc[
        :, INPUT_INDEX
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 提取10个输出
    y_df = df.iloc[
        :, OUTPUT_INDEX
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 检查是否有缺失值
    x_valid = np.isfinite(
        X_df.to_numpy(dtype=float)
    ).all()

    y_valid = np.isfinite(
        y_df.to_numpy(dtype=float)
    ).all()

    if not (x_valid and y_valid):

        raise ValueError(
            "13个输入或10个输出中存在NaN/Inf。\n"
            "注意：C-F列的NAN不影响本模型。"
        )

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = y_df.to_numpy(
        dtype=np.float32
    )

    # 样本编号
    sample_ids = (
        df.iloc[:, 0].to_numpy()
    )

    # 文献来源
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

    # 记录数据文件哈希值
    checksum = hashlib.sha256(
        INPUT_FILE.read_bytes()
    ).hexdigest()

    print("\n========== 数据读取 ==========")

    print("输入文件：", INPUT_FILE)

    print("数据SHA256：", checksum)

    print("X.shape：", X.shape)

    print("y.shape：", y.shape)

    print(
        "注意：ETII完整输出包含插补标签，"
        "OOF结果不等同于独立实测标签验证。"
    )

    return (
        X,
        y,
        sample_ids,
        references,
        checksum
    )


# =====================================================
# 9. 残差模块与普通MLP模块
# =====================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        dropout_p,
        use_residual
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

        self.use_residual = use_residual

    def forward(self, x):

        residual = x

        out = self.fc1(x)

        out = self.bn1(out)

        out = torch.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.bn2(out)

        # ==========================================
        # 两种模型唯一的结构差异
        # ==========================================

        if self.use_residual:

            out = out + residual

        out = torch.relu(out)

        return out


# =====================================================
# 10. ResNet-MLP / MLP统一模型
# =====================================================

class HybridResNetMLP(nn.Module):

    def __init__(
        self,
        input_size=13,
        output_size=10,
        hidden_size=128,
        num_blocks=3,
        dropout_p=0.4,
        use_residual=True
    ):

        super().__init__()

        # 输入层
        self.entry_layer = nn.Linear(
            input_size,
            hidden_size
        )

        self.bn_entry = nn.BatchNorm1d(
            hidden_size
        )

        # 三个网络块
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_size,
                    dropout_p,
                    use_residual
                )
                for _ in range(num_blocks)
            ]
        )

        # 输出层
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
# 11. 五折划分
# =====================================================

def make_folds(
    X,
    references
):

    if CV_TYPE == "random":

        kf = KFold(
            n_splits=K_FOLDS,
            shuffle=True,
            random_state=SEED
        )

        return list(
            kf.split(X)
        )

    elif CV_TYPE == "reference":

        if len(set(references)) < K_FOLDS:

            raise ValueError(
                "独立Reference组数少于5。"
            )

        gkf = GroupKFold(
            n_splits=K_FOLDS
        )

        return list(
            gkf.split(
                X,
                groups=references
            )
        )

    else:

        raise ValueError(
            "CV_TYPE只能为random或reference。"
        )


# =====================================================
# 12. 评价指标
# =====================================================

def metrics(
    y_true,
    y_pred
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }


# =====================================================
# 13. 单个模型完整五折训练
# =====================================================

def train_full_fivefold(
    model_name,
    X,
    y,
    folds,
    device
):

    """
    对一个模型连续完成全部五折。

    随机种子仅在当前模型开始时设置一次。

    折与折之间不重置随机种子。
    """

    # 与原始ResNet程序保持一致
    seed_all(SEED)

    use_residual = (
        model_name == "ResNet-MLP"
    )

    # 初始化OOF预测
    oof = np.full(
        y.shape,
        np.nan,
        dtype=np.float32
    )

    fold_rows = []

    history = []

    fold_labels = np.zeros(
        len(X),
        dtype=int
    )

    param_count = None

    # ==============================================
    # 逐折训练
    # ==============================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        folds,
        start=1
    ):

        print(
            f"\n[{model_name}] "
            f"Fold {fold}/{K_FOLDS}"
        )

        print(
            "训练样本：",
            len(train_idx)
        )

        print(
            "测试样本：",
            len(test_idx)
        )

        # ------------------------------------------
        # X标准化
        # ------------------------------------------

        x_scaler = StandardScaler()

        X_train = (
            x_scaler.fit_transform(
                X[train_idx]
            ).astype(np.float32)
        )

        X_test = (
            x_scaler.transform(
                X[test_idx]
            ).astype(np.float32)
        )

        # ------------------------------------------
        # Y标准化
        # ------------------------------------------

        y_scaler = StandardScaler()

        y_train = (
            y_scaler.fit_transform(
                y[train_idx]
            ).astype(np.float32)
        )

        # ------------------------------------------
        # 构建网络
        # 不在这里重新设置随机种子
        # ------------------------------------------

        model = HybridResNetMLP(
            input_size=13,
            output_size=10,
            hidden_size=HIDDEN_SIZE,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P,
            use_residual=use_residual
        ).to(device)

        # 参数数量
        this_count = sum(
            p.numel()
            for p in model.parameters()
        )

        if param_count is None:
            param_count = this_count

        # ------------------------------------------
        # 优化器
        # ------------------------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # 损失函数
        criterion = nn.MSELoss()

        # ------------------------------------------
        # DataLoader
        # 恢复原始ResNet代码的随机方式
        # ------------------------------------------

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
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

        if len(train_loader) == 0:

            raise RuntimeError(
                "DataLoader为空。"
            )

        # ------------------------------------------
        # 开始训练
        # ------------------------------------------

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            model.train()

            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:

                batch_X = batch_X.to(device)

                batch_y = batch_y.to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(batch_X)

                loss = criterion(
                    outputs,
                    batch_y
                )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            # 打印训练日志
            if (
                epoch == 1
                or epoch % 200 == 0
                or epoch == EPOCHS
            ):

                mean_loss = (
                    epoch_loss / len(train_loader)
                )

                print(
                    f"{model_name} | "
                    f"Fold {fold} | "
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Loss={mean_loss:.6f}",
                    flush=True
                )

                history.append({
                    "Model": model_name,
                    "Fold": fold,
                    "Epoch": epoch,
                    "Train_MSE_scaled": mean_loss
                })

        # ------------------------------------------
        # 测试集OOF预测
        # ------------------------------------------

        model.eval()

        with torch.no_grad():

            test_tensor = (
                torch.from_numpy(
                    X_test
                ).to(device)
            )

            pred_scaled = (
                model(test_tensor)
                .cpu()
                .numpy()
            )

        # 反标准化
        pred_original = (
            y_scaler.inverse_transform(
                pred_scaled
            )
        )

        # 保存OOF
        oof[test_idx] = pred_original

        fold_labels[test_idx] = fold

        # ------------------------------------------
        # 当前折性能
        # ------------------------------------------

        fold_r2s = []

        for j, target in enumerate(
            TARGETS
        ):

            score = metrics(
                y[test_idx, j],
                pred_original[:, j]
            )

            fold_r2s.append(
                score["R2"]
            )

            fold_rows.append({
                "Model": model_name,
                "Fold": fold,
                "Target": target,
                "N_test": len(test_idx),
                **score
            })

        print(
            f"{model_name} Fold {fold} "
            f"平均R²={np.mean(fold_r2s):.4f}",
            flush=True
        )

        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # ----------------------------------------------
    # 检查OOF预测完整性
    # ----------------------------------------------

    if not np.isfinite(oof).all():

        raise RuntimeError(
            f"{model_name}存在未完成的OOF预测。"
        )

    fold_df = pd.DataFrame(
        fold_rows
    )

    history_df = pd.DataFrame(
        history
    )

    return (
        oof,
        fold_labels,
        fold_df,
        history_df,
        param_count
    )


# =====================================================
# 14. 绘制R2比较图
# =====================================================

def draw_bar_charts(comparison):

    idx = np.arange(
        len(TARGETS)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.bar(
        idx - width / 2,
        comparison["ResNet_R2"],
        width,
        color="#4C78A8",
        label="ResNet-MLP"
    )

    ax.bar(
        idx + width / 2,
        comparison["MLP_R2"],
        width,
        color="#D97757",
        label="MLP (without shortcut)"
    )

    ax.set_xticks(idx)

    ax.set_xticklabels(
        TARGETS,
        rotation=38,
        ha="right"
    )

    ax.set_ylabel(
        "5-fold OOF R²"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):

        fig.savefig(
            OUTPUT_DIR
            / f"Ablation_R2_Comparison.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)

    # ----------------------------------------------
    # Delta R2
    # ----------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    delta = comparison[
        "Delta_R2_ResNet_minus_MLP"
    ].to_numpy()

    colors = np.where(
        delta >= 0,
        "#4C78A8",
        "#D97757"
    )

    ax.bar(
        idx,
        delta,
        color=colors
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xticks(idx)

    ax.set_xticklabels(
        TARGETS,
        rotation=38,
        ha="right"
    )

    ax.set_ylabel(
        "Delta R² = ResNet-MLP - MLP"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):

        fig.savefig(
            OUTPUT_DIR
            / f"Ablation_Delta_R2.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# =====================================================
# 15. 绘制OOF散点图
# =====================================================

def draw_oof_scatter(
    y,
    predictions,
    model_name,
    result_rows
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(16, 18)
    )

    axes = axes.ravel()

    for j, target in enumerate(TARGETS):

        ax = axes[j]

        true_values = y[:, j]

        pred_values = predictions[:, j]

        color = (
            "#4C78A8"
            if model_name == "ResNet-MLP"
            else "#D97757"
        )

        ax.scatter(
            true_values,
            pred_values,
            s=15,
            alpha=0.55,
            color=color,
            edgecolors="none"
        )

        low = float(
            min(
                true_values.min(),
                pred_values.min()
            )
        )

        high = float(
            max(
                true_values.max(),
                pred_values.max()
            )
        )

        pad = max(
            0.04 * (high - low),
            1e-6
        )

        limits = [
            low - pad,
            high + pad
        ]

        ax.plot(
            limits,
            limits,
            "--",
            color="black",
            linewidth=1
        )

        ax.set_xlim(limits)

        ax.set_ylim(limits)

        ax.set_xlabel(
            "ETII value"
        )

        ax.set_ylabel(
            "OOF prediction"
        )

        ax.set_title(
            f"{target} "
            f"R²={result_rows[j]['R2']:.3f}"
        )

        ax.grid(
            alpha=0.15
        )

    for ax in axes[len(TARGETS):]:

        ax.set_visible(False)

    fig.suptitle(
        f"{model_name}: "
        "5-fold OOF on ETII-completed targets",
        fontsize=14
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    safe_name = (
        "ResNet"
        if model_name == "ResNet-MLP"
        else "MLP"
    )

    for ext in ("png", "pdf"):

        fig.savefig(
            OUTPUT_DIR
            / f"OOF_Scatter_{safe_name}.{ext}",
            dpi=DPI,
            bbox_inches="tight"
        )

    plt.close(fig)


# =====================================================
# 16. 与旧版ResNet结果对比
# =====================================================

def compare_to_old_resnet(
    new_resnet_df
):

    if not OLD_RESNET_EXCEL.is_file():

        return pd.DataFrame([{
            "Status": "旧ResNet结果Excel未找到",
            "Note": str(OLD_RESNET_EXCEL)
        }])

    try:

        old = pd.read_excel(
            OLD_RESNET_EXCEL,
            sheet_name="OOF_Metrics"
        )

        if not {
            "Target",
            "OOF_R2"
        }.issubset(old.columns):

            raise ValueError(
                "旧文件缺少Target或OOF_R2列"
            )

        joined = old[
            ["Target", "OOF_R2"]
        ].merge(
            new_resnet_df[
                ["Target", "R2"]
            ],
            on="Target",
            how="inner"
        )

        joined = joined.rename(
            columns={
                "OOF_R2": "Old_ResNet_R2",
                "R2": "New_ResNet_R2"
            }
        )

        joined[
            "Difference_New_minus_Old"
        ] = (
            joined["New_ResNet_R2"]
            - joined["Old_ResNet_R2"]
        )

        return joined

    except Exception as exc:

        return pd.DataFrame([{
            "Status": "无法读取旧文件",
            "Note": str(exc)
        }])


# =====================================================
# 17. 主函数
# =====================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    seed_all(SEED)

    print("=" * 65)

    print(
        "ResNet-MLP vs MLP 残差消融实验"
    )

    print("=" * 65)

    # ----------------------------------------------
    # 读取数据
    # ----------------------------------------------

    (
        X,
        y,
        sample_ids,
        references,
        checksum
    ) = load_data()

    folds = make_folds(
        X,
        references
    )

    # ----------------------------------------------
    # GPU配置
    # ----------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n计算设备：",
        device
    )

    if device.type == "cuda":

        print(
            "GPU型号：",
            torch.cuda.get_device_name(0)
        )

    print(
        f"\n训练参数："
        f"Epoch={EPOCHS}, "
        f"Hidden={HIDDEN_SIZE}, "
        f"Blocks={NUM_BLOCKS}, "
        f"Batch={BATCH_SIZE}, "
        f"Seed={SEED}"
    )

    # ----------------------------------------------
    # 初始化结果容器
    # ----------------------------------------------

    oof_results = {}

    metric_rows = []

    fold_dfs = []

    history_dfs = []

    counts = {}

    fold_labels = None

    # ==============================================
    # 关键修正：
    # 先训练ResNet完整5折
    # 再训练MLP完整5折
    # ==============================================

    for model_name in MODELS:

        print("\n" + "=" * 65)

        print(
            "开始训练：",
            model_name
        )

        print("=" * 65)

        (
            oof,
            labels,
            fold_df,
            history_df,
            n_parameters
        ) = train_full_fivefold(
            model_name,
            X,
            y,
            folds,
            device
        )

        oof_results[
            model_name
        ] = oof

        counts[
            model_name
        ] = n_parameters

        fold_labels = labels

        fold_dfs.append(
            fold_df
        )

        history_dfs.append(
            history_df
        )

        # ------------------------------------------
        # 计算该模型10个输出的OOF指标
        # ------------------------------------------

        model_metrics = []

        for j, target in enumerate(
            TARGETS
        ):

            score = metrics(
                y[:, j],
                oof[:, j]
            )

            model_metrics.append(
                score
            )

            metric_rows.append({
                "Model": model_name,
                "Target": target,
                "N": len(y),
                **score
            })

        # 绘制OOF散点图
        draw_oof_scatter(
            y,
            oof,
            model_name,
            model_metrics
        )

    # ==============================================
    # 18. 检查参数量是否一致
    # ==============================================

    if (
        counts["ResNet-MLP"]
        != counts["MLP"]
    ):

        raise AssertionError(
            "两种模型参数量不一致！"
        )

    # ----------------------------------------------
    # 整理各项指标
    # ----------------------------------------------

    oof_df = pd.DataFrame(
        metric_rows
    )

    fold_df = pd.concat(
        fold_dfs,
        ignore_index=True
    )

    training_df = pd.concat(
        history_dfs,
        ignore_index=True
    )

    fold_means = (
        fold_df.groupby(
            ["Model", "Fold"],
            as_index=False
        )["R2"]
        .mean()
    )

    fold_means = fold_means.rename(
        columns={
            "R2": "Mean_R2_10_targets"
        }
    )

    # ----------------------------------------------
    # 分别取出两种模型的OOF指标
    # ----------------------------------------------

    resnet_df = oof_df[
        oof_df["Model"] == "ResNet-MLP"
    ].copy()

    mlp_df = oof_df[
        oof_df["Model"] == "MLP"
    ].copy()

    comparison = pd.DataFrame({

        "Target": TARGETS,

        "ResNet_R2":
            resnet_df["R2"].to_numpy(),

        "MLP_R2":
            mlp_df["R2"].to_numpy(),

        "ResNet_RMSE":
            resnet_df["RMSE"].to_numpy(),

        "MLP_RMSE":
            mlp_df["RMSE"].to_numpy(),

        "ResNet_MAE":
            resnet_df["MAE"].to_numpy(),

        "MLP_MAE":
            mlp_df["MAE"].to_numpy()
    })

    # 计算R2差值
    comparison[
        "Delta_R2_ResNet_minus_MLP"
    ] = (
        comparison["ResNet_R2"]
        - comparison["MLP_R2"]
    )

    # 绘制对比图
    draw_bar_charts(
        comparison
    )

    # ==============================================
    # 19. 汇总结果
    # ==============================================

    summary = pd.DataFrame([{

        "Model": name,

        "Parameters": counts[name],

        "Mean_OOF_R2_10_targets": float(
            oof_df.loc[
                oof_df["Model"] == name,
                "R2"
            ].mean()
        ),

        "Mean_OOF_RMSE_unscaled": float(
            oof_df.loc[
                oof_df["Model"] == name,
                "RMSE"
            ].mean()
        ),

        "Epochs": EPOCHS,

        "Dataset": INPUT_FILE.name,

        "CV_type": CV_TYPE

    } for name in MODELS])

    # ----------------------------------------------
    # 逐样本OOF预测
    # ----------------------------------------------

    detail = pd.DataFrame({

        "Index_No": sample_ids,

        "Reference": references,

        "Fold": fold_labels
    })

    for j, target in enumerate(
        TARGETS
    ):

        detail[
            f"{target}_ETII_Value"
        ] = y[:, j]

        detail[
            f"{target}_ResNet_OOF"
        ] = (
            oof_results["ResNet-MLP"][:, j]
        )

        detail[
            f"{target}_MLP_OOF"
        ] = (
            oof_results["MLP"][:, j]
        )

    # ----------------------------------------------
    # 旧版ResNet结果核对
    # ----------------------------------------------

    old_comparison = compare_to_old_resnet(
        resnet_df
    )

    # ----------------------------------------------
    # 参数记录
    # ----------------------------------------------

    settings = pd.DataFrame(
        [
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
            (
                "Randomness",
                "Reset seed once before "
                "each complete five-fold run"
            ),
            (
                "Labels",
                "ETII imputed targets; "
                "not independent measured-label validation"
            )
        ],
        columns=[
            "Parameter",
            "Value"
        ]
    )

    # ==============================================
    # 20. 保存Excel
    # ==============================================

    output_path = (
        OUTPUT_DIR
        / "Ablation_ResNet_vs_MLP_Fixed.xlsx"
    )

    try:

        with pd.ExcelWriter(
            output_path,
            engine="openpyxl"
        ) as writer:

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

            comparison.to_excel(
                writer,
                sheet_name="R2_Comparison",
                index=False
            )

            oof_df.to_excel(
                writer,
                sheet_name="OOF_PerTarget",
                index=False
            )

            fold_df.to_excel(
                writer,
                sheet_name="Fold_PerTarget",
                index=False
            )

            fold_means.to_excel(
                writer,
                sheet_name="Fold_Mean_R2",
                index=False
            )

            detail.to_excel(
                writer,
                sheet_name="OOF_Predictions",
                index=False
            )

            training_df.to_excel(
                writer,
                sheet_name="Training_Log",
                index=False
            )

            old_comparison.to_excel(
                writer,
                sheet_name="Vs_Old_ResNet",
                index=False
            )

            settings.to_excel(
                writer,
                sheet_name="Settings",
                index=False
            )

    except PermissionError as exc:

        raise PermissionError(
            f"无法写入：{output_path}\n"
            "请关闭Excel中已打开的同名文件。"
        ) from exc

    # ==============================================
    # 21. 打印结果
    # ==============================================

    print(
        "\n========== 残差消融结果 =========="
    )

    print(
        comparison[
            [
                "Target",
                "ResNet_R2",
                "MLP_R2",
                "Delta_R2_ResNet_minus_MLP"
            ]
        ].round(4).to_string(
            index=False
        )
    )

    print(
        "\n10个输出平均OOF R²："
    )

    print(
        summary[
            [
                "Model",
                "Parameters",
                "Mean_OOF_R2_10_targets"
            ]
        ].round(4).to_string(
            index=False
        )
    )

    if (
        "Difference_New_minus_Old"
        in old_comparison.columns
    ):

        print(
            "\n修正版与旧版ResNet的R²差异："
        )

        print(
            old_comparison.round(6).to_string(
                index=False
            )
        )

    print(
        "\n结果已保存：",
        output_path
    )

    print(
        "\n注意：更换ETII文件、PyTorch版本、"
        "GPU驱动或计算设备，"
        "仍可能使结果发生变化。"
    )


# =====================================================
# 22. 程序入口
# =====================================================

if __name__ == "__main__":

    main()
