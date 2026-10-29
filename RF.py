
# -*- coding: utf-8 -*-

"""
Random Forest (RF) 五折OOF预测
数据：ETII(6).xlsx
工作表：MICE_All_Exploratory

13个输入：G-I、K-T
10个输出：U-AD

输出：
RF_ETII_Results文件夹
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ==================================================
# 1. 基本设置
# ==================================================

SEED = 42
K_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ETII.xlsx"
SHEET_NAME = "MICE_All_Exploratory"

OUTPUT_DIR = BASE_DIR / "RF_ETII_Results"

OUTPUT_EXCEL = (
    OUTPUT_DIR / "RF_ETII_5Fold_OOF_Results.xlsx"
)

PLOT_PNG = (
    OUTPUT_DIR / "RF_ETII_5Fold_OOF_Scatter.png"
)

PLOT_PDF = (
    OUTPUT_DIR / "RF_ETII_5Fold_OOF_Scatter.pdf"
)

DPI = 300


# ==================================================
# 2. RF超参数
# ==================================================

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 16,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": SEED
}


# ==================================================
# 3. Excel列位置
# ==================================================

INPUT_INDEX = [
    6, 7, 8
] + list(range(10, 20))

OUTPUT_INDEX = list(range(20, 30))


# ==================================================
# 4. 输入输出名称
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


# ==================================================
# 5. 绘图设置
# ==================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ==================================================
# 6. 读取ETII数据
# ==================================================

def load_data():

    print("\n--- 1. 正在读取ETII数据 ---")

    if not INPUT_FILE.is_file():

        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl"
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "Excel列数不足30列。"
        )

    # 前4行为表头
    body = raw.iloc[4:, :].copy()

    # 根据A列筛选有效记录
    sample_no = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    body = body.loc[
        sample_no.notna()
    ].reset_index(drop=True)

    # 提取13个输入
    X_df = body.iloc[
        :, INPUT_INDEX
    ].copy()

    X_df.columns = FEATURES

    X_df = X_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 提取10个输出
    y_df = body.iloc[
        :, OUTPUT_INDEX
    ].copy()

    y_df.columns = TARGETS

    y_df = y_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X_df.to_numpy(
        dtype=np.float64
    )

    y = y_df.to_numpy(
        dtype=np.float64
    )

    # 完整性检查
    if not np.isfinite(X).all():

        raise ValueError(
            "13个输入变量存在NaN或Inf。"
        )

    if not np.isfinite(y).all():

        raise ValueError(
            "10个输出变量存在NaN或Inf。"
        )

    sample_ids = (
        body.iloc[:, 0].to_numpy()
    )

    if body.shape[1] > 30:

        references = (
            body.iloc[:, 30]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .to_numpy()
        )

    else:

        references = np.array(
            ["Unknown"] * len(body)
        )

    print("有效样本：", len(body))
    print("X.shape：", X.shape)
    print("y.shape：", y.shape)

    return X, y, sample_ids, references


# ==================================================
# 7. 计算指标
# ==================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "R2": float(
            r2_score(y_true, y_pred)
        ),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "MSE": float(mse)
    }


# ==================================================
# 8. OOF散点图
# ==================================================

def plot_oof(
    y,
    predictions,
    final_df
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 18)
    )

    axes = axes.ravel()

    for j, target in enumerate(TARGETS):

        ax = axes[j]

        actual = y[:, j]
        pred = predictions[:, j]

        ax.scatter(
            actual,
            pred,
            s=18,
            alpha=0.58,
            color="#4C78A8",
            edgecolors="none"
        )

        low = float(
            min(actual.min(), pred.min())
        )

        high = float(
            max(actual.max(), pred.max())
        )

        margin = max(
            0.04 * (high - low),
            1e-8
        )

        limits = [
            low - margin,
            high + margin
        ]

        # 1:1线
        ax.plot(
            limits,
            limits,
            "--",
            color="#D97757",
            linewidth=1.5,
            label="1:1 line"
        )

        ax.set_xlim(limits)
        ax.set_ylim(limits)

        ax.set_xlabel(
            "ETII value"
        )

        ax.set_ylabel(
            "RF OOF prediction"
        )

        r2_value = final_df.loc[
            j, "OOF_R2"
        ]

        ax.set_title(
            f"{target}\n"
            f"R² = {r2_value:.4f}"
        )

        ax.grid(
            linestyle="--",
            alpha=0.18
        )

        ax.legend(
            frameon=False,
            fontsize=8
        )

    for ax in axes[len(TARGETS):]:

        ax.set_visible(False)

    fig.suptitle(
        "Random Forest: 5-fold OOF predictions",
        fontsize=16
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    fig.savefig(
        PLOT_PNG,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        PLOT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==================================================
# 9. 主程序
# ==================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y, sample_ids, references = load_data()

    print(
        f"\n--- 2. 开始RF {K_FOLDS}折交叉验证 ---"
    )

    # 与ResNet-MLP、SVR一致
    kf = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    # 初始化OOF矩阵
    oof = np.full(
        y.shape,
        np.nan,
        dtype=np.float64
    )

    fold_labels = np.zeros(
        len(X),
        dtype=int
    )

    fold_rows = []
    importance_rows = []

    # ==============================================
    # 10. 五折训练
    # ==============================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        kf.split(X),
        start=1
    ):

        print(
            f"\n正在训练Fold {fold}/{K_FOLDS}"
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # X标准化
        # 只在训练折上拟合
        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        # RF直接支持10个输出
        model = RandomForestRegressor(
            **RF_PARAMS
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        # 测试折预测
        y_pred = model.predict(
            X_test_scaled
        )

        # 保存OOF预测值
        oof[test_idx] = y_pred

        fold_labels[test_idx] = fold

        current_r2 = []

        # 10个目标分别评价
        for j, name in enumerate(TARGETS):

            scores = calculate_metrics(
                y_test[:, j],
                y_pred[:, j]
            )

            fold_rows.append({
                "Fold": fold,
                "Target": name,
                "N_test": len(test_idx),
                **scores
            })

            current_r2.append(
                scores["R2"]
            )

        # RF特征重要性
        for name, value in zip(
            FEATURES,
            model.feature_importances_
        ):

            importance_rows.append({
                "Fold": fold,
                "Feature": name,
                "Importance": float(value)
            })

        print(
            f"Fold {fold} "
            f"平均R² = {np.mean(current_r2):.4f}"
        )

    # ==============================================
    # 11. OOF检查
    # ==============================================

    if not np.isfinite(oof).all():

        raise RuntimeError(
            "存在未完成的OOF预测。"
        )

    # ==============================================
    # 12. 最终OOF指标
    # ==============================================

    final_rows = []

    for j, name in enumerate(TARGETS):

        score = calculate_metrics(
            y[:, j],
            oof[:, j]
        )

        final_rows.append({
            "Target": name,
            "N": len(y),
            "OOF_R2": score["R2"],
            "OOF_RMSE": score["RMSE"],
            "OOF_MAE": score["MAE"],
            "OOF_MSE": score["MSE"]
        })

    final_df = pd.DataFrame(
        final_rows
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    # ==============================================
    # 13. 五折R2汇总
    # ==============================================

    cv_detail = fold_df.pivot(
        index="Fold",
        columns="Target",
        values="R2"
    )

    cv_detail = cv_detail.reindex(
        columns=TARGETS
    )

    cv_detail.index = [
        f"Fold {i}"
        for i in cv_detail.index
    ]

    cv_detail.loc["Average"] = (
        cv_detail.mean(axis=0)
    )

    # ==============================================
    # 14. 逐样本OOF预测
    # ==============================================

    predictions_df = pd.DataFrame({
        "Index_No": sample_ids,
        "Reference": references,
        "Fold": fold_labels
    })

    for j, name in enumerate(TARGETS):

        predictions_df[
            name + "_ETII_Value"
        ] = y[:, j]

        predictions_df[
            name + "_RF_OOF"
        ] = oof[:, j]

        predictions_df[
            name + "_Residual"
        ] = (
            y[:, j] - oof[:, j]
        )

    # ==============================================
    # 15. 特征重要性
    # ==============================================

    importance_df = (
        pd.DataFrame(importance_rows)
        .pivot(
            index="Feature",
            columns="Fold",
            values="Importance"
        )
        .reindex(FEATURES)
    )

    importance_df[
        "Mean_Importance"
    ] = importance_df.mean(axis=1)

    importance_df = (
        importance_df
        .sort_values(
            "Mean_Importance",
            ascending=False
        )
        .reset_index()
    )

    # ==============================================
    # 16. 总体指标
    # ==============================================

    summary = pd.DataFrame([{
        "Model": "Random Forest",
        "N_samples": len(y),
        "N_inputs": len(FEATURES),
        "N_outputs": len(TARGETS),
        "CV": "Random 5-fold OOF",
        "Mean_OOF_R2_10_targets":
            final_df["OOF_R2"].mean(),
        "Mean_OOF_RMSE":
            final_df["OOF_RMSE"].mean(),
        "Mean_OOF_MAE":
            final_df["OOF_MAE"].mean()
    }])

    # ==============================================
    # 17. 模型参数记录
    # ==============================================

    settings = pd.DataFrame(
        [
            ("Input_file", INPUT_FILE.name),
            ("Sheet", SHEET_NAME),
            ("SEED", SEED),
            ("K_FOLDS", K_FOLDS),
            ("n_estimators", 150),
            ("max_depth", 16),
            ("min_samples_split", 4),
            ("min_samples_leaf", 1),
            ("n_jobs", -1),
            ("X_scaling",
             "StandardScaler fitted on training fold"),
            ("Y_scaling", "None"),
            ("Label_source", "ETII-completed outputs")
        ],
        columns=[
            "Parameter",
            "Value"
        ]
    )

    # ==============================================
    # 18. Excel导出
    # ==============================================

    print(
        "\n--- 3. 正在导出结果 ---"
    )

    try:

        with pd.ExcelWriter(
            OUTPUT_EXCEL,
            engine="openpyxl"
        ) as writer:

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

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

            cv_detail.to_excel(
                writer,
                sheet_name="CV_Detail",
                index=True
            )

            predictions_df.to_excel(
                writer,
                sheet_name="OOF_Predictions",
                index=False
            )

            importance_df.to_excel(
                writer,
                sheet_name="RF_Feature_Importance",
                index=False
            )

            settings.to_excel(
                writer,
                sheet_name="Settings",
                index=False
            )

    except PermissionError as exc:

        raise PermissionError(
            "无法保存Excel，请关闭同名文件。"
        ) from exc

    # ==============================================
    # 19. 绘制OOF图
    # ==============================================

    plot_oof(
        y,
        oof,
        final_df
    )

    # ==============================================
    # 20. 打印结果
    # ==============================================

    print(
        "\n========== RF最终OOF结果 =========="
    )

    print(
        final_df[
            [
                "Target",
                "OOF_R2",
                "OOF_RMSE",
                "OOF_MAE"
            ]
        ].round(4).to_string(
            index=False
        )
    )

    print(
        "\n10输出平均OOF R²：",
        round(
            final_df["OOF_R2"].mean(),
            4
        )
    )

    print(
        "\nExcel：",
        OUTPUT_EXCEL
    )

    print(
        "\n图片：",
        PLOT_PNG
    )

    print(
        "\nRF训练完成。"
    )


# ==================================================
# 程序入口
# ==================================================

if __name__ == "__main__":

    main()
