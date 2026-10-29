
# -*- coding: utf-8 -*-

"""
SVR（RBF核）五折交叉验证

数据：
ETII(6).xlsx
MICE_All_Exploratory

13个输入：
G-I、K-T

10个输出：
U-AD

模型：
MultiOutputRegressor(SVR)

结果：
5折OOF预测
R2、RMSE、MAE、MSE
各Fold指标
Excel数据
PNG和PDF散点图

与ResNet-MLP使用相同的随机五折划分。
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.svm import SVR

from sklearn.multioutput import MultiOutputRegressor


# ==================================================
# 1. 参数设置
# ==================================================

SEED = 42

K_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)


# ==================================================
# 2. 文件路径
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ETII.xlsx"

SHEET_NAME = "MICE_All_Exploratory"

OUTPUT_DIR = BASE_DIR / "SVR_ETII_Results"

OUTPUT_EXCEL = (
    OUTPUT_DIR
    / "SVR_ETII_5Fold_OOF_Results.xlsx"
)

PLOT_PNG = (
    OUTPUT_DIR
    / "SVR_ETII_5Fold_OOF_Scatter.png"
)

PLOT_PDF = (
    OUTPUT_DIR
    / "SVR_ETII_5Fold_OOF_Scatter.pdf"
)


# ==================================================
# 3. SVR超参数
# ==================================================

SVM_PARAMS = {

    "kernel": "rbf",

    "C": 100,

    "gamma": "scale",

    "epsilon": 1,

    "verbose": False
}


# ==================================================
# 4. Excel列位置
# ==================================================

# 输入：
# G、H、I、K、L、M、N、O、P、Q、R、S、T

INPUT_INDEX = [
    6, 7, 8
] + list(range(10, 20))

# 输出：
# U、V、W、X、Y、Z、AA、AB、AC、AD

OUTPUT_INDEX = list(range(20, 30))


# ==================================================
# 5. 13个输入变量
# ==================================================

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


# ==================================================
# 6. 10个输出变量
# ==================================================

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

assert len(INPUT_INDEX) == len(feature_cols) == 13

assert len(OUTPUT_INDEX) == len(target_cols) == 10


# ==================================================
# 7. 绘图参数
# ==================================================

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "axes.unicode_minus": False,

    "pdf.fonttype": 42
})


# ==================================================
# 8. 读取Excel
# ==================================================

def load_data():

    print("\n--- 1. 正在加载ETII数据... ---")

    if not INPUT_FILE.is_file():

        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}\n"
            "请把Excel和Python代码放在同一文件夹。"
        )

    # 不使用第一行作为列名
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

    # ==============================================
    # 跳过前4行表头
    # ==============================================

    body = raw.iloc[4:, :].copy()

    # A列为样本编号
    ids = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    # 保留具有有效编号的行
    body = body.loc[
        ids.notna()
    ].reset_index(drop=True)

    # ==============================================
    # 提取13个输入
    # ==============================================

    X_df = body.iloc[
        :, INPUT_INDEX
    ].copy()

    X_df.columns = feature_cols

    X_df = X_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ==============================================
    # 提取10个输出
    # ==============================================

    y_df = body.iloc[
        :, OUTPUT_INDEX
    ].copy()

    y_df.columns = target_cols

    y_df = y_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # sklearn使用float64
    X = X_df.to_numpy(
        dtype=np.float64
    )

    y = y_df.to_numpy(
        dtype=np.float64
    )

    # 样本编号
    sample_ids = (
        body.iloc[:, 0].to_numpy()
    )

    # 文献来源
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

    # ==============================================
    # 数据检查
    # ==============================================

    print("数据加载成功！")

    print("样本数：", len(body))

    print("X.shape：", X.shape)

    print("y.shape：", y.shape)

    print(
        "输入NaN：",
        int(np.isnan(X).sum())
    )

    print(
        "输出NaN：",
        int(np.isnan(y).sum())
    )

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):

        bad_x = np.argwhere(
            ~np.isfinite(X)
        )

        bad_y = np.argwhere(
            ~np.isfinite(y)
        )

        raise ValueError(
            "所选23个变量仍有NaN/Inf。\n"
            f"输入异常位置：{bad_x[:5].tolist()}\n"
            f"输出异常位置：{bad_y[:5].tolist()}"
        )

    print("数据检查通过。")

    return (
        X,
        y,
        sample_ids,
        references
    )


# ==================================================
# 9. 计算评价指标
# ==================================================

def evaluate(
    y_true,
    y_pred
):

    r2 = r2_score(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return {

        "R2": float(r2),

        "RMSE": float(rmse),

        "MAE": float(mae),

        "MSE": float(mse)
    }


# ==================================================
# 10. 绘制OOF散点图
# ==================================================

def draw_scatter(
    y,
    predictions,
    metrics_df
):

    print("\n--- 正在绘制OOF散点图... ---")

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(16, 18)
    )

    axes = axes.flatten()

    for j, name in enumerate(target_cols):

        ax = axes[j]

        true_values = y[:, j]

        pred_values = predictions[:, j]

        # 散点
        ax.scatter(
            true_values,
            pred_values,
            s=22,
            alpha=0.60,
            color="#4C78A8",
            edgecolors="none"
        )

        # ==========================================
        # 1:1 理想预测线
        # ==========================================

        minimum = float(
            min(
                true_values.min(),
                pred_values.min()
            )
        )

        maximum = float(
            max(
                true_values.max(),
                pred_values.max()
            )
        )

        margin = max(
            0.04 * (maximum - minimum),
            1e-8
        )

        limits = [
            minimum - margin,
            maximum + margin
        ]

        ax.plot(
            limits,
            limits,
            "--",
            linewidth=1.4,
            color="#D97757",
            label="1:1 line"
        )

        ax.set_xlim(limits)

        ax.set_ylim(limits)

        ax.set_xlabel(
            "ETII value"
        )

        ax.set_ylabel(
            "SVR OOF prediction"
        )

        r2_value = metrics_df.loc[
            j, "OOF_R2"
        ]

        ax.set_title(
            f"{name}\nR² = {r2_value:.4f}"
        )

        ax.grid(
            alpha=0.20,
            linestyle="--"
        )

        ax.legend(
            frameon=False,
            fontsize=9
        )

    # 隐藏空白子图
    for ax in axes[len(target_cols):]:

        ax.set_visible(False)

    fig.suptitle(
        "SVR (RBF): 5-fold OOF predictions",
        fontsize=15
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    # 保存PNG
    fig.savefig(
        PLOT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    # 保存PDF
    fig.savefig(
        PLOT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("散点图已保存。")


# ==================================================
# 11. 主程序
# ==================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==============================================
    # Step 1：读取数据
    # ==============================================

    (
        X,
        y,
        sample_ids,
        references
    ) = load_data()

    print(
        f"\n--- 2. 开始{K_FOLDS}折交叉验证（SVR）---"
    )

    # ==============================================
    # Step 2：设置五折
    # 与ResNet-MLP保持一致
    # ==============================================

    kf = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    # OOF预测矩阵
    oof_predictions = np.full(
        y.shape,
        np.nan,
        dtype=np.float64
    )

    fold_labels = np.zeros(
        len(y),
        dtype=int
    )

    fold_rows = []

    # ==============================================
    # Step 3：依次训练5折
    # ==============================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        kf.split(X),
        start=1
    ):

        print(
            f"\nFold {fold}/{K_FOLDS}"
        )

        print(
            "训练样本数：",
            len(train_idx)
        )

        print(
            "测试样本数：",
            len(test_idx)
        )

        # ------------------------------------------
        # 提取训练集和测试集
        # ------------------------------------------

        X_train = X[train_idx]

        X_test = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        # ------------------------------------------
        # X标准化
        # 只使用训练数据拟合
        # ------------------------------------------

        x_scaler = StandardScaler()

        X_train_scaled = (
            x_scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            x_scaler.transform(
                X_test
            )
        )

        # ------------------------------------------
        # Y标准化
        # ------------------------------------------

        y_scaler = StandardScaler()

        y_train_scaled = (
            y_scaler.fit_transform(
                y_train
            )
        )

        # ------------------------------------------
        # 创建SVR
        # ------------------------------------------

        base_svr = SVR(
            **SVM_PARAMS
        )

        # 每个输出训练一个SVR
        model = MultiOutputRegressor(
            base_svr,
            n_jobs=1
        )

        # ------------------------------------------
        # 训练
        # ------------------------------------------

        model.fit(
            X_train_scaled,
            y_train_scaled
        )

        # ------------------------------------------
        # 预测测试折
        # ------------------------------------------

        y_pred_scaled = model.predict(
            X_test_scaled
        )

        # 反标准化
        y_pred_orig = (
            y_scaler.inverse_transform(
                y_pred_scaled
            )
        )

        # ------------------------------------------
        # 保存OOF预测
        # ------------------------------------------

        oof_predictions[test_idx] = (
            y_pred_orig
        )

        fold_labels[test_idx] = fold

        # ------------------------------------------
        # 当前Fold性能
        # ------------------------------------------

        current_r2 = []

        for j, name in enumerate(target_cols):

            score = evaluate(
                y_test[:, j],
                y_pred_orig[:, j]
            )

            fold_rows.append({

                "Fold": fold,

                "Target": name,

                "N_test": len(test_idx),

                **score
            })

            current_r2.append(
                score["R2"]
            )

        mean_r2 = np.mean(
            current_r2
        )

        print(
            f"Fold {fold} "
            f"平均R² = {mean_r2:.4f}"
        )

    # ==============================================
    # Step 4：检查OOF预测
    # ==============================================

    if not np.isfinite(
        oof_predictions
    ).all():

        raise RuntimeError(
            "部分样本没有生成OOF预测值。"
        )

    # ==============================================
    # Step 5：计算最终OOF性能
    # ==============================================

    print(
        "\n--- 3. 正在计算最终OOF指标... ---"
    )

    final_rows = []

    for j, name in enumerate(target_cols):

        score = evaluate(
            y[:, j],
            oof_predictions[:, j]
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
    # Step 6：五折R²汇总
    # ==============================================

    cv_detail = fold_df.pivot(
        index="Fold",
        columns="Target",
        values="R2"
    )

    # 保持10个目标的顺序
    cv_detail = cv_detail.reindex(
        columns=target_cols
    )

    cv_detail.index = [
        f"Fold {i}"
        for i in cv_detail.index
    ]

    # 添加平均行
    cv_detail.loc["Average"] = (
        cv_detail.mean()
    )

    # ==============================================
    # Step 7：逐样本预测数据
    # ==============================================

    predictions_df = pd.DataFrame({

        "Index_No": sample_ids,

        "Reference": references,

        "Fold": fold_labels
    })

    for j, name in enumerate(target_cols):

        predictions_df[
            f"{name}_ETII_Value"
        ] = y[:, j]

        predictions_df[
            f"{name}_SVR_OOF_Pred"
        ] = oof_predictions[:, j]

        predictions_df[
            f"{name}_Residual"
        ] = (
            y[:, j]
            - oof_predictions[:, j]
        )

    # ==============================================
    # Step 8：模型总体结果
    # ==============================================

    summary_df = pd.DataFrame([{

        "Model": "SVR (RBF)",

        "N_samples": len(y),

        "N_features": len(feature_cols),

        "N_targets": len(target_cols),

        "CV": "Random 5-fold OOF",

        "Mean_OOF_R2_10_targets":
            final_df["OOF_R2"].mean(),

        "Mean_OOF_RMSE_10_targets":
            final_df["OOF_RMSE"].mean(),

        "Mean_OOF_MAE_10_targets":
            final_df["OOF_MAE"].mean()
    }])

    # ==============================================
    # Step 9：记录模型参数
    # ==============================================

    settings_df = pd.DataFrame([

        ("Input_file", INPUT_FILE.name),

        ("Sheet", SHEET_NAME),

        ("CV_seed", SEED),

        ("K_FOLDS", K_FOLDS),

        ("Kernel", SVM_PARAMS["kernel"]),

        ("C", SVM_PARAMS["C"]),

        ("gamma", SVM_PARAMS["gamma"]),

        ("epsilon", SVM_PARAMS["epsilon"]),

        (
            "Scaling",
            "X and y StandardScaler "
            "fitted within each training fold"
        ),

        (
            "Label_source",
            "ETII-completed targets"
        )

    ], columns=[
        "Parameter",
        "Value"
    ])

    # ==============================================
    # Step 10：保存Excel
    # ==============================================

    print(
        "\n--- 4. 正在保存Excel... ---"
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
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

        settings_df.to_excel(
            writer,
            sheet_name="Settings",
            index=False
        )

    # ==============================================
    # Step 11：绘制散点图
    # ==============================================

    draw_scatter(
        y,
        oof_predictions,
        final_df
    )

    # ==============================================
    # Step 12：打印结果
    # ==============================================

    print(
        "\n========== 最终5折OOF结果 =========="
    )

    print(
        final_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\n10个目标平均OOF R²：",
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
        "\n全部完成。"
    )


# ==================================================
# 程序入口
# ==================================================

if __name__ == "__main__":

    main()
