
# -*- coding: utf-8 -*-

"""
生物炭数据集：优化KNN缺失值填补 + R2验证

方法：
1. RobustScaler标准化
2. KNNImputer(k=3, weights='distance')
3. 五折人工遮蔽验证
4. 计算R2、RMSE、MAE
5. 保留原始实验测量值
6. 仅对缺失值进行物理范围约束
7. 根据元素含量计算缺失的O/C和H/C原子比

输入：
    数据集.xlsx

输出：
    Biochar_KNN_优化填补_R2.xlsx

无需PyTorch和CUDA
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =====================================================
# 1. 配置
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "数据集.xlsx"

OUTPUT_FILE = (
    BASE_DIR / "Biochar_KNN_优化填补_R2.xlsx"
)

SHEET_NAME = "Raw dataset"

HEADER_ROWS = 4

RANDOM_SEED = 2024

N_FOLDS = 5

N_NEIGHBORS = 3

WEIGHTS = "distance"


# =====================================================
# 2. 数据列
# =====================================================

# Excel列位置从0开始

INPUT_COLS = [
    6, 7, 8,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19
]

TARGET_COLS = list(range(20, 30))

ALL_COLS = INPUT_COLS + TARGET_COLS

VARIABLE_NAMES = [

    # 13个输入
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
    "Pyrolysis_temperature",
    "Heating_rate",

    # 10个输出
    "Biochar_yield",
    "Total_pore_volume",
    "Biochar_ash",
    "Zeta_potential",
    "pH_or_pHPZC",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O",
    "O_C_atomic_ratio",
    "H_C_atomic_ratio"
]

# 百分比变量在23列矩阵中的位置
PERCENTAGE_COLS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    13, 15, 18, 19, 20
}


# =====================================================
# 3. KNN填补
# =====================================================

def knn_fill(matrix, improved=True):

    if improved:

        # RobustScaler标准化
        scaler = RobustScaler()

        scaled_data = scaler.fit_transform(
            matrix
        )

        # 优化KNN
        imputer = KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS,
            metric="nan_euclidean",
            keep_empty_features=True
        )

        filled_scaled = imputer.fit_transform(
            scaled_data
        )

        # 恢复原始尺度
        filled = scaler.inverse_transform(
            filled_scaled
        )

    else:

        # 原始KNN，用于比较
        imputer = KNNImputer(
            n_neighbors=5,
            weights="distance",
            metric="nan_euclidean",
            keep_empty_features=True
        )

        filled = imputer.fit_transform(
            matrix
        )

    if filled.shape != matrix.shape:
        raise ValueError(
            "填补后数据维度发生变化"
        )

    # 保证所有原始实测值不变
    filled = np.where(
        np.isfinite(matrix),
        matrix,
        filled
    )

    return filled


# =====================================================
# 4. 物理范围约束
# =====================================================

def postprocess_only_missing(original, completed):

    result = completed.copy()

    missing = ~np.isfinite(original)

    for j in range(original.shape[1]):

        mask = missing[:, j]

        if j in PERCENTAGE_COLS:

            # 百分比0-100
            result[mask, j] = np.clip(
                result[mask, j],
                0,
                100
            )

        elif j == 17:

            # pH或pHPZC
            result[mask, j] = np.clip(
                result[mask, j],
                0,
                14
            )

        elif j != 16:

            # Zeta电位允许负值
            result[mask, j] = np.maximum(
                result[mask, j],
                0
            )

    # =================================================
    # 5. O/C和H/C原子比
    # =================================================

    if original.shape[1] == 23:

        M_C = 12.011
        M_H = 1.008
        M_O = 15.999

        # C_biochar：第18列
        # H_biochar：第19列
        # O_biochar：第20列
        # O/C：第21列
        # H/C：第22列

        ratio_settings = [
            (21, 20, M_O),
            (22, 19, M_H)
        ]

        for ratio_col, element_col, mass in ratio_settings:

            mask = (
                missing[:, ratio_col]
                & np.isfinite(result[:, element_col])
                & np.isfinite(result[:, 18])
                & (result[:, 18] > 0)
            )

            result[mask, ratio_col] = (
                (result[mask, element_col] / mass)
                /
                (result[mask, 18] / M_C)
            )

    return result


# =====================================================
# 6. 五折遮蔽验证
# =====================================================

def evaluate_5fold(original):

    print("\n开始五折人工遮蔽验证...")

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    n_features = original.shape[1]

    # 每个变量分别划分五折
    folds = []

    for j in range(n_features):

        observed_indices = np.flatnonzero(
            np.isfinite(original[:, j])
        )

        shuffled = rng.permutation(
            observed_indices
        )

        folds.append(
            np.array_split(
                shuffled,
                N_FOLDS
            )
        )

    # 保存验证预测值
    old_predictions = np.full_like(
        original,
        np.nan
    )

    new_predictions = np.full_like(
        original,
        np.nan
    )

    # 五折验证
    for fold in range(N_FOLDS):

        train_data = original.copy()

        # 暂时隐藏真实值
        for j in range(n_features):

            indices = folds[j][fold]

            train_data[indices, j] = np.nan

        # 原始KNN
        old_filled = knn_fill(
            train_data,
            improved=False
        )

        # 优化KNN
        new_filled = knn_fill(
            train_data,
            improved=True
        )

        # 只对缺失位置施加物理约束
        new_filled = postprocess_only_missing(
            train_data,
            new_filled
        )

        # 保存该折预测结果
        for j in range(n_features):

            indices = folds[j][fold]

            old_predictions[indices, j] = (
                old_filled[indices, j]
            )

            new_predictions[indices, j] = (
                new_filled[indices, j]
            )

        print(
            f"第 {fold + 1}/{N_FOLDS} 折完成"
        )

    # =================================================
    # 7. 逐变量计算R2
    # =================================================

    results = []

    for j in range(n_features):

        mask = np.isfinite(
            original[:, j]
        )

        y_true = original[mask, j]

        y_old = old_predictions[mask, j]

        y_new = new_predictions[mask, j]

        r2_old = r2_score(
            y_true,
            y_old
        )

        r2_new = r2_score(
            y_true,
            y_new
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_new
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_new
        )

        results.append({

            "Variable": VARIABLE_NAMES[j],

            "Observed": int(mask.sum()),

            "Original_KNN_R2": r2_old,

            "Optimized_KNN_R2": r2_new,

            "Optimized_KNN_RMSE": rmse,

            "Optimized_KNN_MAE": mae

        })

    results_df = pd.DataFrame(results)

    return results_df


# =====================================================
# 8. 读取原始数据
# =====================================================

def read_original_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到文件：{INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        keep_default_na=False
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "数据列数不足，请检查Excel结构"
        )

    header = raw.iloc[
        :HEADER_ROWS
    ].copy()

    body = raw.iloc[
        HEADER_ROWS:
    ].copy()

    # 排除空编号行
    valid_rows = (
        body.iloc[:, 0]
        .astype(str)
        .str.strip()
        .ne("")
    )

    body = body.loc[
        valid_rows
    ].copy()

    # 提取23个数值变量
    numeric = np.column_stack([

        pd.to_numeric(
            body.iloc[:, col],
            errors="coerce"
        ).to_numpy(dtype=float)

        for col in ALL_COLS

    ])

    return raw, header, body, numeric


# =====================================================
# 9. 保存Excel
# =====================================================

def reconstruct_excel(
    header,
    body,
    original,
    filled
):

    output_body = body.copy()

    for j, excel_col in enumerate(ALL_COLS):

        missing = ~np.isfinite(
            original[:, j]
        )

        if missing.any():

            output_body.iloc[
                np.flatnonzero(missing),
                excel_col
            ] = filled[missing, j]

    output = pd.concat(
        [header, output_body],
        ignore_index=True
    )

    return output


# =====================================================
# 10. 主程序
# =====================================================

def main():

    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("优化KNN缺失值填补及R2验证")
    print("=" * 60)

    print("计算设备：CPU")
    print("不需要PyTorch")

    # 读取数据
    raw, header, body, numeric = (
        read_original_data()
    )

    print(f"\n数据记录数：{len(body)}")

    print(
        f"变量数量：{numeric.shape[1]}"
    )

    print(
        f"原始缺失值：{np.isnan(numeric).sum()}"
    )

    # 确保每列有足够观测值
    observed_counts = np.isfinite(
        numeric
    ).sum(axis=0)

    if (observed_counts < 10).any():

        raise ValueError(
            "部分变量有效数不足10，无法进行当前验证"
        )

    # -------------------------------------------------
    # 10.1 R2验证
    # -------------------------------------------------

    results = evaluate_5fold(
        numeric
    )

    old_r2 = results[
        "Original_KNN_R2"
    ].mean()

    optimized_r2 = results[
        "Optimized_KNN_R2"
    ].mean()

    print("\n" + "=" * 60)

    print("R2验证结果")

    print("=" * 60)

    print(
        f"原始KNN平均R2：{old_r2:.4f}"
    )

    print(
        f"优化KNN平均R2：{optimized_r2:.4f}"
    )

    print(
        "\n注意：该R2为23个变量"
        "分别计算后取平均。"
    )

    print("\n各变量R2：")

    print(
        results[
            [
                "Variable",
                "Observed",
                "Optimized_KNN_R2"
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------
    # 10.2 仅填补13个输入变量
    # -------------------------------------------------

    input_data = numeric[:, :13]

    input_filled = knn_fill(
        input_data,
        improved=True
    )

    input_filled = (
        postprocess_only_missing(
            input_data,
            input_filled
        )
    )

    input_only = numeric.copy()

    input_only[:, :13] = input_filled

    # -------------------------------------------------
    # 10.3 完整23个变量填补
    # -------------------------------------------------

    all_filled = knn_fill(
        numeric,
        improved=True
    )

    all_filled = (
        postprocess_only_missing(
            numeric,
            all_filled
        )
    )

    # -------------------------------------------------
    # 10.4 生成数据表
    # -------------------------------------------------

    input_excel = reconstruct_excel(
        header,
        body,
        numeric,
        input_only
    )

    complete_excel = reconstruct_excel(
        header,
        body,
        numeric,
        all_filled
    )

    # -------------------------------------------------
    # 10.5 缺失值统计
    # -------------------------------------------------

    missing_summary = pd.DataFrame({

        "Variable": VARIABLE_NAMES,

        "Type": (
            ["Input"] * 13
            + ["Target"] * 10
        ),

        "Observed": np.isfinite(
            numeric
        ).sum(axis=0),

        "Missing": np.isnan(
            numeric
        ).sum(axis=0),

        "Missing_Ratio (%)": (
            np.isnan(numeric).mean(axis=0)
            * 100
        )

    })

    flags = pd.DataFrame(
        (~np.isfinite(numeric)).astype(int),
        columns=VARIABLE_NAMES
    )

    flags.insert(
        0,
        "Index No.",
        body.iloc[:, 0].to_numpy()
    )

    summary = pd.DataFrame({

        "Metric": [
            "Original KNN mean R2",
            "Optimized KNN mean R2",
            "Number of samples",
            "Number of features",
            "KNN neighbors",
            "Scaling",
            "Cross-validation"
        ],

        "Value": [
            old_r2,
            optimized_r2,
            len(body),
            numeric.shape[1],
            N_NEIGHBORS,
            "RobustScaler",
            "5-fold masked validation"
        ]

    })

    # -------------------------------------------------
    # 10.6 保存Excel
    # -------------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        raw.to_excel(
            writer,
            sheet_name="Original",
            index=False,
            header=False
        )

        input_excel.to_excel(
            writer,
            sheet_name="KNN_InputOnly",
            index=False,
            header=False
        )

        complete_excel.to_excel(
            writer,
            sheet_name="KNN_All_Exploratory",
            index=False,
            header=False
        )

        results.to_excel(
            writer,
            sheet_name="R2_5fold",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="R2_Summary",
            index=False
        )

        missing_summary.to_excel(
            writer,
            sheet_name="Missing_Summary",
            index=False
        )

        flags.to_excel(
            writer,
            sheet_name="Imputation_Flags",
            index=False
        )

    print("\n" + "=" * 60)

    print("KNN填补及R2验证完成")

    print("=" * 60)

    print(f"\n输出文件：{OUTPUT_FILE}")

    print("\n请查看以下工作表：")
    print("KNN_InputOnly：输入变量填补")
    print("KNN_All_Exploratory：全部变量填补")
    print("R2_5fold：各变量验证R2")
    print("R2_Summary：平均R2")


# =====================================================
# 11. 运行
# =====================================================

if __name__ == "__main__":
    main()
