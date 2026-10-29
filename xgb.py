
# -*- coding: utf-8 -*-

"""
纯 XGBoost 缺失值填补 + 五折R2验证

不使用：
MICE、IterativeImputer、KNN、ExtraTrees、PyTorch

输入文件：数据集.xlsx
工作表：Raw dataset

13个输入变量：Excel G-I、K-T列
10个输出变量：Excel U-AD列

其中Y列统一更名为pH。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# =====================================================
# 1. 基本设置
# =====================================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "数据集.xlsx"

OUTPUT_FILE = (
    ROOT / "Biochar_Pure_XGBoost_Imputation_R2.xlsx"
)

SHEET_NAME = "Raw dataset"

HEADER_ROWS = 4
SEED = 2024
FOLDS = 5

# 是否进行R2验证
RUN_R2_VALIDATION = True

# 只对预测值实施物理范围约束
# False表示完全使用XGBoost原始预测
APPLY_BOUNDS = True


# =====================================================
# 2. XGBoost参数
# =====================================================

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "missing": np.nan,
    "n_jobs": 4,
    "random_state": SEED
}


# =====================================================
# 3. 数据列
# =====================================================

INPUT_MAP = {
    6: "FC_dry",
    7: "VM_dry",
    8: "Ash_dry",
    10: "Feedstock_C",
    11: "Feedstock_H",
    12: "Feedstock_O",
    13: "Feedstock_N",
    14: "Cellulose",
    15: "Hemicellulose",
    16: "Lignin",
    17: "Residence_time",
    18: "Temperature",
    19: "Heating_rate"
}

OUTPUT_MAP = {
    20: "Biochar_yield",
    21: "Total_pore_volume",
    22: "Biochar_ash",
    23: "Zeta_potential",
    24: "pH",
    25: "Biochar_C",
    26: "Biochar_H",
    27: "Biochar_O",
    28: "O_C_ratio",
    29: "H_C_ratio"
}

COLUMN_MAP = {
    **INPUT_MAP,
    **OUTPUT_MAP
}

NAMES = list(COLUMN_MAP.values())

N_INPUTS = len(INPUT_MAP)

PERCENTAGE_NAMES = {
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
    "Biochar_yield",
    "Biochar_ash",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O"
}

MISSING_TEXT = {
    "", "nan", "na", "n/a",
    "none", "null", "nr", "-", "—"
}


# =====================================================
# 4. 读取数据
# =====================================================

def read_data():

    if not INPUT_FILE.is_file():

        raise FileNotFoundError(
            f"找不到数据文件：{INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        dtype=object,
        keep_default_na=False
    )

    if raw.shape[1] < 30:

        raise ValueError(
            "Excel至少需要30列"
        )

    # 保留前4行表头
    header = raw.iloc[
        :HEADER_ROWS, :
    ].copy()

    # PZC改成pH
    header.iat[
        HEADER_ROWS - 1, 24
    ] = "pH"

    body = raw.iloc[
        HEADER_ROWS:, :
    ].copy()

    # 通过编号筛选有效数据
    record_ids = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    body = body.loc[
        record_ids.notna()
    ].reset_index(drop=True)

    if body.empty:

        raise ValueError(
            "没有找到有效数据"
        )

    values = pd.DataFrame(
        index=body.index
    )

    invalid_rows = []

    # 提取23个变量
    for excel_col, name in COLUMN_MAP.items():

        source = body.iloc[
            :, excel_col
        ]

        text = (
            source.astype(str)
            .str.strip()
        )

        numeric = pd.to_numeric(
            text.str.replace(
                ",", "", regex=False
            ),
            errors="coerce"
        )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan
        )

        bad_mask = (
            numeric.isna()
            & ~text.str.lower().isin(
                MISSING_TEXT
            )
        )

        for index in np.flatnonzero(
            bad_mask.to_numpy()
        ):

            invalid_rows.append({
                "Excel_row":
                    int(index) + HEADER_ROWS + 1,

                "Variable": name,

                "Original_text":
                    str(source.iloc[index]),

                "Note":
                    "非数值或低于检出限，需人工核查"
            })

        values[name] = numeric.astype(float)

    # 检查数据是否能够训练
    for name in NAMES:

        count = values[name].notna().sum()

        if count < FOLDS + 2:

            raise ValueError(
                f"{name} 实测值过少：{count}"
            )

    invalid = pd.DataFrame(
        invalid_rows,
        columns=[
            "Excel_row",
            "Variable",
            "Original_text",
            "Note"
        ]
    )

    return raw, header, body, values, invalid


# =====================================================
# 5. 选择XGBoost预测特征
# =====================================================

def predictor_columns(j):

    # 输入变量：
    # 利用其他12个输入变量预测
    if j < N_INPUTS:

        return [
            k for k in range(N_INPUTS)
            if k != j
        ]

    # 输出变量：
    # 只使用13个输入变量预测
    else:

        return list(range(N_INPUTS))


# =====================================================
# 6. 预测值范围约束
# =====================================================

def bound_prediction(pred, name):

    pred = np.asarray(
        pred,
        dtype=float
    )

    if not APPLY_BOUNDS:

        return pred

    # Zeta电位可以为负
    if name == "Zeta_potential":

        return pred

    # 百分比变量
    if name in PERCENTAGE_NAMES:

        return np.clip(
            pred, 0, 100
        )

    # pH
    if name == "pH":

        return np.clip(
            pred, 0, 14
        )

    # 其他变量不得为负
    return np.maximum(
        pred, 0
    )


# =====================================================
# 7. 独立XGBoost模型训练
# =====================================================

def train_predict(
    data_matrix,
    j,
    train_indices,
    pred_indices
):

    features = predictor_columns(j)

    model = XGBRegressor(
        **XGB_PARAMS
    )

    # 只使用当前目标的实测值训练
    X_train = data_matrix[
        np.ix_(
            train_indices,
            features
        )
    ]

    y_train = data_matrix[
        train_indices, j
    ]

    # XGBoost原生支持输入特征NaN
    model.fit(
        X_train,
        y_train
    )

    # 预测缺失位置
    X_pred = data_matrix[
        np.ix_(
            pred_indices,
            features
        )
    ]

    prediction = model.predict(
        X_pred
    )

    prediction = bound_prediction(
        prediction,
        NAMES[j]
    )

    return prediction


# =====================================================
# 8. 使用纯XGBoost填补23个变量
# =====================================================

def impute_all(values):

    matrix = values.to_numpy(
        dtype=float
    )

    completed = matrix.copy()

    rows = np.arange(
        len(matrix)
    )

    missing_flags = ~np.isfinite(
        matrix
    )

    for j, name in enumerate(NAMES):

        # 有实测值的样本
        known = rows[
            ~missing_flags[:, j]
        ]

        # 缺失值样本
        missing = rows[
            missing_flags[:, j]
        ]

        if len(missing) > 0:

            if len(known) < 8:

                raise ValueError(
                    f"{name} 实测值不足"
                )

            # 当前变量单独训练XGBoost
            predictions = train_predict(
                matrix,
                j,
                known,
                missing
            )

            completed[
                missing, j
            ] = predictions

        print(
            f"填补 {name:24s}"
            f" 实测={len(known):3d}"
            f" 缺失={len(missing):3d}"
        )

    # 核查原始实测值
    if not np.array_equal(
        completed[~missing_flags],
        matrix[~missing_flags]
    ):

        raise AssertionError(
            "原始实测数值被修改"
        )

    # 检查填补完整性
    if not np.isfinite(
        completed
    ).all():

        raise AssertionError(
            "填补后仍存在缺失值"
        )

    return pd.DataFrame(
        completed,
        columns=NAMES,
        index=values.index
    )


# =====================================================
# 9. 五折交叉验证计算R2
# =====================================================

def fivefold_r2(values):

    matrix = values.to_numpy(
        dtype=float
    )

    results = []

    print("\n开始五折R2验证...")

    for j, name in enumerate(NAMES):

        # 仅对原始实测值进行验证
        measured = np.flatnonzero(
            np.isfinite(
                matrix[:, j]
            )
        )

        # 保存折外预测结果
        oof = np.full(
            len(matrix),
            np.nan
        )

        cv = KFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED
        )

        for train_loc, test_loc in cv.split(
            measured
        ):

            train_rows = measured[
                train_loc
            ]

            test_rows = measured[
                test_loc
            ]

            # 重新训练当前折模型
            predictions = train_predict(
                matrix,
                j,
                train_rows,
                test_rows
            )

            oof[
                test_rows
            ] = predictions

        if np.isnan(
            oof[measured]
        ).any():

            raise AssertionError(
                f"{name} 存在未预测的验证值"
            )

        y_true = matrix[
            measured, j
        ]

        y_pred = oof[
            measured
        ]

        # 计算R2
        r2 = r2_score(
            y_true,
            y_pred
        )

        # 计算RMSE
        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        # 计算MAE
        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        results.append({
            "Variable": name,

            "Measured_count":
                len(measured),

            "R2_5fold_OOF": r2,

            "RMSE_5fold_OOF": rmse,

            "MAE_5fold_OOF": mae
        })

        print(
            f"验证 {name:24s}"
            f" R2={r2:.4f}"
        )

    return pd.DataFrame(results)


# =====================================================
# 10. 恢复原Excel结构
# =====================================================

def restore_sheet(
    header,
    body,
    raw_numeric,
    completed,
    input_only=False
):

    out = body.copy()

    if input_only:

        allowed = INPUT_MAP

    else:

        allowed = COLUMN_MAP

    for excel_col, name in allowed.items():

        missing = (
            raw_numeric[name]
            .isna()
            .to_numpy()
        )

        if missing.any():

            out.iloc[
                np.flatnonzero(missing),
                excel_col
            ] = completed.loc[
                missing,
                name
            ].to_numpy()

    return pd.concat(
        [header, out],
        ignore_index=True
    )


# =====================================================
# 11. 主程序
# =====================================================

def main():

    print("=" * 60)
    print("纯XGBoost缺失值填补")
    print("=" * 60)

    print("不使用MICE、KNN和PyTorch")

    # -----------------------------------------------
    # 11.1 读取数据
    # -----------------------------------------------

    (
        raw,
        header,
        body,
        numeric,
        invalid
    ) = read_data()

    print(f"\n样本数：{len(body)}")

    print(
        f"变量数：{len(NAMES)}"
    )

    print(
        "原始缺失数量："
        f"{int(numeric.isna().sum().sum())}"
    )

    # -----------------------------------------------
    # 11.2 XGBoost填补
    # -----------------------------------------------

    completed = impute_all(
        numeric
    )

    # -----------------------------------------------
    # 11.3 R2验证
    # -----------------------------------------------

    scores = None

    if RUN_R2_VALIDATION:

        scores = fivefold_r2(
            numeric
        )

    # -----------------------------------------------
    # 11.4 缺失位置标记
    # -----------------------------------------------

    flags = (
        numeric.isna().astype(int)
    )

    flags.insert(
        0,
        "Index_No",
        body.iloc[:, 0].to_numpy()
    )

    # -----------------------------------------------
    # 11.5 缺失统计
    # -----------------------------------------------

    missing_summary = pd.DataFrame({

        "Variable": NAMES,

        "Role": (
            ["Input"] * N_INPUTS
            + ["Target"] * len(OUTPUT_MAP)
        ),

        "Measured": (
            numeric.notna()
            .sum()
            .to_numpy()
        ),

        "Missing": (
            numeric.isna()
            .sum()
            .to_numpy()
        ),

        "Missing_pct": (
            numeric.isna()
            .mean()
            .mul(100)
            .round(2)
            .to_numpy()
        )

    })

    # -----------------------------------------------
    # 11.6 输出两个填补版本
    # -----------------------------------------------

    original_out = pd.concat(
        [header, body],
        ignore_index=True
    )

    input_out = restore_sheet(
        header,
        body,
        numeric,
        completed,
        input_only=True
    )

    full_out = restore_sheet(
        header,
        body,
        numeric,
        completed,
        input_only=False
    )

    # -----------------------------------------------
    # 11.7 保存Excel
    # -----------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        original_out.to_excel(
            writer,
            sheet_name="Original",
            index=False,
            header=False
        )

        input_out.to_excel(
            writer,
            sheet_name="XGB_InputOnly",
            index=False,
            header=False
        )

        full_out.to_excel(
            writer,
            sheet_name="XGB_All_Exploratory",
            index=False,
            header=False
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

        invalid.to_excel(
            writer,
            sheet_name="Invalid_Cells",
            index=False
        )

        if scores is not None:

            scores.to_excel(
                writer,
                sheet_name="R2_5fold",
                index=False
            )

            summary = pd.DataFrame({

                "Metric": [
                    "23变量平均R2",
                    "13输入平均R2",
                    "10输出平均R2"
                ],

                "Value": [

                    scores[
                        "R2_5fold_OOF"
                    ].mean(),

                    scores.iloc[:N_INPUTS][
                        "R2_5fold_OOF"
                    ].mean(),

                    scores.iloc[N_INPUTS:][
                        "R2_5fold_OOF"
                    ].mean()

                ]

            })

            summary.to_excel(
                writer,
                sheet_name="R2_Summary",
                index=False
            )

            print("\n====== R2验证结果 ======")

            for row in summary.itertuples(
                index=False
            ):

                print(
                    f"{row.Metric}: "
                    f"{row.Value:.4f}"
                )

    print("\n" + "=" * 60)

    print("纯XGBoost填补完成")

    print("=" * 60)

    print(
        f"\n输出文件：{OUTPUT_FILE}"
    )

    print(
        "原始实测数值全部保留"
    )

    print(
        "PZC列已更名为pH"
    )


# =====================================================
# 12. 程序入口
# =====================================================

if __name__ == "__main__":
    main()
