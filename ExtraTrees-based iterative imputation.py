
# -*- coding: utf-8 -*-

"""
ETII + GHI干基组成约束 + 五折人工遮蔽验证

核心算法：
IterativeImputer + ExtraTreesRegressor

保留原版ETII参数：
125棵树、4次迭代、五折验证

新增：
1. FC_dry + VM_dry + Ash_dry = 100
2. 原始实测值不修改
3. 多缺失组分非负、定和投影
4. 比较约束前后的R2
5. 导出逐行GHI校验结果

输入：
数据集.xlsx / Raw dataset

输出：
ETII_GHI_Physics_Constrained.xlsx

无需artifact_tool、PyTorch或CUDA。
"""

from pathlib import Path
from collections import Counter
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ==================================================
# 1. 参数设置
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "数据集.xlsx"

OUTPUT_FILE = (
    BASE_DIR / "ETII_GHI_Physics_Constrained.xlsx"
)

SHEET_NAME = "Raw dataset"

HEADER_ROWS = 4
SEED = 2024
FOLDS = 5

TOP_FEEDSTOCK_TYPES = 15

N_TREES = 125
MIN_SAMPLES_LEAF = 1
MAX_ITER = 4
N_JOBS = 4

# 是否运行跨文献五折验证
RUN_GROUP_CV = True

# GHI组成约束参数
SUM_TARGET = 100.0
SUM_TOL = 1e-8


# ==================================================
# 2. Excel输入输出变量
# ==================================================

# 13个输入：G-I、K-T
# 10个输出：U-AD

EXCEL_COLS = [
    6, 7, 8,
    10, 11, 12, 13,
    14, 15, 16, 17, 18, 19
] + list(range(20, 30))

NAMES = [
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
    "Heating_rate",
    "Biochar_yield",
    "Total_pore_volume",
    "Biochar_ash",
    "Zeta_potential",
    "pH",
    "Biochar_C",
    "Biochar_H",
    "Biochar_O",
    "O_C_ratio",
    "H_C_ratio"
]

N_INPUTS = 13

PERCENT_NAMES = {
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
    "",
    "nan",
    "na",
    "n/a",
    "nr",
    "null",
    "none",
    "-",
    "—"
}


# ==================================================
# 3. 读取数据
# ==================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到Excel：{INPUT_FILE}"
        )

    raw = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=None,
        dtype=object,
        keep_default_na=False
    )

    if raw.shape[1] < 31:
        raise ValueError(
            "原始数据应至少有31列"
        )

    headers = raw.iloc[:HEADER_ROWS].copy()

    # 与原代码保持一致
    # 若该指标实际是PZC，应按原文核实名称
    headers.iat[3, 24] = "pH"

    body = raw.iloc[HEADER_ROWS:].copy()

    record_numbers = pd.to_numeric(
        body.iloc[:, 0],
        errors="coerce"
    )

    body = body.loc[
        record_numbers.notna()
    ].reset_index(drop=True)

    if body.empty:
        raise ValueError(
            "没有读取到有效样本"
        )

    cols = []
    invalid = []

    for pos, name in zip(EXCEL_COLS, NAMES):

        raw_col = body.iloc[:, pos]

        txt = (
            raw_col.astype(str)
            .str.strip()
            .str.replace(
                ",",
                "",
                regex=False
            )
        )

        numeric = pd.to_numeric(
            txt,
            errors="coerce"
        )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan
        )

        bad = (
            numeric.isna()
            & ~txt.str.lower().isin(MISSING_TEXT)
        )

        for i in np.flatnonzero(
            bad.to_numpy()
        ):

            invalid.append({
                "Excel_row": i + HEADER_ROWS + 1,
                "Variable": name,
                "Original_text": str(raw_col.iloc[i]),
                "Note": "非数值文本，按缺失处理，请核对"
            })

        cols.append(
            numeric.to_numpy(dtype=float)
        )

    data = np.column_stack(cols)

    if np.any(
        np.isfinite(data).sum(axis=0) < 10
    ):
        raise ValueError(
            "部分变量实测值少于10条"
        )

    # 原料类型辅助特征
    types = (
        body.iloc[:, 1]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    top_types = [
        k for k, _ in Counter(types).most_common(
            TOP_FEEDSTOCK_TYPES
        )
    ]

    feedstock_aux = np.column_stack([
        np.array(
            [float(t == kind) for t in types],
            dtype=float
        )
        for kind in top_types
    ])

    # 文献来源
    sources = (
        body.iloc[:, 30]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    sources = np.array([
        " ".join(s.split())
        if s else f"unknown_ref_{i}"
        for i, s in enumerate(sources)
    ])

    return (
        raw,
        headers,
        body,
        data,
        feedstock_aux,
        sources,
        pd.DataFrame(invalid)
    )


# ==================================================
# 4. 创建ETII插补器
# ==================================================

def make_imputer(seed=SEED):

    regressor = ExtraTreesRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=0.90,
        random_state=SEED,
        n_jobs=N_JOBS
    )

    return IterativeImputer(
        estimator=regressor,
        initial_strategy="median",
        max_iter=MAX_ITER,
        imputation_order="ascending",
        skip_complete=True,
        sample_posterior=False,
        keep_empty_features=True,
        random_state=seed,
        tol=1e-3
    )


# ==================================================
# 5. 数值范围约束
# ==================================================

def apply_bounds(original, completed, names):

    output = completed.copy()

    missing = ~np.isfinite(original)

    for j, name in enumerate(names):

        mask = missing[:, j]

        if name == "Zeta_potential":
            continue

        if name == "pH":

            output[mask, j] = np.clip(
                output[mask, j],
                0,
                14
            )

        elif name in PERCENT_NAMES:

            output[mask, j] = np.clip(
                output[mask, j],
                0,
                100
            )

        else:

            output[mask, j] = np.maximum(
                output[mask, j],
                0
            )

    # 保留实测值
    output[~missing] = original[~missing]

    return output


# ==================================================
# 6. 新增：非负定和投影
# ==================================================

def project_nonnegative_simplex(
    prediction,
    target_total
):

    """
    将预测值投影到：
    x >= 0
    sum(x) = target_total

    适用于两个或三个组分同时缺失。
    """

    vector = np.asarray(
        prediction,
        dtype=float
    )

    if not np.isfinite(vector).all():
        raise ValueError(
            "GHI预测存在非有限数值"
        )

    if target_total < -SUM_TOL:
        raise ValueError(
            "实测组分之和超过100%"
        )

    target_total = max(
        0.0,
        float(target_total)
    )

    if len(vector) == 1:
        return np.array(
            [target_total],
            dtype=float
        )

    if target_total == 0:
        return np.zeros_like(vector)

    ordered = np.sort(
        vector
    )[::-1]

    cumulative = (
        np.cumsum(ordered)
        - target_total
    )

    candidates = np.flatnonzero(
        ordered
        - cumulative
        / np.arange(
            1,
            len(vector) + 1
        )
        > 0
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "单纯形投影失败"
        )

    rho = int(candidates[-1])

    theta = (
        cumulative[rho]
        / (rho + 1)
    )

    projected = np.maximum(
        vector - theta,
        0.0
    )

    # 修正浮点数舍入误差
    index = int(np.argmax(projected))

    projected[index] += (
        target_total
        - float(projected.sum())
    )

    return projected


# ==================================================
# 7. 新增：GHI总和100%约束
# ==================================================

def enforce_ghi_sum(observed, completed):

    """
    G列：FC_dry
    H列：VM_dry
    I列：Ash_dry

    FC_dry + VM_dry + Ash_dry = 100

    只修改原始缺失位置。
    原始实测值严格不变。
    """

    observed = np.asarray(
        observed,
        dtype=float
    )

    adjusted = np.asarray(
        completed,
        dtype=float
    ).copy()

    if observed.shape != adjusted.shape:
        raise ValueError(
            "输入矩阵尺寸不一致"
        )

    for i in range(len(adjusted)):

        missing = ~np.isfinite(
            observed[i, :3]
        )

        # 三项均为实测值：不修改
        if not missing.any():
            continue

        known = observed[
            i, :3
        ][~missing]

        if np.any(known < -SUM_TOL):
            raise ValueError(
                f"第{i+1}条存在负的实测组分"
            )

        known_sum = float(
            known.sum()
        )

        remaining = (
            SUM_TARGET - known_sum
        )

        if remaining < -SUM_TOL:
            raise ValueError(
                f"Excel第{i+HEADER_ROWS+1}行"
                "实测组分之和超过100%，"
                "请检查原始数据"
            )

        predicted_missing = (
            adjusted[i, :3][missing]
        )

        corrected = (
            project_nonnegative_simplex(
                predicted_missing,
                remaining
            )
        )

        # 仅更新缺失组分
        adjusted[i, :3][missing] = (
            corrected
        )

        final_sum = float(
            adjusted[i, :3].sum()
        )

        if abs(final_sum - 100) > SUM_TOL:
            raise AssertionError(
                f"第{i+1}条GHI约束失败"
            )

    # 检查原始实测值
    measured = np.isfinite(observed)

    if not np.array_equal(
        adjusted[measured],
        observed[measured]
    ):
        raise AssertionError(
            "原始实测值被修改"
        )

    if not np.isfinite(adjusted).all():
        raise AssertionError(
            "约束后仍有缺失值"
        )

    return adjusted


# ==================================================
# 8. 新增：GHI组成校验表
# ==================================================

def composition_report(observed, final):

    sums = final[:, :3].sum(axis=1)

    missing_count = (
        np.isnan(
            observed[:, :3]
        ).sum(axis=1)
    )

    report = pd.DataFrame({

        "Excel_row":
            np.arange(len(final)) + HEADER_ROWS + 1,

        "FC_dry":
            final[:, 0],

        "VM_dry":
            final[:, 1],

        "Ash_dry":
            final[:, 2],

        "GHI_sum":
            sums,

        "Deviation_from_100":
            sums - 100,

        "Original_GHI_missing_count":
            missing_count,

        "Record_type":
            np.where(
                missing_count == 0,
                "All measured",
                "Contains imputation"
            )
    })

    mask = missing_count > 0

    if np.any(
        np.abs(sums[mask] - 100) > SUM_TOL
    ):
        raise AssertionError(
            "存在不满足GHI约束的插补记录"
        )

    return report


# ==================================================
# 9. ETII插补
# ==================================================

def fit_impute(
    matrix,
    feedstock_aux,
    names,
    seed=SEED
):

    model_input = np.column_stack([
        matrix,
        feedstock_aux
    ])

    model = make_imputer(seed)

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message=(
                ".*Early stopping criterion "
                "not reached.*"
            )
        )

        filled_all = model.fit_transform(
            model_input
        )

    completed = filled_all[
        :, :matrix.shape[1]
    ]

    measured = np.isfinite(matrix)

    completed[measured] = matrix[measured]

    if not np.isfinite(completed).all():
        raise RuntimeError(
            "ETII填补后仍存在NaN"
        )

    return completed


# ==================================================
# 10. 插补性能指标
# ==================================================

def metrics(
    truth,
    raw_predictions,
    bounded_predictions,
    constrained_predictions,
    names
):

    records = []

    for j, name in enumerate(names):

        mask = (
            np.isfinite(truth[:, j])
            & np.isfinite(
                raw_predictions[:, j]
            )
        )

        y = truth[mask, j]

        a = raw_predictions[mask, j]

        b = bounded_predictions[mask, j]

        c = constrained_predictions[mask, j]

        records.append({

            "Variable": name,

            "Observed": int(mask.sum()),

            "Originally_missing": int(
                np.isnan(
                    truth[:, j]
                ).sum()
            ),

            "R2_model":
                r2_score(y, a),

            "R2_after_bounds":
                r2_score(y, b),

            "RMSE_after_bounds":
                np.sqrt(
                    mean_squared_error(y, b)
                ),

            "MAE_after_bounds":
                mean_absolute_error(y, b),

            "R2_after_constraints":
                r2_score(y, c),

            "RMSE_after_constraints":
                np.sqrt(
                    mean_squared_error(y, c)
                ),

            "MAE_after_constraints":
                mean_absolute_error(y, c)
        })

    return pd.DataFrame(records)


# ==================================================
# 11. 五折随机遮蔽验证
# ==================================================

def random_mask_fivefold(data, aux, names):

    n, p = data.shape

    rng = np.random.default_rng(SEED)

    folds = []

    for j in range(p):

        known = np.flatnonzero(
            np.isfinite(data[:, j])
        )

        folds.append(
            np.array_split(
                rng.permutation(known),
                FOLDS
            )
        )

    raw_pred = np.full(
        (n, p),
        np.nan
    )

    bounded_pred = np.full(
        (n, p),
        np.nan
    )

    constrained_pred = np.full(
        (n, p),
        np.nan
    )

    for fold in range(FOLDS):

        masked = data.copy()

        # 每个变量隐藏本折的实测值
        for j in range(p):

            masked[
                folds[j][fold],
                j
            ] = np.nan

        # 原始ETII插补
        filled = fit_impute(
            masked,
            aux,
            names,
            SEED
        )

        # 数值范围约束
        bounded = apply_bounds(
            masked,
            filled,
            names
        )

        # 新增GHI质量守恒约束
        constrained = enforce_ghi_sum(
            masked,
            bounded
        )

        # 记录隐藏值的预测结果
        for j in range(p):

            idx = folds[j][fold]

            raw_pred[idx, j] = (
                filled[idx, j]
            )

            bounded_pred[idx, j] = (
                bounded[idx, j]
            )

            constrained_pred[idx, j] = (
                constrained[idx, j]
            )

        print(
            f"随机遮蔽验证 "
            f"Fold {fold+1}/{FOLDS} 完成",
            flush=True
        )

    return metrics(
        data,
        raw_pred,
        bounded_pred,
        constrained_pred,
        names
    )


# ==================================================
# 12. 跨文献五折验证
# ==================================================

def grouped_target_cv(
    data,
    aux,
    references
):

    if len(set(references)) < FOLDS:

        print(
            "文献数量不足，跳过分组验证"
        )

        return None

    pred = np.full(
        (len(data), 10),
        np.nan
    )

    cv = GroupKFold(
        n_splits=FOLDS
    )

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            np.arange(len(data)),
            groups=references
        ),
        start=1
    ):

        model = make_imputer(SEED)

        train = np.column_stack([
            data[train_idx],
            aux[train_idx]
        ])

        test = data[
            test_idx
        ].copy()

        # 隐藏测试集全部输出
        test[:, N_INPUTS:] = np.nan

        test = np.column_stack([
            test,
            aux[test_idx]
        ])

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    ".*Early stopping criterion "
                    "not reached.*"
                )
            )

            model.fit(train)

            predictions = model.transform(test)

        targets_pred = predictions[
            :, N_INPUTS:23
        ]

        # 仅限制输出物理范围
        fully_masked_targets = (
            np.full_like(
                targets_pred,
                np.nan
            )
        )

        targets_bounded = apply_bounds(
            fully_masked_targets,
            targets_pred,
            NAMES[N_INPUTS:]
        )

        pred[test_idx] = targets_bounded

        print(
            f"跨文献验证 "
            f"Fold {fold}/{FOLDS} 完成",
            flush=True
        )

    scores = []

    for j, name in enumerate(
        NAMES[N_INPUTS:]
    ):

        mask = (
            np.isfinite(
                data[:, N_INPUTS + j]
            )
            & np.isfinite(
                pred[:, j]
            )
        )

        if mask.sum() < 2:
            continue

        y = data[
            mask,
            N_INPUTS + j
        ]

        p = pred[mask, j]

        scores.append({

            "Variable": name,

            "Observed": int(mask.sum()),

            "R2_grouped":
                r2_score(y, p),

            "RMSE_grouped":
                np.sqrt(
                    mean_squared_error(y, p)
                ),

            "MAE_grouped":
                mean_absolute_error(y, p)
        })

    return pd.DataFrame(scores)


# ==================================================
# 13. 恢复原Excel格式
# ==================================================

def rebuild_sheet(
    header,
    body,
    observed,
    filled,
    columns_to_fill
):

    output = body.copy()

    for j in columns_to_fill:

        missing_rows = np.flatnonzero(
            ~np.isfinite(
                observed[:, j]
            )
        )

        if len(missing_rows):

            output.iloc[
                missing_rows,
                EXCEL_COLS[j]
            ] = filled[
                missing_rows,
                j
            ]

    return pd.concat(
        [header, output],
        ignore_index=True
    )


# ==================================================
# 14. 主程序
# ==================================================

def main():

    print("=" * 65)
    print("ETII + GHI质量守恒约束")
    print("=" * 65)

    (
        raw,
        header,
        body,
        data,
        aux,
        sources,
        invalid
    ) = load_data()

    print(
        f"样本量：{len(body)}"
    )

    print(
        f"变量数：{data.shape[1]}"
    )

    print(
        f"原始缺失值：{np.isnan(data).sum()}"
    )

    # ----------------------------------------------
    # Step 1：随机五折验证
    # ----------------------------------------------

    print("\n开始随机五折验证...")

    scores = random_mask_fivefold(
        data,
        aux,
        NAMES
    )

    # ----------------------------------------------
    # Step 2：跨文献验证
    # ----------------------------------------------

    if RUN_GROUP_CV:

        group_scores = grouped_target_cv(
            data,
            aux,
            sources
        )

    else:

        group_scores = None

    # ----------------------------------------------
    # Step 3：13输入插补
    # ----------------------------------------------

    print("\n开始13个输入变量插补...")

    input_filled = fit_impute(
        data[:, :N_INPUTS],
        aux,
        NAMES[:N_INPUTS]
    )

    input_filled = apply_bounds(
        data[:, :N_INPUTS],
        input_filled,
        NAMES[:N_INPUTS]
    )

    # 新增GHI约束
    input_filled = enforce_ghi_sum(
        data[:, :N_INPUTS],
        input_filled
    )

    input_only = data.copy()

    input_only[:, :N_INPUTS] = (
        input_filled
    )

    # ----------------------------------------------
    # Step 4：23变量插补
    # ----------------------------------------------

    print("\n开始23个变量完整插补...")

    completed = fit_impute(
        data,
        aux,
        NAMES
    )

    completed = apply_bounds(
        data,
        completed,
        NAMES
    )

    # 新增GHI约束
    completed = enforce_ghi_sum(
        data,
        completed
    )

    # ----------------------------------------------
    # Step 5：校验
    # ----------------------------------------------

    measured = np.isfinite(data)

    assert np.array_equal(
        completed[measured],
        data[measured]
    ), "实测值被修改"

    assert np.isfinite(
        completed
    ).all(), "仍有缺失值"

    check = composition_report(
        data,
        completed
    )

    imputed_rows = (
        check["Original_GHI_missing_count"] > 0
    )

    print(
        "GHI需要插补的记录数：",
        int(imputed_rows.sum())
    )

    print(
        "插补记录的最大GHI总和误差：",
        float(
            check.loc[
                imputed_rows,
                "Deviation_from_100"
            ].abs().max()
        )
    )

    # ----------------------------------------------
    # Step 6：R2汇总
    # ----------------------------------------------

    eligible = (
        scores["Originally_missing"] > 0
    )

    summary = pd.DataFrame([

        {
            "Metric":
                "23变量平均R2_原始ETII",

            "Value":
                scores["R2_model"].mean()
        },

        {
            "Metric":
                "22缺失变量平均R2_原始ETII",

            "Value":
                scores.loc[
                    eligible,
                    "R2_model"
                ].mean()
        },

        {
            "Metric":
                "22缺失变量平均R2_范围约束",

            "Value":
                scores.loc[
                    eligible,
                    "R2_after_bounds"
                ].mean()
        },

        {
            "Metric":
                "22缺失变量平均R2_GHI约束后",

            "Value":
                scores.loc[
                    eligible,
                    "R2_after_constraints"
                ].mean()
        },

        {
            "Metric":
                "13输入平均R2_约束后",

            "Value":
                scores.iloc[
                    :N_INPUTS
                ]["R2_after_constraints"].mean()
        },

        {
            "Metric":
                "10输出平均R2_随机遮蔽",

            "Value":
                scores.iloc[
                    N_INPUTS:
                ]["R2_after_constraints"].mean()
        },

        {
            "Metric":
                "10输出平均R2_跨文献",

            "Value":
                (
                    group_scores[
                        "R2_grouped"
                    ].mean()
                    if group_scores is not None
                    else np.nan
                )
        }
    ])

    # ----------------------------------------------
    # Step 7：缺失值统计
    # ----------------------------------------------

    missing_stats = pd.DataFrame({

        "Variable": NAMES,

        "Role":
            ["Input"] * N_INPUTS
            + ["Target"] * (
                len(NAMES) - N_INPUTS
            ),

        "Observed":
            np.isfinite(data).sum(axis=0),

        "Missing":
            np.isnan(data).sum(axis=0),

        "Missing_percent":
            np.isnan(data).mean(axis=0) * 100
    })

    flags = pd.DataFrame(
        np.isnan(data).astype(int),
        columns=NAMES
    )

    flags.insert(
        0,
        "Index_No",
        body.iloc[:, 0].to_numpy()
    )

    # ----------------------------------------------
    # Step 8：保存Excel
    # ----------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        rebuild_sheet(
            header,
            body,
            data,
            data,
            []
        ).to_excel(
            writer,
            sheet_name="Original",
            header=False,
            index=False
        )

        rebuild_sheet(
            header,
            body,
            data,
            input_only,
            list(range(N_INPUTS))
        ).to_excel(
            writer,
            sheet_name="MICE_InputOnly",
            header=False,
            index=False
        )

        rebuild_sheet(
            header,
            body,
            data,
            completed,
            list(range(len(NAMES)))
        ).to_excel(
            writer,
            sheet_name="MICE_All_Exploratory",
            header=False,
            index=False
        )

        scores.to_excel(
            writer,
            sheet_name="R2_Random5fold",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="R2_Summary",
            index=False
        )

        if group_scores is not None:

            group_scores.to_excel(
                writer,
                sheet_name="R2_Group5fold",
                index=False
            )

        missing_stats.to_excel(
            writer,
            sheet_name="Missing_Stats",
            index=False
        )

        flags.to_excel(
            writer,
            sheet_name="Missing_Flags",
            index=False
        )

        invalid.to_excel(
            writer,
            sheet_name="Invalid_Cells",
            index=False
        )

        check.to_excel(
            writer,
            sheet_name="GHI_Constraint_Check",
            index=False
        )

    print("\n========== R2汇总 ==========")

    print(
        summary.to_string(index=False)
    )

    print(
        "\n结果保存到：",
        OUTPUT_FILE
    )

    print(
        "\n全部完成。"
    )


# ==================================================
# 程序入口
# ==================================================

if __name__ == "__main__":
    main()
