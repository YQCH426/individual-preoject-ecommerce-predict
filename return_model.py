"""
随机森林退货预测模型（基于 Kaggle fashion_returns_master.csv）
====================================================================

一、任务说明
--------------------------------------------------------------------
输入订单在下单前即可获得的特征（品类、面料、价格、尺码匹配度、历史行为等），
预测该订单是否会被退货（``returned`` = 1 表示退货，0 表示未退货）。
这是一个典型的**二分类**问题，评价重点在"能不能把会退货的订单捞出来"，
因此除准确率外，还需要重点关注**召回率 Recall** 与 **AUC**。

二、建模流程（与课程报告章节一一对应）
--------------------------------------------------------------------
1. 读取数据，检查目标分布与缺失值；
2. 定义特征与目标，**剔除泄漏字段**（``return_reason`` 与目标同源，
   ``order_id`` 无预测意义）；
3. 类别特征做 One-Hot 编码，数值特征原样保留，
   用 ``ColumnTransformer + Pipeline`` 封装，避免预处理泄漏；
4. 分层抽样划分训练集 / 测试集（默认 8 : 2）；
5. 训练随机森林（200 棵树，``class_weight="balanced"`` 提升召回率）；
6. 输出准确率、召回率、F1、AUC、混淆矩阵、特征重要性；
7. 封装预测函数，对单笔 / 批量订单输出退货概率、风险等级与建议。

三、对外接口（可直接在 app.py / Streamlit 中调用）
--------------------------------------------------------------------
    train_return_model(...)        # 一步到位：读数据 -> 训练 -> 评估
    predict_return_risk(...)       # 预测函数：订单 -> 退货概率 / 风险等级 / 建议
    plot_target_distribution(...)  # 目标变量分布图
    plot_confusion_matrix(...)     # 混淆矩阵图
    plot_roc_curve(...)            # ROC 曲线
    plot_feature_importance(...)   # 特征重要性条形图
    business_conclusions(...)      # 业务结论（字符串列表，可直接渲染）

命令行运行（生成 output/ 下的全部图表并打印指标）：

    python return_model.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 无交互后端：Streamlit / 命令行都能稳定出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 路径与全局配置（基于当前文件定位，避免依赖运行时工作目录）
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "fashion_returns_master.csv"
FONT_PATH = BASE_DIR / "NotoSansSC-Regular.ttf"
OUTPUT_DIR = BASE_DIR / "output"  # 已被 .gitignore 忽略，不入库

# 随机种子：保证实验结果可复现
RANDOM_STATE = 42

# 中文字体：让图表标题、坐标轴、中文特征名正常显示
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ---------------------------------------------------------------------------
# 字段定义
# ---------------------------------------------------------------------------
TARGET = "returned"

# 必须剔除的列：目标本身、泄漏字段、无意义编号
DROP_COLUMNS = [
    "returned",       # 目标变量本身
    "return_reason",  # 退货原因：not_returned 与目标完全对应，属于严重数据泄漏
    "order_id",       # 订单编号：无业务预测意义
]

# 类别特征：需要 One-Hot 编码
CATEGORICAL_FEATURES = [
    "marketplace_region",  # 市场 / 地区
    "category",            # 商品品类
    "fabric",              # 面料
    "fit_type",            # 版型
    "device",              # 下单设备
]

# 数值 / 二值特征：直接送入模型（随机森林对量纲不敏感，无需标准化）
NUMERIC_FEATURES = [
    "price_usd",                  # 价格（美元）
    "discount_pct",               # 折扣比例
    "is_premium",                 # 是否高价 / 精选商品
    "size_ordered",               # 下单尺码
    "size_usual",                 # 平时尺码
    "size_mismatch",              # 尺码偏差程度（|下单尺码 - 平时尺码|）
    "ordered_multiple_sizes",     # 是否一次购买多个尺码
    "customer_prior_orders",      # 历史订单数
    "customer_prior_return_rate", # 历史退货率
    "avg_review_rating",          # 平均评分
    "num_reviews",                # 评论数
    "reviews_read",               # 是否阅读评论
    "size_chart_viewed",          # 是否查看尺码表
    "model_shown",                # 是否展示模特图
    "is_gift",                    # 是否礼物
    "days_to_delivery",           # 送达天数
    "return_shipping_free",       # 退货是否包邮
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# 中文特征名：用于图表展示与业务结论，避免报告里出现英文列名
FEATURE_CN: Dict[str, str] = {
    "marketplace_region": "市场地区",
    "category": "商品品类",
    "fabric": "面料",
    "fit_type": "版型",
    "device": "下单设备",
    "price_usd": "价格",
    "discount_pct": "折扣比例",
    "is_premium": "是否精选商品",
    "size_ordered": "下单尺码",
    "size_usual": "平时尺码",
    "size_mismatch": "尺码偏差",
    "ordered_multiple_sizes": "一次买多个尺码",
    "customer_prior_orders": "历史订单数",
    "customer_prior_return_rate": "历史退货率",
    "avg_review_rating": "平均评分",
    "num_reviews": "评论数",
    "reviews_read": "是否阅读评论",
    "size_chart_viewed": "是否查看尺码表",
    "model_shown": "是否展示模特图",
    "is_gift": "是否礼物",
    "days_to_delivery": "送达天数",
    "return_shipping_free": "退货包邮",
}


# ---------------------------------------------------------------------------
# 数据结构：一次训练的全部产物
# ---------------------------------------------------------------------------
@dataclass
class ReturnModelResult:
    """
    训练结果容器。

    把"模型 + 指标 + 图表数据 + 原始数据"打包在一起，
    方便 Streamlit 侧缓存一次、多处复用（指标卡、混淆矩阵、特征重要性、单笔预测）。

    :param model: 训练好的 sklearn Pipeline（预处理 + 随机森林）
    :param metrics: 测试集指标字典（accuracy / precision / recall / f1 / auc）
    :param confusion: 2x2 混淆矩阵，顺序为 [[TN, FP], [FN, TP]]
    :param importance_df: 特征重要性表（feature / feature_clean / feature_cn / importance）
    :param y_test: 测试集真实标签
    :param y_pred: 测试集预测标签
    :param y_proba: 测试集"退货"预测概率
    :param train_size: 训练集样本数
    :param test_size: 测试集样本数
    :param defaults: 各特征在训练集中的中位数 / 众数，供单笔预测时兜底填充
    :param data: 清洗后的完整数据（含 returned），用于分组退货率佐证
    """

    model: Pipeline
    metrics: Dict[str, float]
    confusion: np.ndarray
    importance_df: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    train_size: int
    test_size: int
    defaults: Dict[str, object]
    data: pd.DataFrame


# ---------------------------------------------------------------------------
# 1. 数据读取与特征准备
# ---------------------------------------------------------------------------
def load_returns_dataset(csv_path: str | Path = CSV_PATH) -> pd.DataFrame:
    """
    读取服装退货数据集，并做最小必要清洗。

    清洗原则：只做"不引入未来信息"的处理，避免污染建模。
      - 删除目标变量缺失的行（无标签无法训练）；
      - 类别列统一转成字符串，防止数值 / 字符串混用导致编码报错。

    :param csv_path: CSV 文件路径
    :return: 清洗后的 DataFrame
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到退货数据文件：{path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET]).copy()

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    构造特征矩阵 X 与目标向量 y。

    注意：这里**只取 FEATURE_COLUMNS**，天然把 ``return_reason``、``order_id``
    等泄漏 / 无用字段排除在外，避免模型"偷看答案"。

    :param df: 原始 DataFrame
    :return: (X, y)
    """
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段：{missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# 2. 预处理 + 模型管线
# ---------------------------------------------------------------------------
def build_pipeline(random_state: int = RANDOM_STATE, n_estimators: int = 200) -> Pipeline:
    """
    构建"One-Hot 编码 + 随机森林"的完整 Pipeline。

    为什么用 Pipeline：
      - 预处理只在训练集上 fit，测试集只 transform，**避免数据泄漏**；
      - 模型对象自带预处理，预测新订单时不需要手工重复编码步骤。

    参数说明：
      - ``handle_unknown="ignore"``：测试 / 新数据出现未见过的类别时不报错；
      - ``class_weight="balanced"``：提高少数类（已退货）权重，重点提升召回率；
      - ``n_jobs=-1``：使用全部 CPU 核心并行训练。

    :return: 未训练的 sklearn Pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,   # 决策树数量
        class_weight="balanced",     # 平衡类别权重
        random_state=random_state,   # 随机种子，保证可复现
        n_jobs=-1,                   # 多核并行
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _extract_importance_df(model: Pipeline) -> pd.DataFrame:
    """
    从训练好的 Pipeline 中提取特征重要性，并还原可读的特征名。

    One-Hot 编码后特征名形如 ``cat__category_shoes`` / ``num__price_usd``，
    这里去掉前缀，并尽量映射为中文名（类别特征保留"字段_取值"，便于定位）。

    :return: 按重要性降序排列的 DataFrame
    """
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["feature_clean"] = (
        df["feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
    )

    def to_cn(name: str) -> str:
        """把 num__price_usd / cat__category_shoes 转成中文可读名称。"""
        for column, cn in FEATURE_CN.items():
            if name == column:
                return cn
            if name.startswith(column + "_"):
                return f"{cn}={name[len(column) + 1:]}"
        return name

    df["feature_cn"] = df["feature_clean"].map(to_cn)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _build_defaults(X: pd.DataFrame) -> Dict[str, object]:
    """
    计算各特征的"典型值"，用于单笔预测时补全未填写的字段。

    数值特征取中位数（抗异常值），类别特征取众数。

    :return: {特征名: 典型值}
    """
    defaults: Dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            mode = X[column].mode()
            defaults[column] = mode.iloc[0] if len(mode) else ""
        else:
            defaults[column] = float(X[column].median())
    return defaults


# ---------------------------------------------------------------------------
# 3. 训练与评估
# ---------------------------------------------------------------------------
def train_return_model(
    csv_path: str | Path = CSV_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 200,
) -> ReturnModelResult:
    """
    一步到位：读数据 -> 定义特征 -> 划分数据集 -> 训练随机森林 -> 评估。

    划分策略：``stratify=y`` 分层抽样，保证训练 / 测试集中退货比例一致，
    避免因随机划分造成评估结果波动。

    :param csv_path: 退货数据 CSV 路径
    :param test_size: 测试集占比，默认 0.2（8 : 2）
    :param random_state: 随机种子
    :param n_estimators: 随机森林树的数量
    :return: ReturnModelResult，包含模型、指标、混淆矩阵、特征重要性等
    """
    df = load_returns_dataset(csv_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(random_state=random_state, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # 预测类别（0/1）与"退货"概率（用于 AUC / ROC）
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }

    return ReturnModelResult(
        model=model,
        metrics=metrics,
        confusion=confusion_matrix(y_test, y_pred),
        importance_df=_extract_importance_df(model),
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
        defaults=_build_defaults(X_train),
        data=df,
    )


# ---------------------------------------------------------------------------
# 4. 预测函数（单笔 / 批量）
# ---------------------------------------------------------------------------
def predict_return_risk(
    result: ReturnModelResult,
    orders: pd.DataFrame | Dict[str, object] | List[Dict[str, object]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    对订单做退货风险预测（支持单笔字典、字典列表、DataFrame 三种入参）。

    缺失的特征字段会用训练集典型值自动补全，因此调用方只需提供关心的字段，
    例如在 Streamlit 表单里只填"品类 + 尺码偏差 + 历史退货率"即可。

    风险分级（基于退货概率）：
      - < 0.35          -> 低风险
      - 0.35 ~ 0.60     -> 中风险
      - >= 0.60         -> 高风险

    :param result: train_return_model 的返回值
    :param orders: 订单数据
    :param threshold: 判定为"会退货"的概率阈值，默认 0.5
    :return: 原订单数据 + return_probability / predicted_returned / risk_level / advice
    """
    if isinstance(orders, dict):
        orders_df = pd.DataFrame([orders])
    elif isinstance(orders, list):
        orders_df = pd.DataFrame(orders)
    else:
        orders_df = orders.copy()

    # 用训练集典型值补全缺失字段，保证 Pipeline 能拿到完整特征
    full = orders_df.copy()
    for column, default in result.defaults.items():
        if column not in full.columns:
            full[column] = default
        else:
            full[column] = full[column].fillna(default)

    # 类别列统一转字符串，与训练时保持一致
    for column in CATEGORICAL_FEATURES:
        full[column] = full[column].astype(str)

    probabilities = result.model.predict_proba(full[FEATURE_COLUMNS])[:, 1]

    # 结果拼接在原始输入右侧，方便对照
    output = orders_df.copy()
    output["return_probability"] = np.round(probabilities, 4)
    output["predicted_returned"] = (probabilities >= threshold).astype(int)
    output["risk_level"] = [
        "低风险" if p < 0.35 else ("中风险" if p < 0.60 else "高风险")
        for p in probabilities
    ]
    output["advice"] = [
        _risk_advice(p, row) for p, (_, row) in zip(probabilities, full.iterrows())
    ]
    return output


def _risk_advice(probability: float, row: pd.Series) -> str:
    """
    生成单条订单的避雷建议（规则 + 关键风险点提示）。

    :param probability: 退货概率
    :param row: 已补全特征的订单行
    :return: 一句话建议
    """
    if probability >= 0.60:
        head = "退货风险高，建议暂缓下单"
    elif probability >= 0.35:
        head = "退货风险中等，建议下单前再核对商品信息"
    else:
        head = "退货风险较低，可放心下单"

    hints: List[str] = []
    if float(row.get("size_mismatch", 0)) >= 1:
        hints.append("尺码存在偏差，请对照尺码表或咨询客服")
    if float(row.get("ordered_multiple_sizes", 0)) == 1:
        hints.append("多码试穿会提高退货概率，确认尺码后只买一码")
    if float(row.get("customer_prior_return_rate", 0)) >= 0.3:
        hints.append("你的历史退货率偏高，注意克制冲动消费")
    if float(row.get("avg_review_rating", 5)) < 4:
        hints.append("商品评分偏低，建议多看看差评与买家秀")
    if float(row.get("return_shipping_free", 1)) == 0:
        hints.append("该订单退货不包邮，退货成本需要自行承担")

    if hints:
        return f"{head}；" + "；".join(hints) + "。"
    return f"{head}。"


# ---------------------------------------------------------------------------
# 5. 可视化（统一返回 matplotlib Figure，便于 st.pyplot 直接展示）
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame):
    """
    绘制目标变量分布（未退货 vs 已退货），用于说明类别是否失衡。

    :param df: 含 returned 列的 DataFrame
    :return: matplotlib Figure
    """
    counts = df[TARGET].value_counts().sort_index()
    labels = ["未退货 (0)", "已退货 (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#C44E52"])
    ax.set_title("目标变量 returned 分布")
    ax.set_ylabel("订单数")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray):
    """
    绘制混淆矩阵热力图，并在格子中标注样本数与业务含义。

    :param cm: 2x2 混淆矩阵，顺序 [[TN, FP], [FN, TP]]
    :return: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测未退货", "预测已退货"])
    ax.set_yticklabels(["真实未退货", "真实已退货"])
    ax.set_xlabel("预测结果")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵（测试集）")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, auc: float):
    """
    绘制 ROC 曲线，直观展示模型区分"退货 / 未退货"的能力。

    :param y_test: 真实标签
    :param y_proba: 预测为退货的概率
    :param auc: AUC 数值（用于图例展示）
    :return: matplotlib Figure
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"随机森林 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="随机猜测")
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR（召回率）")
    ax.set_title("ROC 曲线")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """
    绘制特征重要性 Top N 横向条形图（重要性最高的排在最上方）。

    :param importance_df: 含 feature_cn / importance 两列的重要性表
    :param top_n: 展示的特征数量
    :return: matplotlib Figure
    """
    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(top_df))))
    ax.barh(top_df["feature_cn"], top_df["importance"], color="#4C72B0")
    ax.set_xlabel("重要性 Importance")
    ax.set_title(f"随机森林特征重要性 Top {top_n}")
    for i, value in enumerate(top_df["importance"]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. 业务结论（供课程报告与 Streamlit 页面直接使用）
# ---------------------------------------------------------------------------
def business_conclusions(result: ReturnModelResult, top_n: int = 5) -> List[str]:
    """
    结合"模型特征重要性"与"真实数据分组退货率"，生成可写入报告的业务结论。

    每条结论都由数据支撑（重要性排名 / 分组退货率），避免空泛表述。

    :param result: 训练结果
    :param top_n: 取重要性前 N 的特征做解读
    :return: 结论字符串列表（Markdown 项目符号格式）
    """
    df = result.data
    metrics = result.metrics

    # 指标层面的整体结论
    conclusions: List[str] = [
        f"**整体表现**：测试集准确率 {metrics['accuracy']:.2%}、召回率 {metrics['recall']:.2%}、"
        f"F1 {metrics['f1']:.2%}、AUC {metrics['auc']:.4f}。"
        f"AUC 接近或超过 0.8 说明模型对退货订单具备较好的排序与筛查能力。"
    ]

    # 特征重要性结论
    top_features = result.importance_df.head(top_n)
    feature_text = "、".join(
        f"{row.feature_cn}（{row.importance:.3f}）" for row in top_features.itertuples()
    )
    conclusions.append(f"**关键驱动因素**：重要性排名前 {top_n} 的特征为 {feature_text}。")

    # 分组退货率佐证：尺码偏差
    if "size_mismatch" in df.columns:
        grouped = df.groupby("size_mismatch")[TARGET].mean()
        detail = "、".join(f"偏差 {int(idx)} 码 {rate:.1%}" for idx, rate in grouped.items())
        conclusions.append(
            f"**尺码是头号风险源**：{detail}。尺码偏差越大，退货率越高，"
            f"这解释了模型为何高度依赖尺码类特征。"
        )

    # 分组退货率佐证：多码下单
    if "ordered_multiple_sizes" in df.columns:
        grouped = df.groupby("ordered_multiple_sizes")[TARGET].mean()
        if 0 in grouped.index and 1 in grouped.index:
            conclusions.append(
                f"**多码试穿不可取**：一次购买多个尺码的订单退货率 {grouped[1]:.1%}，"
                f"明显高于只买一码的 {grouped[0]:.1%}，本质仍是“对尺码没把握”。"
            )

    # 分组退货率佐证：历史退货率
    if "customer_prior_return_rate" in df.columns:
        binned = pd.cut(
            df["customer_prior_return_rate"],
            bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
            include_lowest=True,
        )
        grouped = df.groupby(binned, observed=True)[TARGET].mean()
        lowest, highest = grouped.iloc[0], grouped.iloc[-1]
        conclusions.append(
            f"**退货习惯会延续**：历史退货率最低区间的订单实际退货率约 {lowest:.1%}，"
            f"最高区间升至约 {highest:.1%}，可作为下单前的重要自检指标。"
        )

    # 品类差异
    if "category" in df.columns:
        grouped = df.groupby("category")[TARGET].mean().sort_values(ascending=False)
        conclusions.append(
            f"**品类差异明显**：退货率最高的品类为 {grouped.index[0]}（{grouped.iloc[0]:.1%}），"
            f"最低的为 {grouped.index[-1]}（{grouped.iloc[-1]:.1%}），"
            f"裤装、鞋靴等应重点核对尺码与版型。"
        )

    # 落地建议
    conclusions.append(
        "**落地建议**：下单前对照尺码表、优先选择高评分商品、避免多码试穿；"
        "平台侧可用该模型对高风险订单做尺码提醒或运费险推荐，降低退货成本。"
    )
    return conclusions


# ---------------------------------------------------------------------------
# 7. 命令行入口
# ---------------------------------------------------------------------------
def main() -> None:
    """命令行演示：训练模型、打印指标、保存全部图表到 output/。"""
    result = train_return_model()
    metrics = result.metrics

    print("=" * 46)
    print("随机森林退货预测模型 —— 测试集表现")
    print("=" * 46)
    print(f"训练集样本数 : {result.train_size:,}")
    print(f"测试集样本数 : {result.test_size:,}")
    print(f"准确率 Accuracy  : {metrics['accuracy']:.4f}")
    print(f"精确率 Precision : {metrics['precision']:.4f}")
    print(f"召回率 Recall    : {metrics['recall']:.4f}")
    print(f"F1 分数 F1-score : {metrics['f1']:.4f}")
    print(f"AUC              : {metrics['auc']:.4f}")

    print("\n分类报告：")
    print(
        classification_report(
            result.y_test,
            result.y_pred,
            target_names=["未退货 (0)", "已退货 (1)"],
            digits=4,
        )
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    figures = {
        "target_distribution.png": plot_target_distribution(result.data),
        "confusion_matrix.png": plot_confusion_matrix(result.confusion),
        "roc_curve.png": plot_roc_curve(result.y_test, result.y_proba, metrics["auc"]),
        "feature_importance.png": plot_feature_importance(result.importance_df),
    }
    for filename, fig in figures.items():
        fig.savefig(OUTPUT_DIR / filename, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存：{OUTPUT_DIR / filename}")

    print("\n业务结论：")
    for index, conclusion in enumerate(business_conclusions(result), start=1):
        print(f"{index}. {conclusion.replace('**', '')}")


if __name__ == "__main__":
    main()
