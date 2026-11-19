"""
退货数据清洗 + 退货原因词云（供 app.py / Streamlit 直接调用）

对外函数：
    load_returns_data(csv_path=CSV_PATH)      -> DataFrame   # 读取、过滤空值、英文原因转中文
    get_return_reason_frequency(df)           -> dict        # {中文退货原因: 次数}
    make_wordcloud_figure(freq, font_path)    -> Figure      # 用 NotoSansSC-Regular.ttf 生成词云
    build_wordcloud_from_csv(csv_path)        -> (fig, freq, df)  # 一步到位

命令行预览：
    python returns_analysis.py    # 生成 return_reasons_wordcloud.png 并打印频次
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 无交互后端，Streamlit / 命令行都能稳定出图
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# 路径配置：基于当前文件定位，不依赖运行时的工作目录
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fashion_returns_master.csv")
FONT_PATH = os.path.join(BASE_DIR, "NotoSansSC-Regular.ttf")

# ---------------------------------------------------------------------------
# 退货原因：英文 -> 中文映射字典
# 既覆盖本数据集的真实取值，也兼容题目示例里的别名（defective / too_large ...）
# ---------------------------------------------------------------------------
RETURN_REASON_CN: Dict[str, str] = {
    "not_returned": "未退货",
    "size_too_large": "尺码偏大",
    "too_large": "尺码偏大",
    "size_too_small": "尺码偏小",
    "too_small": "尺码偏小",
    "poor_fit": "版型不合",
    "wrong_color": "颜色不符",
    "changed_mind": "改变主意",
    "quality_defect": "质量破损",
    "defective": "质量破损",
    "damaged": "商品破损",
}

# 这些不是真正的“退货原因”，词云统计时要剔除
NON_RETURN_REASONS = {"未退货"}


def map_reason(reason: object) -> Optional[str]:
    """把单个英文退货原因翻译成中文；无法识别时返回 None。"""
    if not isinstance(reason, str):
        return None
    return RETURN_REASON_CN.get(reason.strip().lower())


def load_returns_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    读取并清洗退货数据。

    清洗步骤：
      1. 过滤空值（整行 dropna，并剔除 return_reason 为空的行）；
      2. return_reason 英文原因映射成中文，写入新列 return_reason_cn。

    :return: 清洗后的 DataFrame（含 return_reason_cn 列）
    """
    df = pd.read_csv(csv_path)

    # 1) 过滤空值
    df = df.dropna()
    df["return_reason"] = df["return_reason"].astype(str).str.strip()
    df = df[df["return_reason"] != ""].copy()

    # 2) 英文原因 -> 中文（未知原因归入“其他”，避免丢数据）
    df["return_reason_cn"] = df["return_reason"].map(map_reason).fillna("其他")
    return df


def get_return_reason_frequency(df: pd.DataFrame) -> Dict[str, int]:
    """
    统计退货原因的中文频次（自动剔除“未退货”）。

    :return: {中文退货原因: 出现次数}，可直接传给 WordCloud.generate_from_frequencies
    """
    if "return_reason_cn" not in df.columns:
        df = df.copy()
        df["return_reason_cn"] = df["return_reason"].map(map_reason).fillna("其他")

    reasons = df.loc[~df["return_reason_cn"].isin(NON_RETURN_REASONS), "return_reason_cn"]
    return reasons.value_counts().to_dict()


def make_wordcloud_figure(
    freq: Dict[str, int],
    font_path: str = FONT_PATH,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
):
    """
    根据 {原因: 次数} 生成中文词云，返回 matplotlib Figure（供 st.pyplot 使用）。

    中文字体固定使用项目根目录的 NotoSansSC-Regular.ttf。
    """
    if not freq:
        raise ValueError("词频为空，无法生成词云。")

    wc = WordCloud(
        font_path=font_path,
        background_color=background_color,
        width=width,
        height=height,
        max_words=200,
        colormap="viridis",
        relative_scaling=0.5,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def build_wordcloud_from_csv(
    csv_path: str = CSV_PATH,
    font_path: str = FONT_PATH,
) -> Tuple[object, Dict[str, int], pd.DataFrame]:
    """
    一步到位：读 CSV -> 清洗 -> 统计频次 -> 生成词云。

    :return: (matplotlib Figure, 原因频次 dict, 清洗后的 DataFrame)
    """
    df = load_returns_data(csv_path)
    freq = get_return_reason_frequency(df)
    fig = make_wordcloud_figure(freq, font_path=font_path)
    return fig, freq, df


def main() -> None:
    """命令行预览：生成词云 PNG（*.png 已被 .gitignore 忽略）并打印原因频次。"""
    fig, freq, df = build_wordcloud_from_csv()

    out_path = os.path.join(BASE_DIR, "return_reasons_wordcloud.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"清洗后数据行数：{len(df)}")
    print("退货原因频次（已剔除未退货）：")
    for reason, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    print(f"词云已保存：{out_path}")


if __name__ == "__main__":
    main()
