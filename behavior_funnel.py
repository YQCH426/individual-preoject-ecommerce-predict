"""
用户行为转化漏斗模块（天池 UserBehavior.csv）

====================================================================
一、数据背景与抽样说明（可直接写进课程报告）
====================================================================
天池电商用户行为数据集 UserBehavior.csv 体积约 3.42GB，行数上亿，
无法一次性读入内存。本模块采用如下策略：

1. 使用 pandas 的 ``chunksize`` **分块读取**（默认每块 100 万行）；
2. 对每一块随机抽取 **10%** 的样本（``sample(frac=0.10)``）；
3. 抽样后**立即统计各行为发生次数并累加**，不把所有抽样明细拼接进内存，
   从而把内存占用控制在一个数据块量级，避免内存溢出；
4. 关键位置使用 ``del`` + ``gc.collect()`` 主动释放，并捕获 ``MemoryError``，
   读取/绘图失败时返回友好提示而不是让页面崩溃。

因此报告里可以这样写：
「因原始数据集过大（约 3.42GB），为保证分析可执行与内存安全，
  采用 pandas 分块读取 + 每块随机抽样 10% 的方式进行建模与可视化。」

====================================================================
二、字段说明
====================================================================
原始文件无表头，字段依次为：
    user_id, item_id, category_id, behavior_type, timestamp

behavior_type 取值：
    pv   -> 浏览
    cart -> 加购
    fav  -> 收藏
    buy  -> 下单

====================================================================
三、对外函数（可直接在 app.py / Streamlit 中调用）
====================================================================
    sample_behavior_counts(...)      # 分块抽样 + 统计各行为次数
    compute_conversion_rates(...)    # 计算各阶段转化率（表格）
    make_behavior_funnel_figure(...) # 生成 Plotly 漏斗图
    build_behavior_funnel(...)       # 一步到位：抽样 -> 计数 -> 出图

Streamlit 中用法：
    from behavior_funnel import build_behavior_funnel
    fig, meta = build_behavior_funnel()
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(meta["error"])
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 路径与常量配置（基于当前文件定位，避免依赖运行时工作目录）
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
USER_BEHAVIOR_CSV = BASE_DIR / "UserBehavior.csv"

# 原始文件无表头，这里显式指定字段名
BEHAVIOR_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]

# 行为英文 -> 中文
BEHAVIOR_CN: Dict[str, str] = {
    "pv": "浏览",
    "cart": "加购",
    "fav": "收藏",
    "buy": "下单",
}

# 漏斗层级顺序（按需求：浏览 -> 加购 -> 收藏 -> 下单）
FUNNEL_ORDER = ["pv", "cart", "fav", "buy"]

# 默认抽样 10%；默认每块 100 万行
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_CHUNK_SIZE = 1_000_000


def sample_behavior_counts(
    csv_path: str | Path = USER_BEHAVIOR_CSV,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    分块读取 UserBehavior.csv，并对每块随机抽样，累计各行为发生次数。

    内存安全要点：
      - 只加载 ``behavior_type`` 一列（usecols），显著降低单块内存；
      - 抽样后立刻 ``value_counts`` 并累加，不保留全部抽样记录；
      - 每处理若干块主动 ``gc.collect()`` 回收内存。

    注意：本函数以“各行为的发生次数”作为漏斗各层数值；
    若改为按去重用户数统计，需要保存抽样后的 user_id 集合，内存开销更大。

    :param csv_path: UserBehavior.csv 路径
    :param sample_ratio: 抽样比例，默认 0.10（即 10%）
    :param chunk_size: 每块读取行数，默认 1,000,000
    :param random_state: 随机种子，保证结果可复现
    :return: {
        "counts":      {中文行为: 次数},
        "raw_counts":  {英文行为: 次数},
        "total_rows":  扫描到的总行数,
        "sampled_rows": 抽样到的总行数,
        "sample_ratio": 实际抽样比例,
    }
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到用户行为数据文件：{path}")

    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio 必须位于 (0, 1] 区间内。")

    raw_counts: Dict[str, int] = {behavior: 0 for behavior in FUNNEL_ORDER}
    total_rows = 0
    sampled_rows = 0

    # 分块读取：header=None（原始文件无表头）；只取 behavior_type 列省内存
    reader = pd.read_csv(
        path,
        header=None,
        names=BEHAVIOR_COLUMNS,
        usecols=["behavior_type"],
        dtype={"behavior_type": "category"},
        chunksize=chunk_size,
        engine="c",
    )

    for chunk_index, chunk in enumerate(reader):
        total_rows += len(chunk)

        # 每块随机抽样 10%；random_state + 块序号，保证可复现且各块抽样不同
        sampled = chunk.sample(
            frac=sample_ratio,
            random_state=random_state + chunk_index,
        )
        sampled_rows += len(sampled)

        # 统计本块抽样中各行为次数，并累加到总计数
        value_counts = sampled["behavior_type"].value_counts()
        for behavior in FUNNEL_ORDER:
            raw_counts[behavior] += int(value_counts.get(behavior, 0))

        # 主动释放本块对象，降低内存峰值
        del chunk, sampled, value_counts

        # 每处理 10 块回收一次内存，进一步防止内存持续增长
        if (chunk_index + 1) % 10 == 0:
            gc.collect()

    # 英文行为名 -> 中文行为名
    counts: Dict[str, int] = {
        BEHAVIOR_CN[behavior]: raw_counts[behavior] for behavior in FUNNEL_ORDER
    }

    actual_ratio = (sampled_rows / total_rows) if total_rows else 0.0

    return {
        "counts": counts,
        "raw_counts": raw_counts,
        "total_rows": total_rows,
        "sampled_rows": sampled_rows,
        "sample_ratio": actual_ratio,
    }


def compute_conversion_rates(counts: Dict[str, int]) -> pd.DataFrame:
    """
    根据各阶段行为次数计算转化率，返回可直接展示的表格。

    - 相对首层转化率 = 当前层 / 浏览层
    - 相对上一层转化率 = 当前层 / 上一层
    """
    stages = [BEHAVIOR_CN[behavior] for behavior in FUNNEL_ORDER]
    values = [int(counts.get(stage, 0)) for stage in stages]

    first_value = values[0] if values else 0
    rows = []
    for i, (stage, value) in enumerate(zip(stages, values)):
        if i == 0:
            previous_rate = 1.0
        else:
            previous_value = values[i - 1]
            previous_rate = (value / previous_value) if previous_value else 0.0

        to_first_rate = (value / first_value) if first_value else 0.0
        rows.append(
            {
                "阶段": stage,
                "次数": value,
                "相对首层转化率": f"{to_first_rate:.2%}",
                "相对上一层转化率": f"{previous_rate:.2%}",
            }
        )
    return pd.DataFrame(rows)


def make_behavior_funnel_figure(
    counts: Dict[str, int],
    title: str = "用户行为转化漏斗（随机抽样 10%）",
) -> go.Figure:
    """
    根据行为计数生成 Plotly 漏斗图。

    :param counts: {中文行为: 次数}，如 {"浏览": 100, "加购": 30, ...}
    :param title: 图表标题
    :return: plotly.graph_objects.Figure，可直接 st.plotly_chart(fig) 展示
    """
    stages = [BEHAVIOR_CN[behavior] for behavior in FUNNEL_ORDER]
    values = [int(counts.get(stage, 0)) for stage in stages]

    if not any(values):
        raise ValueError("各阶段行为次数均为 0，无法绘制漏斗图。")

    funnel_df = pd.DataFrame({"阶段": stages, "次数": values})

    fig = px.funnel(
        funnel_df,
        x="次数",
        y="阶段",
        title=title,
    )
    # 显示具体次数 + 相对首层（浏览）的百分比
    fig.update_traces(textinfo="value+percent initial")
    fig.update_layout(
        height=480,
        margin=dict(l=90, r=40, t=70, b=40),
    )
    return fig


def build_behavior_funnel(
    csv_path: str | Path = USER_BEHAVIOR_CSV,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = 42,
    title: Optional[str] = None,
) -> Tuple[Optional[go.Figure], Dict[str, object]]:
    """
    一步到位：分块抽样 -> 统计行为次数 -> 生成 Plotly 漏斗图。

    内置异常捕获（文件缺失、内存不足、解析失败等），
    返回 ``(fig, meta)``；失败时 ``fig`` 为 None，``meta["error"]`` 为错误提示，
    便于 Streamlit 页面友好展示而不至于崩溃。

    :return: (plotly Figure 或 None, 包含计数与错误信息的 meta 字典)
    """
    meta: Dict[str, object] = {
        "counts": {},
        "raw_counts": {},
        "total_rows": 0,
        "sampled_rows": 0,
        "sample_ratio": 0.0,
        "error": None,
    }

    try:
        info = sample_behavior_counts(
            csv_path=csv_path,
            sample_ratio=sample_ratio,
            chunk_size=chunk_size,
            random_state=random_state,
        )
        meta.update(info)

        if title is None:
            title = f"用户行为转化漏斗（分块随机抽样 {info['sample_ratio']:.1%}）"

        fig = make_behavior_funnel_figure(info["counts"], title=title)
        return fig, meta

    except FileNotFoundError as exc:
        meta["error"] = str(exc)
        return None, meta
    except MemoryError:
        meta["error"] = "内存不足：请调小 chunk_size 或 sample_ratio 后重试。"
        return None, meta
    except ValueError as exc:
        meta["error"] = f"参数或数据异常：{exc}"
        return None, meta
    except Exception as exc:  # 兜底捕获，避免页面因未知错误崩溃
        meta["error"] = f"读取或绘制用户行为漏斗失败：{exc}"
        return None, meta


def main() -> None:
    """命令行演示：抽样统计并保存漏斗图到 output/behavior_funnel.html。"""
    fig, meta = build_behavior_funnel()

    if fig is None:
        print("生成失败：", meta.get("error"))
        return

    print(f"扫描总行数：{meta['total_rows']:,}")
    print(f"抽样行数：{meta['sampled_rows']:,}（实际抽样比例 {meta['sample_ratio']:.2%}）")
    print("\n各阶段行为次数与转化率：")
    print(compute_conversion_rates(meta["counts"]).to_string(index=False))

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "behavior_funnel.html"
    fig.write_html(str(out_path))
    print(f"\n漏斗图已保存：{out_path}")


if __name__ == "__main__":
    main()
