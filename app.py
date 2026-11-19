import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from openai import OpenAI

# 新增：天池用户行为转化漏斗（分块抽样 + Plotly 绘图）
from behavior_funnel import build_behavior_funnel, compute_conversion_rates

# 新增：随机森林退货预测（训练、评估、预测、业务结论）
from return_model import (
    business_conclusions,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    predict_return_risk,
    train_return_model,
)

st.set_page_config(page_title="电商消费避雷透视镜", page_icon="🛒", layout="wide")

st.title("🛒 电商消费避雷数据分析与 AI 可视化平台")
st.markdown("基于真实行为与售后数据，打破信息不对称。")

# ================= 侧边栏 =================
st.sidebar.header("筛选条件")
category = st.sidebar.selectbox("选择商品品类", ["女装", "鞋靴", "美妆", "数码"])


@st.cache_resource(ttl=0, show_spinner=False)
def load_behavior_funnel_cached(random_state: int = 42):
    """
    缓存天池大数据的分块抽样结果。

    按需求设置 ttl=0：缓存条目立即过期，每次页面 rerun 都会重新扫描文件，
    保证看到的是最新抽样结果。若希望长期复用缓存，可改为 ttl=None。

    random_state 作为缓存键的一部分：点击“重新抽样”按钮时传入新的随机种子，
    即可得到一组不同的抽样结果。
    """
    return build_behavior_funnel(random_state=random_state)


@st.cache_resource(ttl=0, show_spinner=False)
def load_return_model_cached():
    """
    缓存随机森林训练结果。

    按需求设置 ttl=0：缓存条目立即过期，每次页面 rerun 都会重新训练模型。
    若希望长期复用缓存，可改为 ttl=None。
    """
    return train_return_model()


# ================= 顶部标签页 =================
tab_avoidance, tab_behavior, tab_model = st.tabs(
    ["🛒 电商避雷分析", "📊 用户行为转化分析", "🤖 退货预测建模"]
)

# ===========================================================================
# 标签页一：原有电商避雷分析
# 说明：以下模块一/二/三为原有代码，仅整体缩进放入标签页，业务逻辑保持不变。
# ===========================================================================
with tab_avoidance:
    # ================= 模块一：事前预警 =================
    st.header("1. 事前预警：高弃单率雷达")
    st.write(f"当前品类：**{category}**。观察消费者的加购与最终购买转化情况。")

    # 模拟漏斗图数据（后续可替换为真实的 behavior_sample.csv）
    funnel_data = dict(
        number=[1000, 600, 400, 200],
        stage=["浏览 (pv)", "加购 (cart)", "收藏 (fav)", "购买 (buy)"]
    )
    fig_funnel = px.funnel(funnel_data, x='number', y='stage', title=f"{category} 购买转化漏斗")
    st.plotly_chart(fig_funnel, use_container_width=True)

    # ================= 模块二：事后实锤 =================
    st.header("2. 事后实锤：退货原因高频词")
    st.write("基于真实退货数据统计，暴露商品质量问题。")

    # 调用 returns_analysis.py 中的函数：读取 CSV -> 清洗 -> 英文原因转中文 -> 生成词云
    try:
        from returns_analysis import build_wordcloud_from_csv

        fig_wc, reason_freq, returns_df = build_wordcloud_from_csv()
        st.pyplot(fig_wc)

        with st.expander("查看退货原因频次明细"):
            freq_table = sorted(reason_freq.items(), key=lambda item: item[1], reverse=True)
            st.dataframe(
                pd.DataFrame(freq_table, columns=["退货原因", "出现次数"]),
                use_container_width=True,
            )
    except FileNotFoundError:
        st.info("提示：未找到 fashion_returns_master.csv，本次跳过词云图展示。")
    except Exception as e:
        st.warning(f"词云图生成失败：{e}")

    # ================= 模块三：AI 智能导购 =================
    st.header("3. AI 智能避雷导购")
    st.write("结合弃单率与退货词频，由大语言模型生成避雷建议。")

    if st.button("生成避雷建议"):
        with st.spinner("AI 正在研判当前品类的风险..."):
            
            # 读取 API Key (从 .streamlit/secrets.toml 读取)
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            
            if not api_key:
                st.error("⚠️ 未检测到 API Key！请在 .streamlit/secrets.toml 中填入 DEEPSEEK_API_KEY。这里先展示模拟数据：")
                st.success("🔍 避雷建议：该品类弃单率高达 65%，主要退货原因为材质不符。建议您避开无运费险的店铺。")
            else:
                try:
                    # 初始化 DeepSeek 客户端 (如果你用的是 OpenAI，改 base_url 和 model 即可)
                    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                    
                    prompt = f"你是一个专业的电商避雷导购。当前品类是：{category}。已知该品类加购弃单率高，且退货原因多为材质不符、货不对板、起球异味。请给消费者提供3条精炼、实用的避雷建议，总字数在100字以内。"
                    
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个直言不讳的电商避雷专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    advice = response.choices[0].message.content
                    st.success(f"🔍 AI 避雷建议：\n\n{advice}")
                    
                except Exception as e:
                    st.error(f"调用 AI 接口失败，请检查 API Key 或网络。报错信息：{e}")

# ===========================================================================
# 标签页二：用户行为转化分析（新增）
# ===========================================================================
with tab_behavior:
    st.header("天池电商用户行为转化漏斗分析")

    # 1) 抽样说明
    st.markdown(
        """
        **抽样说明**：天池电商用户行为数据集 `UserBehavior.csv` 体积约 **3.42GB**（上亿行），
        无法一次性读入内存。因此采用 pandas 的 `chunksize` **分块读取**（默认每块 100 万行），
        并对每一块**随机抽取 10%** 的数据进行分析，逐块累计各行为发生次数，
        在保证内存安全的前提下完成漏斗统计与可视化。
        """
    )

    # 2) 重新抽样按钮：递增随机种子并清空缓存，本次 rerun 会读取到不同的样本
    if "behavior_random_state" not in st.session_state:
        st.session_state.behavior_random_state = 42

    resample_column, seed_column = st.columns([1, 4])
    if resample_column.button("🔄 重新抽样", use_container_width=True):
        st.session_state.behavior_random_state += 1
        load_behavior_funnel_cached.clear()
        st.rerun()
    seed_column.caption(
        f"当前随机种子：{st.session_state.behavior_random_state}"
        "（点击“重新抽样”将换一组随机样本）"
    )

    # 3) 加载提示：大文件分块读取较慢，先提示；加载完成后自动移除
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.info("正在分块读取并抽样天池大数据，请稍候...")

    fig_behavior, behavior_meta = load_behavior_funnel_cached(
        st.session_state.behavior_random_state
    )
    loading_placeholder.empty()

    if fig_behavior is not None:
        # 4) 渲染 Plotly 漏斗图
        st.plotly_chart(fig_behavior, use_container_width=True)

        st.caption(
            f"扫描总行数：{behavior_meta['total_rows']:,} ｜ "
            f"抽样行数：{behavior_meta['sampled_rows']:,} ｜ "
            f"实际抽样比例：{behavior_meta['sample_ratio']:.2%}"
        )

        # 5) 转化表格：浏览、加购、收藏、下单 的数量与转化率
        st.subheader("各阶段数量与转化率")
        conversion_df = compute_conversion_rates(behavior_meta["counts"])
        st.dataframe(conversion_df, use_container_width=True)

        # 6) 业务解读
        st.subheader("业务解读")
        st.markdown(
            """
            - **浏览 → 加购流失极高**：绝大多数浏览用户停留在浏览阶段，加购转化率仅约 6%，
              说明大量流量尚未形成明确购买意向，优化商品详情页、尺码信息与推荐精准度是关键。
            - **下单环节转化相对不错**：从收藏到下单的转化率约 70%，
              说明用户一旦产生明确意向，最终下单的意愿较强。
            - **收藏是重要中间环节**：加购 → 收藏转化约 52%，
              可通过收藏夹提醒、限时优惠等运营手段促成最终下单。
            """
        )
    else:
        st.warning(f"用户行为漏斗生成失败：{behavior_meta.get('error')}")

# ===========================================================================
# 标签页三：退货预测建模（随机森林，新增）
# 说明：训练逻辑全部封装在 return_model.py，本页只负责展示与交互。
# ===========================================================================
with tab_model:
    st.header("退货预测建模：随机森林")
    st.markdown(
        """
        以 `fashion_returns_master.csv` 为数据源，预测订单**是否会被退货**（`returned`）。

        - **特征**：品类、面料、版型、价格、尺码匹配度、历史退货率、评分、物流等下单前可获得的信息；
        - **防泄漏**：剔除 `return_reason`（与目标同源）与 `order_id`（无预测意义）；
        - **预处理**：类别特征 One-Hot 编码，数值特征原样保留，预处理与模型封装在同一个 Pipeline 中；
        - **划分**：按目标分层抽样，训练集 / 测试集 = 8 : 2；
        - **模型**：200 棵决策树的随机森林，`class_weight="balanced"` 提升退货订单的召回率。
        """
    )

    try:
        with st.spinner("正在训练随机森林模型（首次约需数秒）..."):
            model_result = load_return_model_cached()
    except FileNotFoundError:
        model_result = None
        st.info("提示：未找到 fashion_returns_master.csv，退货预测模块暂不可用。")
    except Exception as exc:  # 兜底：避免训练异常导致整页崩溃
        model_result = None
        st.warning(f"模型训练失败：{exc}")

    if model_result is not None:
        metrics = model_result.metrics

        # ---------- 1) 核心指标 ----------
        st.subheader("1. 测试集模型指标")
        metric_columns = st.columns(5)
        metric_columns[0].metric("准确率 Accuracy", f"{metrics['accuracy']:.2%}")
        metric_columns[1].metric("精确率 Precision", f"{metrics['precision']:.2%}")
        metric_columns[2].metric("召回率 Recall", f"{metrics['recall']:.2%}")
        metric_columns[3].metric("F1 分数", f"{metrics['f1']:.2%}")
        metric_columns[4].metric("AUC", f"{metrics['auc']:.4f}")

        st.caption(
            f"训练集 {model_result.train_size:,} 条 ｜ "
            f"测试集 {model_result.test_size:,} 条 ｜ "
            f"特征 {len(model_result.importance_df['feature_clean'].unique())} 个（编码后）"
        )

        # ---------- 2) 混淆矩阵与 ROC 曲线 ----------
        st.subheader("2. 混淆矩阵与 ROC 曲线")
        cm_column, roc_column = st.columns(2)

        with cm_column:
            fig_cm = plot_confusion_matrix(model_result.confusion)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            tn, fp, fn, tp = model_result.confusion.ravel()
            st.caption(
                f"正确识别退货 {tp} 笔，漏报 {fn} 笔，误报 {fp} 笔，"
                f"正确识别未退货 {tn} 笔。"
            )

        with roc_column:
            fig_roc = plot_roc_curve(
                model_result.y_test,
                model_result.y_proba,
                metrics["auc"],
            )
            st.pyplot(fig_roc)
            plt.close(fig_roc)
            st.caption("曲线越靠近左上角，模型区分“会退货 / 不会退货”的能力越强。")

        # ---------- 3) 特征重要性 ----------
        st.subheader("3. 特征重要性")
        fig_importance = plot_feature_importance(model_result.importance_df, top_n=15)
        st.pyplot(fig_importance)
        plt.close(fig_importance)

        with st.expander("查看特征重要性明细（Top 20）"):
            importance_table = model_result.importance_df.head(20)[
                ["feature_cn", "importance"]
            ].rename(columns={"feature_cn": "特征", "importance": "重要性"})
            st.dataframe(importance_table, use_container_width=True)

        # ---------- 4) 业务结论 ----------
        st.subheader("4. 业务结论")
        st.markdown(
            "\n".join(
                f"- {conclusion}"
                for conclusion in business_conclusions(model_result)
            )
        )

        # ---------- 5) 单笔订单退货风险速查（预测函数演示） ----------
        st.subheader("5. 单笔订单退货风险速查")
        st.caption(
            "下面的表单只填写关键字段，其余特征自动用训练集典型值补全，"
            "预测结果来自 `return_model.predict_return_risk()`。"
        )

        category_options = sorted(model_result.data["category"].astype(str).unique())
        fit_options = sorted(model_result.data["fit_type"].astype(str).unique())

        with st.form("return_risk_form"):
            col_left, col_middle, col_right = st.columns(3)

            with col_left:
                category_input = st.selectbox("商品品类", category_options)
                fit_input = st.selectbox("版型", fit_options)
                price_input = st.slider(
                    "价格（美元）", 0, 500,
                    int(model_result.defaults["price_usd"]), 5,
                )

            with col_middle:
                size_mismatch_input = st.selectbox("尺码偏差（码）", [0, 1, 2, 3])
                multiple_sizes_input = st.checkbox("一次购买多个尺码", value=False)
                prior_rate_input = st.slider(
                    "历史退货率", 0.0, 1.0,
                    float(model_result.defaults["customer_prior_return_rate"]), 0.01,
                )

            with col_right:
                rating_input = st.slider(
                    "商品平均评分", 1.0, 5.0,
                    float(model_result.defaults["avg_review_rating"]), 0.1,
                )
                free_return_input = st.checkbox("退货包邮", value=True)
                st.write("")
                submitted = st.form_submit_button("开始预测")

        if submitted:
            # 以训练集典型值为底稿，覆盖表单里用户填写的字段
            order = dict(model_result.defaults)
            order.update(
                {
                    "category": category_input,
                    "fit_type": fit_input,
                    "price_usd": float(price_input),
                    "size_mismatch": int(size_mismatch_input),
                    "ordered_multiple_sizes": int(multiple_sizes_input),
                    "customer_prior_return_rate": float(prior_rate_input),
                    "avg_review_rating": float(rating_input),
                    "return_shipping_free": int(free_return_input),
                }
            )

            prediction = predict_return_risk(model_result, order).iloc[0]
            probability = prediction["return_probability"]

            result_column, advice_column = st.columns([1, 3])
            result_column.metric("退货概率", f"{probability:.1%}")
            with advice_column:
                st.markdown(
                    f"**风险等级：{prediction['risk_level']}**\n\n"
                    f"{prediction['advice']}"
                )

            st.progress(float(probability))
            st.caption("风险分级规则：退货概率 < 35% 为低风险，35% ~ 60% 为中风险，≥ 60% 为高风险。")

