import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from openai import OpenAI

st.set_page_config(page_title="电商消费避雷透视镜", page_icon="🛒", layout="wide")

st.title("🛒 电商消费避雷数据分析与 AI 可视化平台")
st.markdown("基于真实行为与售后数据，打破信息不对称。")

# ================= 侧边栏 =================
st.sidebar.header("筛选条件")
category = st.sidebar.selectbox("选择商品品类", ["女装", "鞋靴", "美妆", "数码"])

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
st.write("基于真实退货文本分析，暴露商品质量问题。")

# 模拟的退货文本数据（真实情况需读取 return_reasons.csv）
mock_text = """
衣服质量太差了 穿了一次就起球 货不对板 和图片完全不一样 卖家秀买家秀差别太大
材质不舒服 有异味 面料很差 尺码偏大 退货运费还要自己出 商家态度不好
颜色有色差 做工粗糙 线头多 直播间买的被坑了 严重缩水 根本不是纯棉的
"""
# 使用 jieba 分词
words = jieba.cut(mock_text)
text_for_cloud = " ".join(words)

# 生成词云图
# 注意：这里为了兼容 Windows，直接指定了默认中文字体
try:
    wc = WordCloud(
        font_path="msyh.ttc",  # Windows 默认微软雅黑字体，Mac 可换为 "PingFang.ttc"
        background_color="white",
        width=800,
        height=400
    ).generate(text_for_cloud)
    
    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
except OSError:
    # 字体文件打不开（如缺少 msyh.ttc）时，只提示文字，不抛出错误
    st.info("提示：未能打开中文字体文件（msyh.ttc），本次跳过词云图展示。可将 font_path 换成本机已有的中文字体。")
except Exception as e:
    st.warning(f"词云图生成失败，可能是缺少中文字体文件。错误信息：{e}")

# ================= 模块三：AI 智能导购 =================
st.header("3. AI 智能避雷导购")
st.write("结合弃单率与退货词频，由大语言模型生成避雷建议。")

if st.button("生成避雷建议"):
    with st.spinner("AI 正在研判当前品类的风险..."):
        
        # 读取 API Key (从 .streamlit/secrets.toml 读取)
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
        except KeyError:
            api_key = None
        
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