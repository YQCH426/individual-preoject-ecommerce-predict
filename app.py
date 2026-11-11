import streamlit as st
import pandas as pd

st.set_page_config(page_title="电商预测", page_icon="📈", layout="wide")

st.title("📈 电商销量预测")
st.markdown("这是我的数据分析与可视化课程的个人项目。")

# 侧边栏：数据上传
st.sidebar.header("数据")
uploaded_file = st.sidebar.file_uploader("上传 CSV 数据文件", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("数据预览")
    st.dataframe(df.head(20))

    st.subheader("统计描述")
    st.write(df.describe())

    # 选择数值列绘制折线图
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if numeric_cols:
        col = st.selectbox("选择要可视化的数值列", numeric_cols)
        st.line_chart(df[col])
else:
    st.info("请在左侧上传 CSV 文件开始分析。")
    # 无数据时展示示例
    st.subheader("示例数据")
    sample = pd.DataFrame(
        {"月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
         "销量": [120, 150, 135, 180, 210, 195]}
    )
    st.bar_chart(sample.set_index("月份"))
