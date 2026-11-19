# 电商退货风险预测与用户行为转化分析平台

> 退货可视化 · 退货预测 · 行为漏斗 · AI 避雷助手 —— 把消费决策从"凭感觉"变成"看数据"

## 项目简介

网购的信息不对称贯穿消费全流程：**下单前**只能靠精修图和精选好评做判断，容易冲动下单；**下单后**又常常因尺码、色差、质量等问题退货，赔上运费和时间。

本平台把真实的服装退货数据与电商用户行为数据转化为可交互的分析界面：用词云复盘退货原因，用随机森林模型预测订单退货风险，用转化漏斗定位流失环节，再由 AI 消费避雷助手给出落地建议，形成"事前预警 — 事后复盘"的完整分析闭环。

## 项目功能

| 功能模块 | 说明 |
|---|---|
| **退货可视化** | 从真实退货原因文本生成中文词云，字号反映出现频次，并支持展开查看各原因的明细次数 |
| **退货预测** | 基于服装订单特征训练随机森林分类模型，预测订单是否会被退货，并输出准确率 / 召回率 / F1 / AUC、混淆矩阵、ROC 曲线与特征重要性 |
| **用户转化漏斗** | 对天池用户行为数据分块抽样，统计浏览 → 加购 → 收藏 → 下单各阶段的数量与转化率，绘制交互式漏斗图 |
| **AI 消费避雷助手** | 接入大语言模型，结合品类风险与退货词频，生成简明实用的避雷建议 |

## 数据集介绍

本项目使用两份真实数据集，**原始 CSV 文件体积较大，未纳入版本管理**（见下方"数据文件说明"）。

### ① Kaggle `fashion_returns_master.csv` —— 服装订单退货数据集

- **用途**：退货原因词云可视化、退货预测随机森林模型。
- **内容**：服装类订单记录，包含品类、材质、价格、尺码匹配度、历史行为等特征，以及是否退货（`returned`）与退货原因（`return_reason`）字段。
- **字段说明**：`returned = 1` 表示退货、`0` 表示未退货；`return_reason` 为英文取值（如 `size_too_large`、`wrong_color`、`quality_defect`），分析时统一映射为中文。
- **建模要点**：预测时剔除 `return_reason` 等会造成标签泄漏的字段，仅使用下单前可获得的信息，保证模型具备真实预警价值。
- **存放位置**：项目根目录 `fashion_returns_master.csv`。

### ② 天池 `UserBehavior.csv`（约 3.42GB）—— 电商用户行为数据集

- **用途**：浏览 → 加购 → 收藏 → 下单转化漏斗分析。
- **内容**：原始文件无表头，字段依次为 `user_id, item_id, category_id, behavior_type, timestamp`；`behavior_type` 取值为 `pv`（浏览）、`cart`（加购）、`fav`（收藏）、`buy`（下单）。
- **处理方式**：数据量过亿，无法一次性读入内存，因此采用 **pandas 分块读取（默认每块 100 万行）+ 每块随机抽取 10%** 的策略，逐块累计各行为发生次数，并主动释放内存，保证分析可执行且不溢出。结果通过 `st.cache_resource` 缓存，避免每次页面交互都重新扫描大文件。
- **存放位置**：项目根目录 `UserBehavior.csv`。

## 技术栈

| 技术 | 用途 |
|---|---|
| **Streamlit** | Web 应用框架，搭建交互式分析界面 |
| **Pandas / NumPy** | 分块读取、数据清洗、抽样与聚合统计 |
| **Scikit-learn** | 随机森林分类模型与评估指标 |
| **Plotly** | 交互式漏斗图 |
| **WordCloud / Matplotlib / jieba** | 中文词云生成与文本处理 |
| **OpenAI SDK（DeepSeek）** | 驱动 AI 消费避雷助手 |

## 运行依赖

- Python 3.9 及以上（建议 3.10+）
- 已安装 git
- 内存建议 4GB 以上（分块抽样已大幅降低占用）

Python 依赖见 `requirements.txt`：

```text
streamlit
pandas
scikit-learn
plotly
openai
wordcloud
matplotlib
jieba
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<你的用户名>/ecommerce-predict.git
cd ecommerce-predict
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 准备数据文件

将两份原始数据集放到项目根目录：

```text
ecommerce-predict/
├── fashion_returns_master.csv
└── UserBehavior.csv
```

> 缺少 `fashion_returns_master.csv` 时应用会跳过词云模块；缺少 `UserBehavior.csv` 时漏斗页会给出友好提示，页面不会崩溃。

### 4. 配置 LLM API Key（AI 消费避雷助手需要）

在项目根目录下新建 `.streamlit/secrets.toml`：

```toml
# .streamlit/secrets.toml
DEEPSEEK_API_KEY = "你的密钥"
```

应用中通过 Streamlit 原生 secrets 接口读取：

```python
api_key = st.secrets["DEEPSEEK_API_KEY"]
```

> `.streamlit/` 已在 `.gitignore` 中忽略，**secrets.toml 不会被提交**。请勿把密钥硬编码进代码，也不要用 `.env` / `python-dotenv` 方案。未配置密钥时，AI 模块会展示模拟建议而不报错中断。

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器访问 <http://localhost:8501> 即可使用。

## 数据文件说明

> ⚠️ **原始 CSV 文件不存入 git，`.gitignore` 已屏蔽**（`*.csv`、`fashion_returns_master.csv`、`UserBehavior.csv` 均在忽略规则内）。

这样做是因为 `UserBehavior.csv` 约 3.42GB、`fashion_returns_master.csv` 也有数十 MB，直接入库会让仓库体积失控并触及 GitHub 文件大小限制。请按"准备数据文件"一节自行下载后放入根目录，克隆仓库的其他人也需要同样操作。

同样被忽略的还有：`output/`（模型图表输出）、`*.png`、`__pycache__/`、`.venv/`、`.streamlit/`。

## 项目结构

```text
ecommerce-predict/
├── app.py                      # Streamlit 应用入口（四个功能模块）
├── returns_analysis.py         # 退货数据清洗 + 中文词云生成
├── behavior_funnel.py          # 用户行为分块抽样 + 转化漏斗
├── return_model.py             # 随机森林训练 / 评估 / 预测封装（供 app 调用）
├── return_model.ipynb          # 随机森林退货预测建模 Notebook（完整实验记录）
├── requirements.txt            # 依赖清单
├── NotoSansSC-Regular.ttf      # 词云/图表中文字体
├── fashion_returns_master.csv  # 服装退货数据（不入库）
├── UserBehavior.csv            # 天池用户行为数据（不入库）
├── output/                     # 模型评估图表（不入库）
├── .streamlit/
│   └── secrets.toml            # LLM API Key（本地配置，不入库）
├── .gitignore                  # 忽略规则（数据文件、密钥、输出等）
└── README.md
```

## 单独运行分析脚本（可选）

```bash
# 生成退货原因词云 PNG 并打印原因频次
python returns_analysis.py

# 训练随机森林、打印指标与业务结论，并导出评估图表到 output/
python return_model.py

# 执行分块抽样、打印漏斗转化率并导出 output/behavior_funnel.html
python behavior_funnel.py
```
