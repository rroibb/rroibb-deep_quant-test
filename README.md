# DeepQuant — 多模态深度学习量化交易系统

融合 LSTM / Transformer / CNN / XGBoost / BERT / LLM 六种模型的多模态量化选股系统，支持 A 股 同花顺模拟盘 每周 Top5 推荐。

## 系统架构

```
                    ┌──────────────────────┐
                    │    数据源 (Tushare)    │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │     特征工程          │
                    │  技术指标 · 量价关系   │
                    └──────────┬───────────┘
                               ▼
         ┌──────────────────────────────────────┐
         │          模型层 (6种模态)             │
         ├──────────────┬───────────────┬───────┤
         │  时序预测     │ 图表识别      │ 语义  │
         │  LSTM        │ CNN           │ BERT  │
         │  Transformer │               │ LLM   │
         ├──────────────┴───────────────┴───────┤
         │  XGBoost (趋势/震荡/反转三态基线)     │
         └──────────────────┬───────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │  多模态融合 (MultiModalFusionModel)   │
         └──────────────────┬───────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │  回测系统 · 虚拟盘交易 · 飞书推送     │
         └──────────────────────────────────────┘
```

## 功能

- **多模态预测** — 6 种模型集成：LSTM（时序）、Transformer（注意力）、CNN（K线图）、XGBoost（传统 ML）、BERT（中文情感）、LLM（宏观分析）
- **市场状态感知** — 自动识别趋势/震荡/反转三种市场状态，不同状态下使用不同的 XGBoost 特征
- **多模态融合** — 自适应权重融合所有模型预测
- **同花顺模拟盘推荐** — 生成含挂单价、手数、权重的买入指令，支持可购买性优化（高价股替换）
- **飞书推送** — 交易指令 + 回测报告自动推送到飞书
- **可视化仪表盘** — 累计收益、回撤、月度收益、绩效指标四合一

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置密钥 (复制并编辑)
copy .env.example .env
# 填入 TUSHARE_TOKEN (必填, https://tushare.pro)
# 可选: FEISHU_WEBHOOK_URL / FEISHU_WEBHOOK_SECRET

# 3. 运行虚拟盘交易
python virtual_trade.py

# 4. 运行全流程回测
python main.py
```

## 虚拟盘交易

针对同花顺模拟盘设计，每周一开盘前执行：

```bash
python virtual_trade.py
```

输出：
- `output/thss_weekly_picks_*.txt` — 同花顺买入指令
- `output/virtual_trade_*.csv` — 交易明细

### 可购买性优化

从 Top15 候选股中逐只检查价格，确保每只都能买入 ≥1 手。若某只价格过高（如宁德时代 > 400 元/股，1 手需 >4 万元），自动跳过并尝试下一候选。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INITIAL_CAPITAL` | 200,000 | 初始资金 |
| `TOP_K` | 5 | 持仓数量 |
| `MIN_LOTS` | 1 | 最少买入手数 |
| `CANDIDATE_POOL` | 15 | 候选池大小 |
| `PRICE_TOLERANCE` | 2% | 挂单缓冲 |

## 模型支持

| 模型 | 输入 | 用途 |
|------|------|------|
| LSTM | 60天量价序列 | 时序预测 |
| Transformer | 60天量价序列 | 注意力预测 |
| CNN | K线图表 | 形态识别 |
| XGBoost | 技术指标 | 传统基线 |
| BERT | 新闻文本 | 情感分析 |
| LLM | 新闻文本 | 宏观分析 |

## 回测对比

运行 `main.py` 自动对比三种模式：

- **多模态融合** — DL + XGBoost + NLP + LLM（全模型）
- **深度学习仅** — LSTM + Transformer + CNN
- **XGBoost仅** — 传统机器学习

## 股票池

默认 电子科技50股，涵盖半导体、消费电子、新能源等领域。可在 `config.py` 中 `SECTOR_MAP` 替换。

## 项目结构

```
deep_quant/
├── main.py                  # 全流程回测入口
├── virtual_trade.py         # 虚拟盘交易 (同花顺)
├── config.py                # 全局配置
├── backtest.py              # 回测引擎
├── trainer.py               # 模型训练
├── features.py              # 特征工程
├── data_layer.py            # 数据加载
├── regime.py                # 市场状态识别
├── feishu_pusher.py         # 飞书推送
├── visualization.py         # 可视化
├── models/
│   ├── lstm.py              # LSTM 时序模型
│   ├── transformer.py       # Transformer 模型
│   ├── cnn_chart.py         # CNN K线识别
│   ├── nlp_sentiment.py     # BERT 情感分析
│   ├── llm_analyzer.py      # LLM 分析
│   └── fusion.py            # 多模态融合
├── output/                  # 输出目录
├── .env.example             # 环境变量模板
└── THSS_OPERATION_GUIDE.md  # 同花顺操作指南
```

## 技术栈

- **深度学习:** PyTorch 2.0+
- **传统 ML:** XGBoost / scikit-learn
- **NLP:** Transformers (BERT-base-Chinese)
- **数据:** Tushare Pro
- **可视化:** Matplotlib
