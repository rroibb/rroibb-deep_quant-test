import os
import torch

# ======================== 全局设置 ========================
RISK_FREE_RATE = 0.03
TRADING_DAYS = 240
INITIAL_CAPITAL = 100000

# 交易成本
COMMISSION_RATE = 0.00025
STAMP_DUTY_RATE = 0.001
TRANSFER_FEE_RATE = 0.00001

# Tushare API
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

# 股票池 50只电子科技
SECTOR_MAP = {
    '688981.SH': '中芯国际', '688012.SH': '中微公司', '002371.SZ': '北方华创',
    '603501.SH': '韦尔股份', '603986.SH': '兆易创新', '688008.SH': '澜起科技',
    '600584.SH': '长电科技', '688126.SH': '沪硅产业', '300373.SZ': '扬杰科技',
    '688396.SH': '华润微', '002049.SZ': '紫光国微', '300782.SZ': '卓胜微',
    '603893.SH': '瑞芯微',
    '002475.SZ': '立讯精密', '002241.SZ': '歌尔股份', '000725.SZ': '京东方A',
    '300433.SZ': '蓝思科技', '601138.SH': '工业富联', '688036.SH': '传音控股',
    '002600.SZ': '领益智造', '300136.SZ': '信维通信',
    '000063.SZ': '中兴通讯', '300308.SZ': '中际旭创', '300502.SZ': '新易盛',
    '300394.SZ': '天孚通信', '600487.SH': '亨通光电', '600522.SH': '中天科技',
    '600498.SH': '烽火通信',
    '600570.SH': '恒生电子', '688111.SH': '金山办公', '300033.SZ': '同花顺',
    '600536.SH': '中国软件', '000977.SZ': '浪潮信息', '300496.SZ': '中科创达',
    '688561.SH': '奇安信', '300454.SZ': '深信服',
    '601012.SH': '隆基绿能', '600438.SH': '通威股份', '002459.SZ': '晶澳科技',
    '300274.SZ': '阳光电源', '002594.SZ': '比亚迪', '300750.SZ': '宁德时代',
    '603659.SH': '璞泰来', '002709.SZ': '天赐材料', '300014.SZ': '亿纬锂能',
    '000938.SZ': '中芯国际(重复)', '688041.SH': '海光信息', '688256.SH': '寒武纪',
    '300476.SZ': '胜宏科技', '002463.SZ': '沪电股份',
}

STOCK_POOL = list(SECTOR_MAP.keys())
MAX_SECTOR_PCT = 0.30
TOP_N = 8

# 深度学习参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 序列长度
SEQ_LEN = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 15

# 各领域模型配置
MODEL_CONFIGS = {
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True,
    },
    'transformer': {
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
    },
    'cnn': {
        'in_channels': 5,
        'num_classes': 5,
    },
    'nlp': {
        'model_name': 'bert-base-chinese',
        'max_len': 128,
        'num_labels': 3,
    },
    'fusion': {
        'hidden_dim': 256,
    }
}

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
