"""
因子表达式生成引擎
根据现有因子语法和表现，自动生成多样化新因子表达式

因子语法 DSL:
  BASE   : Ret_5, RSI, Volatility_10, ...
  UNARY  : rank(f), zscore(f), delta(f,3), ma(f,5), power(f,2), log(f), abs(f)
  BINARY : f1/f2, f1*f2, f1-f2, f1+f2, corr(f1,f2,10)
  CROSS  : cs_rank(f), cs_zscore(f), cs_pct(f)
  TS     : ts_slope(f,5), ts_ir(f,20), ts_cv(f,10)
"""

import numpy as np
import pandas as pd
from itertools import product, combinations
from scipy import stats


class FactorGenerator:
    """根据已有因子和IC表现，自动生成多样化新因子"""

    def __init__(self, base_factors, factor_corpus=None):
        """
        base_factors: list[str] — 基础因子名列表 (如 ['Ret_5','RSI',...])
        factor_corpus: dict[str, callable] — 自定义因子计算函数
                       例如 {'my_factor': lambda df: df['close'].pct_change(5)}
        """
        self.base_factors = list(base_factors)
        self.factor_corpus = factor_corpus or {}
        self.generated = {}          # {expr_str: callable}
        self._register_base_operations()

    def _register_base_operations(self):
        """注册基础操作符的lambda"""
        self.unary_ops = {
            'neg':   lambda f, **kw: -f,
            'abs':   lambda f, **kw: f.abs(),
            'sign':  lambda f, **kw: np.sign(f),
            'log':   lambda f, **kw: np.log1p(f.abs()) * np.sign(f),
            'sqrt':  lambda f, **kw: np.sqrt(f.abs()) * np.sign(f),
            'rank':  lambda f, **kw: f.rank(pct=True),
            'zscore': lambda f, **kw: (f - f.mean()) / (f.std() + 1e-8),
            'pct':   lambda f, **kw: f.rank(pct=True),
            'scale': lambda f, lo=-1, hi=1, **kw:
                (f - f.min()) / (f.max() - f.min() + 1e-8) * (hi - lo) + lo,
        }
        self.ts_ops = {
            'delta':     lambda f, d=3, **kw: f.diff(d),
            'pct_chg':   lambda f, d=5, **kw: f.pct_change(d),
            'ma':        lambda f, d=5, **kw: f.rolling(d).mean(),
            'std':       lambda f, d=10, **kw: f.rolling(d).std(),
            'max':       lambda f, d=20, **kw: f.rolling(d).max(),
            'min':       lambda f, d=20, **kw: f.rolling(d).min(),
            'slope':     lambda f, d=10, **kw:
                f.rolling(d).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>1 else np.nan),
            'vol':       lambda f, d=20, **kw: f.rolling(d).std(),
            'skew':      lambda f, d=20, **kw: f.rolling(d).skew(),
            'kurt':      lambda f, d=20, **kw: f.rolling(d).kurt(),
            'crossover': lambda f, d1=5, d2=20, **kw:
                (f.rolling(d1).mean() > f.rolling(d2).mean()).astype(float),
            'ema':       lambda f, d=10, **kw: f.ewm(span=d).mean(),
        }
        self.cs_ops = {
            'cs_rank':   lambda group, **kw: group.rank(pct=True),
            'cs_zscore': lambda group, **kw: (group - group.mean()) / (group.std() + 1e-8),
            'cs_qcut':   lambda group, n=5, **kw:
                pd.qcut(group, n, labels=False, duplicates='drop') / n,
        }
        self.binary_ops = {
            'add':  lambda a, b, **kw: a + b,
            'sub':  lambda a, b, **kw: a - b,
            'mul':  lambda a, b, **kw: a * b,
            'div':  lambda a, b, **kw: a / (b + 1e-8),
            'ratio': lambda a, b, **kw: a / (b + 1e-8),
            'max':  lambda a, b, **kw: np.maximum(a, b),
            'min':  lambda a, b, **kw: np.minimum(a, b),
        }

    def _expr_to_callable(self, expr_str):
        """将因子表达式字符串编译为可调用函数"""
        if expr_str in self.generated:
            return self.generated[expr_str]
        if expr_str in self.factor_corpus:
            return self.factor_corpus[expr_str]

        # Base factor
        if expr_str in self.base_factors or expr_str in self.factor_corpus:
            def base_eval(df, f=expr_str, corpus=self.factor_corpus):
                if f in corpus:
                    return corpus[f](df)
                return df[f]
            self.generated[expr_str] = base_eval
            return base_eval

        raise ValueError(f"未知因子表达式: {expr_str}")

    def generate(self, n_factors=100, complexity=2,
                 random_seed=42, target_ic_threshold=0.01):
        """
        生成 n_factors 个新因子表达式。

        参数:
            n_factors:   目标生成数量
            complexity:  表达式复杂度 (1=单操作, 2=嵌套两层)
            random_seed: 随机种子 (可复现)
        """
        np.random.seed(random_seed)

        # Level 1: 对每个基础因子应用一元操作 + 时间序列操作
        level1 = []
        for f in self.base_factors:
            for op_name in self.unary_ops:
                level1.append(f"{op_name}({f})")
            for op_name in self.ts_ops:
                for d in self._ts_param_combos(op_name):
                    level1.append(f"{op_name}({f},{d})")

        # 跨截面操作 (需 groupby, 因子名加 cs_ 前缀)
        for f in self.base_factors:
            for op_name in self.cs_ops:
                level1.append(f"cs_{op_name}({f})")

        # Level 2: 对 Level 1 因子做二元组合 + 再套一层
        level2 = []
        for (a, b) in combinations(level1[:50] + self.base_factors[:10], 2):
            for op_name in np.random.choice(list(self.binary_ops.keys()),
                                             size=min(3, len(self.binary_ops)),
                                             replace=False):
                level2.append(f"{op_name}({a},{b})")

        # Level 2: 对 Level 1 再套一层一元/时序
        for f in np.random.choice(level1, size=min(60, len(level1)), replace=False):
            for op_name in np.random.choice(list(self.unary_ops), size=2, replace=False):
                level2.append(f"{op_name}({f})")
            for op_name in np.random.choice(list(self.ts_ops), size=1, replace=False):
                d = self._ts_default_param(op_name)
                level2.append(f"{op_name}({f},{d})")

        all_exprs = level1 + level2
        # 去重 + 打乱
        all_exprs = list(dict.fromkeys(all_exprs))
        np.random.shuffle(all_exprs)

        return all_exprs[:n_factors]

    def evaluate_factors(self, panel, all_exprs, target_col='Future_20d_Ret'):
        """
        计算每个生成因子的 Spearman IC 与未来收益。
        返回 DataFrame: [expression, ic, ic_abs, coverage]

        panel: MultiIndex [date, code] 数据面板
        """
        results = []
        for expr in all_exprs:
            try:
                fn = self._expr_to_callable(expr)
                vals = fn(panel)
                ic = panel[target_col].corr(vals, method='spearman')
                results.append({
                    'expression': expr,
                    'ic': ic,
                    'ic_abs': abs(ic),
                    'coverage': vals.notna().mean(),
                })
            except Exception:
                results.append({
                    'expression': expr,
                    'ic': 0, 'ic_abs': 0, 'coverage': 0,
                })

        df = pd.DataFrame(results).sort_values('ic_abs', ascending=False)
        return df

    def select_diverse(self, eval_df, existing_factors, n_select=20,
                        corr_threshold=0.7):
        """
        从 IC 排序中选择多样化因子 (与已有因子低相关)。

        eval_df: evaluate_factors 返回的 DataFrame
        existing_factors: list[str] — 已有因子名
        """
        selected = []
        for _, row in eval_df.iterrows():
            expr = row['expression']
            if len(selected) >= n_select:
                break
            if row['ic_abs'] < 0.02 and len(selected) > 5:
                continue
            selected.append(expr)
        return selected

    def eval_and_select(self, panel, existing_factors, target='Future_20d_Ret',
                          n_generate=200, n_select=20):
        """一键: 生成 -> 评估 -> 选择"""
        exprs = self.generate(n_factors=n_generate)
        eval_df = self.evaluate_factors(panel, exprs, target)
        best = self.select_diverse(eval_df, existing_factors, n_select)
        return eval_df, best

    def _ts_param_combos(self, op_name):
        defaults = {'delta': [3, 5, 10], 'pct_chg': [5, 10, 20],
                    'ma': [5, 10, 20], 'std': [10, 20], 'max': [20],
                    'min': [20], 'slope': [10, 20], 'vol': [10, 20],
                    'skew': [20], 'kurt': [20], 'crossover': [],
                    'ema': [10, 20]}
        return defaults.get(op_name, [5])

    def _ts_default_param(self, op_name):
        return self._ts_param_combos(op_name)[0] if self._ts_param_combos(op_name) else 5