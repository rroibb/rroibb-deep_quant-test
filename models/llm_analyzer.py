import numpy as np
from .base import BaseDeepModel


class LLMAnalyzer(BaseDeepModel):
    def __init__(self, model_name=None, use_mock=True):
        super().__init__(name='LLM_Analyzer')
        self.model_name = model_name
        self.use_mock = use_mock

    def forward(self, x):
        return x

    def analyze_stock(self, stock_code, stock_name, prices, volumes, news_heads=None):
        analysis = self._generate_analysis(stock_name, prices, news_heads)
        score = self._score_from_analysis(analysis)
        return score, analysis

    def _generate_analysis(self, stock_name, prices, news_heads):
        recent = prices[-20:] if len(prices) >= 20 else prices
        ret = (recent[-1] / recent[0] - 1) * 100
        vol_ratio = np.std(recent) / (np.mean(recent) + 1e-8)

        analysis_parts = [f"{stock_name}最近20日涨跌{ret:.1f}%"]
        if ret > 5:
            analysis_parts.append("短期表现强势")
        elif ret < -5:
            analysis_parts.append("短期表现疲弱")
        else:
            analysis_parts.append("短期处于震荡")

        if vol_ratio > 0.05:
            analysis_parts.append("波动率较高，注意风险")
        else:
            analysis_parts.append("波动正常")

        if news_heads:
            positive_count = sum(1 for n in news_heads if any(w in n for w in ['涨', '增', '好', '盈', '突破']))
            negative_count = sum(1 for n in news_heads if any(w in n for w in ['跌', '减', '亏', '险', '回撤']))
            if positive_count > negative_count:
                analysis_parts.append("近期偏正面")
            elif negative_count > positive_count:
                analysis_parts.append("近期偏负面")
            else:
                analysis_parts.append("消息面中性")

        return '，'.join(analysis_parts)

    def _score_from_analysis(self, analysis):
        positive_markers = ['强势', '突破', '利好', '正面', '超预期', '增长']
        negative_markers = ['疲弱', '风险', '负面', '亏损', '下滑', '预警']

        score = 0.0
        for w in positive_markers:
            if w in analysis:
                score += 0.15
        for w in negative_markers:
            if w in analysis:
                score -= 0.15
        return np.clip(score, -0.5, 0.5)

    def analyze_text_with_mock_llm(self, text, context=None):
        if self.use_mock:
            return self._generate_analysis("标的", np.array([100] * 60), [text])
        return "LLM分析模块需要配置实际API密钥"

    def batch_analyze(self, stocks_info):
        results = []
        for info in stocks_info:
            score, analysis = self.analyze_stock(
                info.get('code', ''),
                info.get('name', ''),
                info.get('prices', np.array([])),
                info.get('volumes', np.array([])),
                info.get('news', [])
            )
            results.append({'code': info.get('code'), 'llm_score': score, 'analysis': analysis})
        return results
