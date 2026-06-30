import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .base import BaseDeepModel


class NLPSentimentAnalyzer(BaseDeepModel):
    def __init__(self, model_name='bert-base-chinese', max_len=128, num_labels=3):
        super().__init__(name='NLP_Sentiment')
        self.max_len = max_len
        self.num_labels = num_labels

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            self.bert_dim = self.bert.config.hidden_size
        except Exception as e:
            print(f"NLP模型加载失败: {e}, 使用随机初始化")
            self.tokenizer = None
            self.bert = None
            self.bert_dim = 768

        self.fc = nn.Sequential(
            nn.Linear(self.bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # 情感数据缓存
        self._forum_data = None

    def forward(self, input_ids, attention_mask=None):
        if self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output
        else:
            pooled = torch.mean(input_ids.float(), dim=1)
        return self.fc(pooled).squeeze(-1)

    def predict_sentiment(self, texts):
        self.eval()
        if self.tokenizer is None:
            return np.zeros(len(texts))

        sentiments = []
        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text, max_length=self.max_len, truncation=True, padding='max_length',
                    return_tensors='pt'
                )
                score = self.forward(encoded['input_ids'], encoded['attention_mask'])
                sentiments.append(float(score.numpy()))
        return np.array(sentiments)

    def load_forum_sentiment(self, tickers, days=20, use_cache=True):
        if use_cache and self._forum_data is not None:
            return self._forum_data
        try:
            from sentiment_sources import fetch_all_sentiment_features
            raw = fetch_all_sentiment_features(tickers, days)
            self._forum_data = raw
            return raw
        except Exception as e:
            print(f"  [NLP] 论坛情感加载失败: {e}")
            return {}

    def get_combined_sentiment(self, ticker, date, text_sentiment=0.0):
        forum = self._forum_data or {}
        df = forum.get(ticker)
        if df is not None and not df.empty:
            date = pd.Timestamp(date)
            if date in df.index:
                row = df.loc[date]
            else:
                row = df.iloc[-1] if len(df) > 0 else None

            if row is not None:
                forum_score = 0.0
                n = 0
                for col in ['composite_score', 'buy_desire', 'attention_index']:
                    val = row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        forum_score += (float(val) / 50.0 - 1.0)
                        n += 1
                if n > 0:
                    forum_score /= n
                    return 0.6 * text_sentiment + 0.4 * forum_score
        return text_sentiment

    def predict_batch_combined(self, tickers, dates, texts):
        bert_scores = self.predict_sentiment(texts)
        self.load_forum_sentiment(tickers)
        combined = []
        for ticker, date, bs in zip(tickers, dates, bert_scores):
            combined.append(self.get_combined_sentiment(ticker, date, bs))
        return np.array(combined)

    def extract_news_keywords(self, texts):
        keywords = []
        for text in texts:
            words = re.findall(r'[\u4e00-\u9fff\w]+', text.lower())
            keywords.append(words[:20])
        return keywords

    @staticmethod
    def estimate_sentiment_rule_based(text):
        positive_words = {'增长', '上涨', '突破', '利好', '盈利', '创新高', '扩张', '超预期',
                          '强劲', '回升', '反弹', '买入', '推荐', '升级', '领先'}
        negative_words = {'下跌', '亏损', '利空', '风险', '减持', '暴跌', '下滑', '预警',
                          '违约', '降级', '回落', '卖出', '做空', '调查', '处罚'}
        score = 0
        for w in positive_words:
            if w in text:
                score += 1
        for w in negative_words:
            if w in text:
                score -= 1
        return np.tanh(score / 3)
