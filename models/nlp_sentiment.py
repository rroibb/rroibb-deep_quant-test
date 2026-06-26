import re
import numpy as np
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
