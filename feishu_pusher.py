"""
飞书消息推送模块
支持: 文本消息、富文本卡片、图片上传
"""
import os
import json
import hmac
import hashlib
import base64
import time
import requests
from datetime import datetime


class FeishuPusher:
    def __init__(self, webhook_url=None, secret=None, app_id=None, app_secret=None):
        self.webhook_url = webhook_url or os.environ.get('FEISHU_WEBHOOK_URL', '')
        self.secret = secret or os.environ.get('FEISHU_WEBHOOK_SECRET', '')
        self.app_id = app_id or os.environ.get('FEISHU_APP_ID', '')
        self.app_secret = app_secret or os.environ.get('FEISHU_APP_SECRET', '')
        self._tenant_token = None
        self._token_expire = 0

    def _sign(self):
        ts = int(time.time())
        sign_str = f"{ts}\n{self.secret}"
        sign = base64.b64encode(hmac.new(
            sign_str.encode('utf-8'), b'', hashlib.sha256
        ).digest()).decode('utf-8')
        return ts, sign

    def _get_tenant_token(self):
        if self._tenant_token and datetime.now().timestamp() < self._token_expire:
            return self._tenant_token
        resp = requests.post(
            'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
            json={"app_id": self.app_id, "app_secret": self.app_secret}
        )
        data = resp.json()
        if data.get('code') != 0:
            raise RuntimeError(f"获取飞书token失败: {data}")
        self._tenant_token = data['tenant_access_token']
        self._token_expire = datetime.now().timestamp() + data.get('expire', 7200) - 300
        return self._tenant_token

    def _post(self, payload):
        if not self.webhook_url:
            return None
        if self.secret:
            ts, sign = self._sign()
            payload["timestamp"] = str(ts)
            payload["sign"] = sign
        resp = requests.post(self.webhook_url, json=payload)
        return resp.json()

    def send_text(self, text):
        if not self.webhook_url:
            print("[飞书] 未配置webhook，跳过发送")
            return
        payload = {"msg_type": "text", "content": {"text": text}}
        result = self._post(payload)
        if result:
            ok = result.get('code') == 0 or result.get('StatusCode') == 0
            print(f"[飞书] 文本消息: {'成功' if ok else '失败 - ' + str(result)}")
            return ok
        return False

    def send_card(self, title, elements, color='blue'):
        if not self.webhook_url:
            print("[飞书] 未配置webhook，跳过发送")
            return
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color
                },
                "elements": elements
            }
        }
        result = self._post(payload)
        if result:
            ok = result.get('code') == 0 or result.get('StatusCode') == 0
            print(f"[飞书] 卡片消息: {'成功' if ok else '失败 - ' + str(result)}")
            return ok
        return False

    def upload_image(self, image_path):
        if not self.app_id:
            print("[飞书] 未配置app凭证，跳过图片上传")
            return None
        token = self._get_tenant_token()
        with open(image_path, 'rb') as f:
            resp = requests.post(
                'https://open.feishu.cn/open-apis/im/v1/images',
                headers={'Authorization': f'Bearer {token}'},
                files={'image_type': (None, 'message'),
                       'image': (os.path.basename(image_path), f, 'image/png')}
            )
        data = resp.json()
        if data.get('code') != 0:
            print(f"[飞书] 图片上传失败: {data}")
            return None
        return data['data']['image_key']

    def send_image(self, image_path):
        image_key = self.upload_image(image_path)
        if not image_key:
            return False
        if not self.webhook_url:
            return False
        payload = {"msg_type": "image", "content": {"image_key": image_key}}
        result = self._post(payload)
        if result:
            ok = result.get('code') == 0 or result.get('StatusCode') == 0
            print(f"[飞书] 图片消息: {'成功' if ok else '失败 - ' + str(result)}")
            return ok
        return False

    def send_backtest_report(self, results, benchmark_name='等权指数',
                              pool_name='电子科技50股', period='', generated=''):
        if not self.webhook_url:
            print("[飞书] 未配置webhook，跳过发送")
            return

        import numpy as np
        bench = next(iter(results.values()))['Cum_Benchmark']
        bench_total = bench.iloc[-1] * 100

        # 构建Markdown表格
        header  = "| 策略 | 总收益 | 年化收益 | 夏普比率 | 最大回撤 | 超额收益 |\n"
        header += "| :--- | ---: | ---: | ---: | ---: | ---: |\n"
        rows = ""
        cn_names = {
            'Multimodal Fusion (DL+XGB)': '多模态融合 (DL+XGB)',
            'Deep Learning Only (LSTM+Transformer+CNN)': '纯深度学习 (LSTM+Transformer+CNN)',
            'XGBoost Only': '纯 XGBoost',
        }
        for name, daily in results.items():
            total = daily['Cum_Strategy'].iloc[-1] * 100
            ann = daily['Strategy_Ret'].mean() * 240 * 100
            vol = daily['Strategy_Ret'].std() * np.sqrt(240)
            sharpe = (ann/100 - 0.03) / vol if vol > 1e-8 else 0
            mdd = (daily['Cum_Strategy'].cummax() - daily['Cum_Strategy']).max() * 100
            excess = total - bench_total
            display = cn_names.get(name, name)
            emoji = '🟢' if excess > 0 else ('🔴' if excess < 0 else '⚪')
            rows += f"| {emoji} {display} | {total:.2f}% | {ann:.2f}% | {sharpe:.2f} | {mdd:.2f}% | {excess:+.2f}% |\n"

        rows += f"| **{benchmark_name}(基准)** | **{bench_total:.2f}%** | | | | |\n"

        # 找最佳策略
        best_name = max(results, key=lambda n: results[n]['Cum_Strategy'].iloc[-1])
        best_display = cn_names.get(best_name, best_name)
        best_total = results[best_name]['Cum_Strategy'].iloc[-1] * 100

        elements = [
            {"tag": "markdown", "content": f"**回测区间:** {period}\n**股票池:** {pool_name} · 等权基准\n**生成时间:** {generated}"},
            {"tag": "hr"},
            {"tag": "markdown", "content": header + rows},
            {"tag": "hr"},
            {"tag": "column_set", "flex_mode": "bisect",
             "columns": [
                 {"tag": "column", "width": "weighted", "weight": 1,
                  "elements": [{"tag": "markdown",
                                "content": f"🏆 **最佳策略**\n**{best_display}**\n累计收益 **{best_total:.2f}%**"}]},
                 {"tag": "column", "width": "weighted", "weight": 1,
                  "elements": [{"tag": "markdown",
                                "content": f"📊 **基准指数**\n{benchmark_name}\n累计收益 **{bench_total:.2f}%**"}]}
             ]},
            {"tag": "hr"},
            {"tag": "note",
             "elements": [{"tag": "plain_text",
                           "content": f"📁 完整图表及数据 → output 目录 | deep_quant v2.0"}]}
        ]

        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": "深度量化 多模型回测报告"},
                    "template": "blue"
                },
                "elements": elements
            }
        }
        result = self._post(payload)
        if result:
            ok = result.get('code') == 0 or result.get('StatusCode') == 0
            print(f"[飞书] 回测报告卡片: {'成功' if ok else '失败 - ' + str(result)}")
            return ok
        return False