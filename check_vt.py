import os
f = r'D:\opencode工作区\deep_quant\output\virtual_trade_log.txt'
try:
    raw = open(f, 'rb').read()
    text = raw.decode('utf-16-le', errors='replace')
    print(text[-3000:])
except Exception as e:
    print(f"Error: {e}", "filePath": "D:\\opencode工作区\\deep_quant\\check_vt.py")