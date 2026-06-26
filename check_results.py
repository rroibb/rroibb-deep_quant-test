import os

# Check latest output files
output_dir = r'D:\opencode工作区\deep_quant\output'
for f in sorted(os.listdir(output_dir), reverse=True)[:5]:
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f}  ({size} bytes)")

# Check if virtual_trade CSV was created
for f in os.listdir(output_dir):
    if 'virtual' in f.lower():
        print(f"\nVIRTUAL TRADE FILE: {f}")