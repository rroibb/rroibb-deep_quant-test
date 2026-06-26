import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow", "-q"], check=True)
print("Done")
print(sys.executable)