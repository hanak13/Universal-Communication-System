import os, subprocess
# Create junction
src = r"c:\Users\pradh\Documents\Universal Communication System"
dst = r"c:\Users\pradh\UCS"
if not os.path.exists(dst):
    subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True)
    print(f"Created junction {dst} -> {src}")
else:
    print(f"Junction {dst} already exists")

