"""ng.pt 端到端推理冒烟测试。

用法（在仓库根目录）：
    NitroGen\\.venv\\Scripts\\python.exe scripts\\smoke_test.py

验证内容：
1. 从官方 ng.pt 权重加载 NitroGen 模型（含 SigLIP2 视觉编码器，首次运行自动下载）；
2. 对一张随机 1280x720 RGB 帧执行 zero-shot 推理；
3. 校验输出动作块的形状与数值合法性。
"""
import builtins
import time

builtins.input = lambda *a: ""  # 跳过交互式游戏选择，使用无条件模式

import numpy as np
from PIL import Image

print("[1/3] loading model from ng.pt ...", flush=True)
t0 = time.time()
from nitrogen.inference_session import InferenceSession

session = InferenceSession.from_ckpt(r"NitroGen\ng.pt")
print(f"      model loaded in {time.time()-t0:.1f}s", flush=True)

print("[2/3] running predict on a dummy 1280x720 frame ...", flush=True)
img = Image.fromarray(np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8))
t0 = time.time()
result = session.predict(img)
print(f"      inference done in {time.time()-t0:.1f}s", flush=True)

print("[3/3] outputs:", flush=True)
buttons = np.asarray(result["buttons"])
j_left = np.asarray(result["j_left"])
j_right = np.asarray(result["j_right"])
print("      buttons shape:", buttons.shape)
print("      j_left  shape:", j_left.shape)
print("      j_right shape:", j_right.shape)

# 形状校验：18 步动作块，21 维按键 + 左/右摇杆 (x, y)
assert buttons.shape == (18, 21), f"unexpected buttons shape: {buttons.shape}"
assert j_left.shape == (18, 2), f"unexpected j_left shape: {j_left.shape}"
assert j_right.shape == (18, 2), f"unexpected j_right shape: {j_right.shape}"
assert j_left.min() >= -1.05 and j_left.max() <= 1.05, "j_left out of [-1,1] range"
assert j_right.min() >= -1.05 and j_right.max() <= 1.05, "j_right out of [-1,1] range"

print("      buttons[0]:", buttons[0])
print("      j_left[0]:", j_left[0])
print("SMOKE TEST PASSED")
