"""評価用: カメラ JPEG 風のブラケット（リニア → 肩特性 → sRGB → S 字のコントラスト → 8bit）を作る。

使い方:
    python make_jpeg_brackets.py ../data/test jpeg_scenes シーン名1,シーン名2,...
ヘッダの 4 番目の値を 1 にして、render_eval.ts に sRGB 符号値のフレームだと伝える。
"""
import json, os, struct, sys
import numpy as np
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
names = sys.argv[3].split(',')
os.makedirs(dst, exist_ok=True)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
def shoulder(x, k=0.75):
    r = 1 - k
    return np.where(x <= k, x, k + r * (1 - np.exp(-(x - k) / r)))
def l2s(v):
    v = np.clip(v, 0, 1); return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
def camera_curve(x):
    v = l2s(shoulder(x * 1.4))
    return v + 0.3 * (v * v * (3 - 2 * v) - v)
for n in names:
    z = np.load(os.path.join(src, n + '.npz'))
    x = z['in1024'].astype(np.float32)
    scale = 0.125 / max(float((x @ LUMA).mean()), 1e-8)
    h, w = x.shape[:2]
    with open(os.path.join(dst, n + '.bin'), 'wb') as f:
        f.write(struct.pack('<4I', w, h, 3, 1))  # 4 番目 = 1: sRGB 符号値
        f.write(np.array([0.25, 1, 4], np.float32).tobytes())
        for e in (-2, 0, 2):
            v8 = np.round(camera_curve(x * scale * 2.0 ** e) * 255)
            f.write((v8 * 257).astype('<u2').tobytes())
    Image.fromarray(z['tgt1024']).save(os.path.join(dst, n + '_target.png'))
json.dump(names, open(os.path.join(dst, 'index.json'), 'w'))
print('ok', len(names))
