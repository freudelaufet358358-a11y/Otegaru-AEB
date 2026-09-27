"""評価用: 学習に使っていない HDR+ のシーンから、カメラの AEB（-2 / 0 / +2 EV）を模したブラケットを作る。

使い方:
    python make_brackets.py ../data/test scenes 1024

0 EV のフレームは平均測光のように露出を決め（平均輝度 → 飽和点の 12.5%。Canon の RAW で中間グレーが
来るあたり）、各フレームはセンサーの飽和点でクリップし、アプリの LibRaw 出力と同じ 16bit に量子化する。
"""
import glob, json, os, struct, sys
import numpy as np
from PIL import Image

src = sys.argv[1] if len(sys.argv) > 1 else 'data/test'
dst = sys.argv[2] if len(sys.argv) > 2 else 'eval/scenes'
side = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
os.makedirs(dst, exist_ok=True)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
EVS = [-2, 0, 2]
rng = np.random.default_rng(0)
names = []
for f in sorted(glob.glob(os.path.join(src, '*.npz'))):
    z = np.load(f)
    key = f'in{side}' if f'in{side}' in z else 'in512'
    x = z[key].astype(np.float32)
    t = z[key.replace('in', 'tgt')]
    y = x @ LUMA
    scale = 0.125 / max(float(y.mean()), 1e-8)
    name = os.path.basename(f)[:-4]
    h, w = x.shape[:2]
    with open(os.path.join(dst, name + '.bin'), 'wb') as o:
        o.write(struct.pack('<4I', w, h, len(EVS), 0))
        o.write(np.array([2.0 ** e for e in EVS], np.float32).tobytes())
        for e in EVS:
            fr = np.clip(x * scale * 2.0 ** e, 0, 1)
            o.write(np.round(fr * 65535).astype('<u2').tobytes())
    Image.fromarray(t).save(os.path.join(dst, name + '_target.png'))
    names.append(name)
json.dump(names, open(os.path.join(dst, 'index.json'), 'w'))
print('scenes', len(names))
