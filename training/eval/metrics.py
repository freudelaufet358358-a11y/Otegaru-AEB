"""各モードの仕上がりと HDR+ の完成画像との差（PSNR / SSIM / CIELAB の色差）を測る。

使い方:
    python metrics.py scenes out [ref,fusion,hdr,learned]
"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

scenes, outdir = sys.argv[1], sys.argv[2]
modes = sys.argv[3].split(',') if len(sys.argv) > 3 else ['ref', 'fusion', 'hdr', 'learned']
names = json.load(open(os.path.join(scenes, 'index.json')))

def load(name, mode, w, h):
    return np.fromfile(os.path.join(outdir, f'{name}_{mode}.rgb'), np.uint8).reshape(h, w, 3)

def gauss(img, s=1.5):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, s)

def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        mx, my = gauss(x), gauss(y)
        sxx = gauss(x * x) - mx * mx; syy = gauss(y * y) - my * my; sxy = gauss(x * y) - mx * my
        m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        vals.append(m.mean())
    return float(np.mean(vals))

def srgb2lab(img):
    v = img.astype(np.float64) / 255
    v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = v @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

res = {m: dict(psnr=[], ssim=[], de=[], dL=[]) for m in modes}
for name in names:
    meta = json.load(open(os.path.join(outdir, name + '.json')))
    w, h = meta['w'], meta['h']
    t = np.asarray(Image.open(os.path.join(scenes, name + '_target.png')).convert('RGB'))
    lt = srgb2lab(t)
    for m in modes:
        o = load(name, m, w, h)
        mse = ((o.astype(np.float64) - t) ** 2).mean()
        res[m]['psnr'].append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
        res[m]['ssim'].append(ssim(o, t))
        lo = srgb2lab(o)
        res[m]['de'].append(float(np.sqrt(((lo - lt) ** 2).sum(-1)).mean()))
        res[m]['dL'].append(float(np.abs(lo[..., 0] - lt[..., 0]).mean()))
summary = {m: {k: float(np.mean(v)) for k, v in r.items()} for m, r in res.items()}
for m in modes:
    s = summary[m]
    print(f"{m:8s} PSNR {s['psnr']:.2f} dB  SSIM {s['ssim']:.4f}  ΔE {s['de']:.2f}  |ΔL*| {s['dL']:.2f}")
json.dump(dict(summary=summary, per_scene={m: {k: list(map(float, v)) for k, v in r.items()} for m, r in res.items()}, names=names),
          open(os.path.join(outdir, 'metrics.json'), 'w'))
