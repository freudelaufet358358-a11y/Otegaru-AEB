"""ドキュメント用の比較画像を作る（docs/leica-m10-*.jpg）。

1. 実写（EOS R6 の CR3）: アプリの標準 / Leica M10（既定: 明るさはそのまま）/ Leica M10（階調も）/ Canon のカメラ内 JPEG
2. 色見本（ColorChecker 24 色 + 肌 6 種、D65）: EOS R6 Mark II（アプリの標準） / Leica M10

使い方: python sheet.py ../../docs
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from common import DATA, illuminant, libraw_matrix, load_cm, load_ssf, oetf, reflectances, simulate
from engine import decode
from export_model import apply_look, camera_key

KNEE = 0.75


def shoulder(x):
    r = 1 - KNEE
    return np.where(x <= KNEE, x, KNEE + r * (1 - np.exp(-(x - KNEE) / r)))


def app_display(x):
    """アプリの素の表示: RAW_DISPLAY_GAIN (√2) は露出の一部とみなし、中間グレーを 0.18 に合わせたものを渡す"""
    return oetf(shoulder(x))


def to8(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return im


def main(outdir):
    model = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'models', 'leica-m10.json')))
    engine = json.load(open(os.path.join(os.path.dirname(__file__), 'out', 'engine.json')))
    cam, wb, jpg = decode(os.path.join(DATA, 'raw-canon-eos-r6.cr3'))
    x = (cam * wb) @ libraw_matrix(load_cm('Canon-EOS-R6', 2)).T
    # カメラ内 JPEG と同じ露出にそろえる（Canon の中間グレーの位置 → 0.18）
    x = x * (0.18 / 2 ** engine['canon_r6']['tone']['mid_log2'])
    d = app_display(x)
    h, w = d.shape[:2]
    cct = 5200.0  # この CR3 の撮影時の色温度（EXIF の ColorTemperature）
    key = camera_key('Canon', 'Canon EOS R6')
    look = apply_look(model, d.reshape(-1, 3), 'linear', key, cct).reshape(h, w, 3)
    look_tone = apply_look(model, d.reshape(-1, 3), 'linear', key, cct, tone=1.0).reshape(h, w, 3)
    j = np.asarray(Image.fromarray((jpg * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)) / 255
    tiles = [label(to8(d), 'app standard'), label(to8(look), 'Leica M10'), label(to8(look_tone), 'Leica M10 + tone'),
             label(to8(j), 'Canon in-camera JPEG')]
    for t in tiles:
        t.thumbnail((640, 640))
    W = sum(t.width for t in tiles)
    sheet = Image.new('RGB', (W, tiles[0].height))
    xo = 0
    for t in tiles:
        sheet.paste(t, (xo, 0))
        xo += t.width
    sheet.save(os.path.join(outdir, 'leica-m10-comparison.jpg'), quality=88)

    # 色見本
    refl = reflectances()
    skin = refl['SKIN'][np.linspace(0, len(refl['SKIN']) - 1, 6).astype(int)]
    R = np.concatenate([refl['CC24'], skin])
    E = illuminant('D65')
    raw, white = simulate(load_ssf('Canon-EOS-R6-Mark-II'), E, R)
    xc = (raw / white) @ libraw_matrix(load_cm('Canon-EOS-R6-Mark-II', 2)).T * (0.18 / 0.19)  # CC の 22 番（中間グレー）≒ 0.19
    dc = app_display(xc)
    dl = apply_look(model, dc, 'linear', camera_key('Canon', 'Canon EOS R6 Mark II'), 6000.0)
    s = 64
    img = Image.new('RGB', (6 * s, 5 * s * 2 + 30), (40, 40, 40))
    dr = ImageDraw.Draw(img)
    for k in range(len(R)):
        r, c = divmod(k, 6)
        for half, col in ((0, dc[k]), (1, dl[k])):
            y0 = r * 2 * s + half * s
            dr.rectangle([c * s, y0, c * s + s - 2, y0 + s - 2], fill=tuple(int(v * 255 + 0.5) for v in np.clip(col, 0, 1)))
    dr.text((4, 10 * s + 8), 'top: EOS R6 Mark II (app standard) / bottom: Leica M10', fill=(230, 230, 230))
    img.save(os.path.join(outdir, 'leica-m10-patches.png'))
    print('wrote', outdir)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out')
