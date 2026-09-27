"""sensor.py と engine.py の結果を、アプリ（src/core/look.ts）が読む JSON にまとめる。

使い方:
    python export_model.py ../../src/models/leica-m10.json --fixture ../../test/fixtures/leica.json

--fixture を付けると、決まった入力に対するこの Python 実装の出力を書き出す。
test/look.test.ts はこれと TypeScript 実装の出力が一致することを確かめる。
"""
import argparse
import json
import os
import re

import numpy as np

from common import KNOTS, LUMA, OUT, curve, eotf, oetf

CURVE_N = 1001  # log2 の 0.01 段刻み
DOMAIN = [float(KNOTS[0]), float(KNOTS[-1])]
KNEE = 0.75  # src/core/color.ts の shoulder と同じ
MID = 118.9 / 255  # 中間グレー（18%）のカメラ内 JPEG での値


def sample_curve(c):
    u = np.linspace(DOMAIN[0], DOMAIN[1], CURVE_N)
    return np.round(curve(u, np.array(c)), 6)


# --- ここから下は src/core/look.ts と 1 対 1 に対応させた参照実装 -------------------------

def curve_eval(samples, v):
    """log2(v) で標本を線形補間。範囲外は 0 / 1"""
    v = np.asarray(v, float)
    u = np.log2(np.maximum(v, 1e-30))
    t = (u - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * (CURVE_N - 1)
    t = np.clip(t, 0, CURVE_N - 1)
    i = np.minimum(np.floor(t).astype(int), CURVE_N - 2)
    y = samples[i] + (samples[i + 1] - samples[i]) * (t - i)
    return np.where(v > 0, y, 0.0)


def curve_inverse(samples, y):
    """curve_eval の逆関数（y → v）"""
    k = np.searchsorted(samples, y)
    k = np.clip(k, 1, CURVE_N - 1)
    y0, y1 = samples[k - 1], samples[k]
    t = np.where(y1 > y0, (y - y0) / np.maximum(y1 - y0, 1e-12), 0)
    u = DOMAIN[0] + (k - 1 + t) * (DOMAIN[1] - DOMAIN[0]) / (CURVE_N - 1)
    return 2.0 ** u


def shoulder_inv(y):
    r = 1 - KNEE
    y = np.minimum(y, 1 - 1e-6)
    return np.where(y <= KNEE, y, KNEE - r * np.log(1 - (y - KNEE) / r))


def camera_key(make, model):
    """機種名の表記ゆれをそろえたキー（src/core/look.ts の cameraKey と同じ）。
    例: ('Canon', 'Canon EOS R6 Mark II') / ('Canon', 'Canon EOS R6m2') → 'canon:r6m2'"""
    mk = (make or '').strip().lower().split(' ')[0] if make else ''
    m = (model or '').lower()
    for w in (mk, 'eos'):
        if w:
            m = m.replace(w, ' ')
    m = re.sub(r'[^a-z0-9]', '', m)
    for a, b in (('markiii', 'm3'), ('markiv', 'm4'), ('markii', 'm2'), ('mark3', 'm3'), ('mark4', 'm4'), ('mark2', 'm2')):
        m = m.replace(a, b)
    return f'{mk}:{m}'


def sensor_matrix(model, camera, cct):
    ent = None
    for c in model['sensor']['cameras']:
        if camera in c['models']:
            ent = c
    ent = ent or model['sensor']['generic']
    A, D = np.array(ent['A']).reshape(3, 3), np.array(ent['D']).reshape(3, 3)
    lo, hi = ent['cct']
    if cct is None or cct >= hi:
        return D
    if cct <= lo:
        return A
    g = (1 / cct - 1 / hi) / (1 / lo - 1 / hi)
    return g * A + (1 - g) * D


def keep_luminance(lin, yi):
    """色度はそのままで輝度（リニア）を yi にそろえる。はみ出す色は輝度を保ったまま彩度を落とす"""
    yo = lin @ LUMA
    p = np.where(yo[:, None] > 1e-6, lin * (yi / np.maximum(yo, 1e-6))[:, None], yi[:, None])
    mx = p.max(1)
    f = np.where(mx > 1, np.where(mx > yi, (1 - yi) / np.maximum(mx - yi, 1e-12), 0), 1)
    return oetf(yi[:, None] + (p - yi[:, None]) * f[:, None])


def apply_look(model, d, encoding, camera=None, cct=None, jpeg_base=0.0, tone=0.0):
    """表示用 sRGB (0..1, n×3) → Leica M10 の色 (0..1)。
    encoding='linear': RAW（アプリの素の表示を戻す。jpeg_base の割合でカメラ内 JPEG 並みのトーンカーブの逆を混ぜる）
    encoding='srgb': JPEG（Canon のピクチャースタイル「スタンダード」を打ち消す）
    tone: Leica のトーンカーブの割合。0 なら輝度は元のまま、色相・彩度だけを Leica にする"""
    L, C = model['leica'], model['canon']
    tl, tc = np.array(L['curve']), np.array(C['curve'])
    M = np.array(L['matrix']).reshape(3, 3) @ sensor_matrix(model, camera, cct)
    vl = curve_inverse(tl, MID)  # Leica の中間グレー（行列・露出を掛けた後の値）
    vc = curve_inverse(tc, MID)  # Canon の中間グレー
    jpeg = lambda: curve_inverse(tc, np.clip(d, 0, 1)) * 0.18 / vc
    if encoding == 'linear':
        # アプリの素の表示（sRGB + 肩特性）を戻したリニア値。0.18 が中間グレー
        x = shoulder_inv(eotf(d))
        if jpeg_base > 0:
            x = x + (jpeg() - x) * min(1.0, jpeg_base)
    else:
        M = M @ np.linalg.inv(np.array(C['matrix']).reshape(3, 3))
        x = jpeg()
    K = (vl / 0.18) * M
    o = curve_eval(tl, x @ K.T)
    q = keep_luminance(eotf(o), eotf(d) @ LUMA)
    return q + (o - q) * tone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--fixture', default='')
    args = ap.parse_args()
    sensor = json.load(open(os.path.join(OUT, 'sensor.json')))
    engine = json.load(open(os.path.join(OUT, 'engine.json')))
    flat = lambda m: [round(float(v), 6) for v in np.array(m).ravel()]
    key = lambda name: camera_key(name.split(' ')[0], name)
    cams = [dict(models=[key(m) for m in c['models']], cct=c['cct'], A=flat(c['S_A']), D=flat(c['S_D'])) for c in sensor['cameras'].values()]
    # Leica M10 系（同じセンサー）の RAW はセンサーの変換をしない
    eye = flat(np.eye(3))
    cams.append(dict(models=[key(m) for m in ('Leica M10', 'Leica M10-P', 'Leica M10-D')], cct=[2856, 6504], A=eye, D=eye))
    lei, can = engine['leica_m10r'], engine['canon_r6']
    model = dict(
        version=1,
        curve_domain=DOMAIN,
        leica=dict(matrix=flat(lei['matrix']), curve=sample_curve(lei['curve']).tolist()),
        canon=dict(matrix=flat(can['matrix']), curve=sample_curve(can['curve']).tolist()),
        sensor=dict(cameras=cams, generic=dict(models=[], cct=sensor['generic']['cct'], A=flat(sensor['generic']['S_A']),
                                               D=flat(sensor['generic']['S_D']))),
        info=dict(
            target='Leica M10 (Leica embedded DNG profile + in-camera JPEG, Standard)',
            sensor='Canon → Leica M10 3×3 fitted by spectral simulation (predicted SSFs: Solomatov & Akkaynak, ICCP 2023; '
                   'reflectances: ColorChecker, CIE 2017 CES 99, ISSA skin); Leica matrices read from an M10 DNG',
            leica_engine=f'fitted to raw + in-camera JPEG of a Leica M10-R (CC0, raw.pixls.us); ΔE00 held-out {lei["fit"]["test"][0]}',
            canon_engine=f'fitted to raw + in-camera JPEG of a Canon EOS R6, Picture Style Standard (CC0); ΔE00 held-out {can["fit"]["test"][0]}',
            training='training/leica'),
    )
    # 露出の倍率は中間グレーを合わせるときに打ち消されるので、行列とカーブだけを持てばよい
    with open(args.out, 'w') as f:
        json.dump(model, f, separators=(',', ':'), ensure_ascii=False)
    print('wrote', args.out, os.path.getsize(args.out), 'bytes')

    if args.fixture:
        g = np.linspace(0, 1, 9)
        grid = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
        extra = np.array([[0.46, 0.46, 0.46], [0.8, 0.45, 0.35], [0.25, 0.5, 0.2], [0.2, 0.35, 0.7], [0.93, 0.8, 0.7], [0.999, 0.999, 0.999]])
        inp = np.concatenate([grid, extra])
        r6m2 = camera_key('Canon', 'Canon EOS R6m2')
        cases = [dict(encoding='linear', camera=r6m2, cct=5200.0),
                 dict(encoding='linear', camera=r6m2, cct=3300.0),
                 dict(encoding='linear', camera=r6m2, cct=None),
                 dict(encoding='linear', camera=camera_key('NIKON CORPORATION', 'NIKON Z 6'), cct=6000.0),
                 dict(encoding='linear', camera=r6m2, cct=5200.0, jpeg_base=1.0),
                 dict(encoding='linear', camera=r6m2, cct=5200.0, jpeg_base=0.5),
                 dict(encoding='srgb', camera=r6m2, cct=None),
                 dict(encoding='linear', camera=r6m2, cct=5200.0, tone=1.0),
                 dict(encoding='linear', camera=r6m2, cct=5200.0, jpeg_base=1.0, tone=1.0),
                 dict(encoding='srgb', camera=r6m2, cct=None, tone=1.0)]
        for c in cases:
            c.setdefault('jpeg_base', 0.0)
            c.setdefault('tone', 0.0)
            c['out'] = np.round(apply_look(model, inp, c['encoding'], c['camera'], c['cct'], c['jpeg_base'], c['tone']), 6).ravel().tolist()
        with open(args.fixture, 'w') as f:
            json.dump(dict(input=np.round(inp, 6).ravel().tolist(), cases=cases), f, separators=(',', ':'))
        mid = apply_look(model, np.array([[oetf(0.18)] * 3]), 'linear', r6m2, 5200.0, tone=1.0)
        print('wrote', args.fixture, '| mid-grey in', round(float(oetf(0.18)) * 255, 1), '→ out', np.round(mid * 255, 1))


if __name__ == '__main__':
    main()
