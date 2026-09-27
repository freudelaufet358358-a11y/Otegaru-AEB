"""調査結果の表（Markdown）を out/report.md に書き出す。docs/leica-m10-color.md の数値はここから取っている。

使い方: python report.py
"""
import json
import os

import colour
import numpy as np

from common import (OUT, eotf, illuminant, lab, libraw_matrix, load_cm, load_ssf, oetf, reflectances,
                    simulate, curve)
from export_model import apply_look, camera_key

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE = 0.75


def shoulder(x):
    r = 1 - KNEE
    return np.where(x <= KNEE, x, KNEE + r * (1 - np.exp(-(x - KNEE) / r)))


def lch(srgb_encoded):
    return colour.Lab_to_LCHab(lab(eotf(srgb_encoded)))


def hue_table(a, b, names=('chroma', 'hue', 'L*'), min_chroma=10):
    """a → b の違いを a の色相ごとに（彩度の比・色相のずれ・明度の差の中央値）"""
    A, B = lch(a), lch(b)
    rows = []
    labels = ['赤紫〜赤', '赤〜橙', '橙〜黄', '黄〜黄緑', '黄緑〜緑', '緑', '緑〜青緑', '青緑〜空色', '空色〜青', '青〜紫', '紫〜赤紫', '赤紫']
    for k, h0 in enumerate(range(0, 360, 30)):
        s = (A[:, 1] > min_chroma) & (A[:, 2] >= h0) & (A[:, 2] < h0 + 30)
        if s.sum() < 2:
            continue
        cr = np.median(B[s, 1] / A[s, 1])
        dh = np.median(((B[s, 2] - A[s, 2] + 180) % 360) - 180)
        dl = np.median(B[s, 0] - A[s, 0])
        rows.append(f'| {h0}〜{h0 + 30}° | {labels[k]} | {s.sum()} | ×{cr:.2f} | {dh:+.1f}° | {dl:+.1f} |')
    return '| CIELAB 色相 | 目安 | 色数 | 彩度 | 色相 | 明度 L* |\n| --- | --- | --- | --- | --- | --- |\n' + '\n'.join(rows)


def main():
    model = json.load(open(os.path.join(HERE, '..', '..', 'src', 'models', 'leica-m10.json')))
    engine = json.load(open(os.path.join(OUT, 'engine.json')))
    sensor = json.load(open(os.path.join(OUT, 'sensor.json')))
    refl = reflectances()
    R = np.concatenate([refl['CC24'], refl['CES99']])
    E = illuminant('D65')
    raw, white = simulate(load_ssf('Canon-EOS-R6-Mark-II'), E, R)
    x = (raw / white) @ libraw_matrix(load_cm('Canon-EOS-R6-Mark-II', 2)).T  # アプリの RAW 現像（白 = 1）
    x = x * 0.18 / 0.2  # 白 = 0.9 の反射率を中間グレー基準で写した明るさに
    std = oetf(shoulder(x))  # アプリの標準の表示
    key = camera_key('Canon', 'Canon EOS R6 Mark II')
    leica_color = apply_look(model, std, 'linear', key, 6000.0)
    leica_full = apply_look(model, std, 'linear', key, 6000.0, tone=1.0)
    # Canon のピクチャースタイル「スタンダード」: 当てはめたエンジンを同じ露出で
    c = engine['canon_r6']
    vmid = 2 ** c['tone']['mid_log2']
    canon = curve(np.log2(np.maximum((x / 0.18 * vmid) @ np.array(c['matrix']).T * c['exposure'], 1e-9)), np.array(c['curve']))
    out = []
    out.append('## 1. センサー + 色行列の違い（Canon EOS R6 Mark II → Leica M10、D65、ColorChecker + CIE 2017 の 99 色）\n')
    rows = []
    for h in sensor['report']['hue_D65']:
        rows.append(f"| {h['hue'][0]}〜{h['hue'][1]}° | {h['n']} | ×{h['chroma']:.2f} | {h['dhue']:+.1f}° | {h['dL']:+.1f} |")
    out.append('| CIELAB 色相 | 色数 | 彩度 | 色相 | 明度 L* |\n| --- | --- | --- | --- | --- |\n' + '\n'.join(rows))
    out.append(f"\n肌（ISSA 15,256 件）: {sensor['report']['skin_D65']}\n")
    out.append('照明ごとの差（ΔE00 平均 / 95 パーセンタイル、変換前 → 変換後）:\n')
    out.append('| 照明 | ColorChecker | CIE 99 色 | 肌 |\n| --- | --- | --- | --- |')
    for ill, row in sensor['report']['illuminants'].items():
        cells = [f"{row[k]['before'][0]:.2f} / {row[k]['before'][1]:.2f} → {row[k]['after'][0]:.2f} / {row[k]['after'][1]:.2f}" for k in ('CC24', 'CES99', 'SKIN')]
        out.append(f'| {ill} | ' + ' | '.join(cells) + ' |')
    out.append(f"\n色の正確さ（人の目との ΔE00、D65）: {sensor['report']['accuracy_D65']}\n")
    out.append('## 2. カメラ内 JPEG のトーンカーブ（中間グレーからの段数 → sRGB 値）\n')
    out.append('| | ' + ' | '.join(f'{s:+d}' for s in range(-5, 4)) + ' | 中間調の傾き (L*/段) | 白飛びまで |')
    out.append('| --- | ' + ' | '.join('---' for _ in range(-5, 4)) + ' | --- | --- |')
    for name, t in [('Leica M10-R', engine['leica_m10r']['tone']), ('Canon EOS R6', engine['canon_r6']['tone']),
                    ('ACR 既定', engine['acr_default'])]:
        tab = {int(k): v for k, v in t['sRGB_by_stop'].items()}
        out.append(f'| {name} | ' + ' | '.join(f'{tab[s]:.0f}' for s in range(-5, 4)) + f" | {t['slope']:.1f} | +{t['headroom']:.2f} 段 |")
    u = np.arange(-5, 4)
    app = oetf(shoulder(0.18 * 2.0 ** u)) * 255
    out.append('| アプリ標準 | ' + ' | '.join(f'{v:.0f}' for v in app) + ' | 15.3 | (肩特性) |')
    out.append('\n## 3. JPEG エンジンの色の行列（メーカーの色行列で現像した色 → JPEG）\n')
    for name, k in [('Leica M10-R', 'leica_m10r'), ('Canon EOS R6', 'canon_r6')]:
        A = np.array(engine[k]['matrix'])
        out.append(f"- {name}（当てはめの誤差 ΔE00: 学習 {engine[k]['fit']['train'][0]}、検証 {engine[k]['fit']['test'][0]}）\n\n```\n{np.array2string(A, precision=3, suppress_small=True)}\n```\n")
    out.append('## 4. 仕上がりの違い（R6 Mark II で撮った色見本、D65）\n')
    out.append('### アプリ標準 → Leica M10（既定: 明るさはそのまま）\n')
    out.append(hue_table(std, leica_color))
    out.append('\n### アプリ標準 → Leica M10（階調も Leica）\n')
    out.append(hue_table(std, leica_full))
    out.append('\n### Canon のカメラ内 JPEG（スタンダード）→ Leica M10 のカメラ内 JPEG\n')
    out.append(hue_table(canon, leica_full))
    dE = colour.delta_E(lab(eotf(canon)), lab(eotf(leica_full)), method='CIE 2000')
    out.append(f'\nΔE00: 平均 {dE.mean():.2f}、95 パーセンタイル {np.percentile(dE, 95):.2f}、最大 {dE.max():.2f}\n')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
