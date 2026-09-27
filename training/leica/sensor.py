"""センサー（分光感度）と色行列の違いを分光シミュレーションで求める。

Canon の RAW をアプリと同じ方法（LibRaw、Adobe の D65 行列）でリニア sRGB にしたものを、
「同じ被写体を Leica M10 で撮り、Leica が DNG に埋め込んだ色行列で現像したもの」に写す 3×3 行列 S を当てはめる。

- 分光感度: Solomatov & Akkaynak (ICCP 2023) が色行列から推定した 1,000 機種以上の分光感度
- 反射率: ColorChecker 24 色 + CIE 2017 の 99 色 + 肌（ISSA）
- 照明: 昼光（D50〜D75）と標準光源 A でそれぞれ当てはめ、アプリでは撮影時の色温度で補間する

使い方: python sensor.py   → out/sensor.json と、違いの分析表を表示
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

from common import (OUT, DATA, dng_matrix, dng_profile, illuminant, lab, libraw_cct, libraw_matrix, load_cm,
                    load_ssf, neutral_matrix, reflectances, simulate, truth_srgb, de00)
import colour

np.set_printoptions(precision=4, suppress=True)

# アプリの EXIF の機種名（LibRaw の make + model）→ 推定分光感度の名前
CANON = {
    'Canon EOS R6 Mark II': 'Canon-EOS-R6-Mark-II',
    'Canon EOS R8': 'Canon-EOS-R6-Mark-II',  # R6 Mark II と同じセンサー
    'Canon EOS R6': 'Canon-EOS-R6',
    'Canon EOS R5': 'Canon-EOS-R5',
    'Canon EOS R5 C': 'Canon-EOS-R5-C',
    'Canon EOS R3': 'Canon-EOS-R3',
    'Canon EOS R': 'Canon-EOS-R',
    'Canon EOS Ra': 'Canon-EOS-Ra',
    'Canon EOS RP': 'Canon-EOS-RP',
    'Canon EOS R7': 'Canon-EOS-R7',
    'Canon EOS R10': 'Canon-EOS-R10',
    'Canon EOS 5D Mark IV': 'Canon-EOS-5D-Mark-IV',
    'Canon EOS 5DS': 'Canon-EOS-5DS',
    'Canon EOS 5DS R': 'Canon-EOS-5DS-R',
    'Canon EOS 6D Mark II': 'Canon-EOS-6D-Mark-II',
    'Canon EOS 90D': 'Canon-EOS-90D',
    'Canon EOS 1D X Mark III': 'Canon-EOS-1D-X-Mark-III',
    'Canon EOS M50 Mark II': 'Canon-EOS-M50-Mark-II',
    'Canon EOS M6 Mark II': 'Canon-EOS-M6-Mark-II',
}
TARGET = 'LEICA-M10'
DAYLIGHT = ['D50', 'D55', 'D65', 'D75']


def leica_m10_profile():
    """Leica M10 の DNG に Leica が埋め込んだプロファイル（ファームウェア 2017 年 1 月のファイル）"""
    p = dng_profile(os.path.join(DATA, 'mime-fixtures', 'fixture-Leica-M10.dng'))
    assert p['model'] == 'LEICA M10' and p['profile_name'] == 'LEICA M10'
    assert p['illuminant1'] == [17] and p['illuminant2'] == [21]
    return p


def split_sets(refl, rng):
    skin = refl['SKIN']
    idx = rng.permutation(len(skin))
    train = [refl['CC24'], refl['CES99'], skin[idx[:300]]]
    test = {'CC24': refl['CC24'], 'CES99': refl['CES99'], 'SKIN': skin[idx[300:]]}
    return train, test


class Pair:
    """ある照明の下での「アプリで見た Canon の色」と「Leica M10 で撮った色」"""

    def __init__(self, src_ssf, src_cm65, prof):
        self.src_ssf = src_ssf
        self.src = libraw_matrix(src_cm65) if src_ssf is not None else None
        self.ssfL = load_ssf(TARGET)
        self.prof = prof

    def __call__(self, ill, R):
        E = illuminant(ill)
        if self.src_ssf is None:  # 汎用: 色を正しく測れるカメラとみなす
            x = truth_srgb(E, R)
        else:
            raw, w = simulate(self.src_ssf, E, R)
            x = (raw / w) @ self.src.T
        rawL, wL = simulate(self.ssfL, E, R)
        y = (rawL / wL) @ dng_matrix(self.prof['color_matrix1'], self.prof['color_matrix2'], wL).T
        return x, y


def fit_S(pair, ills, sets, weights=(1.0, 1.0, 0.3)):
    X, Y, W = [], [], []
    for ill in ills:
        for R, w in zip(sets, weights):
            x, y = pair(ill, R)
            X.append(x)
            Y.append(y)
            W.append(np.full(len(x), w))
    X, Y, W = np.concatenate(X), np.concatenate(Y), np.sqrt(np.concatenate(W))[:, None]
    LY = lab(Y)
    r = least_squares(lambda p: ((lab(X @ neutral_matrix(p).T) - LY) * W).ravel(), np.zeros(6))
    return neutral_matrix(r.x)


def interp_S(S_A, S_D, lo, hi, cct):
    if cct <= lo:
        return S_A
    if cct >= hi:
        return S_D
    g = (1 / cct - 1 / hi) / (1 / lo - 1 / hi)
    return g * S_A + (1 - g) * S_D


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    refl = reflectances()
    train, test = split_sets(refl, rng)
    prof = leica_m10_profile()
    print('Leica M10 embedded profile:', prof['profile_name'], 'BaselineExposure', prof['baseline_exposure'])
    print(' ColorMatrix1 (A)\n', prof['color_matrix1'], '\n ColorMatrix2 (D65)\n', prof['color_matrix2'])
    print(' LookTable dims', prof['look_table_dims'], 'data', prof['look_table_data'], 'HueSatMap', prof['hue_sat_map_dims'],
          'ToneCurve', prof['tone_curve'] is not None, 'ForwardMatrix', prof['forward_matrix1'] is not None)
    adobe_m10 = (load_cm(TARGET, 1), load_cm(TARGET, 2))

    result = {'target': 'Leica M10 (Leica embedded DNG profile)', 'cameras': {}}
    names = {}
    for model, ssf_name in CANON.items():
        names.setdefault(ssf_name, []).append(model)

    rep_models = {}
    for ssf_name, models in names.items():
        ssf = load_ssf(ssf_name)
        cm65 = load_cm(ssf_name, 2)
        pair = Pair(ssf, cm65, prof)
        S_D = fit_S(pair, DAYLIGHT, train)
        S_A = fit_S(pair, ['A'], train)
        # 補間の節点: アプリと同じ方法（D65 の行列だけ）で推定した色温度で決める
        lo = libraw_cct(cm65, illuminant('A') @ ssf)
        hi = libraw_cct(cm65, illuminant('BB4000') @ ssf)
        entry = dict(models=models, S_A=S_A.round(5).tolist(), S_D=S_D.round(5).tolist(), cct=[round(lo), round(hi)])
        result['cameras'][ssf_name] = entry
        rep_models[ssf_name] = (pair, S_A, S_D, lo, hi, ssf, cm65)
        print(f'{ssf_name:24s} cct anchors {lo:.0f}K / {hi:.0f}K')

    # 汎用（機種がわからない・Canon 以外）: 色を正しく測れるカメラ → Leica M10
    gpair = Pair(None, None, prof)
    result['generic'] = dict(S_A=fit_S(gpair, ['A'], train).round(5).tolist(), S_D=fit_S(gpair, DAYLIGHT, train).round(5).tolist(),
                             cct=result['cameras']['Canon-EOS-R6-Mark-II']['cct'])

    # --- 評価と分析（EOS R6 Mark II）
    pair, S_A, S_D, lo, hi, ssf, cm65 = rep_models['Canon-EOS-R6-Mark-II']
    report = {'illuminants': {}, 'hue_D65': [], 'skin_D65': {}}
    print('\n== EOS R6 Mark II → Leica M10: ΔE00 (mean / p95), before → after S(CCT)  [held-out skin]')
    for ill in ['D65', 'D50', 'D75', 'BB5000', 'BB4000', 'LED-B3', 'FL2', 'FL11', 'BB3200', 'A']:
        cct = libraw_cct(cm65, illuminant(ill) @ ssf)
        S = interp_S(S_A, S_D, lo, hi, cct)
        row = {}
        line = f'  {ill:7s} ({cct:5.0f}K)'
        for nm, R in test.items():
            x, y = pair(ill, R)
            d0, d1 = de00(x, y), de00(x @ S.T, y)
            row[nm] = dict(before=[round(d0.mean(), 2), round(np.percentile(d0, 95), 2)],
                           after=[round(d1.mean(), 2), round(np.percentile(d1, 95), 2)])
            line += f' | {nm} {d0.mean():4.2f}/{np.percentile(d0, 95):5.2f} → {d1.mean():4.2f}/{np.percentile(d1, 95):4.2f}'
        report['illuminants'][ill] = row
        print(line)

    # 色の違いの傾向（D65、色相ごと）: Leica M10 / Canon
    x, y = pair('D65', np.concatenate([refl['CC24'], refl['CES99']]))
    lc, ll = colour.Lab_to_LCHab(lab(x)), colour.Lab_to_LCHab(lab(y))
    print('\n== D65: Leica M10 (Leica matrix) vs EOS R6 Mark II (app), CC24 + CES99, by Canon hue (chroma > 15)')
    for h0 in range(0, 360, 30):
        s = (lc[:, 1] > 15) & (lc[:, 2] >= h0) & (lc[:, 2] < h0 + 30)
        if s.sum() == 0:
            continue
        dh = np.median(((ll[s, 2] - lc[s, 2] + 180) % 360) - 180)
        cr = np.median(ll[s, 1] / lc[s, 1])
        dL = np.median(ll[s, 0] - lc[s, 0])
        report['hue_D65'].append(dict(hue=[h0, h0 + 30], n=int(s.sum()), chroma=round(cr, 3), dhue=round(dh, 1), dL=round(dL, 2)))
        print(f'  hue {h0:3d}-{h0 + 30:3d} n={s.sum():2d}  chroma x{cr:.3f}  hue {dh:+5.1f}°  L* {dL:+5.2f}')
    x, y = pair('D65', refl['SKIN'])
    la, lb = lab(x), lab(y)
    report['skin_D65'] = dict(da=round(float(np.mean(lb[:, 1] - la[:, 1])), 2), db=round(float(np.mean(lb[:, 2] - la[:, 2])), 2),
                              dL=round(float(np.mean(lb[:, 0] - la[:, 0])), 2),
                              chroma=round(float(np.median(np.hypot(lb[:, 1], lb[:, 2]) / np.hypot(la[:, 1], la[:, 2]))), 3),
                              dhue=round(float(np.median(((np.degrees(np.arctan2(lb[:, 2], lb[:, 1])) - np.degrees(np.arctan2(la[:, 2], la[:, 1])) + 180) % 360) - 180)), 1))
    print('  skin:', report['skin_D65'])

    # 色の正確さ（人の目で見た色との差、D65）: 各カメラ + 行列の組み合わせ
    E = illuminant('D65')
    R = np.concatenate([refl['CC24'], refl['CES99']])
    t = truth_srgb(E, R)
    rawC, wC = simulate(ssf, E, R)
    rawL, wL = simulate(load_ssf(TARGET), E, R)
    acc = {
        'EOS R6 Mark II (LibRaw / Adobe matrix)': de00((rawC / wC) @ libraw_matrix(cm65).T, t).mean(),
        'Leica M10 (Leica embedded matrix)': de00((rawL / wL) @ dng_matrix(prof['color_matrix1'], prof['color_matrix2'], wL).T, t).mean(),
        'Leica M10 (Adobe matrix)': de00((rawL / wL) @ dng_matrix(*adobe_m10, wL).T, t).mean(),
    }
    report['accuracy_D65'] = {k: round(float(v), 2) for k, v in acc.items()}
    print('\n== colour accuracy vs. the eye (D65, CC24 + CES99, mean ΔE00):')
    for k, v in acc.items():
        print(f'  {k:42s} {v:.2f}')
    result['report'] = report
    result['leica_profile'] = dict(color_matrix1=prof['color_matrix1'].tolist(), color_matrix2=prof['color_matrix2'].tolist(),
                                   baseline_exposure=prof['baseline_exposure'], look_table=prof['look_table_data'])
    with open(os.path.join(OUT, 'sensor.json'), 'w') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print('\nwrote out/sensor.json')


if __name__ == '__main__':
    main()
