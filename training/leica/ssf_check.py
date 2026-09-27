"""推定分光感度（Solomatov & Akkaynak）の確からしさの確認。

分光感度が実測されているカメラで、実測値と推定値から同じ色行列で色を作り、どれだけ違うかを見る。
Canon → Leica M10 の変換（sensor.py）は推定値どうしの差なので、この程度の誤差を含む。

使い方: python ssf_check.py
"""
import csv
import os

import numpy as np

from common import DATA, WL, de00, illuminant, libraw_matrix, load_cm, load_ssf, reflectances, simulate, truth_srgb

GT = os.path.join(DATA, 'ssf-estimation', 'data', 'ground-truths')
# （実測値のファイル, 推定値の名前）。Nikon D80 は推定に使うモデルの学習に使われていない
CAMERAS = [('test-data/Nikon_D80_Jiang-et-al-2013.csv', 'Nikon-D80'),
           ('Canon_5D-Mk-II_Jiang-et-al-2013.csv', 'Canon-EOS-5D-Mark-II'),
           ('Canon_600D_Jiang-et-al-2013.csv', 'Canon-EOS-600D'),
           ('Canon_60D_Jiang-et-al-2013.csv', 'Canon-EOS-60D'),
           ('Nikon_D700_Jiang-et-al-2013.csv', 'Nikon-D700'),
           ('Sony_NEX5N_Jiang-et-al-2013.csv', 'Sony-NEX-5N'),
           ('Leica_M8_Mauer-2009.csv', 'Leica-M8')]


def read_measured(path):
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    a = np.array([[float(x) for x in r] for r in rows])
    return np.array([np.interp(WL, a[:, 0], a[:, k]) for k in (1, 2, 3)]).T


def main():
    refl = reflectances()
    R = np.concatenate([refl['CC24'], refl['CES99']])
    E = illuminant('D65')
    truth = truth_srgb(E, R)
    print('camera                 | ΔE00 measured vs predicted SSF (mean / p95) | accuracy vs eye: measured / predicted')
    for f, name in CAMERAS:
        M = libraw_matrix(load_cm(name, 2))
        rows = []
        for ssf in (read_measured(os.path.join(GT, f)), load_ssf(name)):
            raw, w = simulate(ssf, E, R)
            rows.append((raw / w) @ M.T)
        d = de00(rows[0], rows[1])
        print(f'{name:22s} | {d.mean():5.2f} / {np.percentile(d, 95):5.2f} | {de00(rows[0], truth).mean():5.2f} / {de00(rows[1], truth).mean():5.2f}')


if __name__ == '__main__':
    main()
