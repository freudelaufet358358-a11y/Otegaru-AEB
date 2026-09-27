"""カメラ内 JPEG の「仕上げ方」（JPEG エンジン）を、RAW と同じコマに埋め込まれたカメラ内 JPEG の組から当てはめる。

  JPEG ≈ T( e · A · x )   （T は R, G, B それぞれに同じトーンカーブ）

- x: RAW をメーカーの色行列でリニア sRGB にしたもの（Leica は DNG に埋め込まれた Leica の行列、Canon は Adobe の行列 = アプリと同じ）
- A: 白を保つ 3×3 行列（JPEG エンジンの色づくり）、e: 露出の倍率、T: 単調なトーンカーブ
- 画像は位置を合わせてから 16×16 画素のブロックで平均し、模様の少ないブロックだけを使う

データ（raw.pixls.us 由来、CC0）:
- Leica M10-R（DNG の中のフルサイズのカメラ内 JPEG、JPEG 設定は標準）
- Canon EOS R6（CR3 の中のカメラ内 JPEG、ピクチャースタイル「スタンダード」）

使い方: python engine.py   → out/engine.json と out/*.jpg（検証用の比較画像）
"""
import json
import io
import os

import colour
import cv2
import numpy as np
import rawpy
from PIL import Image
from scipy.optimize import least_squares

from common import (DATA, KNOTS, LUMA, OUT, acr_default_curve, curve, dng_matrix, dng_profile, eotf, lab, libraw_matrix,
                    load_cm, neutral_matrix, oetf)

np.set_printoptions(precision=4, suppress=True)
BLOCK = 8  # 半分の解像度での 8×8 = 元の 16×16 画素


def decode(path):
    """RAW を半分の解像度でデモザイク（リニア、黒レベルを引いて飽和 = 1、ホワイトバランス・色変換なし）と、カメラ内 JPEG"""
    with rawpy.imread(path) as r:
        cam = r.postprocess(output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True, user_wb=[1, 1, 1, 1],
                            output_bps=16, half_size=True, user_flip=0, highlight_mode=rawpy.HighlightMode.Clip)
        wb = np.array(r.camera_whitebalance[:3], float)
        thumb = r.extract_thumb()
    jpg = np.asarray(Image.open(io.BytesIO(thumb.data)).convert('RGB')).astype(np.float32) / 255
    return cam.astype(np.float32) / 65535, wb / wb[1], jpg


def to_scene(path, cam, wb):
    """カメラ RGB → リニア sRGB（メーカーの色行列）"""
    if path.endswith('.dng'):
        p = dng_profile(path)
        n = np.array(p['as_shot_neutral'])
        return (cam / (n / n[1])) @ dng_matrix(p['color_matrix1'], p['color_matrix2'], n).T, p
    return (cam * wb) @ libraw_matrix(load_cm('Canon-EOS-R6', 2)).T, None


def pairs(path):
    cam, wb, jpg = decode(path)
    x, prof = to_scene(path, cam, wb)
    jh = cv2.resize(jpg, (jpg.shape[1] // 2, jpg.shape[0] // 2), interpolation=cv2.INTER_AREA)
    # 位置合わせ（ECC、アフィン）。明るさの違いは ECC が吸収する
    g1 = (np.clip((cam / wb).mean(2), 0, 1) ** (1 / 2.2)).astype(np.float32)
    g2 = jh.mean(2).astype(np.float32)
    small = lambda g: cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    cc, warp = cv2.findTransformECC(small(g2), small(g1), warp, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7), None, 5)
    warp[:, 2] *= 2
    xw = cv2.warpAffine(x.astype(np.float32), warp, (jh.shape[1], jh.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    camw = cv2.warpAffine(cam, warp, (jh.shape[1], jh.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    H, W = jh.shape[0] // BLOCK, jh.shape[1] // BLOCK
    blk = lambda a: a[:H * BLOCK, :W * BLOCK].reshape(H, BLOCK, W, BLOCK, -1)
    Y = np.maximum(xw @ LUMA, 1e-5)
    sd = np.log2(blk(Y[..., None])).std((1, 3))[..., 0]
    ok = (blk(camw).min((1, 3, 4)) >= 0) & (blk(camw).max((1, 3, 4)) < 0.98)
    xs, js = blk(xw).mean((1, 3)), blk(jh).mean((1, 3))
    m = ok & (sd < 0.2) & (xs.min(-1) > 1e-5) & (js.max(-1) < 0.985) & (js.min(-1) > 0.01)
    cols = np.broadcast_to(np.arange(W)[None, :], (H, W))
    print(f'{os.path.basename(path)}: ECC {cc:.4f}, blocks {H * W}, used {m.sum()}')
    return dict(x=xs[m], j=js[m], col=cols[m] / W, x_full=x, jpg=jpg, prof=prof, warp=warp)


def weights(x):
    """色相 × 明るさの出現頻度の平方根の逆数（緑ばかりの風景などで偏らないように）"""
    L = lab(x / np.median(x @ LUMA) * 0.18)
    h = (np.degrees(np.arctan2(L[:, 2], L[:, 1])) % 360) // 30
    C = np.hypot(L[:, 1], L[:, 2])
    key = np.where(C < 6, 12, h) * 20 + np.clip(L[:, 0] // 5, 0, 19)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = 1 / np.sqrt(cnt[inv])
    return w / w.mean()


def render(p, x):
    A = neutral_matrix(p[:6])
    return curve(np.log2(np.maximum(x @ A.T * np.exp(p[6]), 1e-7)), p[7:])


def fit(d, ridge=1.0):
    x, j = d['x'], d['j']
    w = np.sqrt(weights(x))[:, None]
    Lj = lab(eotf(j))
    test = d['col'] >= 0.8  # 右端 20% を検証用に取っておく（隣り合うブロックが似ているため、ランダムではなく場所で分ける）
    tr = ~test
    p0 = np.concatenate([np.zeros(6), [0.0], np.zeros(len(KNOTS) - 1)])

    def res(p):
        r = ((lab(eotf(render(p, x[tr]))) - Lj[tr]) * w[tr]).ravel()
        return np.concatenate([r, np.sqrt(ridge * tr.sum()) * p[:6]])

    p = least_squares(res, p0, max_nfev=500, x_scale='jac').x
    e = colour.delta_E(lab(eotf(render(p, x))), Lj, method='CIE 2000')
    stats = dict(train=[round(float(e[tr].mean()), 2), round(float(np.percentile(e[tr], 95)), 2)],
                 test=[round(float(e[test].mean()), 2), round(float(np.percentile(e[test], 95)), 2)], n=int(len(x)))
    return p, stats


def tone_stats(c, exposure):
    """トーンカーブの形: 中間グレー (sRGB 118) からの段数ごとの出力、中間調の傾き、白飛びまでの余裕"""
    u = np.linspace(-12, 2, 14001)
    y = curve(u, c)
    i = np.argmin(np.abs(y - 118.9 / 255))
    Ly = 116 * np.cbrt(np.maximum(eotf(y), 0.008856)) - 16
    slope = np.gradient(Ly, u)[i]
    clip = u[np.argmax(y >= 254.5 / 255)] - u[i]
    table = {s: round(float(np.interp(u[i] + s, u, y) * 255), 1) for s in range(-6, 4)}
    return dict(mid_log2=round(float(u[i] - np.log(exposure) / np.log(2)), 3), slope=round(float(slope), 2),
                headroom=round(float(clip), 2), sRGB_by_stop=table)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for key, path in [('leica_m10r', 'raw-leica-m10-r.dng'), ('canon_r6', 'raw-canon-eos-r6.cr3')]:
        d = pairs(os.path.join(DATA, path))
        p, stats = fit(d)
        A = neutral_matrix(p[:6])
        ts = tone_stats(p[7:], np.exp(p[6]))
        print(f'  fit ΔE00 train {stats["train"]}, held-out {stats["test"]}\n  A =\n{A}\n  exposure {np.exp(p[6]):.4f}  tone {ts}')
        out[key] = dict(matrix=A.round(6).tolist(), exposure=float(np.exp(p[6])), curve=p[7:].tolist(), fit=stats, tone=ts)
        # 検証画像: 左 = 色行列だけの素の現像、中 = カメラ内 JPEG、右 = 当てはめたエンジン
        x = d['x_full']
        model = render(p, x)
        mid = 2 ** ts['mid_log2']
        plain = oetf(x * (0.18 / mid))
        jpg = cv2.resize(d['jpg'], (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = min(jpg.shape[0], model.shape[0]), min(jpg.shape[1], model.shape[1])
        sheet = np.concatenate([plain[:h, :w], jpg[:h, :w], model[:h, :w]], 1)
        im = Image.fromarray((np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8))
        im.thumbnail((2400, 2400))
        im.save(os.path.join(OUT, f'engine_{key}.jpg'), quality=88)
    # 参考: Lightroom で Leica の埋め込みプロファイルを使ったときの既定のトーンカーブ（ACR の既定カーブ）
    acr = acr_default_curve()
    u = np.linspace(-12, 2, 14001)
    ya = oetf(np.interp(np.clip(2.0 ** u, 0, 1), np.linspace(0, 1, len(acr)), acr))
    i = np.argmin(np.abs(ya - 118.9 / 255))
    Ly = 116 * np.cbrt(np.maximum(eotf(ya), 0.008856)) - 16
    out['acr_default'] = dict(slope=round(float(np.gradient(Ly, u)[i]), 2), headroom=round(float(u[np.argmax(ya >= 254.5 / 255)] - u[i]), 2),
                              sRGB_by_stop={s: round(float(np.interp(u[i] + s, u, ya) * 255), 1) for s in range(-6, 4)})
    print('ACR default curve:', out['acr_default'])
    with open(os.path.join(OUT, 'engine.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote out/engine.json')


if __name__ == '__main__':
    main()
