"""HDR+ データセットの merged.dng をリニア sRGB（半分の解像度）に現像する。

HDR+ の README にあるメタデータの意味どおりに処理する:
黒レベル / 白レベル → レンズシェーディング（周辺減光と色むら）補正マップをベイヤーの各色に掛ける
→ AsShotNeutral のホワイトバランス → rgb2rgb（センサー RGB → リニア sRGB）。
デモザイクは 2×2 の平均（半分の解像度）で行う。学習ではさらに縮小するので十分。
"""
import numpy as np, rawpy, tifffile
from PIL import Image

def _interp_matrix(n_out, n_in):
    """四隅を合わせた線形補間の重み (n_out × n_in)"""
    pos = np.linspace(0, n_in - 1, n_out)
    i0 = np.clip(np.floor(pos).astype(int), 0, n_in - 2)
    t = pos - i0
    A = np.zeros((n_out, n_in), np.float32)
    A[np.arange(n_out), i0] = 1 - t
    A[np.arange(n_out), i0 + 1] += t
    return A

def bilinear_grid(lsm, H, W):
    """(h, w, 4) のゲインの格子を (H, W, 4) に拡大する。格子の四隅は画像の四隅に対応する"""
    gh, gw, C = lsm.shape
    Ay = _interp_matrix(H, gh)
    Ax = _interp_matrix(W, gw)
    return np.stack([Ay @ lsm[:, :, c] @ Ax.T for c in range(C)], -1)

def decode(dng_path, lsm_path, rgb2rgb):
    r = rawpy.imread(dng_path)
    raw = r.raw_image_visible.astype(np.float32)
    pat = r.raw_pattern  # values index color_desc
    desc = r.color_desc.decode()
    black = np.array(r.black_level_per_channel, np.float32)
    white = float(r.white_level)
    wb = np.array(r.camera_whitebalance[:3], np.float32)  # = 1/AsShotNeutral（G=1 で正規化）
    H, W = raw.shape
    H2, W2 = H // 2, W // 2
    raw = raw[: H2 * 2, : W2 * 2]
    lsm = tifffile.imread(lsm_path).astype(np.float32)  # [R, Gred, Gblue, B]
    if lsm.ndim == 3 and lsm.shape[2] == 4:
        gains = bilinear_grid(lsm, H2, W2)
    else:
        gains = None
    planes = {}
    # 赤のある行の緑が Gred
    red_row = [i for i in range(2) for j in range(2) if desc[pat[i, j]] == 'R'][0]
    for i in range(2):
        for j in range(2):
            ci = pat[i, j]
            col = desc[ci]
            p = (raw[i::2, j::2] - black[ci]) / (white - black[ci])
            if col == 'R': key, li = 'R', 0
            elif col == 'B': key, li = 'B', 3
            else:
                key = 'G' + str(len([k for k in planes if k.startswith('G')]))
                li = 1 if i == red_row else 2
            if gains is not None:
                p = p * gains[:, :, li]
            planes[key] = p
    rgb = np.stack([planes['R'], (planes['G0'] + planes['G1']) * 0.5, planes['B']], -1)
    rgb *= wb[None, None, :] / wb[1]
    M = np.array(rgb2rgb, np.float32).reshape(3, 3)
    rgb = rgb @ M.T
    meta = dict(flip=r.sizes.flip, black=black.tolist(), white=white, wb=wb.tolist())
    return rgb, meta
