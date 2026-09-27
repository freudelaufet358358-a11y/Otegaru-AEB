"""Leica M10 の色の調査・当てはめで共通に使う道具。

- 分光のシミュレーション（カメラの分光感度 × 照明 × 反射率 → RAW の値）
- 色行列による RAW → リニア sRGB の変換を 2 通り
  - LibRaw 方式（アプリの RAW 現像と同じ。D65 の行列 1 つだけ・色順応なし）
  - DNG 方式（Lightroom / DNG SDK と同じ。2 光源の行列を色温度で補間し、Bradford で色順応）
- DNG のタグの読み取り（Leica がファイルに埋め込んだカメラプロファイルを読むため）
- カメラ内 JPEG の階調を表す単調なトーンカーブ
"""
import csv
import os
import re
import shlex
import struct

import colour
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
SSF_DIR = os.path.join(DATA, 'ssf-estimation', 'data')

WL = np.arange(400, 701, 10)
SHAPE = colour.SpectralShape(400, 700, 10)
CMFS = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(SHAPE).values

SRGB = colour.RGB_COLOURSPACES['sRGB']
XYZ_TO_SRGB = SRGB.matrix_XYZ_to_RGB
SRGB_TO_XYZ = SRGB.matrix_RGB_to_XYZ
_CCS = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
D50 = colour.xy_to_XYZ(_CCS['D50'])
D65 = colour.xy_to_XYZ(_CCS['D65'])
LUMA = np.array([0.2126, 0.7152, 0.0722])

# DNG の CalibrationIlluminant（17 = 標準光源 A、21 = D65）の色温度
CCT_A = 2856.0
CCT_D65 = 6504.0


# ---------------------------------------------------------------------------
# 分光データ

def load_ssf(name):
    """推定分光感度（400〜700nm、10nm 刻み、列は R, G, B）"""
    with open(os.path.join(SSF_DIR, 'predictions', f'predicted-{name}.csv')) as f:
        rows = list(csv.reader(f))[1:]
    a = np.array([[float(x) for x in r] for r in rows])
    assert np.allclose(a[:, 0], WL)
    return a[:, 1:]


def load_cm(name, i):
    """Adobe の色行列（XYZ → カメラ RGB）。i = 1: 標準光源 A、2: D65"""
    with open(os.path.join(SSF_DIR, 'color-matrices', f'{name}-cm{i}.csv')) as f:
        return np.array([[float(x) for x in line.split(',')] for line in f.read().split()])


def illuminant(name):
    if name.startswith('BB'):
        return colour.sd_blackbody(float(name[2:]), SHAPE).values
    return colour.SDS_ILLUMINANTS[name].copy().align(SHAPE).values


def reflectances():
    """反射率のデータセット: ColorChecker 24 色、CIE 2017 の 99 色（自然物・人工物）、肌 15,256 件"""
    out = {}
    cc = colour.SDS_COLOURCHECKERS['BabelColor Average']
    out['CC24'] = np.array([sd.copy().align(SHAPE).values for sd in cc.values()])
    from colour.quality.cfi2017 import load_TCS_CIE2017
    tcs = load_TCS_CIE2017(colour.SpectralShape(380, 780, 5))
    w = list(np.array(tcs.wavelengths))
    out['CES99'] = np.array(tcs.values)[[w.index(x) for x in WL]].T
    with open(os.path.join(DATA, 'ssf-data', 'Reference_Spectra', 'ISSA_17_Jan_2025_Yan_Lu.ti3')) as f:
        txt = f.read()
    body = re.search(r'\nBEGIN_DATA\n(.*?)\nEND_DATA', txt, re.S).group(1).splitlines()
    spec = np.array([[float(t) for t in shlex.split(line)[-43:]] for line in body])  # 360〜780nm
    skin = spec[:, [list(range(360, 781, 10)).index(x) for x in WL]] / 100
    out['SKIN'] = skin[(skin > 0).all(1)]
    return out


def simulate(ssf, E, R):
    """反射率 R (n×31) を照明 E で撮ったときの RAW と、完全拡散白色の RAW"""
    return (R * E) @ ssf, E @ ssf


def truth_srgb(E, R):
    """人の目で見た色（照明の白を D65 に Bradford で順応させたリニア sRGB、白 = 1）"""
    XYZ = (R * E) @ CMFS
    W = E @ CMFS
    return (XYZ / W[1]) @ bradford(W / W[1], D65).T @ XYZ_TO_SRGB.T


# ---------------------------------------------------------------------------
# 色行列による変換

def bradford(src, dst):
    return colour.adaptation.matrix_chromatic_adaptation_VonKries(src, dst, 'Bradford')


def cct_of_xy(xy):
    return float(colour.temperature.xy_to_CCT_McCamy1992(np.asarray(xy)))


def interp_cm(cm1, cm2, cct):
    """DNG の規則: 2 つの較正光源の行列を色温度の逆数で線形補間する"""
    if cct <= CCT_A:
        return cm1
    if cct >= CCT_D65:
        return cm2
    g = (1 / cct - 1 / CCT_D65) / (1 / CCT_A - 1 / CCT_D65)
    return g * cm1 + (1 - g) * cm2


def dng_white_xy(cm1, cm2, neutral):
    """カメラのニュートラル（AsShotNeutral）から白の色度を求める（DNG SDK と同じ反復）"""
    xy = np.array([0.3457, 0.3585])
    for _ in range(50):
        XYZ = np.linalg.solve(interp_cm(cm1, cm2, cct_of_xy(xy)), neutral)
        xy = XYZ[:2] / XYZ.sum()
    return xy


def dng_matrix(cm1, cm2, neutral):
    """DNG 方式（ForwardMatrix なし）: ホワイトバランス済みのカメラ RGB（白 = 1）→ リニア sRGB（白 = 1）"""
    neutral = np.asarray(neutral, float) / neutral[1]
    xy = dng_white_xy(cm1, cm2, neutral)
    cam_to_xyz = np.linalg.inv(interp_cm(cm1, cm2, cct_of_xy(xy)))
    white = cam_to_xyz @ neutral
    white = white / white[1]
    M = XYZ_TO_SRGB @ bradford(D50, D65) @ bradford(white, D50) @ cam_to_xyz @ np.diag(neutral)
    return M / (M @ np.ones(3))[:, None]


def libraw_matrix(cm_d65):
    """LibRaw（dcraw）方式: ホワイトバランス済みのカメラ RGB → リニア sRGB。アプリの RAW 現像と同じ計算"""
    cam_rgb = cm_d65 @ SRGB_TO_XYZ
    cam_rgb = cam_rgb / cam_rgb.sum(1, keepdims=True)
    return np.linalg.inv(cam_rgb)


def libraw_cct(cm_d65, neutral):
    """アプリと同じ方法で撮影時の色温度を推定する: ニュートラルを D65 の行列だけで XYZ に戻す"""
    XYZ = np.linalg.solve(cm_d65, np.asarray(neutral, float) / neutral[1])
    return cct_of_xy(XYZ[:2] / XYZ.sum())


def lab(x):
    """リニア sRGB → CIELAB (D65)"""
    return colour.XYZ_to_Lab(np.maximum(x, 1e-7) @ SRGB_TO_XYZ.T, colour.XYZ_to_xy(D65))


def de00(a, b):
    return colour.delta_E(lab(a), lab(b), method='CIE 2000')


def neutral_matrix(p):
    """白を白に保つ 3×3 行列（各行の和 = 1）。p は非対角成分 6 個"""
    M = np.zeros((3, 3))
    k = 0
    for i in range(3):
        for j in range(3):
            if i != j:
                M[i, j] = p[k]
                k += 1
        M[i, i] = 1 - M[i].sum()
    return M


# ---------------------------------------------------------------------------
# sRGB

def oetf(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def eotf(e):
    e = np.asarray(e, float)
    return np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)


# ---------------------------------------------------------------------------
# トーンカーブ: log2(リニア) → sRGB 符号値 の単調な曲線。KNOTS[0] で 0、KNOTS[-1] で 1。
# パラメータは節点の間の増分（len(KNOTS) - 1 個）

KNOTS = np.linspace(-9, 1, 21)


def curve_points(c):
    inc = np.log1p(np.exp(c))  # softplus で増分を正にする → 単調増加
    y = np.concatenate([[0], np.cumsum(inc)])
    return y / y[-1]


def curve(u, c):
    return PchipInterpolator(KNOTS, curve_points(c))(np.clip(u, KNOTS[0], KNOTS[-1]))


# ---------------------------------------------------------------------------
# DNG（TIFF）のタグを読む

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


def read_dng_tags(path, limit=4 << 20):
    """IFD0 のタグを {タグ番号: 値} で返す（ファイルの先頭 limit バイトに収まる値だけ）"""
    with open(path, 'rb') as f:
        d = f.read(limit)
    bo = '<' if d[:2] == b'II' else '>'
    rd = lambda fmt, off: struct.unpack(bo + fmt, d[off:off + struct.calcsize(fmt)])
    off, = rd('I', 4)
    n, = rd('H', off)
    tags = {}
    for i in range(n):
        e = off + 2 + 12 * i
        tag, typ, cnt = rd('HHI', e)
        size = _TYPE_SIZE.get(typ, 1) * cnt
        vo = e + 8 if size <= 4 else rd('I', e + 8)[0]
        if vo + size > len(d):
            continue
        if typ == 2:
            v = d[vo:vo + cnt].rstrip(b'\0').decode('latin1').strip()
        elif typ in (10, 5):
            r = rd(('i' if typ == 10 else 'I') * (2 * cnt), vo)
            v = [r[2 * k] / r[2 * k + 1] if r[2 * k + 1] else 0.0 for k in range(cnt)]
        elif typ == 11:
            v = list(rd('f' * cnt, vo))
        elif typ == 3:
            v = list(rd('H' * cnt, vo))
        elif typ == 4:
            v = list(rd('I' * cnt, vo))
        else:
            v = d[vo:vo + size]
        tags[tag] = v
    return tags


DNG_TAGS = dict(model=0x0110, color_matrix1=0xC621, color_matrix2=0xC622, camera_calibration1=0xC623,
                camera_calibration2=0xC624, as_shot_neutral=0xC628, baseline_exposure=0xC62A,
                illuminant1=0xC65A, illuminant2=0xC65B, profile_name=0xC6F8, forward_matrix1=0xC714,
                hue_sat_map_dims=0xC6F9, tone_curve=0xC6FC, look_table_dims=0xC725, look_table_data=0xC726)


def dng_profile(path):
    """DNG に埋め込まれたカメラプロファイル"""
    t = read_dng_tags(path)
    p = {k: t.get(v) for k, v in DNG_TAGS.items()}
    for k in ('color_matrix1', 'color_matrix2', 'camera_calibration1', 'camera_calibration2', 'forward_matrix1'):
        if p[k] is not None:
            p[k] = np.array(p[k]).reshape(3, 3)
    return p


def acr_default_curve():
    """ACR の既定トーンカーブ（1025 点、リニア → リニア）。RawTherapee の dcp.cc から読む"""
    with open(os.path.join(DATA, 'rawtherapee', 'rtengine', 'dcp.cc')) as f:
        src = f.read()
    i = src.index('adobe_camera_raw_default_curve[] = {')
    return np.array([float(v) for v in re.findall(r'[0-9]+\.[0-9]+', src[i:src.index('};', i)])])
