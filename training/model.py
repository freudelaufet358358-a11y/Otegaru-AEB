"""学習済みトーンレンダラー（「おまかせ」モード）のモデル定義。

シーンのリニアな RGB（露出は任意）→ 表示用の sRGB。HDRNet (Gharbi et al., SIGGRAPH 2017) を
輝度だけに絞って小さくしたもの。

  1. 256×256 に縮小した画像を、対数輝度の中央値（key）で露出を正規化して CNN に入れる
  2. CNN は「双方向グリッド」[係数 3][明るさの段 8][16][16] を出力する
     係数: 対数輝度の傾き a・オフセット b・彩度 s（a と s は節点で exp を取って正にする）
  3. フル解像度の各画素は (x, y, guide(対数輝度)) でグリッドを 3 次元線形補間し、
     lout = a·ln + b（ln は正規化した対数輝度）で明るさを決める。色度（色相）は入力のまま

ブラウザ側の実装 src/core/learned.ts は、この数式と 1 対 1 に対応している。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)
LOG_EPS = 2.0 ** -16
LOW = 256          # CNN に入れる縮小画像の一辺
GRID_XY = 16       # グリッドの空間方向の分割数
GRID_Z = 8         # 明るさの段数
NCOEF = 3          # 傾き・オフセット・彩度
GUIDE_KNOTS = 16
GUIDE_LO, GUIDE_HI = -10.0, 6.0   # ガイドが扱う正規化対数輝度の範囲 [段]
B_INIT = -2.3      # 初期状態で中央値の画素を置く出力の対数輝度（リニアで約 0.2）
IN_SCALE = 0.25    # CNN の入力: (log2 値 - key) * IN_SCALE
KNEE = 0.75        # ハイライトの肩特性（src/core/color.ts と同じ）


def luma(x):  # (..., 3, H, W) -> (..., H, W)
    return LUMA[0] * x[..., 0, :, :] + LUMA[1] * x[..., 1, :, :] + LUMA[2] * x[..., 2, :, :]


def key_of(low):
    """露出の基準: 縮小画像の対数輝度 (log2) の中央値（下側）。low: (B, 3, h, w) リニア"""
    l = torch.log2(luma(low).clamp_min(LOG_EPS)).flatten(1)
    return l.median(dim=1).values  # (B,)


def shoulder(x):
    r = 1 - KNEE
    return torch.where(x <= KNEE, x, KNEE + r * (1 - torch.exp(-(x - KNEE) / r)))


def srgb_oetf(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)


class ToneNet(nn.Module):
    def __init__(self, c1=16, c2=32, c3=64, c4=64, cg=32, fc1=128, fc2=64):
        super().__init__()
        self.splat = nn.ModuleList([
            nn.Conv2d(3, c1, 3, 2, 1), nn.Conv2d(c1, c2, 3, 2, 1),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.Conv2d(c3, c4, 3, 2, 1)])
        self.local1 = nn.Conv2d(c4, c4, 3, 1, 1)
        self.local2 = nn.Conv2d(c4, c4, 3, 1, 1, bias=False)
        self.glob1 = nn.Conv2d(c4, c4, 3, 2, 1)
        self.glob2 = nn.Conv2d(c4, cg, 3, 2, 1)
        self.fc1 = nn.Linear(cg * 4 * 4, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, c4)
        self.out = nn.Conv2d(c4, GRID_Z * NCOEF, 1)
        # ガイド曲線の増分（softplus）→ [0, 1] の単調な折れ線
        self.guide_theta = nn.Parameter(torch.full((GUIDE_KNOTS - 1,), math.log(math.e - 1)))
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            b = torch.zeros(GRID_Z, NCOEF)
            b[:, 1] = B_INIT
            self.out.bias.copy_(b.flatten())

    def grid(self, lowin):
        """lowin: (B, 3, 256, 256) 正規化した対数入力 → グリッド (B, NCOEF, GRID_Z, 16, 16)（exp 前）"""
        x = lowin
        for conv in self.splat:
            x = F.relu(conv(x))
        loc = self.local2(F.relu(self.local1(x)))
        g = F.relu(self.glob1(x))
        g = F.relu(self.glob2(g))
        g = F.relu(self.fc1(g.flatten(1)))
        g = F.relu(self.fc2(g))
        g = self.fc3(g)
        f = F.relu(loc + g[:, :, None, None])
        o = self.out(f)  # (B, GRID_Z*NCOEF, 16, 16)、チャンネルは z*NCOEF + c
        B = o.shape[0]
        return o.view(B, GRID_Z, NCOEF, GRID_XY, GRID_XY).permute(0, 2, 1, 3, 4).contiguous()

    def guide_values(self):
        d = F.softplus(self.guide_theta)
        v = torch.cat([d.new_zeros(1), torch.cumsum(d, 0)])
        return v / v[-1]

    def guide(self, ln):
        """単調な折れ線: 正規化した対数輝度 → [0, 1]"""
        v = self.guide_values()
        t = ((ln - GUIDE_LO) / (GUIDE_HI - GUIDE_LO) * (GUIDE_KNOTS - 1)).clamp(0, GUIDE_KNOTS - 1)
        i = t.floor().clamp(max=GUIDE_KNOTS - 2)
        f = t - i
        i = i.long()
        return v[i] * (1 - f) + v[i + 1] * f


def lowres_input(low, k):
    """low: (B, 3, 256, 256) リニアの縮小画像、k: (B,) key → 正規化した対数入力"""
    return ((torch.log2(low.clamp_min(LOG_EPS)) - k[:, None, None, None]) * IN_SCALE).clamp(-4, 4)


def coef_grid(grid):
    """CNN の出力 → 補間に使う係数。傾きと彩度は節点で exp を取って正にしてから画素ごとに線形補間する"""
    return torch.stack([torch.exp(grid[:, 0]), grid[:, 1], torch.exp(grid[:, 2])], 1)


def render(net, full, low, k=None, grid=None, return_coef=False):
    """full: (B, 3, H, W) リニア、low: 同じ画像の 256×256 縮小（リニア）→ 表示用 sRGB [0, 1]"""
    if k is None:
        k = key_of(low)
    if grid is None:
        grid = net.grid(lowres_input(low, k))
    B, _, H, W = full.shape
    ln = torch.log2(luma(full).clamp_min(LOG_EPS)) - k[:, None, None]  # (B, H, W)
    g = net.guide(ln)
    # 3 次元 grid_sample（align_corners=False, border）= HDRNet のセル中心の規約
    ys = (torch.arange(H, dtype=full.dtype) + 0.5) / H
    xs = (torch.arange(W, dtype=full.dtype) + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([gx.expand(B, H, W) * 2 - 1, gy.expand(B, H, W) * 2 - 1, g * 2 - 1], -1)  # x, y, z
    coef = F.grid_sample(coef_grid(grid), coords[:, None], mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0]
    a = coef[:, 0]
    b = coef[:, 1]
    s = coef[:, 2]
    lout = a * ln + b
    gain = torch.exp2(lout - ln - k[:, None, None])  # 正規化前のリニア入力に掛けるゲイン
    lin = full * gain[:, None]
    disp = srgb_oetf(shoulder(lin))
    y = luma(disp)[:, None]
    out = (y + s[:, None] * (disp - y)).clamp(0, 1)
    if return_coef:
        return out, dict(a=a, b=b, s=s, g=g, ln=ln)
    return out


def curve_penalty(net, grid, min_slope=0.1, max_slope=3.0, samples=65):
    """グリッドの係数を「写真として破綻しない」範囲に保つ正則化。戻り値は (トーンカーブ, 彩度) の罰則。

    - 各セルのトーンカーブ lout(ln) は単調増加で、傾きは [min_slope, max_slope] に収める
      （階調の逆転＝雲より空が暗くなる、のような不自然さを防ぐ）
    - 曲線はなめらかに（2 階差分）。学習画像にほとんど画素が来ない明るさの段も、隣の段から
      なめらかにつながる値になり、フル解像度で細部やノイズがそこへ入っても破綻しない
    - 彩度は 0.5〜2 倍に収め、明るさの段の間でなめらかに変える
    空間方向の補間はガイドを共有する曲線どうしの凸結合なので、節点で調べれば十分。"""
    coef = coef_grid(grid)  # (B, 3, D, H, W)
    ln = torch.linspace(GUIDE_LO - 2, GUIDE_HI + 2, samples, dtype=grid.dtype)
    g = net.guide(ln)
    fz = (g * GRID_Z - 0.5).clamp(0, GRID_Z - 1)
    z0 = fz.floor().clamp(max=GRID_Z - 2).long()
    tz = (fz - z0)[None, :, None, None]
    a = coef[:, 0]
    b = coef[:, 1]
    a_s = a[:, z0] * (1 - tz) + a[:, z0 + 1] * tz  # (B, S, H, W)
    b_s = b[:, z0] * (1 - tz) + b[:, z0 + 1] * tz
    lout = a_s * ln[None, :, None, None] + b_s
    step = float(ln[1] - ln[0])
    d = (lout[:, 1:] - lout[:, :-1]) / step
    tone = (F.relu(min_slope - d) + F.relu(d - max_slope)).mean() + 0.005 * ((d[:, 1:] - d[:, :-1]) / step).abs().mean()
    ls = grid[:, 2]  # log 彩度
    sat = F.relu(ls.abs() - math.log(2)).mean() + 0.01 * (ls[:, 1:] - ls[:, :-1]).abs().mean()
    return tone, sat
