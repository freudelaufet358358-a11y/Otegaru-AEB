// 学習済みトーンレンダラー（「おまかせ」モード）。
// Google の HDR+ データセット（スマートフォンの RAW を合成した HDR データと、仕上がった写真の組）で学習した
// HDRNet 風のモデル（Gharbi et al. 2017）で、リニアな放射輝度から写真らしい仕上がりを作る。
//
//  1. 放射輝度を 256×256 に縮小し、露出を中央値で正規化して小さな CNN に入れる
//  2. CNN は「双方向グリッド」（位置 16×16 × 明るさ 8 段 × 係数 3）を出力する
//  3. フル解像度の各画素は (x, y, 明るさ) でグリッドを補間し、対数輝度の傾き・オフセットと彩度を得る
//     → 明るさの境目をまたがないので、局所的に階調を圧縮してもハロ（縁の光輪）が出にくい
//
// 色相（色度）は入力のまま保ち、輝度と彩度だけを変える。ホワイトバランスは変えない。
// 学習コードは training/ にある。数式は training/model.py と 1 対 1 に対応させている。

import { linearToDisplay, LUMA_B, LUMA_G, LUMA_R } from './color';
import { buildAreaWeights, type View } from './frame';
import type { ProgressFn } from './fusion';
import { mergeRow } from './hdr';

interface Conv {
  cin: number;
  cout: number;
  k: number;
  stride: number;
  weight: Float32Array;
  bias: Float32Array | null;
}

interface Dense {
  cin: number;
  cout: number;
  weight: Float32Array;
  bias: Float32Array;
}

export interface ToneModel {
  /** CNN に入れる縮小画像の一辺 */
  low: number;
  gridXY: number;
  gridZ: number;
  ncoef: number;
  guideLo: number;
  guideHi: number;
  /** 正規化した対数輝度 → [0, 1] の単調な折れ線（等間隔の節点の値） */
  guide: Float32Array;
  inScale: number;
  logEps: number;
  splat: Conv[];
  local1: Conv;
  local2: Conv;
  glob1: Conv;
  glob2: Conv;
  fc: Dense[];
  out: Conv;
  /** 学習データなどの説明 */
  info: Record<string, unknown>;
}

/** 画像ごとに CNN で求めたグリッドと露出の基準値。プレビューと書き出しで共有する */
export interface ToneGrid {
  /** [係数][明るさの段][y][x] */
  grid: Float32Array;
  /** 縮小画像の対数輝度 (log2) の中央値 */
  key: number;
}

export interface LearnedParams {
  /** 露出補正 [EV]。0 で学習どおり */
  exposure: number;
}

export const DEFAULT_LEARNED_PARAMS: LearnedParams = { exposure: 0 };

const MAGIC = 0x314d544f; // "OTM1"

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 5.960464477539063e-8; // 2^-24
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
}

/** モデルファイル（JSON ヘッダ + float16 の重み）を読む */
export function parseToneModel(buf: ArrayBuffer): ToneModel {
  const dv = new DataView(buf);
  if (buf.byteLength < 8 || dv.getUint32(0, true) !== MAGIC) throw new Error('モデルファイルの形式が正しくありません');
  const headerLen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))) as {
    hyper: Record<string, number>;
    info?: Record<string, unknown>;
    tensors: Array<{ name: string; shape: number[]; offset: number; length: number }>;
  };
  const dataStart = 8 + headerLen + ((8 + headerLen) & 1);
  const tensors = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const t of header.tensors) {
    const data = new Float32Array(t.length);
    const base = dataStart + t.offset * 2;
    for (let i = 0; i < t.length; i++) data[i] = halfToFloat(dv.getUint16(base + i * 2, true));
    tensors.set(t.name, { shape: t.shape, data });
  }
  const get = (name: string) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`モデルに ${name} がありません`);
    return t;
  };
  const conv = (name: string, stride: number, hasBias = true): Conv => {
    const w = get(`${name}.weight`);
    const [cout, cin, k] = w.shape;
    return { cin, cout, k, stride, weight: w.data, bias: hasBias ? get(`${name}.bias`).data : null };
  };
  const dense = (name: string): Dense => {
    const w = get(`${name}.weight`);
    return { cout: w.shape[0], cin: w.shape[1], weight: w.data, bias: get(`${name}.bias`).data };
  };
  const h = header.hyper;
  return {
    low: h.low,
    gridXY: h.grid_xy,
    gridZ: h.grid_z,
    ncoef: h.ncoef,
    guideLo: h.guide_lo,
    guideHi: h.guide_hi,
    guide: get('guide').data,
    inScale: h.in_scale,
    logEps: h.log_eps,
    splat: [0, 1, 2, 3].map((i) => conv(`splat.${i}`, 2)),
    local1: conv('local1', 1),
    local2: conv('local2', 1, false),
    glob1: conv('glob1', 2),
    glob2: conv('glob2', 2),
    fc: ['fc1', 'fc2', 'fc3'].map(dense),
    out: conv('out', 1),
    info: header.info ?? {},
  };
}

interface Tensor {
  c: number;
  h: number;
  w: number;
  data: Float32Array;
}

/** 畳み込み（ゼロパディング = (k-1)/2）。データは CHW */
function conv2d(x: Tensor, l: Conv, relu: boolean): Tensor {
  const { k, stride, cout, cin } = l;
  const pad = (k - 1) >> 1;
  const oh = Math.floor((x.h + 2 * pad - k) / stride) + 1;
  const ow = Math.floor((x.w + 2 * pad - k) / stride) + 1;
  const out = new Float32Array(cout * oh * ow);
  const inp = x.data;
  const plane = oh * ow;
  for (let oc = 0; oc < cout; oc++) {
    const ob = oc * plane;
    const b = l.bias ? l.bias[oc] : 0;
    for (let i = 0; i < plane; i++) out[ob + i] = b;
    for (let ic = 0; ic < cin; ic++) {
      const ib = ic * x.h * x.w;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const wv = l.weight[((oc * cin + ic) * k + ky) * k + kx];
          if (wv === 0) continue;
          // 入力の範囲に収まる出力 x の範囲
          let ox0 = 0;
          while (ox0 < ow && ox0 * stride + kx - pad < 0) ox0++;
          let ox1 = ow;
          while (ox1 > ox0 && (ox1 - 1) * stride + kx - pad >= x.w) ox1--;
          for (let oy = 0; oy < oh; oy++) {
            const iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= x.h) continue;
            const irow = ib + iy * x.w + kx - pad;
            const orow = ob + oy * ow;
            for (let ox = ox0; ox < ox1; ox++) out[orow + ox] += wv * inp[irow + ox * stride];
          }
        }
      }
    }
    if (relu) for (let i = ob; i < ob + plane; i++) if (out[i] < 0) out[i] = 0;
  }
  return { c: cout, h: oh, w: ow, data: out };
}

function dense(x: Float32Array, l: Dense, relu: boolean): Float32Array {
  const out = new Float32Array(l.cout);
  for (let o = 0; o < l.cout; o++) {
    let acc = l.bias[o];
    const base = o * l.cin;
    for (let i = 0; i < l.cin; i++) acc += l.weight[base + i] * x[i];
    out[o] = relu && acc < 0 ? 0 : acc;
  }
  return out;
}

/** 下側の中央値（PyTorch の median と同じ） */
function lowerMedian(v: Float32Array): number {
  const s = Float32Array.from(v).sort();
  return s[(s.length - 1) >> 1];
}

/**
 * 縮小画像（リニア RGB、CHW で 3 × low × low）からグリッドを求める。
 */
export function predictGrid(model: ToneModel, thumb: Float32Array): ToneGrid {
  const n = model.low * model.low;
  if (thumb.length !== 3 * n) throw new Error('縮小画像の大きさが違います');
  const eps = model.logEps;
  const ll = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const y = LUMA_R * thumb[i] + LUMA_G * thumb[n + i] + LUMA_B * thumb[2 * n + i];
    ll[i] = Math.log2(y > eps ? y : eps);
  }
  const key = lowerMedian(ll);
  const input = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) {
    const v = thumb[i];
    const t = (Math.log2(v > eps ? v : eps) - key) * model.inScale;
    input[i] = t < -4 ? -4 : t > 4 ? 4 : t;
  }
  let x: Tensor = { c: 3, h: model.low, w: model.low, data: input };
  for (const l of model.splat) x = conv2d(x, l, true);
  const loc = conv2d(conv2d(x, model.local1, true), model.local2, false);
  const g = conv2d(conv2d(x, model.glob1, true), model.glob2, true);
  let v = dense(g.data, model.fc[0], true);
  v = dense(v, model.fc[1], true);
  v = dense(v, model.fc[2], false);
  const plane = loc.h * loc.w;
  const fused = new Float32Array(loc.data.length);
  for (let c = 0; c < loc.c; c++) {
    for (let i = 0; i < plane; i++) {
      const t = loc.data[c * plane + i] + v[c];
      fused[c * plane + i] = t > 0 ? t : 0;
    }
  }
  const o = conv2d({ c: loc.c, h: loc.h, w: loc.w, data: fused }, model.out, false);
  // 出力チャンネルは z * ncoef + c の順。[c][z][y][x] に並べ替え、
  // 傾きと彩度はグリッドの節点で指数を取って正の値にしておく（画素ごとには線形補間するだけ）
  const { gridZ: D, ncoef: C } = model;
  const grid = new Float32Array(C * D * plane);
  for (let z = 0; z < D; z++) {
    for (let c = 0; c < C; c++) {
      const src = o.data.subarray((z * C + c) * plane, (z * C + c + 1) * plane);
      const dst = (c * D + z) * plane;
      for (let i = 0; i < plane; i++) grid[dst + i] = c === 1 ? src[i] : Math.exp(src[i]);
    }
  }
  return { grid, key };
}

/** 合成した放射輝度を面積平均で size × size に縮小する（CHW、縦横比は無視） */
export function radianceThumbnail(
  views: View[],
  luts: Float32Array[],
  exposures: number[],
  size: number,
  progress: ProgressFn = () => {},
): Float32Array {
  const w = views[0].width;
  const h = views[0].height;
  let darkest = 0;
  for (let k = 1; k < views.length; k++) if (exposures[k] < exposures[darkest]) darkest = k;
  const hx = buildAreaWeights(w, size);
  const hy = buildAreaWeights(h, size);
  // 横方向に縮小した行 (h × size × 3)
  const rows = new Float32Array(h * size * 3);
  const row = new Float32Array(w * 3);
  for (let y = 0; y < h; y++) {
    mergeRow(views, luts, exposures, darkest, y, row);
    const o = y * size * 3;
    for (let x = 0; x < size; x++) {
      const s = hx.start[x];
      const n = hx.count[x];
      const wo = hx.offset[x];
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < n; k++) {
        const wt = hx.weights[wo + k];
        const i = (s + k) * 3;
        r += row[i] * wt;
        g += row[i + 1] * wt;
        b += row[i + 2] * wt;
      }
      rows[o + x * 3] = r;
      rows[o + x * 3 + 1] = g;
      rows[o + x * 3 + 2] = b;
    }
    if ((y & 127) === 0) progress(y / h);
  }
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let y = 0; y < size; y++) {
    const s = hy.start[y];
    const cnt = hy.count[y];
    const wo = hy.offset[y];
    for (let k = 0; k < cnt; k++) {
      const wt = hy.weights[wo + k];
      const src = (s + k) * size * 3;
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        out[i] += rows[src + x * 3] * wt;
        out[n + i] += rows[src + x * 3 + 1] * wt;
        out[2 * n + i] += rows[src + x * 3 + 2] * wt;
      }
    }
  }
  progress(1);
  return out;
}

/**
 * 放射輝度の行をグリッドで仕上げて表示用 (sRGB 0..1) にする。
 * 画像の幅・高さごとに x 方向の補間係数などを前計算しておき、行ごとに row() を呼ぶ。
 */
export class LearnedRenderer {
  private readonly x0: Int32Array;
  private readonly tx: Float32Array;
  private readonly rg: Float32Array;

  constructor(
    private readonly model: ToneModel,
    private readonly tg: ToneGrid,
    private readonly params: LearnedParams,
    private readonly width: number,
    private readonly height: number,
  ) {
    const G = model.gridXY;
    this.x0 = new Int32Array(width);
    this.tx = new Float32Array(width);
    for (let x = 0; x < width; x++) {
      let fx = ((x + 0.5) / width) * G - 0.5;
      fx = fx < 0 ? 0 : fx > G - 1 ? G - 1 : fx;
      const i = Math.min(G - 2, Math.floor(fx));
      this.x0[x] = i;
      this.tx[x] = fx - i;
    }
    this.rg = new Float32Array(3 * model.gridZ * G);
  }

  /** src: 放射輝度 (w*3)、y: 行番号、out: 表示用 sRGB (w*3) */
  row(src: Float32Array, y: number, out: Float32Array): void {
    const { model, tg, width: w, rg } = this;
    const G = model.gridXY;
    const D = model.gridZ;
    const plane = G * G;
    const grid = tg.grid;
    // この行の y 方向の補間を先に済ませる: rg[c][z][x]
    let fy = ((y + 0.5) / this.height) * G - 0.5;
    fy = fy < 0 ? 0 : fy > G - 1 ? G - 1 : fy;
    const y0 = Math.min(G - 2, Math.floor(fy));
    const ty = fy - y0;
    for (let cz = 0; cz < 3 * D; cz++) {
      const r0 = cz * plane + y0 * G;
      const r1 = r0 + G;
      const o = cz * G;
      for (let x = 0; x < G; x++) rg[o + x] = grid[r0 + x] + (grid[r1 + x] - grid[r0 + x]) * ty;
    }
    const key = tg.key;
    const eps = model.logEps;
    const guide = model.guide;
    const kMax = guide.length - 1;
    const gScale = kMax / (model.guideHi - model.guideLo);
    const gLo = model.guideLo;
    const offset = this.params.exposure - key;
    const cb = D * G;
    const cs = 2 * D * G;
    const x0s = this.x0;
    const txs = this.tx;
    for (let x = 0, i = 0; x < w; x++, i += 3) {
      const r = src[i];
      const g = src[i + 1];
      const b = src[i + 2];
      const Y = LUMA_R * r + LUMA_G * g + LUMA_B * b;
      const ln = Math.log2(Y > eps ? Y : eps) - key;
      // ガイド（明るさの段の位置）
      let t = (ln - gLo) * gScale;
      t = t < 0 ? 0 : t > kMax ? kMax : t;
      const ti = t >= kMax ? kMax - 1 : t | 0;
      const gv = guide[ti] + (guide[ti + 1] - guide[ti]) * (t - ti);
      let fz = gv * D - 0.5;
      fz = fz < 0 ? 0 : fz > D - 1 ? D - 1 : fz;
      const z0 = fz >= D - 1 ? D - 2 : fz | 0;
      const tz = fz - z0;
      const tx = txs[x];
      const w00 = (1 - tz) * (1 - tx);
      const w01 = (1 - tz) * tx;
      const w10 = tz * (1 - tx);
      const w11 = tz * tx;
      const o0 = z0 * G + x0s[x];
      const o1 = o0 + G;
      const a = rg[o0] * w00 + rg[o0 + 1] * w01 + rg[o1] * w10 + rg[o1 + 1] * w11;
      const bb = rg[cb + o0] * w00 + rg[cb + o0 + 1] * w01 + rg[cb + o1] * w10 + rg[cb + o1 + 1] * w11;
      const s = rg[cs + o0] * w00 + rg[cs + o0 + 1] * w01 + rg[cs + o1] * w10 + rg[cs + o1 + 1] * w11;
      const gain = Math.exp(((a - 1) * ln + bb + offset) * Math.LN2);
      const dr = linearToDisplay(r * gain);
      const dg = linearToDisplay(g * gain);
      const db = linearToDisplay(b * gain);
      const yd = LUMA_R * dr + LUMA_G * dg + LUMA_B * db;
      const vr = yd + s * (dr - yd);
      const vg = yd + s * (dg - yd);
      const vb = yd + s * (db - yd);
      out[i] = vr < 0 ? 0 : vr > 1 ? 1 : vr;
      out[i + 1] = vg < 0 ? 0 : vg > 1 ? 1 : vg;
      out[i + 2] = vb < 0 ? 0 : vb > 1 ? 1 : vb;
    }
  }
}

/** 放射輝度を合成しながら学習済みモデルで仕上げ、表示用 16bit RGB を out に書く */
export function renderLearned(
  views: View[],
  luts: Float32Array[],
  exposures: number[],
  model: ToneModel,
  tg: ToneGrid,
  params: LearnedParams,
  out: Uint16Array,
  progress: ProgressFn = () => {},
): void {
  const w = views[0].width;
  const h = views[0].height;
  let darkest = 0;
  for (let k = 1; k < views.length; k++) if (exposures[k] < exposures[darkest]) darkest = k;
  const row = new Float32Array(w * 3);
  const disp = new Float32Array(w * 3);
  const renderer = new LearnedRenderer(model, tg, params, w, h);
  for (let y = 0; y < h; y++) {
    mergeRow(views, luts, exposures, darkest, y, row);
    renderer.row(row, y, disp);
    const o = y * w * 3;
    for (let i = 0; i < w * 3; i++) out[o + i] = (disp[i] * 65535 + 0.5) | 0;
    if ((y & 63) === 0) progress(y / h);
  }
  progress(1);
}
