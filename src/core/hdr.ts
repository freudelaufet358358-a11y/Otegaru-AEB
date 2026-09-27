// リニアなデータ（主に RAW）から HDR（放射輝度マップ）を合成し、局所トーンマッピングで表示用に圧縮する。
// ・露出比は画像データ自体から推定する（EXIF のシャッター速度表記の丸め誤差を避けるため）
// ・トーンマッピングは対数輝度を高速ガイデッドフィルタでベース／ディテールに分け、ベースだけを圧縮する
//   （Durand & Dorsey 2002 の手法を、ガイデッドフィルタ (He et al.) で高速化したもの）

import { linearToDisplay, LUMA_B, LUMA_G, LUMA_R, RAW_DISPLAY_GAIN } from './color';
import { rowIndex, type View } from './frame';
import type { ProgressFn } from './fusion';

const CLIP_START = 0.8;
const CLIP_END = 0.96;
/** 最も暗いフレームでもこの明るさを超えた画素は、白に向けて彩度を落とす（下の mergeRow を参照） */
const DESAT_START = 0.94;
const DESAT_END = 0.99;

/** 白飛び付近を滑らかに除外する重み */
function clipWeight(m: number): number {
  return m < CLIP_START ? 1 : m >= CLIP_END ? 0 : (CLIP_END - m) / (CLIP_END - CLIP_START);
}

/**
 * 露出比（基準フレーム = 1）を推定する。
 * order は暗い→明るい順のインデックス。隣り合うフレーム同士の輝度比の中央値を鎖状に掛け合わせる。
 * 十分な有効画素がない場合は fallback（EXIF 由来の相対露出。未知なら NaN）を使う。
 */
export function estimateExposures(
  views: View[],
  luts: Float32Array[],
  order: number[],
  ref: number,
  fallback: number[],
): number[] {
  const n = views.length;
  const e = new Array<number>(n).fill(1);
  const w = views[0].width;
  const h = views[0].height;
  const step = Math.max(1, Math.floor(Math.sqrt((w * h) / 250000)));
  const ratios = new Float32Array(Math.ceil(w / step) * Math.ceil(h / step));
  for (let s = 1; s < n; s++) {
    const a = order[s - 1];
    const b = order[s];
    const va = views[a];
    const vb = views[b];
    const la = luts[a];
    const lb = luts[b];
    const da = va.frame.data;
    const db = vb.frame.data;
    let cnt = 0;
    for (let y = 0; y < h; y += step) {
      const ia = rowIndex(va, y);
      const ib = rowIndex(vb, y);
      for (let x = 0; x < w; x += step) {
        const i = ia + x * 3;
        const j = ib + x * 3;
        const ar = la[da[i]];
        const ag = la[da[i + 1]];
        const ab = la[da[i + 2]];
        const br = lb[db[j]];
        const bg = lb[db[j + 1]];
        const bb = lb[db[j + 2]];
        if (Math.max(ar, ag, ab, br, bg, bb) >= CLIP_START) continue;
        const lA = LUMA_R * ar + LUMA_G * ag + LUMA_B * ab;
        const lB = LUMA_R * br + LUMA_G * bg + LUMA_B * bb;
        if (lA < 0.004 || lB < 0.004) continue;
        ratios[cnt++] = lB / lA;
      }
    }
    let r: number;
    if (cnt >= 1000) {
      const sub = ratios.subarray(0, cnt).sort();
      r = sub[cnt >> 1];
    } else {
      const fa = fallback[a];
      const fb = fallback[b];
      r = Number.isFinite(fa) && Number.isFinite(fb) && fa > 0 ? fb / fa : 2;
    }
    e[b] = e[a] * Math.max(r, 1e-6);
  }
  const base = e[ref];
  return e.map((v) => v / base);
}

/** 1 行分の放射輝度（基準フレームの露出に換算したリニア RGB）を out (w*3) に合成する */
export function mergeRow(
  views: View[],
  luts: Float32Array[],
  exposures: number[],
  darkest: number,
  y: number,
  out: Float32Array,
): void {
  const n = views.length;
  const w = views[0].width;
  const starts = new Array<number>(n);
  for (let k = 0; k < n; k++) starts[k] = rowIndex(views[k], y);
  const dk = views[darkest].frame.data;
  const lk = luts[darkest];
  const invDark = 1 / exposures[darkest];
  for (let x = 0; x < w; x++) {
    let sr = 0;
    let sg = 0;
    let sb = 0;
    let sw = 0;
    const off = x * 3;
    for (let k = 0; k < n; k++) {
      const d = views[k].frame.data;
      const lut = luts[k];
      const i = starts[k] + off;
      const r = lut[d[i]];
      const g = lut[d[i + 1]];
      const b = lut[d[i + 2]];
      const m = r > g ? (r > b ? r : b) : g > b ? g : b;
      const cw = clipWeight(m);
      if (cw === 0) continue;
      // 重みを露出に比例させる（ショットノイズに対する最尤推定: Σv / Σe）
      sr += r * cw;
      sg += g * cw;
      sb += b * cw;
      sw += cw * exposures[k];
    }
    const o = off;
    const j = starts[darkest] + off;
    const dr = lk[dk[j]];
    const dg = lk[dk[j + 1]];
    const db = lk[dk[j + 2]];
    if (sw > 0) {
      const inv = 1 / sw;
      out[o] = sr * inv;
      out[o + 1] = sg * inv;
      out[o + 2] = sb * inv;
    } else {
      // 最も暗いフレームでも白飛びしている → そのまま使う
      out[o] = dr * invDark;
      out[o + 1] = dg * invDark;
      out[o + 2] = db * invDark;
    }
    // 最も暗いフレームでも飽和しかけている画素は、一部の色だけが頭打ちになって色相が狂っている
    // （太陽や窓が紫・ピンクに見える）。トーンマッピングで暗く引き下げると目立つので、白に向けて彩度を落とす
    const md = dr > dg ? (dr > db ? dr : db) : dg > db ? dg : db;
    if (md > DESAT_START) {
      let t = md >= DESAT_END ? 1 : (md - DESAT_START) / (DESAT_END - DESAT_START);
      t = t * t * (3 - 2 * t);
      const r = out[o];
      const g = out[o + 1];
      const b = out[o + 2];
      const M = r > g ? (r > b ? r : b) : g > b ? g : b;
      out[o] = r + (M - r) * t;
      out[o + 1] = g + (M - g) * t;
      out[o + 2] = b + (M - b) * t;
    }
  }
}

/** 窓がはみ出す部分を除いた平均を取る箱フィルタ（O(1)/画素） */
export function boxFilter(src: Float32Array, w: number, h: number, r: number, out: Float32Array): void {
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let acc = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) acc += src[o + x];
    for (let x = 0; x < w; x++) {
      const lo = x - r < 0 ? 0 : x - r;
      const hi = x + r >= w ? w - 1 : x + r;
      tmp[o + x] = acc / (hi - lo + 1);
      if (x + r + 1 < w) acc += src[o + x + r + 1];
      if (x - r >= 0) acc -= src[o + x - r];
    }
  }
  const col = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) acc += tmp[y * w + x];
    col[x] = acc;
  }
  for (let y = 0; y < h; y++) {
    const lo = y - r < 0 ? 0 : y - r;
    const hi = y + r >= h ? h - 1 : y + r;
    const inv = 1 / (hi - lo + 1);
    const o = y * w;
    const addRow = y + r + 1 < h ? (y + r + 1) * w : -1;
    const subRow = y - r >= 0 ? (y - r) * w : -1;
    for (let x = 0; x < w; x++) {
      out[o + x] = col[x] * inv;
      if (addRow >= 0) col[x] += tmp[addRow + x];
      if (subRow >= 0) col[x] -= tmp[subRow + x];
    }
  }
}

/** 自己誘導のガイデッドフィルタの係数 (a, b) を平滑化したものを返す */
export function guidedCoefficients(
  I: Float32Array,
  w: number,
  h: number,
  r: number,
  eps: number,
): { a: Float32Array; b: Float32Array } {
  const size = w * h;
  const meanI = new Float32Array(size);
  const corr = new Float32Array(size);
  const sq = new Float32Array(size);
  for (let i = 0; i < size; i++) sq[i] = I[i] * I[i];
  boxFilter(I, w, h, r, meanI);
  boxFilter(sq, w, h, r, corr);
  const a = sq; // 使い回し
  const b = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const m = meanI[i];
    const v = Math.max(0, corr[i] - m * m);
    const ai = v / (v + eps);
    a[i] = ai;
    b[i] = m - ai * m;
  }
  const ma = meanI;
  const mb = corr;
  boxFilter(a, w, h, r, ma);
  boxFilter(b, w, h, r, mb);
  return { a: ma, b: mb };
}

export interface ToneParams {
  /** 階調圧縮の強さ 0..1 */
  strength: number;
  /** ディテール（局所コントラスト）の倍率。1 で元のまま */
  detail: number;
}

export const DEFAULT_TONE_PARAMS: ToneParams = { strength: 0.75, detail: 1.3 };

/** 表示で扱える目安の段数 */
const TARGET_STOPS = 4;
/** 圧縮後のベースの明るい側 (99.5%) の上限（リニア 1.0。ここから上は肩特性で滑らかに白へ） */
const HIGHLIGHT_LIMIT = 0;
/** ハイライトを収めるために中間調を暗くしてよい上限 [段] */
const MAX_DARKEN = 1;
/** 中間調を明るくしてよい上限 [段]（夜景などを昼のように明るくしすぎないため） */
const MAX_LIFT = 4;
/** 自動露出で中間調（ベースの中央値）を合わせる目標のリニア値 */
const MID_KEY = 0.18;

/**
 * HDR 合成 + トーンマッピングを行い、表示用 16bit RGB (sRGB) を out に書き込む。
 */
export function toneMapHDR(
  views: View[],
  luts: Float32Array[],
  exposures: number[],
  params: ToneParams,
  out: Uint16Array,
  progress: ProgressFn = () => {},
): void {
  const w = views[0].width;
  const h = views[0].height;
  let darkest = 0;
  for (let k = 1; k < views.length; k++) if (exposures[k] < exposures[darkest]) darkest = k;
  const row = new Float32Array(w * 3);

  // 1) 対数輝度
  const logL = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    mergeRow(views, luts, exposures, darkest, y, row);
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const i = x * 3;
      const L = LUMA_R * row[i] + LUMA_G * row[i + 1] + LUMA_B * row[i + 2];
      logL[o + x] = Math.log2(L > 1e-6 ? L : 1e-6);
    }
    if ((y & 63) === 0) progress((0.4 * y) / h);
  }

  // 2) 縮小画像上でガイデッドフィルタ（ベース層の係数）
  const s = Math.max(1, Math.floor(Math.min(w, h) / 480));
  const lw = Math.ceil(w / s);
  const lh = Math.ceil(h / s);
  const low = new Float32Array(lw * lh);
  for (let ly = 0; ly < lh; ly++) {
    const y0 = ly * s;
    const y1 = Math.min(h, y0 + s);
    for (let lx = 0; lx < lw; lx++) {
      const x0 = lx * s;
      const x1 = Math.min(w, x0 + s);
      let acc = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) acc += logL[y * w + x];
      low[ly * lw + lx] = acc / ((y1 - y0) * (x1 - x0));
    }
  }
  const radius = Math.max(2, Math.round(0.04 * Math.min(lw, lh)));
  const { a, b } = guidedCoefficients(low, lw, lh, radius, 0.3);

  // ベースの範囲（外れ値を除く）
  const base = new Float32Array(lw * lh);
  for (let i = 0; i < base.length; i++) base[i] = a[i] * low[i] + b[i];
  base.sort();
  const bLo = base[Math.floor(base.length * 0.005)];
  const bMid = base[base.length >> 1];
  const bHi = base[Math.min(base.length - 1, Math.floor(base.length * 0.995))];
  const range = Math.max(1e-3, bHi - bLo);
  const strength = Math.min(1, Math.max(0, params.strength));
  const c = 1 - strength * (1 - Math.min(1, TARGET_STOPS / range));
  // 中間調は自動露出で適正な明るさ（MID_KEY）へ。ただし基準フレームの見た目から
  // 大きく離れすぎないようにし、ハイライトが収まらない分だけ暗くする
  const displayGain = views[0].frame.encoding === 'linear' ? RAW_DISPLAY_GAIN : 1;
  const keepMid = bMid + Math.log2(displayGain);
  const autoMid = Math.min(keepMid + MAX_LIFT, Math.max(keepMid - MAX_DARKEN, Math.log2(MID_KEY)));
  const fitHighlight = HIGHLIGHT_LIMIT - (bHi - bMid) * c;
  const anchor = Math.max(keepMid - MAX_DARKEN, Math.min(autoMid, fitHighlight));
  const detail = params.detail;
  progress(0.5);

  // 3) 各画素を圧縮して出力
  const x0s = new Int32Array(w);
  const txs = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    const f = Math.min(lw - 1, Math.max(0, (x + 0.5) / s - 0.5));
    const i = Math.min(lw - 2, Math.floor(f));
    x0s[x] = Math.max(0, i);
    txs[x] = lw > 1 ? f - Math.max(0, i) : 0;
  }
  const rowA = new Float32Array(lw);
  const rowB = new Float32Array(lw);
  const x1Max = lw - 1;
  for (let y = 0; y < h; y++) {
    const f = Math.min(lh - 1, Math.max(0, (y + 0.5) / s - 0.5));
    const iy = Math.max(0, Math.min(lh - 2, Math.floor(f)));
    const ty = lh > 1 ? f - iy : 0;
    const r0 = iy * lw;
    const r1 = Math.min(lh - 1, iy + 1) * lw;
    for (let x = 0; x < lw; x++) {
      rowA[x] = a[r0 + x] + (a[r1 + x] - a[r0 + x]) * ty;
      rowB[x] = b[r0 + x] + (b[r1 + x] - b[r0 + x]) * ty;
    }
    mergeRow(views, luts, exposures, darkest, y, row);
    const lo = y * w;
    const oo = y * w * 3;
    for (let x = 0; x < w; x++) {
      const xi = x0s[x];
      const xj = xi + 1 > x1Max ? x1Max : xi + 1;
      const tx = txs[x];
      const ca = rowA[xi] + (rowA[xj] - rowA[xi]) * tx;
      const cb = rowB[xi] + (rowB[xj] - rowB[xi]) * tx;
      const l = logL[lo + x];
      const bs = ca * l + cb;
      const nl = (bs - bMid) * c + anchor + (l - bs) * detail;
      const gain = Math.pow(2, nl - l);
      const i = x * 3;
      out[oo + i] = toU16(linearToDisplay(row[i] * gain));
      out[oo + i + 1] = toU16(linearToDisplay(row[i + 1] * gain));
      out[oo + i + 2] = toU16(linearToDisplay(row[i + 2] * gain));
    }
    if ((y & 63) === 0) progress(0.5 + (0.5 * y) / h);
  }
  progress(1);
}

function toU16(v: number): number {
  return (v * 65535 + 0.5) | 0;
}
