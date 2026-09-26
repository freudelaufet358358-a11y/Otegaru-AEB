import type { Encoding } from './color';

/** 16bit RGB（インターリーブ）の画像。RAW はリニア、JPEG などは sRGB 符号値を 16bit に拡張して持つ */
export interface Frame16 {
  width: number;
  height: number;
  data: Uint16Array;
  encoding: Encoding;
}

/** フレームの一部分（位置合わせ後の切り抜き）を参照するビュー。コピーせずにオフセットで扱う */
export interface View {
  frame: Frame16;
  x0: number;
  y0: number;
  width: number;
  height: number;
}

export function fullView(frame: Frame16): View {
  return { frame, x0: 0, y0: 0, width: frame.width, height: frame.height };
}

/** ビューの 1 行目の先頭インデックス（インターリーブ配列上） */
export function rowIndex(v: View, y: number): number {
  return ((y + v.y0) * v.frame.width + v.x0) * 3;
}

/** 1 チャンネルを LUT で変換して平面配列に取り出す */
export function extractChannel(v: View, c: number, lut: Float32Array, out: Float32Array): void {
  const d = v.frame.data;
  const w = v.width;
  let o = 0;
  for (let y = 0; y < v.height; y++) {
    let i = rowIndex(v, y) + c;
    for (let x = 0; x < w; x++, i += 3) out[o++] = lut[d[i]];
  }
}

/**
 * 面積平均による縮小（アンチエイリアスあり）。任意倍率に対応。
 * RAW はリニアのまま平均するので物理的に正しい縮小になる。
 */
export function resizeView(v: View, tw: number, th: number): Frame16 {
  const sw = v.width;
  const sh = v.height;
  const hx = buildAreaWeights(sw, tw);
  const hy = buildAreaWeights(sh, th);
  const d = v.frame.data;
  // 横方向 → Float32 の中間バッファ (tw × sh × 3)
  const tmp = new Float32Array(tw * sh * 3);
  for (let y = 0; y < sh; y++) {
    const base = rowIndex(v, y);
    const orow = y * tw * 3;
    for (let x = 0; x < tw; x++) {
      const s = hx.start[x];
      const n = hx.count[x];
      const wo = hx.offset[x];
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < n; k++) {
        const wt = hx.weights[wo + k];
        const i = base + (s + k) * 3;
        r += d[i] * wt;
        g += d[i + 1] * wt;
        b += d[i + 2] * wt;
      }
      const o = orow + x * 3;
      tmp[o] = r;
      tmp[o + 1] = g;
      tmp[o + 2] = b;
    }
  }
  // 縦方向 → Uint16
  const out = new Uint16Array(tw * th * 3);
  const rowLen = tw * 3;
  const acc = new Float32Array(rowLen);
  for (let y = 0; y < th; y++) {
    acc.fill(0);
    const s = hy.start[y];
    const n = hy.count[y];
    const wo = hy.offset[y];
    for (let k = 0; k < n; k++) {
      const wt = hy.weights[wo + k];
      const src = (s + k) * rowLen;
      for (let i = 0; i < rowLen; i++) acc[i] += tmp[src + i] * wt;
    }
    const o = y * rowLen;
    for (let i = 0; i < rowLen; i++) {
      const val = acc[i] + 0.5;
      out[o + i] = val >= 65535 ? 65535 : val | 0;
    }
  }
  return { width: tw, height: th, data: out, encoding: v.frame.encoding };
}

interface AreaWeights {
  start: Int32Array;
  count: Int32Array;
  offset: Int32Array;
  weights: Float32Array;
}

function buildAreaWeights(src: number, dst: number): AreaWeights {
  const scale = src / dst;
  const start = new Int32Array(dst);
  const count = new Int32Array(dst);
  const offset = new Int32Array(dst);
  const list: number[] = [];
  for (let i = 0; i < dst; i++) {
    const a = i * scale;
    const b = Math.min(src, (i + 1) * scale);
    const s = Math.floor(a);
    const e = Math.min(src, Math.ceil(b));
    start[i] = s;
    count[i] = e - s;
    offset[i] = list.length;
    const total = b - a;
    for (let j = s; j < e; j++) {
      const cover = Math.min(b, j + 1) - Math.max(a, j);
      list.push(cover / total);
    }
  }
  return { start, count, offset, weights: Float32Array.from(list) };
}

/** 長辺が maxSide 以下になるサイズを求める（拡大はしない） */
export function fitSize(w: number, h: number, maxSide: number): [number, number] {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}
