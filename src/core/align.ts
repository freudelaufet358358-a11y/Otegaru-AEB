// Ward の MTB（Median Threshold Bitmap）法による平行移動の位置合わせ。
// 各画像を中央値で二値化するので露出が違うフレーム同士でも比較でき、手持ちのブラケット撮影に強い。

import { fullView, rowIndex, type Frame16, type View } from './frame';

export interface Gray8 {
  width: number;
  height: number;
  data: Uint8Array;
}

/** 表示用 LUT を使って 8bit グレースケールを作る */
export function toGray8(frame: Frame16, lut: Float32Array): Gray8 {
  const v = fullView(frame);
  const { width: w, height: h } = v;
  const d = frame.data;
  const out = new Uint8Array(w * h);
  let o = 0;
  for (let y = 0; y < h; y++) {
    let i = rowIndex(v, y);
    for (let x = 0; x < w; x++, i += 3) {
      const g = (lut[d[i]] * 54 + lut[d[i + 1]] * 183 + lut[d[i + 2]] * 19) * (255 / 256);
      out[o++] = g >= 255 ? 255 : (g + 0.5) | 0;
    }
  }
  return { width: w, height: h, data: out };
}

function half(g: Gray8): Gray8 {
  const w = g.width >> 1;
  const h = g.height >> 1;
  const out = new Uint8Array(w * h);
  const s = g.data;
  const sw = g.width;
  for (let y = 0; y < h; y++) {
    const r0 = 2 * y * sw;
    const r1 = r0 + sw;
    for (let x = 0; x < w; x++) {
      const i = 2 * x;
      out[y * w + x] = (s[r0 + i] + s[r0 + i + 1] + s[r1 + i] + s[r1 + i + 1] + 2) >> 2;
    }
  }
  return { width: w, height: h, data: out };
}

function histogram(g: Gray8): Uint32Array {
  const hist = new Uint32Array(256);
  const d = g.data;
  for (let i = 0; i < d.length; i++) hist[d[i]]++;
  return hist;
}

function percentileOf(hist: Uint32Array, total: number, p: number): number {
  const target = total * p;
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= target) return v;
  }
  return 255;
}

/**
 * 二値化に使うパーセンタイルを選ぶ。基本は中央値 (50%) だが、白飛び・黒つぶれが多いフレームでは
 * 中央値が飽和値に張り付いて比較できなくなるため、両方の画像で階調が残っている割合を探す。
 */
function choosePercentile(a: Gray8, b: Gray8, noise: number): number {
  const ha = histogram(a);
  const hb = histogram(b);
  const lo = noise + 6;
  const hi = 249 - noise;
  for (const p of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.12, 0.88]) {
    const ta = percentileOf(ha, a.data.length, p);
    const tb = percentileOf(hb, b.data.length, p);
    if (ta >= lo && ta <= hi && tb >= lo && tb <= hi) return p;
  }
  return 0.5;
}

/** bit0 = 閾値より明るい, bit1 = 閾値から十分離れている（ノイズ除外） */
function thresholdBitmap(g: Gray8, p: number, noise: number): Uint8Array {
  const t = percentileOf(histogram(g), g.data.length, p);
  const d = g.data;
  const out = new Uint8Array(d.length);
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    out[i] = (v > t ? 1 : 0) | (v > t + noise || v < t - noise ? 2 : 0);
  }
  return out;
}

function shiftError(a: Uint8Array, b: Uint8Array, w: number, h: number, sx: number, sy: number): number {
  // a(x, y) と b(x + sx, y + sy) を比較
  const x0 = Math.max(0, -sx);
  const x1 = Math.min(w, w - sx);
  const y0 = Math.max(0, -sy);
  const y1 = Math.min(h, h - sy);
  if (x1 <= x0 || y1 <= y0) return Infinity;
  let err = 0;
  for (let y = y0; y < y1; y++) {
    let i = y * w + x0;
    let j = (y + sy) * w + x0 + sx;
    for (let x = x0; x < x1; x++, i++, j++) {
      const p = a[i];
      const q = b[j];
      // 二値が異なり、かつ両方ともノイズ除外されていない画素を数える
      err += (p ^ q) & ((p & q) >> 1) & 1;
    }
  }
  // 比較できた画素数で正規化（ずらし量による有利不利をなくす）
  return err / ((x1 - x0) * (y1 - y0));
}

/**
 * target を reference に合わせるずれ量 [dx, dy] を返す。
 * target の (x + dx, y + dy) が reference の (x, y) に対応する。
 */
export function alignMTB(reference: Gray8, target: Gray8, maxShift = 128, noise = 4): [number, number] {
  if (reference.width !== target.width || reference.height !== target.height) {
    throw new Error('位置合わせ: 画像サイズが一致しません');
  }
  // ピラミッド段数: 最上段でも 32px 程度は残す（段数 L で最大 ±(2^(L+1) - 1) px まで探索できる）
  let levels = 0;
  while ((2 << (levels + 1)) - 1 <= maxShift && Math.min(reference.width, reference.height) >> (levels + 1) >= 32) levels++;
  const refPyr: Gray8[] = [reference];
  const tgtPyr: Gray8[] = [target];
  for (let i = 0; i < levels; i++) {
    refPyr.push(half(refPyr[i]));
    tgtPyr.push(half(tgtPyr[i]));
  }
  const p = choosePercentile(reference, target, noise);
  let dx = 0;
  let dy = 0;
  for (let l = levels; l >= 0; l--) {
    dx *= 2;
    dy *= 2;
    const r = refPyr[l];
    const a = thresholdBitmap(r, p, noise);
    const b = thresholdBitmap(tgtPyr[l], p, noise);
    let best = Infinity;
    let bx = dx;
    let by = dy;
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const e = shiftError(a, b, r.width, r.height, dx + i, dy + j);
        if (e < best - 1e-9 || (Math.abs(e - best) <= 1e-9 && i === 0 && j === 0)) {
          best = e;
          bx = dx + i;
          by = dy + j;
        }
      }
    }
    dx = bx;
    dy = by;
  }
  return [dx, dy];
}

/** 全フレームのずれ量から、全フレームに共通して写っている領域を求める */
export function commonCrop(
  width: number,
  height: number,
  shifts: Array<[number, number]>,
): { x0: number; y0: number; width: number; height: number } {
  let x0 = 0;
  let y0 = 0;
  let x1 = width;
  let y1 = height;
  for (const [dx, dy] of shifts) {
    x0 = Math.max(x0, -dx);
    y0 = Math.max(y0, -dy);
    x1 = Math.min(x1, width - dx);
    y1 = Math.min(y1, height - dy);
  }
  if (x1 - x0 < 16 || y1 - y0 < 16) throw new Error('位置合わせ: 重なる領域が小さすぎます');
  return { x0, y0, width: x1 - x0, height: y1 - y0 };
}

/** ずれ量と共通領域から各フレームのビューを作る */
export function alignedViews(
  frames: Frame16[],
  shifts: Array<[number, number]>,
  crop: { x0: number; y0: number; width: number; height: number },
): View[] {
  return frames.map((frame, k) => ({
    frame,
    x0: crop.x0 + shifts[k][0],
    y0: crop.y0 + shifts[k][1],
    width: crop.width,
    height: crop.height,
  }));
}
