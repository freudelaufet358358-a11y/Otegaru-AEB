// 位置合わせ用の相似変換（回転・平行移動・わずかな拡大縮小）と、それに沿った画像の再サンプリング。
// 座標はフル解像度のピクセル単位で、画素の中心を整数とする。

import type { Frame16 } from './frame';

/** 基準画像の座標 (x, y) → フレームの座標 (a·x − b·y + tx, b·x + a·y + ty) */
export interface Similarity {
  a: number;
  b: number;
  tx: number;
  ty: number;
}

export interface Rect {
  x0: number;
  y0: number;
  width: number;
  height: number;
}

export const IDENTITY: Similarity = { a: 1, b: 0, tx: 0, ty: 0 };

export function translation(tx: number, ty: number): Similarity {
  return { a: 1, b: 0, tx, ty };
}

export function apply(t: Similarity, x: number, y: number): [number, number] {
  return [t.a * x - t.b * y + t.tx, t.b * x + t.a * y + t.ty];
}

/** outer ∘ inner（inner を先に適用） */
export function compose(outer: Similarity, inner: Similarity): Similarity {
  return {
    a: outer.a * inner.a - outer.b * inner.b,
    b: outer.b * inner.a + outer.a * inner.b,
    tx: outer.a * inner.tx - outer.b * inner.ty + outer.tx,
    ty: outer.b * inner.tx + outer.a * inner.ty + outer.ty,
  };
}

export function invert(t: Similarity): Similarity {
  const d = t.a * t.a + t.b * t.b;
  const a = t.a / d;
  const b = -t.b / d;
  return { a, b, tx: -(a * t.tx - b * t.ty), ty: -(b * t.tx + a * t.ty) };
}

/** 点 (cx, cy) を中心に angle [rad] 回転し、そのあと (dx, dy) 動かす */
export function rotationAbout(angle: number, cx: number, cy: number, dx = 0, dy = 0): Similarity {
  const a = Math.cos(angle);
  const b = Math.sin(angle);
  return { a, b, tx: cx - (a * cx - b * cy) + dx, ty: cy - (b * cx + a * cy) + dy };
}

/** 縮小画像（1 画素 = フル解像度の s 画素を平均）上の座標系に変換する */
export function toLevel(t: Similarity, s: number): Similarity {
  const o = (s - 1) / 2;
  const S: Similarity = { a: s, b: 0, tx: o, ty: o };
  return compose(invert(S), compose(t, S));
}

export function fromLevel(t: Similarity, s: number): Similarity {
  const o = (s - 1) / 2;
  const S: Similarity = { a: s, b: 0, tx: o, ty: o };
  return compose(S, compose(t, invert(S)));
}

export function angleDegrees(t: Similarity): number {
  return (Math.atan2(t.b, t.a) * 180) / Math.PI;
}

export function scaleOf(t: Similarity): number {
  return Math.hypot(t.a, t.b);
}

/**
 * ほぼ整数画素の平行移動か（再サンプリングを省略できるか）。
 * 既定の許容値は、6000px の画像の端でも 0.02px 未満の誤差に収まる大きさ。
 */
export function isIntegerTranslation(t: Similarity, tolerance = 0.02, size = 8000): boolean {
  const rot = tolerance / size;
  return (
    Math.abs(t.a - 1) < rot &&
    Math.abs(t.b) < rot &&
    Math.abs(t.tx - Math.round(t.tx)) < tolerance &&
    Math.abs(t.ty - Math.round(t.ty)) < tolerance
  );
}

/**
 * 基準座標系で、すべてのフレームに画素が存在する軸平行の矩形を求める。
 * 回転がある場合は各辺を少しずつ内側に寄せていく。
 */
export function validCrop(width: number, height: number, transforms: Similarity[]): Rect {
  let l = 0;
  let r = width - 1;
  let t = 0;
  let b = height - 1;
  const N = 16;
  // 左右の辺は横方向、上下の辺は縦方向のはみ出しだけを見る（回転が小さい前提）
  const overH = (x: number, y: number) => {
    let v = 0;
    for (const tr of transforms) {
      const u = tr.a * x - tr.b * y + tr.tx;
      v = Math.max(v, -u, u - (width - 1));
    }
    return v;
  };
  const overV = (x: number, y: number) => {
    let v = 0;
    for (const tr of transforms) {
      const w = tr.b * x + tr.a * y + tr.ty;
      v = Math.max(v, -w, w - (height - 1));
    }
    return v;
  };
  const eps = 1e-6;
  for (let iter = 0; iter < 200; iter++) {
    let vl = 0;
    let vr = 0;
    let vt = 0;
    let vb = 0;
    for (let i = 0; i <= N; i++) {
      const x = l + ((r - l) * i) / N;
      const y = t + ((b - t) * i) / N;
      vl = Math.max(vl, overH(l, y));
      vr = Math.max(vr, overH(r, y));
      vt = Math.max(vt, overV(x, t));
      vb = Math.max(vb, overV(x, b));
    }
    if (vl <= eps && vr <= eps && vt <= eps && vb <= eps) break;
    if (vl > eps) l += vl + 1e-4;
    if (vr > eps) r -= vr + 1e-4;
    if (vt > eps) t += vt + 1e-4;
    if (vb > eps) b -= vb + 1e-4;
    if (r - l < 16 || b - t < 16) throw new Error('位置合わせ: 重なる領域が小さすぎます');
  }
  const x0 = Math.ceil(l - 1e-3);
  const y0 = Math.ceil(t - 1e-3);
  const x1 = Math.floor(r + 1e-3);
  const y1 = Math.floor(b + 1e-3);
  if (x1 - x0 < 16 || y1 - y0 < 16) throw new Error('位置合わせ: 重なる領域が小さすぎます');
  return { x0, y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
}

/**
 * フレームを基準座標の crop 領域に合わせて描き直す（outW × outH）。
 * src はフル解像度 fullW × fullH のフレームそのもの、または縮小したもの（縦横の比率から自動で判断）。
 * cubic = true で Catmull-Rom（書き出し用）、false でバイリニア（プレビュー用）。
 */
export function warp(
  src: Frame16,
  fullW: number,
  fullH: number,
  t: Similarity,
  crop: Rect,
  outW: number,
  outH: number,
  cubic: boolean,
): Frame16 {
  const sw = src.width;
  const sh = src.height;
  const sX = sw / fullW;
  const sY = sh / fullH;
  const kx = crop.width / outW;
  const ky = crop.height / outH;
  const d = src.data;
  const out = new Uint16Array(outW * outH * 3);
  const rowStride = sw * 3;
  const px = new Float64Array(3);
  for (let v = 0; v < outH; v++) {
    const Y = crop.y0 - 0.5 + (v + 0.5) * ky;
    const X0 = crop.x0 - 0.5 + 0.5 * kx;
    // 出力の 1 行は元画像上の直線になる
    let fx = t.a * X0 - t.b * Y + t.tx;
    let fy = t.b * X0 + t.a * Y + t.ty;
    const stepX = t.a * kx;
    const stepY = t.b * kx;
    let o = v * outW * 3;
    for (let u = 0; u < outW; u++, fx += stepX, fy += stepY, o += 3) {
      const sx = (fx + 0.5) * sX - 0.5;
      const sy = (fy + 0.5) * sY - 0.5;
      if (cubic) sampleCubic(d, sw, sh, rowStride, sx, sy, px);
      else sampleBilinear(d, sw, sh, rowStride, sx, sy, px);
      out[o] = clamp16(px[0]);
      out[o + 1] = clamp16(px[1]);
      out[o + 2] = clamp16(px[2]);
    }
  }
  return { width: outW, height: outH, data: out, encoding: src.encoding };
}

function clamp16(v: number): number {
  const x = v + 0.5;
  return x <= 0 ? 0 : x >= 65535 ? 65535 : x | 0;
}

export function sampleBilinear(
  d: Uint16Array,
  w: number,
  h: number,
  stride: number,
  sx: number,
  sy: number,
  out: Float64Array,
): void {
  let x0 = Math.floor(sx);
  let y0 = Math.floor(sy);
  const fx = sx - x0;
  const fy = sy - y0;
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  if (x0 < 0) x0 = 0;
  else if (x0 > w - 1) x0 = w - 1;
  if (x1 < 0) x1 = 0;
  else if (x1 > w - 1) x1 = w - 1;
  if (y0 < 0) y0 = 0;
  else if (y0 > h - 1) y0 = h - 1;
  if (y1 < 0) y1 = 0;
  else if (y1 > h - 1) y1 = h - 1;
  const i00 = y0 * stride + x0 * 3;
  const i01 = y0 * stride + x1 * 3;
  const i10 = y1 * stride + x0 * 3;
  const i11 = y1 * stride + x1 * 3;
  const w00 = (1 - fx) * (1 - fy);
  const w01 = fx * (1 - fy);
  const w10 = (1 - fx) * fy;
  const w11 = fx * fy;
  for (let c = 0; c < 3; c++) out[c] = d[i00 + c] * w00 + d[i01 + c] * w01 + d[i10 + c] * w10 + d[i11 + c] * w11;
}

const wx = new Float64Array(4);
const wy = new Float64Array(4);
const ix = new Int32Array(4);
const iy = new Int32Array(4);

function catmullRom(t: number, w: Float64Array): void {
  const t2 = t * t;
  const t3 = t2 * t;
  w[0] = -0.5 * t3 + t2 - 0.5 * t;
  w[1] = 1.5 * t3 - 2.5 * t2 + 1;
  w[2] = -1.5 * t3 + 2 * t2 + 0.5 * t;
  w[3] = 0.5 * t3 - 0.5 * t2;
}

export function sampleCubic(
  d: Uint16Array,
  w: number,
  h: number,
  stride: number,
  sx: number,
  sy: number,
  out: Float64Array,
): void {
  const x0 = Math.floor(sx);
  const y0 = Math.floor(sy);
  catmullRom(sx - x0, wx);
  catmullRom(sy - y0, wy);
  for (let k = 0; k < 4; k++) {
    const xx = x0 - 1 + k;
    const yy = y0 - 1 + k;
    ix[k] = (xx < 0 ? 0 : xx > w - 1 ? w - 1 : xx) * 3;
    iy[k] = (yy < 0 ? 0 : yy > h - 1 ? h - 1 : yy) * stride;
  }
  let r = 0;
  let g = 0;
  let b = 0;
  for (let j = 0; j < 4; j++) {
    const row = iy[j];
    const wj = wy[j];
    for (let k = 0; k < 4; k++) {
      const i = row + ix[k];
      const wt = wj * wx[k];
      r += d[i] * wt;
      g += d[i + 1] * wt;
      b += d[i + 2] * wt;
    }
  }
  out[0] = r;
  out[1] = g;
  out[2] = b;
}
