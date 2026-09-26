// Ward の MTB（Median Threshold Bitmap）法による平行移動の位置合わせ。
// 各画像を中央値で二値化するので露出が違うフレーム同士でも比較でき、手持ちのブラケット撮影に強い。

import { fullView, rowIndex, type Frame16 } from './frame';
import { angleDegrees, apply, fromLevel, scaleOf, toLevel, translation, type Similarity } from './transform';

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

/** 2×2 平均で 1/2 に縮小 */
export function half(g: Gray8): Gray8 {
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

// ---------------------------------------------------------------------------
// 精密な位置合わせ: 局所パッチの正規化相互相関 (NCC) で対応点を探し、相似変換を当てはめる。
// 手持ち撮影で生じる回転や 1 画素未満のずれも補正できる。NCC は明るさの倍率・オフセットに
// 影響されないので、露出の違うフレーム同士でもそのまま比較できる。

interface Match {
  x: number;
  y: number;
  u: number;
  v: number;
}

export interface RefineOptions {
  /** パッチを並べる数 [横, 縦] */
  grid: [number, number];
  /** パッチの一辺 [px] */
  patch: number;
  /** 予測位置からの探索範囲 ±[px] */
  search: number;
}

/** 対応点 (x, y) → (u, v) に最小二乗で相似変換を当てはめる */
export function fitSimilarity(m: Match[]): Similarity {
  let mx = 0;
  let my = 0;
  let mu = 0;
  let mv = 0;
  for (const p of m) {
    mx += p.x;
    my += p.y;
    mu += p.u;
    mv += p.v;
  }
  const n = m.length;
  mx /= n;
  my /= n;
  mu /= n;
  mv /= n;
  let sxx = 0;
  let sa = 0;
  let sb = 0;
  for (const p of m) {
    const x = p.x - mx;
    const y = p.y - my;
    const u = p.u - mu;
    const v = p.v - mv;
    sxx += x * x + y * y;
    sa += x * u + y * v;
    sb += x * v - y * u;
  }
  const a = sxx > 0 ? sa / sxx : 1;
  const b = sxx > 0 ? sb / sxx : 0;
  return { a, b, tx: mu - a * mx + b * my, ty: mv - b * mx - a * my };
}

/** 外れ値を除きながら相似変換を当てはめる。対応点が少なすぎれば null */
export function robustFit(m: Match[], minInliers = 6): Similarity | null {
  if (m.length < minInliers) return null;
  let inliers = m.map((_, i) => i);
  let t = fitSimilarity(m);
  for (let it = 0; it < 5; it++) {
    const res = m.map((p) => {
      const [u, v] = apply(t, p.x, p.y);
      return Math.hypot(u - p.u, v - p.v);
    });
    const sorted = inliers.map((i) => res[i]).sort((a, b) => a - b);
    const thr = Math.max(0.75, 3 * sorted[sorted.length >> 1]);
    const next = m.map((_, i) => i).filter((i) => res[i] <= thr);
    if (next.length < minInliers) break;
    const same = next.length === inliers.length;
    inliers = next;
    t = fitSimilarity(inliers.map((i) => m[i]));
    if (same) break;
  }
  return inliers.length >= minInliers ? t : null;
}

/**
 * init（基準 → 対象）の予測位置の周りでパッチを探して対応点を集め、相似変換を求め直す。
 * 座標はこの Gray8 の解像度のピクセル。
 */
export function refineSimilarity(ref: Gray8, tgt: Gray8, init: Similarity, opts: RefineOptions): Similarity | null {
  const P = opts.patch;
  const S = opts.search;
  const h0 = P >> 1;
  const margin = h0 + S + 2;
  const [gx, gy] = opts.grid;
  if (ref.width < margin * 4 || ref.height < margin * 4) return null;
  const n = P * P;
  const pr = new Float32Array(n);
  const scores = new Float64Array((2 * S + 1) * (2 * S + 1));
  const matches: Match[] = [];
  const rd = ref.data;
  const td = tgt.data;
  const rw = ref.width;
  const tw = tgt.width;

  for (let j = 0; j < gy; j++) {
    for (let i = 0; i < gx; i++) {
      const cx = Math.round(margin + ((i + 0.5) * (ref.width - 2 * margin)) / gx);
      const cy = Math.round(margin + ((j + 0.5) * (ref.height - 2 * margin)) / gy);
      // 基準パッチ（平坦・白飛び・黒つぶれの多い場所は使わない）
      let sum = 0;
      let bad = 0;
      let k = 0;
      for (let y = cy - h0; y < cy - h0 + P; y++) {
        const r = y * rw;
        for (let x = cx - h0; x < cx - h0 + P; x++, k++) {
          const val = rd[r + x];
          pr[k] = val;
          sum += val;
          if (val >= 250 || val <= 4) bad++;
        }
      }
      if (bad > n * 0.15) continue;
      const mean = sum / n;
      let norm = 0;
      for (k = 0; k < n; k++) {
        pr[k] -= mean;
        norm += pr[k] * pr[k];
      }
      if (norm < n * 6) continue; // 標準偏差がおよそ 2.5 未満
      norm = Math.sqrt(norm);

      const [px, py] = apply(init, cx, cy);
      const ix = Math.round(px);
      const iy = Math.round(py);
      if (ix - h0 - S < 0 || iy - h0 - S < 0 || ix - h0 + P + S > tw || iy - h0 + P + S > tgt.height) continue;

      let best = -2;
      let bx = 0;
      let by = 0;
      for (let dy = -S; dy <= S; dy++) {
        for (let dx = -S; dx <= S; dx++) {
          let sq = 0;
          let sqq = 0;
          let cross = 0;
          k = 0;
          const oy = iy + dy - h0;
          const ox = ix + dx - h0;
          for (let y = 0; y < P; y++) {
            const r = (oy + y) * tw + ox;
            for (let x = 0; x < P; x++, k++) {
              const q = td[r + x];
              sq += q;
              sqq += q * q;
              cross += pr[k] * q;
            }
          }
          const varq = sqq - (sq * sq) / n;
          const score = varq > 1e-6 ? cross / (norm * Math.sqrt(varq)) : -1;
          scores[(dy + S) * (2 * S + 1) + dx + S] = score;
          if (score > best) {
            best = score;
            bx = dx;
            by = dy;
          }
        }
      }
      // 相関が弱い・探索範囲の端に張り付いた結果は信用しない
      if (best < 0.6 || Math.abs(bx) === S || Math.abs(by) === S) continue;
      const W = 2 * S + 1;
      const at = (dx: number, dy: number) => scores[(dy + S) * W + dx + S];
      const c0 = at(bx, by);
      const sub = (m1: number, p1: number) => {
        const den = m1 - 2 * c0 + p1;
        return den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (m1 - p1)) / den)) : 0;
      };
      const ox = sub(at(bx - 1, by), at(bx + 1, by));
      const oy = sub(at(bx, by - 1), at(bx, by + 1));
      matches.push({ x: cx, y: cy, u: ix + bx + ox, v: iy + by + oy });
    }
  }
  return robustFit(matches);
}

/** 変換が手持ち撮影として妥当な範囲か */
function plausible(t: Similarity, coarse: Similarity, w: number, h: number): boolean {
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const [u1, v1] = apply(t, cx, cy);
  const [u2, v2] = apply(coarse, cx, cy);
  return (
    Math.abs(angleDegrees(t)) < 5 &&
    Math.abs(scaleOf(t) - 1) < 0.03 &&
    Math.hypot(u1 - u2, v1 - v2) < Math.max(w, h) * 0.03
  );
}

export interface AlignResult {
  transform: Similarity;
  /** 精密化に成功したか（false なら平行移動のみ） */
  precise: boolean;
}

/**
 * target を reference に合わせる相似変換（基準の座標 → target の座標、フル解像度）を求める。
 * 1) 1/2 解像度の MTB で大まかな平行移動を求め、
 * 2) 1/8 → 1/4 → 1/2 → 等倍の順にパッチ照合で回転と 1 画素未満のずれまで追い込む。
 */
export function alignFrames(reference: Gray8, target: Gray8, maxShift = 128): AlignResult {
  const refPyr: Gray8[] = [reference];
  const tgtPyr: Gray8[] = [target];
  for (let i = 0; i < 3; i++) {
    refPyr.push(half(refPyr[i]));
    tgtPyr.push(half(tgtPyr[i]));
  }
  const [mx, my] = alignMTB(refPyr[1], tgtPyr[1], Math.ceil(maxShift / 2));
  const coarse = fromLevel(translation(mx, my), 2);
  let t = coarse;
  let precise = false;
  const stages: Array<[number, RefineOptions]> = [
    [3, { grid: [10, 7], patch: 16, search: 5 }],
    [2, { grid: [12, 8], patch: 24, search: 2 }],
    [1, { grid: [14, 10], patch: 32, search: 1 }],
    [0, { grid: [16, 11], patch: 40, search: 1 }],
  ];
  for (const [level, opts] of stages) {
    const s = 1 << level;
    const r = refineSimilarity(refPyr[level], tgtPyr[level], toLevel(t, s), opts);
    if (!r) continue;
    const cand = fromLevel(r, s);
    if (plausible(cand, coarse, reference.width, reference.height)) {
      t = cand;
      precise = true;
    }
  }
  return { transform: t, precise };
}
