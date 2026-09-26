// Mertens らの露出フュージョン（Exposure Fusion, 2007）。
// 各フレームの「コントラスト・彩度・適正露出」から画素ごとの重みを求め、
// ラプラシアンピラミッド上でブレンドする。トーンマッピング不要で自然な仕上がりになる。
//
// 2400 万画素 × 複数枚でもブラウザで動くよう、メモリを節約した構成にしている:
//  - フレームは 16bit のまま保持し、必要なチャンネルだけ都度 Float32 に展開する
//  - 正規化済みの重みピラミッドは 16bit に量子化してフレームごとに保持する
//  - 色チャンネルは 1 つずつ処理し、結果をコールバックで受け渡す

import { extractChannel, rowIndex, type View } from './frame';
import {
  allocLevels,
  buildGaussian,
  collapse,
  gaussianToLaplacian,
  levelCount,
  scratchSize,
  type Level,
} from './pyramid';

export interface FusionWeights {
  /** コントラスト（細部）の重み指数 */
  contrast: number;
  /** 彩度の重み指数 */
  saturation: number;
  /** 適正露出（中間調に近いほど高い）の重み指数 */
  exposure: number;
  /** ディテール（局所コントラスト）の倍率。1 で標準 */
  detail: number;
}

export const DEFAULT_FUSION_WEIGHTS: FusionWeights = { contrast: 1, saturation: 1, exposure: 1, detail: 1 };

/**
 * ディテール強調をかけるピラミッドの段。画像サイズに対する相対的な大きさで決めるので、
 * プレビューと書き出しで見た目がそろう（長辺の 1/800 〜 1/24 程度の模様を強調）。
 */
export function detailLevels(w: number, h: number, levels: number): number[] {
  const side = Math.max(w, h);
  const out: number[] = [];
  for (let l = 0; l < levels - 1; l++) {
    const scale = 1 << l;
    if (scale >= side / 800 && scale <= side / 24) out.push(l);
  }
  return out;
}

const SIGMA = 0.2;
const EPS = 1e-12;

/** フレーム k の重みマップ（未正規化）を out に計算する。gray は作業用 */
export function computeWeightMap(
  v: View,
  lut: Float32Array,
  p: FusionWeights,
  gray: Float32Array,
  out: Float32Array,
): void {
  const w = v.width;
  const h = v.height;
  const d = v.frame.data;
  const k = p.exposure / (2 * SIGMA * SIGMA);
  const ws = p.saturation;
  const wc = p.contrast;
  let o = 0;
  for (let y = 0; y < h; y++) {
    let i = rowIndex(v, y);
    for (let x = 0; x < w; x++, i += 3, o++) {
      const r = lut[d[i]];
      const g = lut[d[i + 1]];
      const b = lut[d[i + 2]];
      const mu = (r + g + b) / 3;
      gray[o] = mu;
      let wv = 1;
      if (k !== 0) {
        const e = (r - 0.5) * (r - 0.5) + (g - 0.5) * (g - 0.5) + (b - 0.5) * (b - 0.5);
        wv = Math.exp(-e * k);
      }
      if (ws !== 0) {
        const sat = Math.sqrt(((r - mu) * (r - mu) + (g - mu) * (g - mu) + (b - mu) * (b - mu)) / 3);
        wv *= ws === 1 ? sat : Math.pow(sat, ws);
      }
      out[o] = wv;
    }
  }
  // コントラスト: グレースケールのラプラシアンの絶対値（端は複製）
  for (let y = 0; y < h; y++) {
    const r = y * w;
    const up = (y > 0 ? y - 1 : 0) * w;
    const dn = (y < h - 1 ? y + 1 : y) * w;
    for (let x = 0; x < w; x++) {
      const c = gray[r + x];
      const l = gray[r + (x > 0 ? x - 1 : 0)];
      const rt = gray[r + (x < w - 1 ? x + 1 : x)];
      let lap = Math.abs(4 * c - l - rt - gray[up + x] - gray[dn + x]);
      if (wc !== 1) lap = wc === 0 ? 1 : Math.pow(lap, wc);
      out[r + x] = out[r + x] * lap + EPS;
    }
  }
}

export type ProgressFn = (fraction: number) => void;

/**
 * 露出フュージョンを実行する。結果は表示用 (sRGB) の値で、チャンネルごとに onChannel に渡される。
 * onChannel に渡す配列は内部で再利用されるため、必要ならコピーすること。
 */
export function exposureFusion(
  views: View[],
  luts: Float32Array[],
  params: FusionWeights,
  onChannel: (c: number, plane: Float32Array) => void,
  progress: ProgressFn = () => {},
): void {
  const n = views.length;
  const w = views[0].width;
  const h = views[0].height;
  const size = w * h;
  const nl = levelCount(w, h);
  const tmp = new Float32Array(scratchSize(w, h));
  const total = n * 2 + n * 3;
  let step = 0;
  const tick = () => progress(++step / total);

  // 1) 重みの総和
  let gray: Float32Array | null = new Float32Array(size);
  let sum: Float32Array | null = new Float32Array(size);
  const work = allocLevels(w, h, nl); // work[0] は重み／チャンネル展開用
  for (let k = 0; k < n; k++) {
    computeWeightMap(views[k], luts[k], params, gray, work[0].data);
    const wd = work[0].data;
    for (let i = 0; i < size; i++) sum[i] += wd[i];
    tick();
  }

  // 2) 正規化した重みのガウシアンピラミッドを 16bit で保持
  const weightPyramids: Uint16Array[][] = [];
  for (let k = 0; k < n; k++) {
    computeWeightMap(views[k], luts[k], params, gray, work[0].data);
    const wd = work[0].data;
    for (let i = 0; i < size; i++) wd[i] /= sum[i];
    buildGaussian(work, tmp);
    weightPyramids.push(work.map((lv) => quantize(lv.data)));
    tick();
  }
  gray = null;
  sum = null;

  // 3) チャンネルごとにラプラシアンピラミッドをブレンドして再構成
  const result = allocLevels(w, h, nl);
  const boost = detailLevels(w, h, nl);
  for (let c = 0; c < 3; c++) {
    for (const lv of result) lv.data.fill(0);
    for (let k = 0; k < n; k++) {
      extractChannel(views[k], c, luts[k], work[0].data);
      buildGaussian(work, tmp);
      gaussianToLaplacian(work, tmp);
      accumulate(result, work, weightPyramids[k]);
      tick();
    }
    if (params.detail !== 1) {
      for (const l of boost) {
        const d = result[l].data;
        for (let i = 0; i < d.length; i++) d[i] *= params.detail;
      }
    }
    collapse(result, tmp);
    onChannel(c, result[0].data);
  }
}

const Q = 65535;

function quantize(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i] * Q + 0.5;
    out[i] = v <= 0 ? 0 : v >= Q ? Q : v | 0;
  }
  return out;
}

function accumulate(result: Level[], lap: Level[], weights: Uint16Array[]): void {
  const s = 1 / Q;
  for (let l = 0; l < result.length; l++) {
    const r = result[l].data;
    const g = lap[l].data;
    const wq = weights[l];
    for (let i = 0; i < r.length; i++) r[i] += wq[i] * s * g[i];
  }
}
