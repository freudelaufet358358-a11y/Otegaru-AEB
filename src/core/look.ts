// 「Leica M10」の色（カラールック）。合成・トーンマッピングを終えた表示用の画像（sRGB）に掛ける。
// Leica M10 と Canon の色の違いの調査（training/leica/）で求めたモデルを使う:
//
//  1. 表示用の値をシーンのリニア値に戻す（LeicaLook.toLinear を参照）
//     RAW: アプリの素の表示（sRGB + 肩特性）の逆 / JPEG: Canon のピクチャースタイル「スタンダード」の逆
//  2. センサーの違い: Canon の RAW（LibRaw・Adobe の色行列）→「Leica M10 で撮り、Leica が DNG に埋め込んだ色行列で現像した色」
//     分光感度のシミュレーションで求めた 3×3 行列で、撮影時の色温度に応じて昼光用と電球光用を補間する
//  3. Leica の JPEG エンジン: 3×3 行列（彩度をやや抑える）+ R・G・B に同じトーンカーブ
//     （Leica M10-R の RAW とカメラ内 JPEG の組から当てはめたもの）
//
// 中間グレー（18%）は Leica のカメラ内 JPEG と同じ明るさ（sRGB 118.9）に合わせる。
//
// 既定では明るさ（輝度）は元の画像のまま残し、色相・彩度だけを Leica のカメラ内 JPEG に合わせる。
// Leica の JPEG のトーンカーブは暗部を深く沈める（中間グレーの 4 段下で sRGB 8 程度）ので、そのまま掛けると
// HDR 合成で起こした暗部がまたつぶれてしまうため。tone を 1 にするとトーンカーブも Leica のものにする。
// 数式は training/leica/export_model.py の apply_look と 1 対 1 に対応させている。

import { fastLinearToSrgb, linearLut, linearToSrgb, LUMA_B, LUMA_G, LUMA_R, shoulderInverse, srgbToLinear, type Encoding } from './color';

export type LookId = 'none' | 'leica-m10';

export interface LookParams {
  id: LookId;
  /** 効き 0..1（0 で元のまま） */
  amount: number;
  /** 明るさ（トーンカーブ）も Leica のカメラ内 JPEG に合わせる */
  tone: boolean;
}

export const DEFAULT_LOOK: LookParams = { id: 'none', amount: 1, tone: false };

/** 3×3 行列（行優先） */
type Mat3 = Float64Array;

interface Engine {
  matrix: Mat3;
  /** log2(リニア値) を curveDomain で等間隔に区切った点での出力（sRGB 符号値） */
  curve: Float32Array;
}

interface SensorEntry {
  models: string[];
  /** この色温度 [K] 以下は電球光用、以上は昼光用の行列（間は色温度の逆数で補間） */
  cct: [number, number];
  tungsten: Mat3;
  daylight: Mat3;
}

export interface LookModel {
  curveDomain: [number, number];
  leica: Engine;
  canon: Engine;
  cameras: SensorEntry[];
  generic: SensorEntry;
}

/** 画像の出どころ。センサーの変換と、表示用の値をリニアに戻す方法を決める */
export interface LookSource {
  encoding: Encoding;
  make?: string;
  model?: string;
  /** カメラのニュートラル（撮影時のホワイトバランスの逆数、G = 1） */
  neutral?: [number, number, number];
  /** XYZ → カメラ RGB の色行列（D65、行優先 9 要素） */
  camXyz?: number[];
  /** 撮影時の色温度 [K] がわかっている場合（neutral と camXyz からの推定より優先する） */
  cct?: number;
}

/** 中間グレー（18%）の、カメラ内 JPEG での値 */
const MID_GREY = 118.9 / 255;

function mat(a: ArrayLike<number>): Mat3 {
  if (a.length !== 9) throw new Error('色の行列は 9 要素です');
  return Float64Array.from(a);
}

export function parseLookModel(json: unknown): LookModel {
  const j = json as {
    curve_domain: [number, number];
    leica: { matrix: number[]; curve: number[] };
    canon: { matrix: number[]; curve: number[] };
    sensor: { cameras: Array<{ models: string[]; cct: [number, number]; A: number[]; D: number[] }>; generic: { models: string[]; cct: [number, number]; A: number[]; D: number[] } };
  };
  const engine = (e: { matrix: number[]; curve: number[] }): Engine => ({ matrix: mat(e.matrix), curve: Float32Array.from(e.curve) });
  const sensor = (s: { models: string[]; cct: [number, number]; A: number[]; D: number[] }): SensorEntry => ({
    models: s.models,
    cct: s.cct,
    tungsten: mat(s.A),
    daylight: mat(s.D),
  });
  return {
    curveDomain: j.curve_domain,
    leica: engine(j.leica),
    canon: engine(j.canon),
    cameras: j.sensor.cameras.map(sensor),
    generic: sensor(j.sensor.generic),
  };
}

/**
 * 機種名の表記ゆれをそろえたキー。
 * 例: ('Canon', 'Canon EOS R6 Mark II') と ('Canon', 'Canon EOS R6m2') → 'canon:r6m2'
 */
export function cameraKey(make: string | undefined, model: string | undefined): string {
  const mk = make ? (make.trim().toLowerCase().split(' ')[0] ?? '') : '';
  let m = (model ?? '').toLowerCase();
  for (const w of [mk, 'eos']) if (w) m = m.split(w).join(' ');
  m = m.replace(/[^a-z0-9]/g, '');
  for (const [a, b] of [
    ['markiii', 'm3'],
    ['markiv', 'm4'],
    ['markii', 'm2'],
    ['mark3', 'm3'],
    ['mark4', 'm4'],
    ['mark2', 'm2'],
  ]) {
    m = m.split(a).join(b);
  }
  return `${mk}:${m}`;
}

/**
 * 撮影時の色温度 [K] を推定する: カメラのニュートラルを D65 の色行列で XYZ に戻し、McCamy の式で色温度にする。
 * （training/leica/common.py の libraw_cct と同じ）
 */
export function estimateCct(neutral: [number, number, number], camXyz: ArrayLike<number>): number | undefined {
  const m = mat(camXyz);
  const inv = invert3(m);
  if (!inv) return undefined;
  const n = [neutral[0] / neutral[1], 1, neutral[2] / neutral[1]];
  const X = inv[0] * n[0] + inv[1] * n[1] + inv[2] * n[2];
  const Y = inv[3] * n[0] + inv[4] * n[1] + inv[5] * n[2];
  const Z = inv[6] * n[0] + inv[7] * n[1] + inv[8] * n[2];
  const s = X + Y + Z;
  if (!(s > 0)) return undefined;
  const x = X / s;
  const y = Y / s;
  const k = (x - 0.332) / (0.1858 - y);
  const cct = 449 * k * k * k + 3525 * k * k + 6823.3 * k + 5520.33;
  return Number.isFinite(cct) && cct > 1000 && cct < 25000 ? cct : undefined;
}

/** 機種と色温度に合うセンサーの変換行列 */
export function sensorMatrix(model: LookModel, key: string | undefined, cct: number | undefined): { matrix: Mat3; matched: boolean } {
  const entry = (key && model.cameras.find((c) => c.models.includes(key))) || null;
  const e = entry ?? model.generic;
  const [lo, hi] = e.cct;
  let m: Mat3;
  if (cct === undefined || cct >= hi) m = e.daylight;
  else if (cct <= lo) m = e.tungsten;
  else {
    const g = (1 / cct - 1 / hi) / (1 / lo - 1 / hi);
    m = new Float64Array(9);
    for (let i = 0; i < 9; i++) m[i] = g * e.tungsten[i] + (1 - g) * e.daylight[i];
  }
  return { matrix: m, matched: entry !== null };
}

function mul3(a: Mat3, b: Mat3): Mat3 {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return o;
}

function invert3(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const r = 1 / det;
  return Float64Array.from([A * r, -(b * i - c * h) * r, (b * f - c * e) * r, B * r, (a * i - c * g) * r, -(a * f - c * d) * r, C * r, -(a * h - b * g) * r, (a * e - b * d) * r]);
}

/** トーンカーブ: リニア値 → sRGB 符号値（範囲外は 0 / 1） */
function curveEval(curve: Float32Array, domain: [number, number], v: number): number {
  if (!(v > 0)) return 0;
  const n = curve.length - 1;
  let t = ((Math.log2(v) - domain[0]) / (domain[1] - domain[0])) * n;
  if (t <= 0) return curve[0];
  if (t >= n) return curve[n];
  const i = t | 0;
  t -= i;
  return curve[i] + (curve[i + 1] - curve[i]) * t;
}

/** トーンカーブの逆: sRGB 符号値 → リニア値 */
function curveInverse(curve: Float32Array, domain: [number, number], y: number): number {
  const n = curve.length - 1;
  let lo = 1;
  let hi = n;
  // y 以上になる最初の点（numpy の searchsorted と同じ）を二分探索
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (curve[mid] < y) lo = mid + 1;
    else hi = mid;
  }
  const k = Math.min(Math.max(lo, 1), n);
  const y0 = curve[k - 1];
  const y1 = curve[k];
  const t = y1 > y0 ? (y - y0) / Math.max(y1 - y0, 1e-12) : 0;
  return Math.pow(2, domain[0] + ((k - 1 + t) * (domain[1] - domain[0])) / n);
}

/** Leica のトーンカーブを速く引くための表の細かさ（平方根で添字を取り、暗部を細かくする） */
const OUT_SIZE = 16384;

/** ある画像の出どころ（機種・ホワイトバランス・形式）に合わせて用意した Leica M10 の色 */
export class LeicaLook {
  /** 撮影時の色温度の推定値 [K]（わからなければ undefined） */
  readonly cct: number | undefined;
  /** 機種の分光感度データでセンサーの違いを変換したか（false なら色を正確に写すカメラとして近似） */
  readonly cameraMatched: boolean;
  /** 中間グレーが 0.18 になるリニア値 → Leica のトーンカーブの入力（センサーの変換・JPEG エンジンの行列・露出をまとめたもの） */
  readonly matrix: Mat3;
  private readonly encoding: Encoding;
  /** Canon のカメラ内 JPEG の中間グレーのリニア値（0.18 にそろえるため） */
  private readonly canonMid: number;
  private inLut: { jpegBase: number; lut: Float32Array } | null = null;
  /** Leica のトーンカーブ（sRGB 符号値）と、それをリニアにしたもの */
  private outLut: { encoded: Float32Array; linear: Float32Array } | null = null;
  private readonly vMax: number;

  constructor(
    private readonly model: LookModel,
    src: LookSource,
  ) {
    this.encoding = src.encoding;
    this.cct = src.cct ?? (src.neutral && src.camXyz ? estimateCct(src.neutral, src.camXyz) : undefined);
    const s = sensorMatrix(model, src.make || src.model ? cameraKey(src.make, src.model) : undefined, this.cct);
    this.cameraMatched = s.matched;
    const { leica, canon, curveDomain } = model;
    let m = mul3(leica.matrix, s.matrix);
    if (src.encoding === 'srgb') {
      const inv = invert3(canon.matrix);
      if (!inv) throw new Error('色の行列が正しくありません');
      m = mul3(m, inv);
    }
    const scale = curveInverse(leica.curve, curveDomain, MID_GREY) / 0.18;
    for (let i = 0; i < 9; i++) m[i] *= scale;
    this.matrix = m;
    this.canonMid = curveInverse(canon.curve, curveDomain, MID_GREY);
    this.vMax = Math.pow(2, curveDomain[1]);
  }

  /**
   * 表示用の値 (0..1) → シーンのリニア値（中間グレー = 0.18）。
   * - RAW: アプリの素の表示（sRGB + 肩特性）の逆。基準フレームや、ナチュラル・HDR の合成結果はこれ
   * - jpegBase (0..1) の割合で「カメラ内 JPEG 並みのトーンカーブの逆」を混ぜる。おまかせの結果は HDR+ の仕上がり
   *   （カメラ内 JPEG 並みのメリハリ）を学習しているので、素の表示として戻すとメリハリが二重にかかる
   * - JPEG などの写真: Canon のピクチャースタイル「スタンダード」のトーンカーブの逆（色の行列の逆は matrix に含む）
   */
  toLinear(d: number, jpegBase = 0): number {
    const jpeg = () => (curveInverse(this.model.canon.curve, this.model.curveDomain, Math.min(1, Math.max(0, d))) * 0.18) / this.canonMid;
    if (this.encoding === 'srgb') return jpeg();
    const x = shoulderInverse(srgbToLinear(d));
    return jpegBase > 0 ? x + (jpeg() - x) * Math.min(1, jpegBase) : x;
  }

  /**
   * 1 画素を変換する（表示用 sRGB 0..1 → Leica の色 0..1）。検証用の正確な計算。
   * tone (0..1): Leica のトーンカーブの割合。0 なら明るさ（輝度）は元のまま
   */
  pixel(r: number, g: number, b: number, out: Float64Array | number[], jpegBase = 0, tone = 0): void {
    const x0 = this.toLinear(r, jpegBase);
    const x1 = this.toLinear(g, jpegBase);
    const x2 = this.toLinear(b, jpegBase);
    const m = this.matrix;
    const { curve } = this.model.leica;
    const dom = this.model.curveDomain;
    const o0 = curveEval(curve, dom, m[0] * x0 + m[1] * x1 + m[2] * x2);
    const o1 = curveEval(curve, dom, m[3] * x0 + m[4] * x1 + m[5] * x2);
    const o2 = curveEval(curve, dom, m[6] * x0 + m[7] * x1 + m[8] * x2);
    const yi = LUMA_R * srgbToLinear(r) + LUMA_G * srgbToLinear(g) + LUMA_B * srgbToLinear(b);
    keepLuminance(srgbToLinear(o0), srgbToLinear(o1), srgbToLinear(o2), yi, out, linearToSrgb);
    out[0] += (o0 - out[0]) * tone;
    out[1] += (o1 - out[1]) * tone;
    out[2] += (o2 - out[2]) * tone;
  }

  private tables(jpegBase: number): { inLut: Float32Array; encoded: Float32Array; linear: Float32Array } {
    if (!this.inLut || this.inLut.jpegBase !== jpegBase) {
      const lut = new Float32Array(65536);
      for (let i = 0; i < 65536; i++) lut[i] = this.toLinear(i / 65535, jpegBase);
      this.inLut = { jpegBase, lut };
    }
    if (!this.outLut) {
      const encoded = new Float32Array(OUT_SIZE + 1);
      const linear = new Float32Array(OUT_SIZE + 1);
      const { curve } = this.model.leica;
      for (let i = 0; i <= OUT_SIZE; i++) {
        const u = i / OUT_SIZE;
        encoded[i] = curveEval(curve, this.model.curveDomain, u * u * this.vMax);
        linear[i] = srgbToLinear(encoded[i]);
      }
      this.outLut = { encoded, linear };
    }
    return { inLut: this.inLut.lut, ...this.outLut };
  }

  /**
   * 表示用 16bit RGB（インターリーブ）に掛ける。amount (0..1) で元の画像と混ぜる。
   * jpegBase は toLinear、tone は pixel を参照。out を省略すると src を書き換える。
   */
  apply(src: Uint16Array, amount: number, out: Uint16Array = src, jpegBase = 0, tone = 0): Uint16Array {
    const k = Math.min(1, Math.max(0, amount));
    if (k === 0) {
      if (out !== src) out.set(src);
      return out;
    }
    const { inLut, encoded, linear } = this.tables(Math.round(Math.min(1, Math.max(0, jpegBase)) * 100) / 100);
    const srgb = linearLut('srgb');
    const t = Math.min(1, Math.max(0, tone));
    const m = this.matrix;
    const scale = OUT_SIZE / Math.sqrt(this.vMax);
    const lookup = (table: Float32Array, x: number): number => {
      if (!(x > 0)) return table[0];
      const f = Math.sqrt(x) * scale;
      if (f >= OUT_SIZE) return table[OUT_SIZE];
      const i = f | 0;
      return table[i] + (table[i + 1] - table[i]) * (f - i);
    };
    const q = new Float64Array(3);
    for (let i = 0; i < src.length; i += 3) {
      const r = src[i];
      const g = src[i + 1];
      const b = src[i + 2];
      const x0 = inLut[r];
      const x1 = inLut[g];
      const x2 = inLut[b];
      const y0 = m[0] * x0 + m[1] * x1 + m[2] * x2;
      const y1 = m[3] * x0 + m[4] * x1 + m[5] * x2;
      const y2 = m[6] * x0 + m[7] * x1 + m[8] * x2;
      let o0: number;
      let o1: number;
      let o2: number;
      if (t < 1) {
        const yi = LUMA_R * srgb[r] + LUMA_G * srgb[g] + LUMA_B * srgb[b];
        keepLuminance(lookup(linear, y0), lookup(linear, y1), lookup(linear, y2), yi, q, fastLinearToSrgb);
        o0 = q[0];
        o1 = q[1];
        o2 = q[2];
        if (t > 0) {
          o0 += (lookup(encoded, y0) - o0) * t;
          o1 += (lookup(encoded, y1) - o1) * t;
          o2 += (lookup(encoded, y2) - o2) * t;
        }
      } else {
        o0 = lookup(encoded, y0);
        o1 = lookup(encoded, y1);
        o2 = lookup(encoded, y2);
      }
      out[i] = r + (o0 * 65535 - r) * k + 0.5;
      out[i + 1] = g + (o1 * 65535 - g) * k + 0.5;
      out[i + 2] = b + (o2 * 65535 - b) * k + 0.5;
    }
    return out;
  }
}

/**
 * 色（色度）はそのままで、輝度（リニア）を yi にそろえて sRGB 符号値にする。
 * 明るくしてはみ出す色は、輝度を保ったまま彩度を落として収める。
 */
function keepLuminance(l0: number, l1: number, l2: number, yi: number, out: Float64Array | number[], encode: (v: number) => number): void {
  const yo = LUMA_R * l0 + LUMA_G * l1 + LUMA_B * l2;
  let p0 = yi;
  let p1 = yi;
  let p2 = yi;
  if (yo > 1e-6) {
    const s = yi / yo;
    p0 = l0 * s;
    p1 = l1 * s;
    p2 = l2 * s;
    const mx = p0 > p1 ? (p0 > p2 ? p0 : p2) : p1 > p2 ? p1 : p2;
    if (mx > 1) {
      const f = mx > yi ? (1 - yi) / (mx - yi) : 0;
      p0 = yi + (p0 - yi) * f;
      p1 = yi + (p1 - yi) * f;
      p2 = yi + (p2 - yi) * f;
    }
  }
  out[0] = encode(p0);
  out[1] = encode(p1);
  out[2] = encode(p2);
}
