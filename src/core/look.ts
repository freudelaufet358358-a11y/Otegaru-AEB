// 「Leica M10」の色（カラールック）。表示用の画像（sRGB）に掛ける: 1 枚の写真（RAW はアプリの素の表示、
// JPEG などはそのまま）か、AEB の合成・トーンマッピングを終えた結果。
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
// 当てはめに使った実写（Leica は庭、Canon は素焼きの置物）には青空のような明るく鮮やかな青がないので、
// その外側で行き過ぎないよう 2 つの手当てをする:
//  - 白飛びした値はトーンカーブの逆を白の点で止める（カーブの肩は平らなので、そのまま戻すと何段も跳ね上がり、
//    行列で他のチャンネルが負になって真っ青緑になる）
//  - R・G・B に同じトーンカーブを別々に掛けると、明るく鮮やかな色では一番明るいチャンネルだけが肩で圧縮されて
//    色相がずれる（青空が水色〜青緑になる）。最大のチャンネルが中間グレーの 1〜2.5 段上にかけて、
//    色相をカーブを掛ける前の色相に戻していく（中間調の色はそのまま）
//
// tone が 0 なら明るさ（輝度）は元の画像のまま残し、色相・彩度だけを Leica のカメラ内 JPEG に合わせる。
// Leica の JPEG のトーンカーブは暗部を深く沈める（中間グレーの 4 段下で sRGB 8 程度）ので、そのまま掛けると
// HDR 合成で起こした暗部がまたつぶれてしまうため（アプリでは AEB の合成結果の既定）。
// tone を 1 にするとトーンカーブも Leica のものにする（アプリでは 1 枚の写真の既定）。
// 数式は training/leica/export_model.py の apply_look と 1 対 1 に対応させている。

import {
  fastLinearToSrgb,
  linearLut,
  linearToSrgb,
  LUMA_B,
  LUMA_G,
  LUMA_R,
  RAW_DISPLAY_GAIN,
  shoulderInverse,
  srgbToLinear,
  type Encoding,
} from './color';

export interface LookParams {
  /** 効き 0..1（0 で元のまま） */
  amount: number;
  /** 明るさ（トーンカーブ）も Leica のカメラ内 JPEG に合わせる */
  tone: boolean;
}

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
/** カメラ内 JPEG でこれ以上の値は白飛びとみなす（トーンカーブの逆をここで止める） */
const WHITE = 0.995;
/** 最大のチャンネルが中間グレーの何段上から何段上までで、色相を保つ割合を 0 → 1 にするか */
const HUE_EV: [number, number] = [1, 2.5];

function smoothstep(t: number): number {
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}

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
  /** Canon のカメラ内 JPEG が白（WHITE）に届くリニア値 */
  private readonly canonWhite: number;
  /** Leica のトーンカーブの入力で、中間グレーになる値 */
  private readonly leicaMid: number;
  private inLut: { jpegBase: number; lut: Float32Array } | null = null;
  /** Leica のトーンカーブ（sRGB 符号値）と、それをリニアにしたもの、明るさごとの色相を保つ割合 */
  private outLut: { encoded: Float32Array; linear: Float32Array; hue: Float32Array } | null = null;
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
    this.leicaMid = curveInverse(leica.curve, curveDomain, MID_GREY);
    const scale = this.leicaMid / 0.18;
    for (let i = 0; i < 9; i++) m[i] *= scale;
    this.matrix = m;
    this.canonMid = curveInverse(canon.curve, curveDomain, MID_GREY);
    this.canonWhite = curveInverse(canon.curve, curveDomain, WHITE);
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
    const { canon, curveDomain } = this.model;
    // 白飛びした値は白の点で止める（RAW は白飛びした画素の値 = RAW_DISPLAY_GAIN）
    const jpeg = () => (Math.min(curveInverse(canon.curve, curveDomain, Math.min(1, Math.max(0, d))), this.canonWhite) * 0.18) / this.canonMid;
    if (this.encoding === 'srgb') return jpeg();
    const x = Math.min(shoulderInverse(srgbToLinear(d)), RAW_DISPLAY_GAIN);
    return jpegBase > 0 ? x + (jpeg() - x) * Math.min(1, jpegBase) : x;
  }

  /** 色相をカーブを掛ける前の色相に戻す割合（トーンカーブの入力の最大のチャンネル ymax から） */
  hueWeight(ymax: number): number {
    return smoothstep((Math.log2(Math.max(ymax, 1e-30) / this.leicaMid) - HUE_EV[0]) / (HUE_EV[1] - HUE_EV[0]));
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
    const y0 = m[0] * x0 + m[1] * x1 + m[2] * x2;
    const y1 = m[3] * x0 + m[4] * x1 + m[5] * x2;
    const y2 = m[6] * x0 + m[7] * x1 + m[8] * x2;
    const o = [curveEval(curve, dom, y0), curveEval(curve, dom, y1), curveEval(curve, dom, y2)];
    const l = Float64Array.from(o, srgbToLinear);
    const mid = keepHue(y0, y1, y2, l, this.hueWeight(Math.max(y0, y1, y2)));
    if (mid >= 0) o[mid] = linearToSrgb(l[mid]);
    const yi = LUMA_R * srgbToLinear(r) + LUMA_G * srgbToLinear(g) + LUMA_B * srgbToLinear(b);
    keepLuminance(l[0], l[1], l[2], yi, out, linearToSrgb);
    out[0] += (o[0] - out[0]) * tone;
    out[1] += (o[1] - out[1]) * tone;
    out[2] += (o[2] - out[2]) * tone;
  }

  private tables(jpegBase: number): { inLut: Float32Array; encoded: Float32Array; linear: Float32Array; hue: Float32Array } {
    if (!this.inLut || this.inLut.jpegBase !== jpegBase) {
      const lut = new Float32Array(65536);
      for (let i = 0; i < 65536; i++) lut[i] = this.toLinear(i / 65535, jpegBase);
      this.inLut = { jpegBase, lut };
    }
    if (!this.outLut) {
      const encoded = new Float32Array(OUT_SIZE + 1);
      const linear = new Float32Array(OUT_SIZE + 1);
      const hue = new Float32Array(OUT_SIZE + 1);
      const { curve } = this.model.leica;
      for (let i = 0; i <= OUT_SIZE; i++) {
        const u = i / OUT_SIZE;
        const v = u * u * this.vMax;
        encoded[i] = curveEval(curve, this.model.curveDomain, v);
        linear[i] = srgbToLinear(encoded[i]);
        hue[i] = this.hueWeight(v);
      }
      this.outLut = { encoded, linear, hue };
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
    const { inLut, encoded, linear, hue } = this.tables(Math.round(Math.min(1, Math.max(0, jpegBase)) * 100) / 100);
    const srgb = linearLut('srgb');
    const t = Math.min(1, Math.max(0, tone));
    const m = this.matrix;
    const scale = OUT_SIZE / Math.sqrt(this.vMax);
    // トーンカーブの入力 → 表の位置（平方根で添字を取る）。同じ位置で符号値・リニア・色相を保つ割合の表を引く
    const pos = (x: number): number => (x > 0 ? Math.sqrt(x) * scale : 0);
    const at = (table: Float32Array, f: number): number => {
      if (f >= OUT_SIZE) return table[OUT_SIZE];
      const i = f | 0;
      return table[i] + (table[i + 1] - table[i]) * (f - i);
    };
    const q = new Float64Array(3);
    const l = new Float64Array(3);
    const e = new Float64Array(3);
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
      const f0 = pos(y0);
      const f1 = pos(y1);
      const f2 = pos(y2);
      // 明るいところだけ、色相をカーブを掛ける前の色相に戻す（中間調より暗いところは w = 0）
      const w = at(hue, f0 > f1 ? (f0 > f2 ? f0 : f2) : f1 > f2 ? f1 : f2);
      let mid = -1;
      if (t < 1 || w > 0) {
        l[0] = at(linear, f0);
        l[1] = at(linear, f1);
        l[2] = at(linear, f2);
        mid = keepHue(y0, y1, y2, l, w);
      }
      if (t > 0) {
        e[0] = at(encoded, f0);
        e[1] = at(encoded, f1);
        e[2] = at(encoded, f2);
        if (mid >= 0) e[mid] = fastLinearToSrgb(l[mid]);
      }
      let o0: number;
      let o1: number;
      let o2: number;
      if (t < 1) {
        const yi = LUMA_R * srgb[r] + LUMA_G * srgb[g] + LUMA_B * srgb[b];
        keepLuminance(l[0], l[1], l[2], yi, q, fastLinearToSrgb);
        o0 = q[0];
        o1 = q[1];
        o2 = q[2];
        if (t > 0) {
          o0 += (e[0] - o0) * t;
          o1 += (e[1] - o1) * t;
          o2 += (e[2] - o2) * t;
        }
      } else {
        o0 = e[0];
        o1 = e[1];
        o2 = e[2];
      }
      out[i] = r + (o0 * 65535 - r) * k + 0.5;
      out[i + 1] = g + (o1 * 65535 - g) * k + 0.5;
      out[i + 2] = b + (o2 * 65535 - b) * k + 0.5;
    }
    return out;
  }
}

/**
 * 明るいところの色相を、トーンカーブを掛ける前（y）の色相に戻す（training/leica/export_model.py の keep_hue_in_highlights）。
 * l は光の強さ（リニア）の出力。最大・最小のチャンネルはそのままに、中間のチャンネルを y と同じ比
 * （(中 - 小) / (大 - 小)）へ w の割合で寄せる。書き換えたチャンネルの番号を返す（書き換えなければ -1）
 */
function keepHue(y0: number, y1: number, y2: number, l: Float64Array, w: number): number {
  if (!(w > 0)) return -1;
  // 小・中・大の順に並べる（同じ値は番号の順のまま = numpy の安定ソートと同じ）。画素ごとに呼ぶので配列を作らない
  let lo = 0;
  let mid = 1;
  let hi = 2;
  let a = y0 > 0 ? y0 : 0;
  let b = y1 > 0 ? y1 : 0;
  let c = y2 > 0 ? y2 : 0;
  let s: number;
  if (a > b) {
    s = a; a = b; b = s;
    s = lo; lo = mid; mid = s;
  }
  if (b > c) {
    s = b; b = c; c = s;
    s = mid; mid = hi; hi = s;
  }
  if (a > b) {
    s = a; a = b; b = s;
    s = lo; lo = mid; mid = s;
  }
  const span = c - a;
  const r = span > 1e-12 ? (b - a) / span : 0;
  l[mid] += (l[lo] + r * (l[hi] - l[lo]) - l[mid]) * w;
  return mid;
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
