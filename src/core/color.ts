// 色空間まわりの変換と、よく使うルックアップテーブル。

/** sRGB の符号値 (0..1) → リニア値 */
export function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** リニア値 → sRGB の符号値 (0..1 にクランプ) */
export function linearToSrgb(v: number): number {
  if (!(v > 0)) return 0;
  if (v >= 1) return 1;
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}

/** ハイライトを飽和させずになだらかに 1 へ近づける肩特性（KNEE 以下はそのまま） */
const KNEE = 0.75;
export function shoulder(x: number): number {
  if (x <= KNEE) return x;
  const r = 1 - KNEE;
  return KNEE + r * (1 - Math.exp(-(x - KNEE) / r));
}

/** Rec.709 / sRGB の輝度係数 */
export const LUMA_R = 0.2126;
export const LUMA_G = 0.7152;
export const LUMA_B = 0.0722;

/**
 * RAW を表示用に変換するときの基準ゲイン（約 +0.5EV）。
 * Canon の RAW は中間グレーが飽和点から 3 段ほど下にあり、そのままガンマだけ掛けると
 * カメラ JPEG より暗く見えるため、少し持ち上げてから肩特性で白に丸める。
 */
export const RAW_DISPLAY_GAIN = Math.SQRT2;

export type Encoding = 'linear' | 'srgb';

/** 16bit 符号値 → リニア値 の LUT */
export function linearLut(encoding: Encoding): Float32Array {
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const v = i / 65535;
    lut[i] = encoding === 'linear' ? v : srgbToLinear(v);
  }
  return lut;
}

/** 16bit 符号値 → 表示用 (sRGB 0..1) の LUT。露出フュージョンの入力に使う */
export function displayLut(encoding: Encoding): Float32Array {
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const v = i / 65535;
    lut[i] = encoding === 'linear' ? linearToSrgb(shoulder(v * RAW_DISPLAY_GAIN)) : v;
  }
  return lut;
}

/**
 * リニア値 (0..1) → sRGB 符号値の高速変換用テーブル。
 * 65536 分割 + 線形補間で 16bit 出力にも十分な精度になる。
 */
const OETF_SIZE = 65536;
let oetfTable: Float32Array | null = null;
export function fastLinearToSrgb(v: number): number {
  if (!(v > 0)) return 0;
  if (v >= 1) return 1;
  if (!oetfTable) {
    oetfTable = new Float32Array(OETF_SIZE + 1);
    for (let i = 0; i <= OETF_SIZE; i++) oetfTable[i] = linearToSrgb(i / OETF_SIZE);
  }
  const f = v * OETF_SIZE;
  const i = f | 0;
  const t = f - i;
  return oetfTable[i] + (oetfTable[i + 1] - oetfTable[i]) * t;
}

/**
 * リニア値 (0..∞) → 肩特性 → sRGB 符号値 (0..1) をまとめた高速変換。
 * 暗部の精度を確保するため、sqrt(x / RANGE) を添字にしたテーブルを線形補間する。
 */
const DISPLAY_RANGE = 16;
const DISPLAY_SIZE = 65536;
let displayTable: Float32Array | null = null;
export function linearToDisplay(x: number): number {
  if (!(x > 0)) return 0;
  if (!displayTable) {
    displayTable = new Float32Array(DISPLAY_SIZE + 1);
    for (let i = 0; i <= DISPLAY_SIZE; i++) {
      const u = i / DISPLAY_SIZE;
      displayTable[i] = linearToSrgb(shoulder(u * u * DISPLAY_RANGE));
    }
  }
  const f = Math.sqrt(x / DISPLAY_RANGE) * DISPLAY_SIZE;
  if (f >= DISPLAY_SIZE) return displayTable[DISPLAY_SIZE];
  const i = f | 0;
  return displayTable[i] + (displayTable[i + 1] - displayTable[i]) * (f - i);
}
