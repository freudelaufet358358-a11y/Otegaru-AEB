// 合成後の仕上げ調整（明るさ・コントラスト・彩度）と出力形式への変換。
// 入力は表示用 (sRGB) の 16bit RGB インターリーブ配列。

export interface Adjustments {
  /** -100..100。中間調を持ち上げる／下げる（白と黒は動かさない） */
  brightness: number;
  /** -100..100。S 字カーブの強さ */
  contrast: number;
  /** -100..100。0 で変化なし */
  saturation: number;
}

export const DEFAULT_ADJUSTMENTS: Adjustments = { brightness: 0, contrast: 0, saturation: 0 };

/** 明るさ・コントラストを 1 本のトーンカーブ (16bit 入力 → 0..1) にまとめる */
export function toneCurveLut(adj: Adjustments): Float32Array {
  const gamma = Math.pow(2, -0.9 * (adj.brightness / 100));
  const c = adj.contrast / 100;
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    let v = Math.pow(i / 65535, gamma);
    const s = v * v * (3 - 2 * v);
    v = v + c * (s - v);
    lut[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return lut;
}

function prepare(adj: Adjustments) {
  return { lut: toneCurveLut(adj), sat: 1 + adj.saturation / 100 };
}

/** 表示用 RGBA (8bit) に変換 */
export function toRGBA8(src: Uint16Array, adj: Adjustments, out: Uint8ClampedArray): void {
  const { lut, sat } = prepare(adj);
  const n = src.length / 3;
  for (let p = 0, i = 0, o = 0; p < n; p++, i += 3, o += 4) {
    let r = lut[src[i]];
    let g = lut[src[i + 1]];
    let b = lut[src[i + 2]];
    if (sat !== 1) {
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = y + (r - y) * sat;
      g = y + (g - y) * sat;
      b = y + (b - y) * sat;
    }
    out[o] = r * 255 + 0.5;
    out[o + 1] = g * 255 + 0.5;
    out[o + 2] = b * 255 + 0.5;
    out[o + 3] = 255;
  }
}

/** 16bit RGB に変換（TIFF 出力用） */
export function toRGB16(src: Uint16Array, adj: Adjustments, out: Uint16Array): void {
  const { lut, sat } = prepare(adj);
  const n = src.length / 3;
  for (let p = 0, i = 0; p < n; p++, i += 3) {
    let r = lut[src[i]];
    let g = lut[src[i + 1]];
    let b = lut[src[i + 2]];
    if (sat !== 1) {
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = y + (r - y) * sat;
      g = y + (g - y) * sat;
      b = y + (b - y) * sat;
    }
    out[i] = clamp16(r);
    out[i + 1] = clamp16(g);
    out[i + 2] = clamp16(b);
  }
}

function clamp16(v: number): number {
  const x = v * 65535 + 0.5;
  return x <= 0 ? 0 : x >= 65535 ? 65535 : x | 0;
}
