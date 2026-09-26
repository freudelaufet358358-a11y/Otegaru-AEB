// Burt & Adelson のガウシアン／ラプラシアンピラミッド（5 タップ [1 4 6 4 1]/16 カーネル）。
// 大きな画像でも扱えるよう、バッファは使い回し、ラプラシアンへの変換と再構成はインプレースで行う。

export interface Level {
  width: number;
  height: number;
  data: Float32Array;
}

/** 反射 (reflect-101) で範囲外インデックスを折り返す */
function reflect(i: number, n: number): number {
  if (n === 1) return 0;
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

export function levelCount(w: number, h: number): number {
  return 1 + Math.floor(Math.log2(Math.max(1, Math.min(w, h))));
}

export function levelSizes(w: number, h: number, n: number): Array<[number, number]> {
  const sizes: Array<[number, number]> = [[w, h]];
  for (let i = 1; i < n; i++) {
    w = (w + 1) >> 1;
    h = (h + 1) >> 1;
    sizes.push([w, h]);
  }
  return sizes;
}

export function allocLevels(w: number, h: number, n: number): Level[] {
  return levelSizes(w, h, n).map(([lw, lh]) => ({ width: lw, height: lh, data: new Float32Array(lw * lh) }));
}

/** reduce/expand で使う作業用バッファのサイズ */
export function scratchSize(w: number, h: number): number {
  return ((w + 1) >> 1) * h;
}

/** ガウシアンぼかし + 1/2 間引き */
export function reduce(src: Level, dst: Level, tmp: Float32Array): void {
  const sw = src.width;
  const sh = src.height;
  const dw = dst.width;
  const dh = dst.height;
  const s = src.data;
  const d = dst.data;
  // 横方向: tmp (dw × sh)
  for (let y = 0; y < sh; y++) {
    const r = y * sw;
    const o = y * dw;
    for (let x = 0; x < dw; x++) {
      const c = 2 * x;
      if (c >= 2 && c + 2 < sw) {
        const i = r + c;
        tmp[o + x] = (s[i - 2] + s[i + 2] + 4 * (s[i - 1] + s[i + 1]) + 6 * s[i]) * 0.0625;
      } else {
        tmp[o + x] =
          (s[r + reflect(c - 2, sw)] +
            s[r + reflect(c + 2, sw)] +
            4 * (s[r + reflect(c - 1, sw)] + s[r + reflect(c + 1, sw)]) +
            6 * s[r + reflect(c, sw)]) *
          0.0625;
      }
    }
  }
  // 縦方向: dst (dw × dh)
  for (let y = 0; y < dh; y++) {
    const c = 2 * y;
    const r0 = reflect(c - 2, sh) * dw;
    const r1 = reflect(c - 1, sh) * dw;
    const r2 = reflect(c, sh) * dw;
    const r3 = reflect(c + 1, sh) * dw;
    const r4 = reflect(c + 2, sh) * dw;
    const o = y * dw;
    for (let x = 0; x < dw; x++) {
      d[o + x] = (tmp[r0 + x] + tmp[r4 + x] + 4 * (tmp[r1 + x] + tmp[r3 + x]) + 6 * tmp[r2 + x]) * 0.0625;
    }
  }
}

/** dst += sign × expand(src)。src は dst の 1/2 サイズ（切り上げ） */
export function expandAdd(src: Level, dst: Level, sign: number, tmp: Float32Array): void {
  const sw = src.width;
  const sh = src.height;
  const dw = dst.width;
  const dh = dst.height;
  const s = src.data;
  const d = dst.data;
  // 縦方向: tmp (sw × dh)
  for (let y = 0; y < dh; y++) {
    const m = y >> 1;
    const o = y * sw;
    if ((y & 1) === 0) {
      const ra = reflect(m - 1, sh) * sw;
      const rb = m * sw;
      const rc = reflect(m + 1, sh) * sw;
      for (let x = 0; x < sw; x++) tmp[o + x] = (s[ra + x] + 6 * s[rb + x] + s[rc + x]) * 0.125;
    } else {
      const ra = m * sw;
      const rb = reflect(m + 1, sh) * sw;
      for (let x = 0; x < sw; x++) tmp[o + x] = (s[ra + x] + s[rb + x]) * 0.5;
    }
  }
  // 横方向: dst に加算
  const k1 = sign * 0.125;
  const k2 = sign * 0.5;
  for (let y = 0; y < dh; y++) {
    const r = y * sw;
    const o = y * dw;
    for (let x = 0; x < dw; x++) {
      const m = x >> 1;
      if ((x & 1) === 0) {
        const a = m > 0 ? m - 1 : reflect(m - 1, sw);
        const c = m + 1 < sw ? m + 1 : reflect(m + 1, sw);
        d[o + x] += (tmp[r + a] + 6 * tmp[r + m] + tmp[r + c]) * k1;
      } else {
        const c = m + 1 < sw ? m + 1 : reflect(m + 1, sw);
        d[o + x] += (tmp[r + m] + tmp[r + c]) * k2;
      }
    }
  }
}

/** levels[0] を元にガウシアンピラミッドを作る */
export function buildGaussian(levels: Level[], tmp: Float32Array): void {
  for (let i = 1; i < levels.length; i++) reduce(levels[i - 1], levels[i], tmp);
}

/** ガウシアンピラミッドをインプレースでラプラシアンピラミッドに変換する */
export function gaussianToLaplacian(levels: Level[], tmp: Float32Array): void {
  for (let i = 0; i < levels.length - 1; i++) expandAdd(levels[i + 1], levels[i], -1, tmp);
}

/** ラプラシアンピラミッドをインプレースで再構成する（結果は levels[0]） */
export function collapse(levels: Level[], tmp: Float32Array): void {
  for (let i = levels.length - 2; i >= 0; i--) expandAdd(levels[i + 1], levels[i], 1, tmp);
}
