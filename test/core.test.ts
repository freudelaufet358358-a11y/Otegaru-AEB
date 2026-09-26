import { describe, expect, it } from 'vitest';
import { alignMTB, alignedViews, commonCrop, toGray8 } from '../src/core/align';
import { toRGBA8, toRGB16, DEFAULT_ADJUSTMENTS } from '../src/core/adjust';
import { displayLut, linearLut, linearToDisplay, linearToSrgb, shoulder, srgbToLinear, fastLinearToSrgb } from '../src/core/color';
import { exposureValue, formatShutter, readExif } from '../src/core/exif';
import { fullView, resizeView, type Frame16 } from '../src/core/frame';
import { exposureFusion, DEFAULT_FUSION_WEIGHTS } from '../src/core/fusion';
import { boxFilter, estimateExposures, mergeRow, toneMapHDR } from '../src/core/hdr';
import { allocLevels, buildGaussian, collapse, gaussianToLaplacian, levelCount, scratchSize } from '../src/core/pyramid';
import { encodeTiff16 } from '../src/core/tiff';

/** 滑らかな模様 + 細かいテクスチャのリニアなシーン（放射輝度） */
function scene(w: number, h: number): Float32Array {
  const s = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const base = Math.pow(2, 6 * (x / w) - 5); // 左は暗く右は明るい（約 6 段）
      const tex = 1 + 0.3 * Math.sin(x * 0.7) * Math.cos(y * 0.5) + 0.2 * Math.sin((x + 2 * y) * 0.13);
      const blob = Math.hypot(x - w * 0.3, y - h * 0.4) < h * 0.15 ? 3 : 1;
      const i = (y * w + x) * 3;
      s[i] = base * tex * blob * 0.9;
      s[i + 1] = base * tex * blob;
      s[i + 2] = base * tex * blob * 0.7;
    }
  }
  return s;
}

/** シーンを露出 e で撮った RAW 風フレーム（リニア 16bit、飽和あり）。dx, dy だけずらす */
function shoot(s: Float32Array, w: number, h: number, e: number, dx = 0, dy = 0): Frame16 {
  const data = new Uint16Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = Math.min(w - 1, Math.max(0, x - dx));
      const sy = Math.min(h - 1, Math.max(0, y - dy));
      for (let c = 0; c < 3; c++) {
        const v = s[(sy * w + sx) * 3 + c] * e;
        data[(y * w + x) * 3 + c] = Math.min(65535, Math.round(v * 65535));
      }
    }
  }
  return { width: w, height: h, data, encoding: 'linear' };
}

describe('color', () => {
  it('sRGB 変換が往復で一致する', () => {
    for (const v of [0, 0.001, 0.04, 0.2, 0.5, 0.9, 1]) {
      expect(srgbToLinear(linearToSrgb(v))).toBeCloseTo(v, 6);
      expect(fastLinearToSrgb(v)).toBeCloseTo(linearToSrgb(v), 5);
    }
  });
  it('linearToDisplay は肩特性 + sRGB と一致する', () => {
    for (const v of [0, 1e-5, 0.002, 0.05, 0.3, 0.75, 1, 2.5, 10, 100]) {
      expect(linearToDisplay(v)).toBeCloseTo(linearToSrgb(shoulder(v)), 4);
    }
  });
  it('表示用 LUT は単調増加で 0..1 に収まる', () => {
    const lut = displayLut('linear');
    for (let i = 1; i < 65536; i++) expect(lut[i]).toBeGreaterThanOrEqual(lut[i - 1]);
    expect(lut[0]).toBe(0);
    expect(lut[65535]).toBeLessThanOrEqual(1);
  });
});

describe('pyramid', () => {
  it('ラプラシアン分解 → 再構成で元に戻る（奇数サイズ含む）', () => {
    for (const [w, h] of [
      [37, 23],
      [64, 64],
      [5, 101],
    ]) {
      const n = levelCount(w, h);
      const lv = allocLevels(w, h, n);
      const orig = new Float32Array(w * h);
      for (let i = 0; i < orig.length; i++) orig[i] = Math.sin(i * 0.37) + (i % 7) * 0.1;
      lv[0].data.set(orig);
      const tmp = new Float32Array(scratchSize(w, h));
      buildGaussian(lv, tmp);
      gaussianToLaplacian(lv, tmp);
      collapse(lv, tmp);
      let maxErr = 0;
      for (let i = 0; i < orig.length; i++) maxErr = Math.max(maxErr, Math.abs(lv[0].data[i] - orig[i]));
      expect(maxErr).toBeLessThan(1e-5);
    }
  });
  it('ガウシアンの縮小は定数を保存する', () => {
    const lv = allocLevels(33, 17, 4);
    lv[0].data.fill(0.25);
    buildGaussian(lv, new Float32Array(scratchSize(33, 17)));
    for (const l of lv) for (const v of l.data) expect(v).toBeCloseTo(0.25, 6);
  });
});

describe('fusion', () => {
  it('同じ画像どうしのフュージョンは元の画像になる', () => {
    const w = 48;
    const h = 32;
    const f = shoot(scene(w, h), w, h, 0.5);
    const lut = displayLut('linear');
    const views = [fullView(f), fullView(f), fullView(f)];
    const out = new Float32Array(w * h * 3);
    exposureFusion(views, [lut, lut, lut], DEFAULT_FUSION_WEIGHTS, (c, plane) => {
      for (let i = 0; i < w * h; i++) out[i * 3 + c] = plane[i];
    });
    let maxErr = 0;
    for (let i = 0; i < out.length; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - lut[f.data[i]]));
    expect(maxErr).toBeLessThan(2e-3);
  });

  it('暗部は明るいフレーム、明部は暗いフレームから取られる', () => {
    const w = 128;
    const h = 64;
    const s = scene(w, h);
    const frames = [0.25, 1, 4].map((e) => shoot(s, w, h, e));
    const lut = displayLut('linear');
    const out = new Float32Array(w * h * 3);
    exposureFusion(frames.map(fullView), [lut, lut, lut], DEFAULT_FUSION_WEIGHTS, (c, plane) => {
      for (let i = 0; i < w * h; i++) out[i * 3 + c] = plane[i];
    });
    const g = (fr: Frame16 | null, x: number, y: number) => {
      const i = (y * w + x) * 3 + 1;
      return fr ? lut[fr.data[i]] : out[i];
    };
    // 暗部: 中間露出より明るい
    expect(g(null, 4, 50)).toBeGreaterThan(g(frames[1], 4, 50));
    // 明部: 中間露出では白飛びして平坦だが、合成結果には模様（階調）が残っている
    const std = (fr: Frame16 | null) => {
      const vals: number[] = [];
      for (let y = 20; y < 44; y++) for (let x = w - 12; x < w; x++) vals.push(g(fr, x, y));
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length);
    };
    expect(std(frames[1])).toBeLessThan(0.01);
    expect(std(null)).toBeGreaterThan(0.03);
  });
});

describe('align', () => {
  it('既知のずれ量を検出できる', () => {
    const w = 400;
    const h = 300;
    const s = scene(w, h);
    const ref = shoot(s, w, h, 1);
    const lut = displayLut('linear');
    for (const [dx, dy, e] of [
      [7, -3, 0.25],
      [-12, 9, 4],
      [0, 0, 2],
    ]) {
      const tgt = shoot(s, w, h, e, dx, dy);
      const shift = alignMTB(toGray8(ref, lut), toGray8(tgt, lut));
      expect(shift).toEqual([dx, dy]);
    }
  });
  it('共通領域の切り抜き', () => {
    const crop = commonCrop(100, 80, [
      [0, 0],
      [5, -3],
      [-2, 4],
    ]);
    expect(crop).toEqual({ x0: 2, y0: 3, width: 93, height: 73 });
    const f: Frame16 = { width: 100, height: 80, data: new Uint16Array(100 * 80 * 3), encoding: 'linear' };
    const views = alignedViews([f, f, f], [[0, 0], [5, -3], [-2, 4]], crop);
    for (const v of views) {
      expect(v.x0).toBeGreaterThanOrEqual(0);
      expect(v.y0).toBeGreaterThanOrEqual(0);
      expect(v.x0 + v.width).toBeLessThanOrEqual(100);
      expect(v.y0 + v.height).toBeLessThanOrEqual(80);
    }
  });
});

describe('hdr', () => {
  it('露出比を画像から推定できる', () => {
    const w = 200;
    const h = 100;
    const s = scene(w, h);
    const frames = [0.3, 1, 3.3].map((e) => shoot(s, w, h, e));
    const lut = linearLut('linear');
    const e = estimateExposures(frames.map(fullView), [lut, lut, lut], [0, 1, 2], 1, [NaN, NaN, NaN]);
    expect(e[0]).toBeCloseTo(0.3, 2);
    expect(e[1]).toBe(1);
    expect(e[2]).toBeCloseTo(3.3, 1);
  });
  it('放射輝度の合成で白飛びしていない値が復元される', () => {
    const w = 64;
    const h = 8;
    const s = scene(w, h);
    const exps = [0.25, 1, 4];
    const frames = exps.map((e) => shoot(s, w, h, e));
    const lut = linearLut('linear');
    const row = new Float32Array(w * 3);
    mergeRow(frames.map(fullView), [lut, lut, lut], exps, 0, 3, row);
    for (let x = 0; x < w; x++) {
      const truth = s[(3 * w + x) * 3 + 1];
      if (truth * 0.25 > 0.9) continue; // 最も暗いフレームでも飽和
      expect(Math.abs(row[x * 3 + 1] - truth) / truth).toBeLessThan(0.02);
    }
  });
  it('トーンマッピングの出力が 16bit の範囲に収まり、暗部が持ち上がる', () => {
    const w = 160;
    const h = 90;
    const s = scene(w, h);
    const exps = [0.25, 1, 4];
    const frames = exps.map((e) => shoot(s, w, h, e));
    const lut = linearLut('linear');
    const out = new Uint16Array(w * h * 3);
    toneMapHDR(frames.map(fullView), [lut, lut, lut], exps, { strength: 0.8, detail: 1.2 }, out);
    const dark = out[(45 * w + 2) * 3 + 1] / 65535;
    const bright = out[(45 * w + w - 3) * 3 + 1] / 65535;
    expect(dark).toBeGreaterThan(0.05);
    expect(bright).toBeLessThan(0.99);
    expect(bright).toBeGreaterThan(dark);
  });
  it('箱フィルタは端を含めて正しい平均を返す', () => {
    const w = 7;
    const h = 5;
    const src = new Float32Array(w * h).map((_, i) => i);
    const out = new Float32Array(w * h);
    boxFilter(src, w, h, 1, out);
    // (0,0) の近傍 {0,1,7,8}
    expect(out[0]).toBeCloseTo((0 + 1 + 7 + 8) / 4, 5);
    // 内部 (3,2) の近傍 3x3
    let acc = 0;
    for (let y = 1; y <= 3; y++) for (let x = 2; x <= 4; x++) acc += y * w + x;
    expect(out[2 * w + 3]).toBeCloseTo(acc / 9, 5);
  });
});

describe('frame', () => {
  it('縮小で平均値が保たれる', () => {
    const f: Frame16 = { width: 10, height: 6, data: new Uint16Array(180).fill(1000), encoding: 'linear' };
    const r = resizeView(fullView(f), 3, 2);
    expect(r.width).toBe(3);
    expect(Array.from(r.data).every((v) => v === 1000)).toBe(true);
  });
});

describe('adjust', () => {
  it('調整なしなら値が変わらない', () => {
    const src = new Uint16Array([0, 32768, 65535, 1000, 2000, 3000]);
    const out8 = new Uint8ClampedArray(8);
    toRGBA8(src, DEFAULT_ADJUSTMENTS, out8);
    expect(Array.from(out8)).toEqual([0, 128, 255, 255, 4, 8, 12, 255]);
    const out16 = new Uint16Array(6);
    toRGB16(src, DEFAULT_ADJUSTMENTS, out16);
    expect(Array.from(out16)).toEqual(Array.from(src));
  });
});

describe('tiff', () => {
  it('ヘッダとサイズが正しい', async () => {
    const w = 5;
    const h = 70;
    const rgb = new Uint16Array(w * h * 3).map((_, i) => i);
    const blob = encodeTiff16(w, h, rgb, { software: 'test' });
    const buf = new DataView(await blob.arrayBuffer());
    expect(buf.getUint16(0)).toBe(0x4949);
    expect(buf.getUint16(2, true)).toBe(42);
    const ifd = buf.getUint32(4, true);
    const n = buf.getUint16(ifd, true);
    const tags = new Map<number, number>();
    let stripOffsetsPtr = 0;
    for (let i = 0; i < n; i++) {
      const p = ifd + 2 + i * 12;
      const tag = buf.getUint16(p, true);
      tags.set(tag, buf.getUint32(p + 8, true));
      if (tag === 273) stripOffsetsPtr = buf.getUint32(p + 8, true);
    }
    expect(tags.get(256)).toBe(w);
    expect(tags.get(257)).toBe(h);
    const first = buf.getUint32(stripOffsetsPtr, true);
    expect(buf.byteLength).toBe(first + w * h * 6);
    expect(buf.getUint16(first + 2, true)).toBe(1);
  });
});

describe('exif', () => {
  it('JPEG の APP1 から露出情報を読む', () => {
    const jpeg = buildExifJpeg();
    const e = readExif(jpeg);
    expect(e.model).toBe('Canon EOS R6m2');
    expect(e.exposureTime).toBeCloseTo(1 / 125, 8);
    expect(e.fNumber).toBeCloseTo(8, 8);
    expect(e.iso).toBe(100);
    expect(e.exposureBias).toBeCloseTo(-2, 8);
    expect(exposureValue(e)).toBeCloseTo(Math.log2(1 / 125 / 64), 8);
    expect(formatShutter(e.exposureTime)).toBe('1/125');
  });
  it('EXIF がなければ空', () => {
    expect(readExif(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]).buffer)).toEqual({});
  });
});

/** テスト用に最小限の EXIF 付き JPEG（ヘッダ部分だけ）を組み立てる（ビッグエンディアン） */
function buildExifJpeg(): ArrayBuffer {
  const b: number[] = [];
  const u16 = (v: number) => b.push((v >> 8) & 255, v & 255);
  const u32 = (v: number) => b.push((v >>> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255);
  const tiff: number[] = [];
  const t16 = (v: number) => tiff.push((v >> 8) & 255, v & 255);
  const t32 = (v: number) => tiff.push((v >>> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255);
  // TIFF ヘッダ
  tiff.push(0x4d, 0x4d);
  t16(42);
  t32(8);
  // IFD0: Model, ExifIFD
  const model = 'Canon EOS R6m2\0';
  const ifd0 = 8;
  const ifd0Size = 2 + 2 * 12 + 4;
  const modelOff = ifd0 + ifd0Size;
  const exifOff = modelOff + model.length + (model.length & 1);
  t16(2);
  t16(0x0110), t16(2), t32(model.length), t32(modelOff);
  t16(0x8769), t16(4), t32(1), t32(exifOff);
  t32(0);
  for (const ch of model) tiff.push(ch.charCodeAt(0));
  if (model.length & 1) tiff.push(0);
  // Exif IFD: ExposureTime, FNumber, ISO, ExposureBias
  const exifSize = 2 + 4 * 12 + 4;
  const ratOff = exifOff + exifSize;
  t16(4);
  t16(0x829a), t16(5), t32(1), t32(ratOff);
  t16(0x829d), t16(5), t32(1), t32(ratOff + 8);
  t16(0x8827), t16(3), t32(1), t16(100), t16(0);
  t16(0x9204), t16(10), t32(1), t32(ratOff + 16);
  t32(0);
  t32(1), t32(125);
  t32(8), t32(1);
  t32(-2 >>> 0), t32(1);
  // JPEG
  u16(0xffd8);
  u16(0xffe1);
  u16(2 + 6 + tiff.length);
  u32(0x45786966);
  u16(0);
  b.push(...tiff);
  u16(0xffd9);
  return new Uint8Array(b).buffer;
}

describe('exif 書き出し', () => {
  it('作った EXIF を差し込んだ JPEG を読み戻せる', async () => {
    const { buildExifApp1, insertExif } = await import('../src/core/exif');
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 4, 1, 2, 0xff, 0xd9]);
    const out = insertExif(jpeg, buildExifApp1({ make: 'Canon', model: 'Canon EOS R6 Mark II', software: 'x', dateTime: '2024:01:02 03:04:05' }));
    const e = readExif(out.buffer as ArrayBuffer);
    expect(e.make).toBe('Canon');
    expect(e.model).toBe('Canon EOS R6 Mark II');
    expect(e.dateTime).toBe('2024:01:02 03:04:05');
    expect(out[out.length - 1]).toBe(0xd9);
  });
});
