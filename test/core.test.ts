import { describe, expect, it } from 'vitest';
import { alignFrames, alignMTB, fitSimilarity, toGray8 } from '../src/core/align';
import { angleDegrees, apply, compose, invert, rotationAbout, validCrop, warp, IDENTITY } from '../src/core/transform';
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

/** 位置合わせのテスト用: ランダムな長方形を重ねた、エッジの多いシーン */
function texturedScene(w: number, h: number): Float32Array {
  const s = new Float32Array(w * h * 3);
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const lum = new Float32Array(w * h).map((_, i) => 0.05 + 0.2 * ((i % w) / w));
  for (let n = 0; n < 160; n++) {
    const rw = 6 + rnd() * w * 0.15;
    const rh = 6 + rnd() * h * 0.15;
    const x0 = rnd() * (w - rw);
    const y0 = rnd() * (h - rh);
    const v = 0.02 + rnd() * 0.5;
    for (let y = Math.floor(y0); y < y0 + rh; y++) for (let x = Math.floor(x0); x < x0 + rw; x++) lum[y * w + x] = v;
  }
  for (let i = 0; i < w * h; i++) {
    s[i * 3] = lum[i] * 0.9;
    s[i * 3 + 1] = lum[i];
    s[i * 3 + 2] = lum[i] * 0.8;
  }
  return s;
}

/** 基準座標 → フレーム座標の変換 t で撮った（= フレーム上の q にはシーンの t⁻¹(q) が写る）フレーム */
function shootWarped(s: Float32Array, w: number, h: number, e: number, t: typeof IDENTITY): Frame16 {
  const inv = invert(t);
  const data = new Uint16Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [sx, sy] = apply(inv, x, y);
      const x0 = Math.min(w - 2, Math.max(0, Math.floor(sx)));
      const y0 = Math.min(h - 2, Math.max(0, Math.floor(sy)));
      const fx = Math.min(1, Math.max(0, sx - x0));
      const fy = Math.min(1, Math.max(0, sy - y0));
      for (let c = 0; c < 3; c++) {
        const at = (xx: number, yy: number) => s[(yy * w + xx) * 3 + c];
        const v = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
        data[(y * w + x) * 3 + c] = Math.min(65535, Math.round(v * e * 65535));
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
  it('回転と 1 画素未満のずれを検出できる（露出違い）', () => {
    const w = 640;
    const h = 480;
    const s = texturedScene(w, h);
    const lut = displayLut('linear');
    const ref = shootWarped(s, w, h, 1, IDENTITY);
    const truth = rotationAbout((0.4 * Math.PI) / 180, (w - 1) / 2, (h - 1) / 2, 5.3, -2.6);
    const tgt = shootWarped(s, w, h, 4, truth);
    const { transform, precise } = alignFrames(toGray8(ref, lut), toGray8(tgt, lut));
    expect(precise).toBe(true);
    expect(angleDegrees(transform)).toBeCloseTo(0.4, 1);
    // 画像全体で 0.3px 以内
    for (const [x, y] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]]) {
      const [u1, v1] = apply(transform, x, y);
      const [u2, v2] = apply(truth, x, y);
      expect(Math.hypot(u1 - u2, v1 - v2)).toBeLessThan(0.3);
    }
  });
  it('相似変換の当てはめ・合成・逆変換', () => {
    const t = rotationAbout(0.1, 50, 40, 3, -2);
    const pts = [[0, 0], [100, 0], [0, 80], [100, 80], [37, 12]].map(([x, y]) => {
      const [u, v] = apply(t, x, y);
      return { x, y, u, v };
    });
    const f = fitSimilarity(pts);
    expect(f.a).toBeCloseTo(t.a, 9);
    expect(f.b).toBeCloseTo(t.b, 9);
    expect(f.tx).toBeCloseTo(t.tx, 9);
    const id = compose(invert(t), t);
    expect(id.a).toBeCloseTo(1, 12);
    expect(id.b).toBeCloseTo(0, 12);
    expect(id.tx).toBeCloseTo(0, 9);
  });
  it('共通領域の切り抜き（平行移動・回転）', () => {
    const W = 100;
    const H = 80;
    // 整数の平行移動なら従来どおりの切り抜きになる
    const crop = validCrop(W, H, [IDENTITY, { a: 1, b: 0, tx: 5, ty: -3 }, { a: 1, b: 0, tx: -2, ty: 4 }]);
    expect(crop).toEqual({ x0: 2, y0: 3, width: 93, height: 73 });
    // 回転があっても切り抜いた範囲の四隅はすべてのフレームの内側に入る
    const ts = [IDENTITY, rotationAbout(0.02, 50, 40, 1.5, 0.5)];
    const c = validCrop(W, H, ts);
    for (const t of ts) {
      for (const [x, y] of [[c.x0, c.y0], [c.x0 + c.width - 1, c.y0], [c.x0, c.y0 + c.height - 1], [c.x0 + c.width - 1, c.y0 + c.height - 1]]) {
        const [u, v] = apply(t, x, y);
        expect(u).toBeGreaterThanOrEqual(-1e-6);
        expect(v).toBeGreaterThanOrEqual(-1e-6);
        expect(u).toBeLessThanOrEqual(W - 1 + 1e-6);
        expect(v).toBeLessThanOrEqual(H - 1 + 1e-6);
      }
    }
    expect(c.width).toBeGreaterThan(80);
  });
  it('恒等変換のワープは元画像と一致する', () => {
    const w = 31;
    const h = 17;
    const f = shoot(scene(w, h), w, h, 0.5);
    const crop = { x0: 0, y0: 0, width: w, height: h };
    for (const cubic of [false, true]) {
      const out = warp(f, w, h, IDENTITY, crop, w, h, cubic);
      expect(Array.from(out.data)).toEqual(Array.from(f.data));
    }
    // 整数の平行移動はずらしたものと一致する
    const shifted = warp(f, w, h, { a: 1, b: 0, tx: 2, ty: 1 }, { x0: 0, y0: 0, width: w - 2, height: h - 1 }, w - 2, h - 1, true);
    expect(shifted.data[0]).toBe(f.data[(1 * w + 2) * 3]);
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
  it('最も暗いフレームでも飽和した画素は白に向けて彩度を落とし、飽和していない色は保つ', () => {
    // 1 画素目: 全フレームで G だけ頭打ちになった明るい画素（そのままだとマゼンタに寄る）
    // 2 画素目: 最も暗いフレームでは飽和していない鮮やかな色
    const px = (e: number): Frame16 => {
      const v = [1.0, 0.8, 1.0, 0.1, 0.3, 0.6].map((c, i) => (i < 3 ? Math.min(1, 4 * e * c) : Math.min(1, e * c)));
      return { width: 2, height: 1, data: new Uint16Array(v.map((c) => Math.round(c * 65535))), encoding: 'linear' };
    };
    const exps = [0.25, 1, 4];
    const views = exps.map((e) => fullView(px(e)));
    const lut = linearLut('linear');
    const row = new Float32Array(6);
    mergeRow(views, [lut, lut, lut], exps, 0, 0, row);
    expect(row[0]).toBeCloseTo(row[1], 5);
    expect(row[1]).toBeCloseTo(row[2], 5);
    expect(row[3] / row[5]).toBeCloseTo(0.1 / 0.6, 2);
    expect(row[4] / row[5]).toBeCloseTo(0.3 / 0.6, 2);
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

describe('効果の調整', () => {
  it('ディテール強調はプレビューと書き出しで同じ相対スケールの段にかかる', async () => {
    const { detailLevels } = await import('../src/core/fusion');
    expect(detailLevels(1600, 1067, 11)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(detailLevels(5936, 3900, 12)).toEqual([3, 4, 5, 6, 7]);
  });
  it('ディテールを上げると局所コントラストが上がる', () => {
    const w = 96;
    const h = 64;
    const frames = [0.25, 1, 4].map((e) => shoot(scene(w, h), w, h, e));
    const lut = displayLut('linear');
    const run = (detail: number) => {
      const g = new Float32Array(w * h);
      exposureFusion(frames.map(fullView), [lut, lut, lut], { ...DEFAULT_FUSION_WEIGHTS, detail }, (c, p) => {
        if (c === 1) g.set(p);
      });
      let acc = 0;
      for (let i = 1; i < g.length; i++) acc += Math.abs(g[i] - g[i - 1]);
      return acc;
    };
    expect(run(1.8)).toBeGreaterThan(run(1) * 1.1);
  });
  it('効果の強さ 0 なら基準フレーム、1 なら合成結果', async () => {
    const { blendWithReference } = await import('../src/core/adjust');
    const f: Frame16 = { width: 2, height: 1, data: new Uint16Array([0, 1000, 65535, 30000, 40000, 50000]), encoding: 'srgb' };
    const merged = new Uint16Array([100, 200, 300, 400, 500, 600]);
    const lut = displayLut('srgb');
    expect(Array.from(blendWithReference(merged, fullView(f), lut, 0, new Uint16Array(6)))).toEqual(Array.from(f.data));
    expect(Array.from(blendWithReference(merged, fullView(f), lut, 1, new Uint16Array(6)))).toEqual(Array.from(merged));
    const half = blendWithReference(merged, fullView(f), lut, 0.5, new Uint16Array(6));
    expect(half[0]).toBe(50);
  });
});
