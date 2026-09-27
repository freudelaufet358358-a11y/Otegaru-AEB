import { describe, expect, it } from 'vitest';
import { LUMA_B, LUMA_G, LUMA_R, linearToSrgb, shoulder, shoulderInverse, srgbToLinear } from '../src/core/color';
import { cameraKey, estimateCct, LeicaLook, parseLookModel, sensorMatrix, type LookSource } from '../src/core/look';
import modelJson from '../src/models/leica-m10.json';
import fixtureText from './fixtures/leica.json?raw';

const model = parseLookModel(modelJson);

/** training/leica/export_model.py の基準データ（決まった入力に対する Python 実装の出力） */
const fixture = JSON.parse(fixtureText) as {
  input: number[];
  cases: Array<{ encoding: 'linear' | 'srgb'; camera: string; cct: number | null; jpeg_base: number; tone: number; out: number[] }>;
};

/** 基準データの機種キーになる make / model */
const CAMERAS: Record<string, { make: string; model: string }> = {
  'canon:r6m2': { make: 'Canon', model: 'Canon EOS R6m2' },
  'nikon:z6': { make: 'NIKON CORPORATION', model: 'NIKON Z 6' },
};

describe('Leica M10 の色', () => {
  it('機種名の表記ゆれをそろえる', () => {
    expect(cameraKey('Canon', 'Canon EOS R6m2')).toBe('canon:r6m2');
    expect(cameraKey('Canon', 'EOS R6 Mark II')).toBe('canon:r6m2');
    expect(cameraKey('Canon', 'Canon EOS 1D X Mark III')).toBe('canon:1dxm3');
    expect(cameraKey('Canon', 'Canon EOS 5DS R')).toBe('canon:5dsr');
    expect(cameraKey('Leica Camera AG', 'LEICA M10')).toBe('leica:m10');
    // Leica M10 と Canon EOS M10 を取り違えない
    expect(cameraKey('Canon', 'Canon EOS M10')).toBe('canon:m10');
    for (const [key, c] of Object.entries(CAMERAS)) expect(cameraKey(c.make, c.model)).toBe(key);
  });

  it('肩特性の逆関数', () => {
    for (const x of [0, 0.1, 0.5, 0.75, 0.9, 1.5, 3]) expect(shoulderInverse(shoulder(x))).toBeCloseTo(x, 6);
  });

  it('撮影時の色温度をホワイトバランスから推定する', () => {
    // EOS R6 Mark II の色行列（D65）と、その推定分光感度から求めた標準光源 A・D65 でのニュートラル
    const camXyz = [0.9539, -0.2795, -0.1224, -0.4175, 1.1998, 0.2458, -0.0465, 0.1755, 0.6048];
    const cctA = estimateCct([0.9, 1, 0.2], camXyz);
    const cctD = estimateCct([0.52, 1, 0.66], camXyz);
    expect(cctA).toBeLessThan(3200);
    expect(cctD).toBeGreaterThan(5000);
    expect(estimateCct([1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0])).toBeUndefined();
  });

  it('機種と色温度でセンサーの変換行列を選ぶ', () => {
    const r6 = sensorMatrix(model, 'canon:r6m2', 5500);
    expect(r6.matched).toBe(true);
    const r8 = sensorMatrix(model, 'canon:r8', 5500);
    expect(Array.from(r8.matrix)).toEqual(Array.from(r6.matrix)); // 同じセンサー
    expect(sensorMatrix(model, 'sony:ilce7m3', 5500).matched).toBe(false);
    // Leica M10 の RAW はセンサーの変換をしない
    expect(Array.from(sensorMatrix(model, 'leica:m10', 5500).matrix)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    // どの行列も白を白のままにする（各行の和 = 1）
    for (const cct of [2500, 3300, 4500, 6500, undefined]) {
      const m = sensorMatrix(model, 'canon:r6m2', cct).matrix;
      for (let r = 0; r < 3; r++) expect(m[r * 3] + m[r * 3 + 1] + m[r * 3 + 2]).toBeCloseTo(1, 4);
    }
  });

  it('変換が学習コード (Python) と一致する', () => {
    const n = fixture.input.length / 3;
    const out = [0, 0, 0];
    for (const c of fixture.cases) {
      const cam = CAMERAS[c.camera];
      const src: LookSource = { encoding: c.encoding, make: cam.make, model: cam.model, cct: c.cct ?? undefined };
      const look = new LeicaLook(model, src);
      let maxErr = 0;
      for (let i = 0; i < n; i++) {
        look.pixel(fixture.input[3 * i], fixture.input[3 * i + 1], fixture.input[3 * i + 2], out, c.jpeg_base, c.tone);
        for (let k = 0; k < 3; k++) maxErr = Math.max(maxErr, Math.abs(out[k] - c.out[3 * i + k]));
      }
      expect(maxErr, `${c.encoding} ${c.camera} ${c.cct} ${c.jpeg_base} ${c.tone}`).toBeLessThan(1e-4);
    }
  });

  it('トーンカーブも Leica にすると、中間グレーは Leica のカメラ内 JPEG と同じ明るさになり、グレーは色づかない', () => {
    const look = new LeicaLook(model, { encoding: 'linear', make: 'Canon', model: 'Canon EOS R6m2', cct: 5200 });
    const out = [0, 0, 0];
    const mid = linearToSrgb(0.18);
    look.pixel(mid, mid, mid, out, 0, 1);
    for (const v of out) expect(v * 255).toBeCloseTo(118.9, 0);
    for (const g of [0.05, 0.2, 0.5, 0.8, 0.95]) {
      look.pixel(g, g, g, out, 0, 1);
      expect(Math.abs(out[0] - out[1])).toBeLessThan(1e-3);
      expect(Math.abs(out[2] - out[1])).toBeLessThan(1e-3);
    }
  });

  it('既定（tone = 0）では明るさ（輝度）は元のまま、色だけが変わる', () => {
    const look = new LeicaLook(model, { encoding: 'linear', make: 'Canon', model: 'Canon EOS R6m2', cct: 5200 });
    const out = [0, 0, 0];
    const lum = (r: number, g: number, b: number) => LUMA_R * srgbToLinear(r) + LUMA_G * srgbToLinear(g) + LUMA_B * srgbToLinear(b);
    let changed = 0;
    for (const [r, g, b] of [
      [0.5, 0.5, 0.5],
      [0.8, 0.45, 0.35],
      [0.25, 0.5, 0.2],
      [0.2, 0.35, 0.7],
      [0.05, 0.08, 0.04],
      [0.95, 0.9, 0.2],
    ]) {
      look.pixel(r, g, b, out);
      expect(lum(out[0], out[1], out[2])).toBeCloseTo(lum(r, g, b), 4);
      changed = Math.max(changed, Math.abs(out[0] - r), Math.abs(out[2] - b));
    }
    expect(changed).toBeGreaterThan(0.01);
  });

  it('おまかせの結果は、カメラ内 JPEG 並みのトーンとして戻すぶんメリハリが二重にかからない', () => {
    const look = new LeicaLook(model, { encoding: 'linear', make: 'Canon', model: 'Canon EOS R6m2', cct: 5200 });
    const a = [0, 0, 0];
    const b = [0, 0, 0];
    look.pixel(118.9 / 255, 118.9 / 255, 118.9 / 255, a, 1, 1);
    expect(a[1] * 255).toBeCloseTo(118.9, 0);
    for (const v of [0.1, 0.15, 0.65]) {
      look.pixel(v, v, v, a, 0, 1);
      look.pixel(v, v, v, b, 1, 1);
      expect(Math.abs(b[1] - v)).toBeLessThan(Math.abs(a[1] - v));
    }
  });

  it('16bit の画像に掛けたときの結果が 1 画素ずつの計算と一致し、効き 0 なら元のまま', () => {
    for (const [encoding, jpegBase, tone] of [
      ['linear', 0, 0],
      ['linear', 1, 0],
      ['linear', 0, 1],
      ['linear', 0.4, 0.6],
      ['srgb', 0, 0],
      ['srgb', 0, 1],
    ] as const) {
      const look = new LeicaLook(model, { encoding, make: 'Canon', model: 'Canon EOS R6m2', cct: 4300 });
      let seed = 3;
      const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      const img = new Uint16Array(3 * 4096).map(() => Math.floor(rnd() * 65536));
      const full = look.apply(img, 1, new Uint16Array(img.length), jpegBase, tone);
      const half = look.apply(img, 0.5, new Uint16Array(img.length), jpegBase, tone);
      const none = look.apply(img, 0, new Uint16Array(img.length), jpegBase, tone);
      expect(Array.from(none)).toEqual(Array.from(img));
      const px = [0, 0, 0];
      let maxErr = 0;
      for (let i = 0; i < img.length; i += 3) {
        look.pixel(img[i] / 65535, img[i + 1] / 65535, img[i + 2] / 65535, px, jpegBase, tone);
        for (let k = 0; k < 3; k++) {
          maxErr = Math.max(maxErr, Math.abs(full[i + k] / 65535 - px[k]));
          expect(Math.abs(half[i + k] - (img[i + k] + full[i + k]) / 2)).toBeLessThanOrEqual(1);
        }
      }
      expect(maxErr, `${encoding} ${jpegBase} ${tone}`).toBeLessThan(0.5 / 255);
    }
  });
});
