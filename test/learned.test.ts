import { describe, expect, it } from 'vitest';
import { linearLut } from '../src/core/color';
import { fullView, type Frame16 } from '../src/core/frame';
import { parseToneModel, predictGrid, radianceThumbnail, renderLearned, LearnedRenderer, DEFAULT_LEARNED_PARAMS } from '../src/core/learned';
import modelDataUrl from '../src/models/tone-hdrplus.bin?inline';
import fixtureText from './fixtures/learned.json?raw';

function loadModel() {
  const b64 = modelDataUrl.slice(modelDataUrl.indexOf(',') + 1);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return parseToneModel(bytes.buffer);
}

/** training/export.py の基準データと同じ合成画像（CHW、約 10 段の明暗差 + 模様） */
function syntheticCHW(h: number, w: number): Float32Array {
  const out = new Float32Array(3 * h * w);
  const n = h * w;
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const yy = (i + 0.5) / h;
      const xx = (j + 0.5) / w;
      const base = Math.pow(2, 10 * xx - 8) * (1 + 0.5 * Math.sin(yy * 20) * Math.cos(xx * 13));
      const p = i * w + j;
      out[p] = base * 0.9;
      out[n + p] = base;
      out[2 * n + p] = base * (0.6 + 0.4 * yy);
    }
  }
  return out;
}

describe('学習済みトーンレンダラー', () => {
  const model = loadModel();
  const fixture = JSON.parse(fixtureText) as {
    key: number;
    grid: number[];
    full_hw: [number, number];
    out: number[];
  };

  it('モデルファイルを読める', () => {
    expect(model.low).toBe(256);
    expect(model.gridXY).toBe(16);
    expect(model.guide[0]).toBe(0);
    expect(model.guide[model.guide.length - 1]).toBeCloseTo(1, 3);
    for (let i = 1; i < model.guide.length; i++) expect(model.guide[i]).toBeGreaterThanOrEqual(model.guide[i - 1]);
  });

  it('グリッドの予測が学習コード (PyTorch) と一致する', () => {
    const tg = predictGrid(model, syntheticCHW(model.low, model.low));
    expect(tg.key).toBeCloseTo(fixture.key, 4);
    expect(tg.grid.length).toBe(fixture.grid.length);
    let maxErr = 0;
    for (let i = 0; i < tg.grid.length; i++) maxErr = Math.max(maxErr, Math.abs(tg.grid[i] - fixture.grid[i]));
    expect(maxErr).toBeLessThan(2e-3);
  });

  it('仕上げの出力が学習コード (PyTorch) と一致する', () => {
    const tg = predictGrid(model, syntheticCHW(model.low, model.low));
    const [h, w] = fixture.full_hw;
    const chw = syntheticCHW(h, w);
    const row = new Float32Array(w * 3);
    const out = new Float32Array(w * 3);
    const renderer = new LearnedRenderer(model, tg, DEFAULT_LEARNED_PARAMS, w, h);
    let maxErr = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) row[x * 3 + c] = chw[c * h * w + y * w + x];
      renderer.row(row, y, out);
      for (let i = 0; i < w * 3; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - fixture.out[y * w * 3 + i]));
    }
    expect(maxErr).toBeLessThan(3e-3);
  });

  it('露出の違う入力でもほぼ同じ仕上がりになり、露出補正で明るさが変わる', () => {
    const w = 64;
    const h = 40;
    const make = (gain: number): Frame16 => {
      const chw = syntheticCHW(h, w);
      const data = new Uint16Array(w * h * 3);
      for (let p = 0; p < w * h; p++) {
        for (let c = 0; c < 3; c++) data[p * 3 + c] = Math.min(65535, Math.round(chw[c * w * h + p] * gain * 65535));
      }
      return { width: w, height: h, data, encoding: 'linear' };
    };
    const lut = linearLut('linear');
    const run = (gain: number, exposure: number) => {
      const views = [fullView(make(gain))];
      const thumb = radianceThumbnail(views, [lut], [1], model.low);
      const tg = predictGrid(model, thumb);
      const out = new Uint16Array(w * h * 3);
      renderLearned(views, [lut], [1], model, tg, { exposure }, out);
      let acc = 0;
      for (let i = 0; i < out.length; i++) acc += out[i] / 65535;
      return acc / out.length;
    };
    // 合成画像の最大値は約 6 なので、どちらの露出でも白飛びしない倍率で比べる
    const a = run(0.04, 0);
    const b = run(0.16, 0);
    expect(Math.abs(a - b)).toBeLessThan(0.01);
    expect(run(0.16, 1)).toBeGreaterThan(b + 0.03);
  });
});
