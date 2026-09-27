// アプリの実際の合成コードで、評価用ブラケットを各モードで仕上げる（出力は 8bit RGB の生データ）。
// 使い方（リポジトリのルートで）:
//   EVAL_DIR=training/eval/scenes EVAL_OUT=training/eval/out npx vitest run --config training/eval/vitest.config.ts
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { it } from 'vitest';
import { displayLut, linearLut } from '../../src/core/color';
import { fullView, type Frame16, type View } from '../../src/core/frame';
import { exposureFusion, DEFAULT_FUSION_WEIGHTS } from '../../src/core/fusion';
import { estimateExposures, toneMapHDR, DEFAULT_TONE_PARAMS } from '../../src/core/hdr';
import { parseToneModel, predictGrid, radianceThumbnail, renderLearned, DEFAULT_LEARNED_PARAMS } from '../../src/core/learned';

const DIR = process.env.EVAL_DIR!;
const OUT = process.env.EVAL_OUT!;
const MODEL = process.env.EVAL_MODEL ?? 'src/models/tone-hdrplus.bin';
const MODES = (process.env.EVAL_MODES ?? 'ref,fusion,hdr,learned').split(',');

function to8(src: Uint16Array): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = Math.round(src[i] / 257);
  return out;
}

it('evaluate', () => {
  mkdirSync(OUT, { recursive: true });
  const buf = readFileSync(MODEL);
  const model = parseToneModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  const names = JSON.parse(readFileSync(`${DIR}/index.json`).toString()) as string[];
  const times: Record<string, number> = {};
  for (const name of names) {
    const b = readFileSync(`${DIR}/${name}.bin`);
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const w = dv.getUint32(0, true);
    const h = dv.getUint32(4, true);
    const n = dv.getUint32(8, true);
    const encoding = dv.getUint32(12, true) === 1 ? 'srgb' : 'linear';
    const nominal: number[] = [];
    for (let k = 0; k < n; k++) nominal.push(dv.getFloat32(16 + k * 4, true));
    const frames: Frame16[] = [];
    let off = 16 + n * 4;
    for (let k = 0; k < n; k++) {
      const data = new Uint16Array(w * h * 3);
      for (let i = 0; i < data.length; i++) data[i] = dv.getUint16(off + i * 2, true);
      off += data.length * 2;
      frames.push({ width: w, height: h, data, encoding });
    }
    const views: View[] = frames.map(fullView);
    const ref = Math.floor((n - 1) / 2);
    const lin = views.map(() => linearLut(encoding));
    // アプリと同じく露出比は画像から推定する
    const exposures = estimateExposures(views, lin, views.map((_, i) => i), ref, nominal.map((e) => e / nominal[ref]));
    for (const mode of MODES) {
      const t0 = performance.now();
      const out = new Uint16Array(w * h * 3);
      if (mode === 'ref') {
        const lut = displayLut(encoding);
        const d = frames[ref].data;
        for (let i = 0; i < d.length; i++) out[i] = lut[d[i]] * 65535 + 0.5;
      } else if (mode === 'fusion') {
        const luts = views.map(() => displayLut(encoding));
        exposureFusion(views, luts, DEFAULT_FUSION_WEIGHTS, (c, plane) => {
          for (let i = 0, o = c; i < plane.length; i++, o += 3) {
            const v = plane[i] * 65535 + 0.5;
            out[o] = v <= 0 ? 0 : v >= 65535 ? 65535 : v | 0;
          }
        });
      } else if (mode === 'hdr') {
        toneMapHDR(views, lin, exposures, DEFAULT_TONE_PARAMS, out);
      } else if (mode === 'learned') {
        const tg = predictGrid(model, radianceThumbnail(views, lin, exposures, model.low));
        renderLearned(views, lin, exposures, model, tg, DEFAULT_LEARNED_PARAMS, out);
      }
      times[mode] = (times[mode] ?? 0) + performance.now() - t0;
      writeFileSync(`${OUT}/${name}_${mode}.rgb`, to8(out));
    }
    writeFileSync(`${OUT}/${name}.json`, JSON.stringify({ w, h, exposures }));
  }
  console.log('ms per scene', Object.fromEntries(Object.entries(times).map(([k, v]) => [k, Math.round(v / names.length)])));
}, 3_600_000);
