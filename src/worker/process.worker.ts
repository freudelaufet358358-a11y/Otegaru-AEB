// 合成処理を担当するワーカー。フレームを保持し、位置合わせ・プレビュー生成・書き出しを行う。

import { toRGB16, toRGBA8 } from '../core/adjust';
import { alignedViews, alignMTB, commonCrop, toGray8 } from '../core/align';
import { displayLut, linearLut, LUMA_B, LUMA_G, LUMA_R } from '../core/color';
import { buildExifApp1, exposureValue, formatExifDate, insertExif, readExif, type ExposureInfo } from '../core/exif';
import { fitSize, fullView, resizeView, rowIndex, type Frame16, type View } from '../core/frame';
import { exposureFusion, type ProgressFn } from '../core/fusion';
import { estimateExposures, toneMapHDR } from '../core/hdr';
import { encodeTiff16 } from '../core/tiff';
import type { ExportOptions, FromWorker, PreparedInfo, RenderParams, ToWorker } from './protocol';

interface Entry {
  id: string;
  name: string;
  frame: Frame16;
  exif: ExposureInfo;
}

interface Prepared {
  info: PreparedInfo;
  entries: Entry[]; // 暗い→明るい順
  refIndex: number;
  views: View[]; // フル解像度（切り抜き済みビュー）
  exposures: number[]; // 基準 = 1
  preview: Frame16[];
}

const ctx = self as unknown as {
  postMessage(msg: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<ToWorker>) => void) | null;
};

const entries = new Map<string, Entry>();
let prepared: Prepared | null = null;
let cache: { key: string; display: Uint16Array; width: number; height: number } | null = null;

const post = (msg: FromWorker, transfer: Transferable[] = []) => ctx.postMessage(msg, transfer);

// 処理は 1 つずつ順番に行う
let queue: Promise<void> = Promise.resolve();
ctx.onmessage = (ev) => {
  const msg = ev.data;
  queue = queue.then(() => handle(msg)).catch((err) => {
    const message =
      err instanceof RangeError
        ? 'メモリが足りません。「RAW の読み込み」を 1/2 サイズにするか、保存サイズを小さくしてください'
        : err instanceof Error
          ? err.message
          : String(err);
    if ('reqId' in msg) post({ type: 'error', reqId: msg.reqId, message });
    else if ('id' in msg) post({ type: 'addFailed', id: msg.id, message });
    console.error(err);
  });
};

async function handle(msg: ToWorker): Promise<void> {
  switch (msg.type) {
    case 'addRaw': {
      const frame: Frame16 = { width: msg.width, height: msg.height, data: msg.data, encoding: 'linear' };
      entries.set(msg.id, { id: msg.id, name: msg.name, frame, exif: msg.exif });
      invalidate();
      post({ type: 'added', id: msg.id, width: msg.width, height: msg.height, exif: msg.exif });
      break;
    }
    case 'addFile': {
      const entry = await decodeImageFile(msg.id, msg.name, msg.file);
      entries.set(msg.id, entry);
      invalidate();
      post({ type: 'added', id: msg.id, width: entry.frame.width, height: entry.frame.height, exif: entry.exif });
      break;
    }
    case 'remove':
      entries.delete(msg.id);
      invalidate();
      break;
    case 'prepare': {
      const t = performance.now();
      prepared = prepare(msg.align, msg.previewSide, (label, f) => post({ type: 'progress', reqId: msg.reqId, label, fraction: f }));
      cache = null;
      const ref = prepared.preview[prepared.refIndex];
      const rgba = new Uint8ClampedArray(ref.width * ref.height * 4);
      const lut = displayLut(ref.encoding);
      const disp = new Uint16Array(ref.data.length);
      for (let i = 0; i < disp.length; i++) disp[i] = lut[ref.data[i]] * 65535 + 0.5;
      toRGBA8(disp, { brightness: 0, contrast: 0, saturation: 0 }, rgba);
      post({ type: 'prepared', reqId: msg.reqId, info: prepared.info, reference: rgba }, [rgba.buffer]);
      console.debug(`prepare: ${(performance.now() - t).toFixed(0)}ms`);
      break;
    }
    case 'render': {
      const p = requirePrepared();
      const t = performance.now();
      const key = baseKey(msg.params);
      if (!cache || cache.key !== key) {
        const views = p.preview.map(fullView);
        const display = renderBase(views, p, msg.params, (f) => post({ type: 'progress', reqId: msg.reqId, label: '合成中', fraction: f }));
        cache = { key, display, width: views[0].width, height: views[0].height };
      }
      const rgba = new Uint8ClampedArray(cache.width * cache.height * 4);
      toRGBA8(cache.display, msg.params.adjust, rgba);
      post(
        { type: 'rendered', reqId: msg.reqId, width: cache.width, height: cache.height, rgba, elapsed: performance.now() - t },
        [rgba.buffer],
      );
      break;
    }
    case 'export': {
      const t = performance.now();
      const { blob, width, height } = await exportImage(msg.params, msg.options, (label, f) =>
        post({ type: 'progress', reqId: msg.reqId, label, fraction: f }),
      );
      post({ type: 'exported', reqId: msg.reqId, blob, width, height, elapsed: performance.now() - t });
      break;
    }
  }
}

function invalidate(): void {
  prepared = null;
  cache = null;
}

function requirePrepared(): Prepared {
  if (!prepared) throw new Error('先に画像を読み込んでください');
  return prepared;
}

async function decodeImageFile(id: string, name: string, file: File): Promise<Entry> {
  const head = await file.slice(0, 512 * 1024).arrayBuffer();
  const exif = readExif(head);
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { colorSpaceConversion: 'default', premultiplyAlpha: 'none' });
  } catch {
    throw new Error('この形式はブラウザで読み込めません');
  }
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  if (!g) throw new Error('Canvas が使えません');
  g.drawImage(bitmap, 0, 0);
  bitmap.close();
  const px = g.getImageData(0, 0, width, height).data;
  const data = new Uint16Array(width * height * 3);
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
    data[j] = px[i] * 257;
    data[j + 1] = px[i + 1] * 257;
    data[j + 2] = px[i + 2] * 257;
  }
  return { id, name, frame: { width, height, data, encoding: 'srgb' }, exif };
}

/** LUT で変換した輝度の平均（間引いて計算） */
function meanLuma(f: Frame16, lut: Float32Array): number {
  const v = fullView(f);
  const step = Math.max(1, Math.floor(Math.sqrt((f.width * f.height) / 40000)));
  let acc = 0;
  let n = 0;
  for (let y = 0; y < f.height; y += step) {
    const r = rowIndex(v, y);
    for (let x = 0; x < f.width; x += step) {
      const i = r + x * 3;
      acc += LUMA_R * lut[f.data[i]] + LUMA_G * lut[f.data[i + 1]] + LUMA_B * lut[f.data[i + 2]];
      n++;
    }
  }
  return acc / n;
}

function prepare(align: boolean, previewSide: number, progress: (label: string, f: number) => void): Prepared {
  const list = [...entries.values()];
  if (list.length < 2) throw new Error('2 枚以上の画像が必要です');
  const w = list[0].frame.width;
  const h = list[0].frame.height;
  for (const e of list) {
    if (e.frame.width !== w || e.frame.height !== h) {
      throw new Error(`画像サイズが一致しません（${e.name}: ${e.frame.width}×${e.frame.height}, 他: ${w}×${h}）`);
    }
  }

  // 暗い→明るい順に並べる。EXIF の露出がそろっていればそれを、なければ平均輝度を使う
  const evs = list.map((e) => exposureValue(e.exif));
  const useExif = evs.every((v) => v !== undefined) && new Set(evs.map((v) => v!.toFixed(2))).size === evs.length;
  const keys = useExif ? (evs as number[]) : list.map((e) => Math.log2(meanLuma(e.frame, linearLut(e.frame.encoding)) + 1e-6));
  const idx = list.map((_, i) => i).sort((a, b) => keys[a] - keys[b]);
  const sorted = idx.map((i) => list[i]);
  const exifEv = idx.map((i) => evs[i]);
  // 基準は中間の露出。偶数枚のときは中央の 2 枚のうち、より適正露出に近い方
  let refIndex = Math.floor((sorted.length - 1) / 2);
  if (sorted.length % 2 === 0) {
    const a = refIndex;
    const b = refIndex + 1;
    const target = 0.42;
    const mean = (k: number) => meanLuma(sorted[k].frame, displayLut(sorted[k].frame.encoding));
    if (Math.abs(mean(b) - target) < Math.abs(mean(a) - target)) refIndex = b;
  }

  // 位置合わせ
  const shifts: Array<[number, number]> = sorted.map(() => [0, 0]);
  if (align) {
    progress('位置合わせ中', 0);
    const refGray = toGray8(sorted[refIndex].frame, displayLut(sorted[refIndex].frame.encoding));
    const maxShift = Math.max(32, Math.round(Math.max(w, h) * 0.04));
    let done = 0;
    for (let k = 0; k < sorted.length; k++) {
      if (k === refIndex) continue;
      const g = toGray8(sorted[k].frame, displayLut(sorted[k].frame.encoding));
      shifts[k] = alignMTB(refGray, g, maxShift);
      progress('位置合わせ中', ++done / (sorted.length - 1));
    }
  }
  const crop = commonCrop(w, h, shifts);
  const views = alignedViews(
    sorted.map((e) => e.frame),
    shifts,
    crop,
  );

  // プレビュー用に縮小
  const [pw, ph] = fitSize(crop.width, crop.height, previewSide);
  const preview = views.map((v, k) => {
    progress('プレビューを準備中', k / views.length);
    return resizeView(v, pw, ph);
  });

  // 露出比を推定（リニア値で）
  const linLuts = preview.map((f) => linearLut(f.encoding));
  const fallback = exifEv.map((ev) => (ev === undefined ? NaN : Math.pow(2, ev)));
  const order = sorted.map((_, i) => i);
  const exposures = estimateExposures(preview.map(fullView), linLuts, order, refIndex, fallback);

  const info: PreparedInfo = {
    order: sorted.map((e) => e.id),
    referenceId: sorted[refIndex].id,
    width: crop.width,
    height: crop.height,
    previewWidth: pw,
    previewHeight: ph,
    shifts: Object.fromEntries(sorted.map((e, k) => [e.id, shifts[k]])),
    relativeEv: Object.fromEntries(sorted.map((e, k) => [e.id, Math.log2(exposures[k])])),
    aligned: align,
  };
  return { info, entries: sorted, refIndex, views, exposures, preview };
}

function baseKey(p: RenderParams): string {
  return p.mode === 'fusion' ? `f:${JSON.stringify(p.fusion)}` : `h:${JSON.stringify(p.tone)}`;
}

/** 仕上げ調整前の合成結果（表示用 16bit RGB）を作る */
function renderBase(views: View[], p: Prepared, params: RenderParams, progress: ProgressFn): Uint16Array {
  const w = views[0].width;
  const h = views[0].height;
  const out = new Uint16Array(w * h * 3);
  if (params.mode === 'fusion') {
    const luts = views.map((v) => displayLut(v.frame.encoding));
    exposureFusion(
      views,
      luts,
      params.fusion,
      (c, plane) => {
        for (let i = 0, o = c; i < plane.length; i++, o += 3) {
          const v = plane[i] * 65535 + 0.5;
          out[o] = v <= 0 ? 0 : v >= 65535 ? 65535 : v | 0;
        }
      },
      progress,
    );
  } else {
    const luts = views.map((v) => linearLut(v.frame.encoding));
    toneMapHDR(views, luts, p.exposures, params.tone, out, progress);
  }
  return out;
}

async function exportImage(
  params: RenderParams,
  opts: ExportOptions,
  progress: (label: string, f: number) => void,
): Promise<{ blob: Blob; width: number; height: number }> {
  const p = requirePrepared();
  const { width: cw, height: ch } = p.info;
  let views = p.views;
  if (opts.maxSide > 0 && Math.max(cw, ch) > opts.maxSide) {
    const [tw, th] = fitSize(cw, ch, opts.maxSide);
    views = views.map((v, k) => {
      progress('縮小中', k / views.length);
      return fullView(resizeView(v, tw, th));
    });
  }
  const w = views[0].width;
  const h = views[0].height;
  let display: Uint16Array | null = renderBase(views, p, params, (f) => progress('書き出し用に合成中', f));
  views = [];
  progress('ファイルを作成中', 1);

  const ref = p.entries[p.refIndex].exif;
  const software = 'お手軽AEB合成';
  if (opts.format === 'tiff') {
    const rgb = new Uint16Array(display.length);
    toRGB16(display, params.adjust, rgb);
    display = null;
    const blob = encodeTiff16(w, h, rgb, { software, make: ref.make, model: ref.model, dateTime: ref.dateTime });
    return { blob, width: w, height: h };
  }
  const rgba = new Uint8ClampedArray(w * h * 4);
  toRGBA8(display, params.adjust, rgba);
  display = null;
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d');
  if (!g) throw new Error('Canvas が使えません');
  g.putImageData(new ImageData(rgba, w, h), 0, 0);
  const type = opts.format === 'png' ? 'image/png' : 'image/jpeg';
  let blob = await canvas.convertToBlob({ type, quality: opts.quality });
  if (opts.format === 'jpeg') {
    const app1 = buildExifApp1({
      make: ref.make,
      model: ref.model,
      software,
      dateTime: ref.dateTime ?? formatExifDate(new Date()),
    });
    const bytes = insertExif(new Uint8Array(await blob.arrayBuffer()), app1);
    blob = new Blob([bytes as Uint8Array<ArrayBuffer>], { type });
  }
  return { blob, width: w, height: h };
}
