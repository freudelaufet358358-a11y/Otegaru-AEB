// 「AEB 合成」の合成処理を担当するワーカー。フレームを保持し、位置合わせ・プレビュー生成・書き出しと、
// 「Leica M10 の色」に渡すフル解像度の合成を行う。

import { blendWithReference, toRGBA8, DEFAULT_ADJUSTMENTS } from '../core/adjust';
import { alignFrames, toGray8 } from '../core/align';
import { displayLut, linearLut, linearToDisplay, LUMA_B, LUMA_G, LUMA_R, RAW_DISPLAY_GAIN, toDisplay16 } from '../core/color';
import { exposureValue, type ExposureInfo } from '../core/exif';
import { fitSize, fullView, resizeView, rowIndex, type Frame16, type View } from '../core/frame';
import { exposureFusion, type ProgressFn } from '../core/fusion';
import { estimateExposures, toneMapHDR } from '../core/hdr';
import { parseToneModel, predictGrid, radianceThumbnail, renderLearned, type ToneGrid, type ToneModel } from '../core/learned';
import modelUrl from '../models/tone-hdrplus.bin?url';
import { decodeImageFile, encodePixels, finishPixels } from './image-io';
import {
  angleDegrees,
  apply,
  compose,
  IDENTITY,
  invert,
  isIntegerTranslation,
  rotationAbout,
  sampleBilinear,
  validCrop,
  warp,
  type Rect,
  type Similarity,
} from '../core/transform';
import type {
  ExportOptions,
  FrameAlignment,
  FromWorker,
  LoupeMode,
  ManualAdjust,
  MergedImage,
  PreparedInfo,
  RawColor,
  RenderParams,
  ToWorker,
} from './protocol';

interface Entry {
  id: string;
  name: string;
  frame: Frame16;
  exif: ExposureInfo;
  color?: RawColor;
}

interface Prepared {
  info: PreparedInfo;
  entries: Entry[]; // 暗い→明るい順
  refIndex: number;
  width: number; // フレームの実寸
  height: number;
  aligned: boolean;
  auto: Similarity[]; // 自動位置合わせ（基準の座標 → 各フレームの座標）
  precise: boolean[];
  manual: ManualAdjust[];
  transforms: Similarity[]; // 手動調整込みの最終的な変換
  crop: Rect;
  exposures: number[]; // 基準 = 1
  previewSide: number;
  previewSrc: Frame16[]; // プレビュー用に縮小した元フレーム（ワープ前）
  preview: Frame16[]; // 位置合わせ済みのプレビュー用フレーム
  layoutDirty: boolean;
  /** おまかせモードのグリッド（プレビュー用フレームから求め、書き出しでも同じものを使う） */
  toneGrid: ToneGrid | null;
}

const NO_MANUAL: ManualAdjust = { x: 0, y: 0, rotation: 0 };

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
    else if (msg.type === 'addRaw' || msg.type === 'addFile') post({ type: 'addFailed', id: msg.id, message });
    console.error(err);
  });
};

async function handle(msg: ToWorker): Promise<void> {
  switch (msg.type) {
    case 'addRaw': {
      const frame: Frame16 = { width: msg.width, height: msg.height, data: msg.data, encoding: 'linear' };
      entries.set(msg.id, { id: msg.id, name: msg.name, frame, exif: msg.exif, color: msg.color });
      invalidate();
      post({ type: 'added', id: msg.id, width: msg.width, height: msg.height, exif: msg.exif });
      break;
    }
    case 'addFile': {
      const { frame, exif } = await decodeImageFile(msg.file);
      entries.set(msg.id, { id: msg.id, name: msg.name, frame, exif });
      invalidate();
      post({ type: 'added', id: msg.id, width: frame.width, height: frame.height, exif });
      break;
    }
    case 'remove':
      entries.delete(msg.id);
      invalidate();
      break;
    case 'prepare': {
      prepared = prepare(msg.align, msg.previewSide, (label, f) => post({ type: 'progress', reqId: msg.reqId, label, fraction: f }));
      const reference = referenceRGBA(prepared);
      post({ type: 'prepared', reqId: msg.reqId, info: prepared.info, reference }, [reference.buffer]);
      break;
    }
    case 'setManual': {
      const p = prepared;
      if (!p) break;
      const k = p.entries.findIndex((e) => e.id === msg.id);
      if (k < 0 || k === p.refIndex) break;
      p.manual[k] = { ...msg.manual };
      updateTransforms(p);
      p.layoutDirty = true;
      break;
    }
    case 'loupe': {
      const p = requirePrepared();
      ensureLayout(p);
      const rgba = loupe(p, msg.id, msg.cx, msg.cy, msg.size, msg.mode);
      post({ type: 'loupe', reqId: msg.reqId, size: msg.size, rgba }, [rgba.buffer]);
      break;
    }
    case 'render': {
      const p = requirePrepared();
      const t = performance.now();
      let layout: { info: PreparedInfo; reference: Uint8ClampedArray } | undefined;
      if (p.layoutDirty) {
        ensureLayout(p);
        layout = { info: p.info, reference: referenceRGBA(p) };
      }
      const key = baseKey(msg.params);
      if (!cache || cache.key !== key) {
        const model = msg.params.mode === 'learned' ? await loadToneModel() : null;
        const views = p.preview.map(fullView);
        const display = renderBase(views, p, msg.params, model, (f) => post({ type: 'progress', reqId: msg.reqId, label: '合成中', fraction: f }));
        cache = { key, display, width: views[0].width, height: views[0].height };
      }
      let display = cache.display;
      if (msg.params.amount < 1) {
        const ref = p.preview[p.refIndex];
        display = blendWithReference(display, fullView(ref), displayLut(ref.encoding), msg.params.amount, new Uint16Array(display.length));
      }
      const rgba = new Uint8ClampedArray(cache.width * cache.height * 4);
      toRGBA8(display, msg.params.adjust, rgba);
      const transfer: Transferable[] = [rgba.buffer];
      if (layout) transfer.push(layout.reference.buffer);
      post(
        { type: 'rendered', reqId: msg.reqId, width: cache.width, height: cache.height, rgba, elapsed: performance.now() - t, layout },
        transfer,
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
    case 'merge': {
      const image = await mergedImage(msg.params, (label, f) => post({ type: 'progress', reqId: msg.reqId, label, fraction: f }));
      post({ type: 'merged', reqId: msg.reqId, image }, [image.data.buffer]);
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

  // 位置合わせ（回転と 1 画素未満のずれまで）
  const auto: Similarity[] = sorted.map(() => IDENTITY);
  const precise = sorted.map(() => false);
  if (align) {
    progress('位置合わせ中', 0);
    const refGray = toGray8(sorted[refIndex].frame, displayLut(sorted[refIndex].frame.encoding));
    const maxShift = Math.max(32, Math.round(Math.max(w, h) * 0.04));
    let done = 0;
    for (let k = 0; k < sorted.length; k++) {
      if (k === refIndex) continue;
      const g = toGray8(sorted[k].frame, displayLut(sorted[k].frame.encoding));
      const r = alignFrames(refGray, g, maxShift);
      auto[k] = r.transform;
      precise[k] = r.precise;
      progress('位置合わせ中', ++done / (sorted.length - 1));
    }
  }

  const p: Prepared = {
    info: null as unknown as PreparedInfo,
    entries: sorted,
    refIndex,
    width: w,
    height: h,
    aligned: align,
    auto,
    precise,
    manual: sorted.map(() => ({ ...NO_MANUAL })),
    transforms: auto,
    crop: { x0: 0, y0: 0, width: w, height: h },
    exposures: sorted.map(() => 1),
    previewSide,
    previewSrc: [],
    preview: [],
    layoutDirty: true,
    toneGrid: null,
  };
  updateTransforms(p);

  // プレビュー用に縮小（ワープは縮小後に行う）
  const crop = validCrop(w, h, p.transforms);
  const [pw] = fitSize(crop.width, crop.height, previewSide);
  const s = pw / crop.width;
  const sw = Math.max(1, Math.round(w * s));
  const sh = Math.max(1, Math.round(h * s));
  p.previewSrc = sorted.map((e, k) => {
    progress('プレビューを準備中', k / sorted.length);
    return resizeView(fullView(e.frame), sw, sh);
  });
  ensureLayout(p);

  // 露出比を推定（リニア値で）
  const linLuts = p.preview.map((f) => linearLut(f.encoding));
  const fallback = exifEv.map((ev) => (ev === undefined ? NaN : Math.pow(2, ev)));
  p.exposures = estimateExposures(p.preview.map(fullView), linLuts, sorted.map((_, i) => i), refIndex, fallback);
  p.info.relativeEv = Object.fromEntries(sorted.map((e, k) => [e.id, Math.log2(p.exposures[k])]));
  return p;
}

/** 手動調整（内容をどれだけ動かすか）を自動の変換に合成する */
function updateTransforms(p: Prepared): void {
  const cx = (p.width - 1) / 2;
  const cy = (p.height - 1) / 2;
  p.transforms = p.auto.map((a, k) => {
    const m = p.manual[k];
    if (m.x === 0 && m.y === 0 && m.rotation === 0) return a;
    const motion = rotationAbout((m.rotation * Math.PI) / 180, cx, cy, m.x, m.y);
    return compose(a, invert(motion));
  });
}

/** 切り抜き範囲とプレビュー用フレームを作り直す */
function ensureLayout(p: Prepared): void {
  if (!p.layoutDirty) return;
  const crop = validCrop(p.width, p.height, p.transforms);
  const [pw, ph] = fitSize(crop.width, crop.height, p.previewSide);
  p.crop = crop;
  p.preview = p.previewSrc.map((src, k) => warp(src, p.width, p.height, p.transforms[k], crop, pw, ph, false));
  p.layoutDirty = false;
  p.toneGrid = null;
  cache = null;
  const cx = (p.width - 1) / 2;
  const cy = (p.height - 1) / 2;
  const alignment: Record<string, FrameAlignment> = {};
  p.entries.forEach((e, k) => {
    const [u, v] = apply(p.auto[k], cx, cy);
    alignment[e.id] = {
      auto: { dx: u - cx, dy: v - cy, rotation: angleDegrees(p.auto[k]), precise: p.precise[k] },
      manual: { ...p.manual[k] },
    };
  });
  p.info = {
    order: p.entries.map((e) => e.id),
    referenceId: p.entries[p.refIndex].id,
    width: crop.width,
    height: crop.height,
    previewWidth: pw,
    previewHeight: ph,
    alignment,
    relativeEv: p.info?.relativeEv ?? {},
    aligned: p.aligned,
  };
}

/** 基準フレームをそのまま表示した画像（比較用） */
function referenceRGBA(p: Prepared): Uint8ClampedArray {
  const ref = p.preview[p.refIndex];
  const rgba = new Uint8ClampedArray(ref.width * ref.height * 4);
  toRGBA8(toDisplay16(ref.data, ref.encoding), DEFAULT_ADJUSTMENTS, rgba);
  return rgba;
}

/**
 * 位置合わせ確認用のルーペ。フル解像度の等倍で、基準フレームと対象フレームを
 * 明るさをそろえて重ねる（blend）か、差の大きさ（diff）を表示する。cx, cy は切り抜き範囲内の 0..1。
 */
function loupe(p: Prepared, id: string, cx: number, cy: number, size: number, mode: LoupeMode): Uint8ClampedArray {
  const k = Math.max(0, p.entries.findIndex((e) => e.id === id));
  const frames = [p.entries[p.refIndex].frame, p.entries[k].frame];
  const ts = [p.transforms[p.refIndex], p.transforms[k]];
  const luts = frames.map((f) => linearLut(f.encoding));
  const gains = frames.map((f, i) => (f.encoding === 'linear' ? RAW_DISPLAY_GAIN : 1) / p.exposures[i === 0 ? p.refIndex : k]);
  const X0 = p.crop.x0 + cx * (p.crop.width - 1) - size / 2 + 0.5;
  const Y0 = p.crop.y0 + cy * (p.crop.height - 1) - size / 2 + 0.5;
  const out = new Uint8ClampedArray(size * size * 4);
  const px = new Float64Array(3);
  const col = [new Float64Array(3), new Float64Array(3)];
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      for (let i = 0; i < 2; i++) {
        const f = frames[i];
        const [sx, sy] = apply(ts[i], X0 + u, Y0 + v);
        sampleBilinear(f.data, f.width, f.height, f.width * 3, sx, sy, px);
        for (let c = 0; c < 3; c++) col[i][c] = linearToDisplay(luts[i][Math.min(65535, px[c] | 0)] * gains[i]);
      }
      const o = (v * size + u) * 4;
      for (let c = 0; c < 3; c++) {
        out[o + c] = mode === 'blend' ? ((col[0][c] + col[1][c]) / 2) * 255 : Math.abs(col[0][c] - col[1][c]) * 3 * 255;
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

function baseKey(p: RenderParams): string {
  if (p.mode === 'fusion') return `f:${JSON.stringify(p.fusion)}`;
  if (p.mode === 'learned') return `l:${JSON.stringify(p.learned)}`;
  return `h:${JSON.stringify(p.tone)}`;
}

let modelPromise: Promise<ToneModel> | null = null;

/** 学習済みモデルを読み込む（初めて使うときに 1 回だけ） */
function loadToneModel(): Promise<ToneModel> {
  modelPromise ??= fetch(modelUrl)
    .then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return parseToneModel(await res.arrayBuffer());
    })
    .catch((e: unknown) => {
      modelPromise = null;
      throw new Error(`学習済みモデルを読み込めませんでした（${e instanceof Error ? e.message : String(e)}）`);
    });
  return modelPromise;
}

/** おまかせモードのグリッド。プレビュー用フレームの放射輝度から 1 回だけ求める */
function ensureToneGrid(p: Prepared, model: ToneModel): ToneGrid {
  if (!p.toneGrid) {
    const views = p.preview.map(fullView);
    const luts = views.map((v) => linearLut(v.frame.encoding));
    p.toneGrid = predictGrid(model, radianceThumbnail(views, luts, p.exposures, model.low));
  }
  return p.toneGrid;
}

/** 仕上げ調整前の合成結果（表示用 16bit RGB）を作る */
function renderBase(views: View[], p: Prepared, params: RenderParams, model: ToneModel | null, progress: ProgressFn): Uint16Array {
  const w = views[0].width;
  const h = views[0].height;
  const out = new Uint16Array(w * h * 3);
  if (params.mode === 'learned') {
    if (!model) throw new Error('学習済みモデルが読み込まれていません');
    const tg = ensureToneGrid(p, model);
    const luts = views.map((v) => linearLut(v.frame.encoding));
    renderLearned(views, luts, p.exposures, model, tg, params.learned, out, progress);
  } else if (params.mode === 'fusion') {
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

/** 仕上げ調整の前の合成結果（表示用 16bit RGB） */
interface Rendered {
  display: Uint16Array;
  width: number;
  height: number;
}

/**
 * 合成結果を書き出す大きさで作る。maxSide が 0 ならフル解像度。
 * mergeLabel は合成している間の進み具合の表示
 */
async function renderFull(
  params: RenderParams,
  maxSide: number,
  mergeLabel: string,
  progress: (label: string, f: number) => void,
): Promise<Rendered> {
  const p = requirePrepared();
  ensureLayout(p);
  const model = params.mode === 'learned' ? await loadToneModel() : null;
  const { crop, width: W, height: H } = p;
  const scaled = maxSide > 0 && Math.max(crop.width, crop.height) > maxSide;
  const [tw, th] = scaled ? fitSize(crop.width, crop.height, maxSide) : [crop.width, crop.height];
  // 位置合わせ済みのフレームを用意（整数の平行移動ならコピーせずにずらして参照する）
  const views: View[] = p.entries.map((e, k) => {
    progress('位置合わせを適用中', k / p.entries.length);
    const t = p.transforms[k];
    if (!scaled && isIntegerTranslation(t, 0.02, Math.max(W, H))) {
      return { frame: e.frame, x0: crop.x0 + Math.round(t.tx), y0: crop.y0 + Math.round(t.ty), width: crop.width, height: crop.height };
    }
    const src = scaled
      ? resizeView(fullView(e.frame), Math.max(1, Math.round((W * tw) / crop.width)), Math.max(1, Math.round((H * th) / crop.height)))
      : e.frame;
    return fullView(warp(src, W, H, t, crop, tw, th, true));
  });
  const display = renderBase(views, p, params, model, (f) => progress(mergeLabel, f));
  if (params.amount < 1) {
    const ref = views[p.refIndex];
    blendWithReference(display, ref, displayLut(ref.frame.encoding), params.amount);
  }
  return { display, width: views[0].width, height: views[0].height };
}

async function exportImage(
  params: RenderParams,
  opts: ExportOptions,
  progress: (label: string, f: number) => void,
): Promise<{ blob: Blob; width: number; height: number }> {
  const p = requirePrepared();
  let full: Rendered | null = await renderFull(params, opts.maxSide, '書き出し用に合成中', progress);
  const { width, height } = full;
  progress('ファイルを作成中', 1);
  const pixels = finishPixels(full.display, params.adjust, opts.format);
  full = null; // 合成結果のメモリを早めに手放す
  const blob = await encodePixels(pixels, width, height, opts, p.entries[p.refIndex].exif);
  return { blob, width, height };
}

/**
 * 「Leica M10 の色」に渡すフル解像度の合成結果。色の変換は基準の写真（形式・機種・ホワイトバランス）に合わせる。
 * おまかせは HDR+ の仕上がり（カメラ内 JPEG 並みのメリハリ）を学習しているので、その分だけ表示用の値を
 * 「カメラ内 JPEG 並みのトーン」として戻す（効果の強さで基準フレームと混ぜた分は素の表示として戻す）
 */
async function mergedImage(params: RenderParams, progress: (label: string, f: number) => void): Promise<MergedImage> {
  const p = requirePrepared();
  const { display, width, height } = await renderFull(params, 0, 'フル解像度で合成中', progress);
  const ref = p.entries[p.refIndex];
  return {
    width,
    height,
    data: display,
    exif: ref.exif,
    look: { encoding: ref.frame.encoding, make: ref.exif.make, model: ref.exif.model, neutral: ref.color?.neutral, camXyz: ref.color?.camXyz },
    jpegBase: params.mode === 'learned' ? params.amount : 0,
  };
}
