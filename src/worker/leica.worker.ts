// 「Leica M10 の色」のワーカー。写真 1 枚（または AEB の合成結果）を保持し、Leica M10 の色を掛けた
// プレビューと書き出しを行う。

import { toRGBA8 } from '../core/adjust';
import { toDisplay16 } from '../core/color';
import type { ExposureInfo } from '../core/exif';
import { fitSize, fullView, resizeView, type Frame16 } from '../core/frame';
import { LeicaLook, parseLookModel, type LookParams, type LookSource } from '../core/look';
import lookModelJson from '../models/leica-m10.json';
import { decodeImageFile, encodePixels, finishPixels } from './image-io';
import type { ExportOptions, FromLeicaWorker, LeicaParams, ToLeicaWorker } from './protocol';

interface Source {
  /** 元の画像。RAW はリニア、JPEG などの写真と AEB の合成結果は表示用 (sRGB) の値 */
  frame: Frame16;
  exif: ExposureInfo;
  /** 元の画像の出どころ（形式・機種・ホワイトバランス）に合わせて用意した Leica M10 の色 */
  look: LeicaLook;
  /** LeicaLook.toLinear の jpegBase（AEB の「おまかせ」の結果だけ 0 より大きい） */
  jpegBase: number;
  /** プレビュー用に縮小した表示用 16bit RGB（Leica M10 の色を掛ける前） */
  preview: Uint16Array;
  previewWidth: number;
  previewHeight: number;
  /** 最後に Leica M10 の色を掛けたプレビュー（仕上げ調整だけを変えたときに使い回す） */
  cache: { key: string; display: Uint16Array } | null;
}

const ctx = self as unknown as {
  postMessage(msg: FromLeicaWorker, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<ToLeicaWorker>) => void) | null;
};

const lookModel = parseLookModel(lookModelJson);
let source: Source | null = null;

const post = (msg: FromLeicaWorker, transfer: Transferable[] = []) => ctx.postMessage(msg, transfer);

// 処理は 1 つずつ順番に行う
let queue: Promise<void> = Promise.resolve();
ctx.onmessage = (ev) => {
  const msg = ev.data;
  queue = queue.then(() => handle(msg)).catch((err) => {
    const message =
      err instanceof RangeError
        ? 'メモリが足りません。保存サイズを小さくするか、ほかのタブやアプリを閉じてから試してください'
        : err instanceof Error
          ? err.message
          : String(err);
    if ('reqId' in msg) post({ type: 'error', reqId: msg.reqId, message });
    console.error(err);
  });
};

async function handle(msg: ToLeicaWorker): Promise<void> {
  switch (msg.type) {
    case 'openRaw': {
      source = null;
      const frame: Frame16 = { width: msg.width, height: msg.height, data: msg.data, encoding: 'linear' };
      const { exif } = msg;
      open(msg.reqId, frame, exif, { encoding: 'linear', make: exif.make, model: exif.model, neutral: msg.color?.neutral, camXyz: msg.color?.camXyz }, 0, msg.previewSide);
      break;
    }
    case 'openFile': {
      source = null; // 先に手放して、読み込み中のメモリを減らす
      const { frame, exif } = await decodeImageFile(msg.file);
      open(msg.reqId, frame, exif, { encoding: 'srgb', make: exif.make, model: exif.model }, 0, msg.previewSide);
      break;
    }
    case 'openMerged': {
      source = null;
      const m = msg.image;
      const frame: Frame16 = { width: m.width, height: m.height, data: m.data, encoding: 'srgb' };
      open(msg.reqId, frame, m.exif, m.look, m.jpegBase, msg.previewSide);
      break;
    }
    case 'close':
      source = null;
      break;
    case 'render': {
      const s = requireSource();
      const t = performance.now();
      const key = lookKey(msg.params.look);
      if (!s.cache || s.cache.key !== key) {
        s.cache = { key, display: applyLook(s, msg.params.look, s.preview, new Uint16Array(s.preview.length)) };
      }
      const size = s.previewWidth * s.previewHeight * 4;
      const rgba = new Uint8ClampedArray(size);
      toRGBA8(s.cache.display, msg.params.adjust, rgba);
      const before = new Uint8ClampedArray(size);
      toRGBA8(s.preview, msg.params.adjust, before);
      post(
        { type: 'rendered', reqId: msg.reqId, width: s.previewWidth, height: s.previewHeight, rgba, before, elapsed: performance.now() - t },
        [rgba.buffer, before.buffer],
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

function requireSource(): Source {
  if (!source) throw new Error('先に写真を開いてください');
  return source;
}

/** 写真を開く: プレビュー用に縮小し、出どころに合わせた Leica M10 の色を用意する */
function open(reqId: number, frame: Frame16, exif: ExposureInfo, from: LookSource, jpegBase: number, previewSide: number): void {
  const [pw, ph] = fitSize(frame.width, frame.height, previewSide);
  const small = pw === frame.width && ph === frame.height ? null : resizeView(fullView(frame), pw, ph);
  // 縮小したものは表示用に置き換えてよいが、元の画像はそのまま残す
  const preview = small ? toDisplay16(small.data, small.encoding, small.data) : toDisplay16(frame.data, frame.encoding);
  const look = new LeicaLook(lookModel, from);
  source = { frame, exif, look, jpegBase, preview, previewWidth: pw, previewHeight: ph, cache: null };
  post({
    type: 'opened',
    reqId,
    info: {
      width: frame.width,
      height: frame.height,
      previewWidth: pw,
      previewHeight: ph,
      exif,
      jpeg: from.encoding === 'srgb',
      cameraMatched: look.cameraMatched,
      cct: look.cct,
    },
  });
}

function lookKey(look: LookParams): string {
  return `${look.amount}:${look.tone}`;
}

/** Leica M10 の色を表示用 16bit RGB に掛ける（out を省略すると src を書き換える） */
function applyLook(s: Source, look: LookParams, src: Uint16Array, out: Uint16Array = src): Uint16Array {
  return s.look.apply(src, look.amount, out, s.jpegBase, look.tone ? 1 : 0);
}

async function exportImage(
  params: LeicaParams,
  opts: ExportOptions,
  progress: (label: string, f: number) => void,
): Promise<{ blob: Blob; width: number; height: number }> {
  const s = requireSource();
  const { frame } = s;
  const scaled = opts.maxSide > 0 && Math.max(frame.width, frame.height) > opts.maxSide;
  const [width, height] = scaled ? fitSize(frame.width, frame.height, opts.maxSide) : [frame.width, frame.height];
  progress(scaled ? '縮小中' : '準備中', 0);
  let display: Uint16Array | null;
  if (scaled) {
    const small = resizeView(fullView(frame), width, height);
    display = toDisplay16(small.data, small.encoding, small.data);
  } else {
    display = toDisplay16(frame.data, frame.encoding); // 元の画像を書き換えないようコピーに掛ける
  }
  progress('Leica M10 の色に変換中', 0.3);
  applyLook(s, params.look, display);
  progress('ファイルを作成中', 0.8);
  const pixels = finishPixels(display, params.adjust, opts.format);
  display = null; // メモリを早めに手放す
  const blob = await encodePixels(pixels, width, height, opts, s.exif);
  return { blob, width, height };
}
