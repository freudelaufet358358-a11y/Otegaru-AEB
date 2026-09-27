// 「Leica M10 の色」のワーカー。開いた写真（または AEB の合成結果）を id ごとに保持し、Leica M10 の色を掛けた
// プレビューと書き出しを行う。まとめて保存では、写真を 1 枚ずつ開いて書き出し、すぐに手放す。

import { toRGBA8 } from '../core/adjust';
import { toDisplay16 } from '../core/color';
import type { ExposureInfo } from '../core/exif';
import { fitSize, fullView, resizeView, type Frame16 } from '../core/frame';
import { LeicaLook, parseLookModel, type LookParams, type LookSource } from '../core/look';
import { crc32 } from '../core/zip';
import lookModelJson from '../models/leica-m10.json';
import { decodeImageFile, encodePixels, finishPixels } from './image-io';
import type { ExportOptions, FromLeicaWorker, LeicaParams, ToLeicaWorker } from './protocol';

interface Preview {
  /** 縮小した表示用 16bit RGB（Leica M10 の色を掛ける前） */
  data: Uint16Array;
  width: number;
  height: number;
  /** 最後に Leica M10 の色を掛けたもの（仕上げ調整だけを変えたときに使い回す） */
  cache: { key: string; display: Uint16Array } | null;
}

interface Source {
  /** 元の画像。RAW はリニア、JPEG などの写真と AEB の合成結果は表示用 (sRGB) の値 */
  frame: Frame16;
  exif: ExposureInfo;
  /** 元の画像の出どころ（形式・機種・ホワイトバランス）に合わせて用意した Leica M10 の色 */
  look: LeicaLook;
  /** LeicaLook.toLinear の jpegBase（AEB の「おまかせ」の結果だけ 0 より大きい） */
  jpegBase: number;
  /** 書き出しだけに開いたときは null */
  preview: Preview | null;
}

const ctx = self as unknown as {
  postMessage(msg: FromLeicaWorker, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<ToLeicaWorker>) => void) | null;
};

const lookModel = parseLookModel(lookModelJson);
const sources = new Map<string, Source>();

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
      sources.delete(msg.id);
      const frame: Frame16 = { width: msg.width, height: msg.height, data: msg.data, encoding: 'linear' };
      const { exif, color } = msg;
      const from: LookSource = { encoding: 'linear', make: exif.make, model: exif.model, neutral: color?.neutral, camXyz: color?.camXyz };
      open(msg.reqId, msg.id, frame, exif, from, 0, msg.previewSide);
      break;
    }
    case 'openFile': {
      sources.delete(msg.id); // 先に手放して、読み込み中のメモリを減らす
      const { frame, exif } = await decodeImageFile(msg.file);
      open(msg.reqId, msg.id, frame, exif, { encoding: 'srgb', make: exif.make, model: exif.model }, 0, msg.previewSide);
      break;
    }
    case 'openMerged': {
      sources.delete(msg.id);
      const m = msg.image;
      const frame: Frame16 = { width: m.width, height: m.height, data: m.data, encoding: 'srgb' };
      open(msg.reqId, msg.id, frame, m.exif, m.look, m.jpegBase, msg.previewSide);
      break;
    }
    case 'release':
      sources.delete(msg.id);
      break;
    case 'render': {
      const s = requireSource(msg.id);
      const p = s.preview;
      if (!p) throw new Error('プレビューを用意していない写真です');
      const t = performance.now();
      const key = lookKey(msg.params.look);
      if (!p.cache || p.cache.key !== key) {
        p.cache = { key, display: applyLook(s, msg.params.look, p.data, new Uint16Array(p.data.length)) };
      }
      const size = p.width * p.height * 4;
      const rgba = new Uint8ClampedArray(size);
      toRGBA8(p.cache.display, msg.params.adjust, rgba);
      const before = new Uint8ClampedArray(size);
      toRGBA8(p.data, msg.params.adjust, before);
      post(
        { type: 'rendered', reqId: msg.reqId, width: p.width, height: p.height, rgba, before, elapsed: performance.now() - t },
        [rgba.buffer, before.buffer],
      );
      break;
    }
    case 'export': {
      const t = performance.now();
      const { blob, width, height } = await exportImage(requireSource(msg.id), msg.params, msg.options, (label, f) =>
        post({ type: 'progress', reqId: msg.reqId, label, fraction: f }),
      );
      const crc = msg.crc ? crc32(new Uint8Array(await blob.arrayBuffer())) : undefined;
      post({ type: 'exported', reqId: msg.reqId, blob, width, height, elapsed: performance.now() - t, crc });
      break;
    }
  }
}

function requireSource(id: string): Source {
  const s = sources.get(id);
  if (!s) throw new Error('先に写真を開いてください');
  return s;
}

/** 写真を開く: 出どころに合わせた Leica M10 の色を用意し、previewSide が 0 でなければプレビュー用に縮小する */
function open(reqId: number, id: string, frame: Frame16, exif: ExposureInfo, from: LookSource, jpegBase: number, previewSide: number): void {
  let preview: Preview | null = null;
  if (previewSide > 0) {
    const [pw, ph] = fitSize(frame.width, frame.height, previewSide);
    const small = pw === frame.width && ph === frame.height ? null : resizeView(fullView(frame), pw, ph);
    // 縮小したものは表示用に置き換えてよいが、元の画像はそのまま残す
    const data = small ? toDisplay16(small.data, small.encoding, small.data) : toDisplay16(frame.data, frame.encoding);
    preview = { data, width: pw, height: ph, cache: null };
  }
  const look = new LeicaLook(lookModel, from);
  sources.set(id, { frame, exif, look, jpegBase, preview });
  post({
    type: 'opened',
    reqId,
    info: {
      width: frame.width,
      height: frame.height,
      previewWidth: preview?.width ?? 0,
      previewHeight: preview?.height ?? 0,
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
  s: Source,
  params: LeicaParams,
  opts: ExportOptions,
  progress: (label: string, f: number) => void,
): Promise<{ blob: Blob; width: number; height: number }> {
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
