// LibRaw (WASM) を使った RAW 現像。Canon EOS R6 Mark II の CR3（通常 RAW / C-RAW）を含め、
// LibRaw が対応する多くの RAW 形式を読み込める。
// 合成に使うため、ガンマ補正や自動明るさ補正をかけない「リニアな 16bit sRGB」で取り出す。

import { formatExifDate, type ExposureInfo } from './core/exif';
import type { RawColor } from './worker/protocol';

const RAW_EXTENSIONS = [
  'cr3', 'cr2', 'crw', 'nef', 'nrw', 'arw', 'srf', 'sr2', 'dng', 'raf', 'orf', 'rw2', 'pef', 'srw',
  '3fr', 'fff', 'iiq', 'erf', 'kdc', 'dcr', 'mef', 'mos', 'mrw', 'x3f', 'rwl', 'gpr',
];

export const RAW_ACCEPT = RAW_EXTENSIONS.map((e) => '.' + e).join(',');

export function isRawFile(file: File): boolean {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  return RAW_EXTENSIONS.includes(ext);
}

interface LibRawMetadata {
  width: number;
  height: number;
  flip: number;
  camera_make: string;
  camera_model: string;
  iso_speed: number;
  shutter: number;
  aperture: number;
  timestamp: Date;
  /** metadata(true) のときだけ入る */
  color_data?: { cam_mul?: number[]; cam_xyz?: number[][] };
}

interface LibRawInstance {
  open(buf: Uint8Array, settings: Record<string, unknown>): Promise<void>;
  metadata(full?: boolean): Promise<LibRawMetadata>;
  imageData(): Promise<{ width: number; height: number; colors: number; bits: number; data: Uint16Array | Uint8Array }>;
  thumbnailData(): Promise<{ data: Uint8Array; width: number; height: number; format: string }>;
  dispose(): void;
}

type LibRawCtor = new () => LibRawInstance;

let ctorPromise: Promise<LibRawCtor> | null = null;

function loadLibRaw(): Promise<LibRawCtor> {
  if (!crossOriginIsolated) {
    return Promise.reject(
      new Error('このページはクロスオリジン分離されていないため RAW を読み込めません（ページを再読み込みしてください）'),
    );
  }
  // バンドラを通さず、public/vendor に置いた LibRaw をそのまま読み込む
  ctorPromise ??= import(/* @vite-ignore */ new URL('vendor/libraw/index.js', document.baseURI).href).then(
    (m: { default: LibRawCtor }) => m.default,
  );
  return ctorPromise;
}

export interface RawMeta {
  exif: ExposureInfo;
  /** 埋め込みプレビュー JPEG（あれば） */
  preview?: Blob;
  /** 画像の向き (LibRaw の flip: 0, 3=180°, 5=90°反時計回り, 6=90°時計回り) */
  flip: number;
  /** 撮影時のホワイトバランスと色行列（取れなければ undefined） */
  color?: RawColor;
}

export interface RawImage {
  width: number;
  height: number;
  data: Uint16Array;
}

/** LibRaw の設定: リニア・16bit・カメラのホワイトバランス・sRGB */
function settings(half: boolean): Record<string, unknown> {
  return {
    outputBps: 16,
    gamm: [1, 1],
    noAutoBright: true,
    useCameraWb: true,
    outputColor: 1,
    highlight: 0,
    userQual: 3,
    halfSize: half,
  };
}

/**
 * RAW を 1 枚ずつ順番に現像するデコーダ。
 * WASM のメモリは解放されても縮まないので、使い終わったら dispose() でワーカーごと破棄する。
 */
export class RawDecoder {
  private instance: LibRawInstance | null = null;

  private async get(): Promise<LibRawInstance> {
    if (!this.instance) {
      const LibRaw = await loadLibRaw();
      this.instance = new LibRaw();
    }
    return this.instance;
  }

  /** ファイルを開いてメタデータと埋め込みプレビューを取り出す（高速） */
  async open(file: File, half: boolean): Promise<RawMeta> {
    const raw = await this.get();
    const buf = new Uint8Array(await file.arrayBuffer());
    try {
      await raw.open(buf, settings(half));
    } catch (e) {
      throw new Error(`RAW として開けませんでした（${(e as Error).message}）`);
    }
    // 色の情報（color_data）は詳細なメタデータにしか入らない
    const m = await raw.metadata(true).catch(() => raw.metadata(false));
    const exif: ExposureInfo = {
      exposureTime: m.shutter > 0 ? m.shutter : undefined,
      fNumber: m.aperture > 0 ? m.aperture : undefined,
      iso: m.iso_speed > 0 ? m.iso_speed : undefined,
      make: m.camera_make || undefined,
      model: m.camera_model ? `${m.camera_make ? m.camera_make + ' ' : ''}${m.camera_model}` : undefined,
      dateTime: m.timestamp instanceof Date && !isNaN(m.timestamp.getTime()) && m.timestamp.getTime() > 0 ? formatExifDate(m.timestamp) : undefined,
    };
    let preview: Blob | undefined;
    try {
      const t = await raw.thumbnailData();
      if (t.format === 'jpeg' && t.data?.length) preview = new Blob([t.data as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' });
    } catch {
      // プレビューがなくても現像はできる
    }
    return { exif, preview, flip: m.flip, color: rawColor(m) };
  }

  /** open() 済みのファイルを現像する（時間がかかる） */
  async develop(): Promise<RawImage> {
    const raw = await this.get();
    const img = await raw.imageData();
    if (img.bits !== 16 || img.colors !== 3 || !(img.data instanceof Uint16Array)) {
      throw new Error('RAW の現像結果が想定外の形式です');
    }
    return { width: img.width, height: img.height, data: img.data };
  }

  dispose(): void {
    this.instance?.dispose();
    this.instance = null;
  }
}

let rawChain: Promise<void> = Promise.resolve();
let rawWaiting = 0;
let sharedDecoder: RawDecoder | null = null;

/**
 * RAW を扱う処理を 1 つずつ順番に行う（LibRaw は一度に 1 枚しか開けないので、どのタブからもこれを通す）。
 * 待っている処理がなくなったら、WASM のメモリを解放するためにデコーダを破棄する。
 */
export function withRawDecoder<T>(task: (decoder: RawDecoder) => Promise<T>): Promise<T> {
  rawWaiting++;
  const run = rawChain.then(() => task((sharedDecoder ??= new RawDecoder())));
  rawChain = run
    .then(
      () => undefined,
      () => undefined,
    )
    .finally(() => {
      if (--rawWaiting === 0) {
        sharedDecoder?.dispose();
        sharedDecoder = null;
      }
    });
  return run;
}

/** 撮影時のホワイトバランスの倍率の逆数（カメラのニュートラル）と、XYZ → カメラ RGB の色行列 */
function rawColor(m: LibRawMetadata): RawColor | undefined {
  const mul = m.color_data?.cam_mul;
  const xyz = m.color_data?.cam_xyz;
  if (!mul || !xyz || xyz.length < 3 || !(mul[0] > 0 && mul[1] > 0 && mul[2] > 0)) return undefined;
  const camXyz = xyz.slice(0, 3).flatMap((row) => row.slice(0, 3));
  if (camXyz.length !== 9 || camXyz.every((v) => v === 0) || camXyz.some((v) => !Number.isFinite(v))) return undefined;
  return { neutral: [mul[1] / mul[0], 1, mul[1] / mul[2]], camXyz };
}

/** サムネイル画像（object URL）を作る。flip は RAW の向き情報 */
export async function makeThumbnail(blob: Blob, flip = 0, size = 240): Promise<string> {
  // 幅だけ指定すると縦横比を保って縮小される
  const bmp = await createImageBitmap(blob, { resizeWidth: size, resizeQuality: 'medium' });
  const w = bmp.width;
  const h = bmp.height;
  const rotate = flip === 3 ? 180 : flip === 5 ? -90 : flip === 6 ? 90 : 0;
  const canvas = document.createElement('canvas');
  const swap = rotate === 90 || rotate === -90;
  canvas.width = swap ? h : w;
  canvas.height = swap ? w : h;
  const g = canvas.getContext('2d')!;
  g.translate(canvas.width / 2, canvas.height / 2);
  g.rotate((rotate * Math.PI) / 180);
  g.drawImage(bmp, -w / 2, -h / 2);
  bmp.close();
  const out = await new Promise<Blob | null>((r) => canvas.toBlob(r, 'image/jpeg', 0.85));
  return out ? URL.createObjectURL(out) : '';
}
