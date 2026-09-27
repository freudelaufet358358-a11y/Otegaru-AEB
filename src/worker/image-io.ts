// ワーカーで使う画像の読み込み（ブラウザで開ける形式）と書き出し（JPEG / PNG / 16bit TIFF）。

import { toRGB16, toRGBA8, type Adjustments } from '../core/adjust';
import { buildExifApp1, formatExifDate, insertExif, readExif, type ExposureInfo } from '../core/exif';
import type { Frame16 } from '../core/frame';
import { encodeTiff16 } from '../core/tiff';
import type { ExportFormat, ExportOptions } from './protocol';

/** 書き出すファイルの EXIF・TIFF タグに入れるソフトウェア名 */
const SOFTWARE = 'お手軽AEB合成';

/** JPEG・PNG などをブラウザで開き、sRGB の符号値を 16bit に広げて取り出す */
export async function decodeImageFile(file: File): Promise<{ frame: Frame16; exif: ExposureInfo }> {
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
  return { frame: { width, height, data, encoding: 'srgb' }, exif };
}

/**
 * 表示用 16bit RGB に仕上げ調整を掛けて、書き出す形式の画素にする（TIFF は 16bit RGB、それ以外は 8bit RGBA）。
 * 大きな画像ではメモリを節約するため、呼び出し側はこの後 display を手放すこと。
 */
export function finishPixels(display: Uint16Array, adjust: Adjustments, format: ExportFormat): Uint16Array | Uint8ClampedArray {
  if (format === 'tiff') {
    const rgb = new Uint16Array(display.length);
    toRGB16(display, adjust, rgb);
    return rgb;
  }
  const rgba = new Uint8ClampedArray((display.length / 3) * 4);
  toRGBA8(display, adjust, rgba);
  return rgba;
}

/** finishPixels の画素をファイルにする。撮影した機種・日時を EXIF（TIFF はタグ）に入れる */
export async function encodePixels(
  pixels: Uint16Array | Uint8ClampedArray,
  w: number,
  h: number,
  opts: ExportOptions,
  exif: ExposureInfo,
): Promise<Blob> {
  if (pixels instanceof Uint16Array) {
    return encodeTiff16(w, h, pixels, { software: SOFTWARE, make: exif.make, model: exif.model, dateTime: exif.dateTime });
  }
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d');
  if (!g) throw new Error('Canvas が使えません');
  g.putImageData(new ImageData(pixels as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
  const type = opts.format === 'png' ? 'image/png' : 'image/jpeg';
  const blob = await canvas.convertToBlob({ type, quality: opts.quality });
  if (opts.format !== 'jpeg') return blob;
  const app1 = buildExifApp1({
    make: exif.make,
    model: exif.model,
    software: SOFTWARE,
    dateTime: exif.dateTime ?? formatExifDate(new Date()),
  });
  const bytes = insertExif(new Uint8Array(await blob.arrayBuffer()), app1);
  return new Blob([bytes as Uint8Array<ArrayBuffer>], { type });
}
