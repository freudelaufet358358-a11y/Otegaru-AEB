// 非圧縮 16bit RGB の TIFF（ベースライン, リトルエンディアン）を書き出す。
// Lightroom / Photoshop などでの追い込み用。

export interface TiffOptions {
  software?: string;
  /** EXIF 日時形式 "YYYY:MM:DD HH:MM:SS" */
  dateTime?: string;
  make?: string;
  model?: string;
}

interface Entry {
  tag: number;
  type: number; // 2=ASCII, 3=SHORT, 4=LONG, 5=RATIONAL
  count: number;
  /** 4 バイトに収まる値。収まらない場合は extra に書く */
  value?: number;
  extra?: Uint8Array;
}

const ROWS_PER_STRIP = 32;

export function encodeTiff16(width: number, height: number, rgb: Uint16Array, opts: TiffOptions = {}): Blob {
  const rowBytes = width * 6;
  const strips = Math.ceil(height / ROWS_PER_STRIP);
  const stripOffsets = new Uint32Array(strips);
  const stripCounts = new Uint32Array(strips);

  const ascii = (s: string) => {
    const b = new TextEncoder().encode(s + '\0');
    return b;
  };
  const shorts = (vals: number[]) => {
    const b = new Uint8Array(vals.length * 2);
    const dv = new DataView(b.buffer);
    vals.forEach((v, i) => dv.setUint16(i * 2, v, true));
    return b;
  };
  const rational = (num: number, den: number) => {
    const b = new Uint8Array(8);
    const dv = new DataView(b.buffer);
    dv.setUint32(0, num, true);
    dv.setUint32(4, den, true);
    return b;
  };

  const entries: Entry[] = [
    { tag: 256, type: 4, count: 1, value: width },
    { tag: 257, type: 4, count: 1, value: height },
    { tag: 258, type: 3, count: 3, extra: shorts([16, 16, 16]) },
    { tag: 259, type: 3, count: 1, value: 1 },
    { tag: 262, type: 3, count: 1, value: 2 },
  ];
  if (opts.make) entries.push({ tag: 271, type: 2, count: 0, extra: ascii(opts.make) });
  if (opts.model) entries.push({ tag: 272, type: 2, count: 0, extra: ascii(opts.model) });
  entries.push(
    { tag: 273, type: 4, count: strips, extra: new Uint8Array(strips * 4) },
    { tag: 277, type: 3, count: 1, value: 3 },
    { tag: 278, type: 4, count: 1, value: ROWS_PER_STRIP },
    { tag: 279, type: 4, count: strips, extra: new Uint8Array(strips * 4) },
    { tag: 282, type: 5, count: 1, extra: rational(300, 1) },
    { tag: 283, type: 5, count: 1, extra: rational(300, 1) },
    { tag: 284, type: 3, count: 1, value: 1 },
    { tag: 296, type: 3, count: 1, value: 2 },
  );
  if (opts.software) entries.push({ tag: 305, type: 2, count: 0, extra: ascii(opts.software) });
  if (opts.dateTime) entries.push({ tag: 306, type: 2, count: 0, extra: ascii(opts.dateTime) });
  for (const e of entries) if (e.type === 2 && e.extra) e.count = e.extra.length;
  // 1 本しかないときは値を直接 IFD に入れる
  const offsetsEntry = entries.find((e) => e.tag === 273)!;
  const countsEntry = entries.find((e) => e.tag === 279)!;

  const ifdOffset = 8;
  const ifdSize = 2 + entries.length * 12 + 4;
  let extraOffset = ifdOffset + ifdSize;
  const extraPos = new Map<Entry, number>();
  for (const e of entries) {
    if (e.extra && e.extra.length > 4) {
      extraPos.set(e, extraOffset);
      extraOffset += e.extra.length + (e.extra.length & 1);
    }
  }
  const dataOffset = extraOffset;
  for (let s = 0; s < strips; s++) {
    const rows = Math.min(ROWS_PER_STRIP, height - s * ROWS_PER_STRIP);
    stripOffsets[s] = dataOffset + s * ROWS_PER_STRIP * rowBytes;
    stripCounts[s] = rows * rowBytes;
  }
  writeLongs(offsetsEntry.extra!, stripOffsets);
  writeLongs(countsEntry.extra!, stripCounts);

  const header = new Uint8Array(dataOffset);
  const dv = new DataView(header.buffer);
  header[0] = 0x49;
  header[1] = 0x49;
  dv.setUint16(2, 42, true);
  dv.setUint32(4, ifdOffset, true);
  dv.setUint16(ifdOffset, entries.length, true);
  entries.sort((a, b) => a.tag - b.tag);
  entries.forEach((e, i) => {
    const p = ifdOffset + 2 + i * 12;
    dv.setUint16(p, e.tag, true);
    dv.setUint16(p + 2, e.type, true);
    dv.setUint32(p + 4, e.count, true);
    if (e.extra) {
      if (e.extra.length <= 4) header.set(e.extra, p + 8);
      else dv.setUint32(p + 8, extraPos.get(e)!, true);
    } else if (e.type === 3) {
      dv.setUint16(p + 8, e.value!, true);
    } else {
      dv.setUint32(p + 8, e.value!, true);
    }
  });
  dv.setUint32(ifdOffset + 2 + entries.length * 12, 0, true);
  for (const [e, pos] of extraPos) header.set(e.extra!, pos);

  // ピクセルはリトルエンディアンで書く（主要なブラウザ環境はリトルエンディアン）
  const pixels = littleEndian() ? rgb : swapBytes(rgb);
  return new Blob([header, pixels as Uint16Array<ArrayBuffer>], { type: 'image/tiff' });
}

function writeLongs(target: Uint8Array, vals: Uint32Array): void {
  const dv = new DataView(target.buffer, target.byteOffset, target.byteLength);
  vals.forEach((v, i) => dv.setUint32(i * 4, v, true));
}

function littleEndian(): boolean {
  return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
}

function swapBytes(src: Uint16Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = ((src[i] & 0xff) << 8) | (src[i] >> 8);
  return out;
}
