// JPEG / TIFF 系ファイルから、合成に必要な最小限の EXIF（露出情報など）を読む。

export interface ExposureInfo {
  /** シャッター速度 [秒] */
  exposureTime?: number;
  fNumber?: number;
  iso?: number;
  /** 露出補正 [EV] */
  exposureBias?: number;
  make?: string;
  model?: string;
  /** "YYYY:MM:DD HH:MM:SS" */
  dateTime?: string;
}

/** JPEG の APP1 (Exif) または TIFF ヘッダから読む。見つからなければ空オブジェクト */
export function readExif(buf: ArrayBuffer): ExposureInfo {
  const dv = new DataView(buf);
  try {
    if (dv.byteLength > 4 && dv.getUint16(0) === 0xffd8) {
      let p = 2;
      while (p + 4 <= dv.byteLength) {
        const marker = dv.getUint16(p);
        if ((marker & 0xff00) !== 0xff00) break;
        const len = dv.getUint16(p + 2);
        if (marker === 0xffe1 && p + 10 <= dv.byteLength && dv.getUint32(p + 4) === 0x45786966) {
          return parseTiff(dv, p + 10);
        }
        if (marker === 0xffda) break;
        p += 2 + len;
      }
      return {};
    }
    return parseTiff(dv, 0);
  } catch {
    return {};
  }
}

function parseTiff(dv: DataView, base: number): ExposureInfo {
  const bo = dv.getUint16(base);
  const le = bo === 0x4949;
  if (!le && bo !== 0x4d4d) return {};
  const u16 = (o: number) => dv.getUint16(base + o, le);
  const u32 = (o: number) => dv.getUint32(base + o, le);
  const i32 = (o: number) => dv.getInt32(base + o, le);
  if (u16(2) !== 42) return {};
  const info: ExposureInfo = {};

  const readIfd = (off: number, cb: (tag: number, type: number, count: number, valOff: number) => void) => {
    const n = u16(off);
    for (let i = 0; i < n; i++) {
      const e = off + 2 + i * 12;
      const type = u16(e + 2);
      const count = u32(e + 4);
      const size = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8][type] ?? 1;
      const valOff = size * count > 4 ? u32(e + 8) : e + 8;
      cb(u16(e), type, count, valOff);
    }
  };
  const str = (o: number, n: number) => {
    let s = '';
    for (let i = 0; i < n; i++) {
      const c = dv.getUint8(base + o + i);
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s.trim();
  };
  const num = (type: number, o: number) => {
    switch (type) {
      case 3:
        return u16(o);
      case 4:
        return u32(o);
      case 5:
        return u32(o) / u32(o + 4);
      case 10:
        return i32(o) / i32(o + 4);
      default:
        return NaN;
    }
  };

  let exifIfd = 0;
  readIfd(u32(4), (tag, type, count, o) => {
    if (tag === 0x010f) info.make = str(o, count);
    else if (tag === 0x0110) info.model = str(o, count);
    else if (tag === 0x0132 && !info.dateTime) info.dateTime = str(o, count);
    else if (tag === 0x8769) exifIfd = num(type, o);
  });
  if (exifIfd) {
    readIfd(exifIfd, (tag, type, count, o) => {
      if (tag === 0x829a) info.exposureTime = num(type, o);
      else if (tag === 0x829d) info.fNumber = num(type, o);
      else if (tag === 0x8827) info.iso = num(type, o);
      else if (tag === 0x9204) info.exposureBias = num(type, o);
      else if (tag === 0x9003) info.dateTime = str(o, count);
    });
  }
  for (const k of ['exposureTime', 'fNumber', 'iso', 'exposureBias'] as const) {
    if (info[k] !== undefined && !Number.isFinite(info[k])) delete info[k];
  }
  return info;
}

/**
 * 相対的な露光量（明るさ）を EV で返す。シャッター速度・絞り・ISO がそろっていれば計算、
 * なければ露出補正値を使う。どちらも無ければ undefined。
 */
export function exposureValue(e: ExposureInfo): number | undefined {
  if (e.exposureTime && e.exposureTime > 0) {
    const n = e.fNumber && e.fNumber > 0 ? e.fNumber : 1;
    const iso = e.iso && e.iso > 0 ? e.iso : 100;
    return Math.log2((e.exposureTime * iso) / 100 / (n * n));
  }
  if (e.exposureBias !== undefined) return e.exposureBias;
  return undefined;
}

/** 1/250 のようなシャッター速度表記 */
export function formatShutter(t?: number): string {
  if (!t || !(t > 0)) return '—';
  if (t >= 0.3) return `${Math.round(t * 10) / 10}″`;
  return `1/${Math.round(1 / t)}`;
}

export interface ExifOutput {
  make?: string;
  model?: string;
  software?: string;
  /** "YYYY:MM:DD HH:MM:SS" */
  dateTime?: string;
}

/** 書き出す JPEG に付ける最小限の EXIF (APP1 セグメント) を作る */
export function buildExifApp1(info: ExifOutput): Uint8Array {
  type E = { tag: number; type: number; count: number; data: Uint8Array };
  const enc = new TextEncoder();
  const ascii = (tag: number, s: string): E => {
    const data = enc.encode(s.replace(/[^\x20-\x7e]/g, '') + '\0');
    return { tag, type: 2, count: data.length, data };
  };
  const short = (tag: number, v: number): E => ({ tag, type: 3, count: 1, data: new Uint8Array([v & 255, v >> 8, 0, 0]) });
  const long = (tag: number, v: number): E => ({
    tag,
    type: 4,
    count: 1,
    data: new Uint8Array([v & 255, (v >> 8) & 255, (v >> 16) & 255, v >>> 24]),
  });

  const ifd0: E[] = [];
  if (info.make) ifd0.push(ascii(0x010f, info.make));
  if (info.model) ifd0.push(ascii(0x0110, info.model));
  ifd0.push(short(0x0112, 1));
  if (info.software) ifd0.push(ascii(0x0131, info.software));
  if (info.dateTime) ifd0.push(ascii(0x0132, info.dateTime));
  const exifPtr = long(0x8769, 0);
  ifd0.push(exifPtr);
  const exif: E[] = [];
  if (info.dateTime) exif.push(ascii(0x9003, info.dateTime));
  exif.push(short(0xa001, 1)); // ColorSpace = sRGB

  const ifdSize = (es: E[]) => 2 + es.length * 12 + 4 + es.reduce((a, e) => a + (e.data.length > 4 ? e.data.length + (e.data.length & 1) : 0), 0);
  const ifd0Off = 8;
  const exifOff = ifd0Off + ifdSize(ifd0);
  exifPtr.data = long(0, exifOff).data;
  const total = exifOff + ifdSize(exif);
  const tiff = new Uint8Array(total);
  const dv = new DataView(tiff.buffer);
  tiff[0] = 0x49;
  tiff[1] = 0x49;
  dv.setUint16(2, 42, true);
  dv.setUint32(4, ifd0Off, true);
  const writeIfd = (off: number, es: E[]) => {
    dv.setUint16(off, es.length, true);
    let extra = off + 2 + es.length * 12 + 4;
    es.forEach((e, i) => {
      const p = off + 2 + i * 12;
      dv.setUint16(p, e.tag, true);
      dv.setUint16(p + 2, e.type, true);
      dv.setUint32(p + 4, e.count, true);
      if (e.data.length <= 4) tiff.set(e.data, p + 8);
      else {
        dv.setUint32(p + 8, extra, true);
        tiff.set(e.data, extra);
        extra += e.data.length + (e.data.length & 1);
      }
    });
    dv.setUint32(off + 2 + es.length * 12, 0, true);
  };
  writeIfd(ifd0Off, ifd0);
  writeIfd(exifOff, exif);

  const seg = new Uint8Array(4 + 6 + total);
  seg[0] = 0xff;
  seg[1] = 0xe1;
  seg[2] = ((seg.length - 2) >> 8) & 255;
  seg[3] = (seg.length - 2) & 255;
  seg.set([0x45, 0x78, 0x69, 0x66, 0, 0], 4);
  seg.set(tiff, 10);
  return seg;
}

/** JPEG に EXIF を差し込む（SOI と JFIF(APP0) の直後） */
export function insertExif(jpeg: Uint8Array, app1: Uint8Array): Uint8Array {
  if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8) return jpeg;
  let p = 2;
  if (jpeg[p] === 0xff && jpeg[p + 1] === 0xe0) p += 2 + ((jpeg[p + 2] << 8) | jpeg[p + 3]);
  const out = new Uint8Array(jpeg.length + app1.length);
  out.set(jpeg.subarray(0, p), 0);
  out.set(app1, p);
  out.set(jpeg.subarray(p), p + app1.length);
  return out;
}

/** Date → "YYYY:MM:DD HH:MM:SS"（ローカル時刻） */
export function formatExifDate(d: Date): string {
  const z = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}:${z(d.getMonth() + 1)}:${z(d.getDate())} ${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
}
