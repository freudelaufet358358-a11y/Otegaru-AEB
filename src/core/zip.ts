// ZIP ファイルの作成（「Leica M10 の色」でまとめて保存するときに使う）。
// JPEG・PNG はもともと圧縮されているので、圧縮せずにそのまま詰める（store）。
// ファイル名は UTF-8（一般目的ビット 11）。大きさや位置が 4GB を超えるときは ZIP64 の形式で書く。

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32（ZIP・PNG と同じもの）。crc に途中までの値を渡すと、続きのデータを足した値になる */
export function crc32(data: Uint8Array, crc = 0): number {
  let c = ~crc >>> 0;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

export interface ZipEntry {
  /** ZIP の中のファイル名 */
  name: string;
  data: Blob;
  /** data の CRC-32 */
  crc: number;
}

export interface ZipOptions {
  /** ファイルの日時（既定は今） */
  date?: Date;
  /** 必要がなくても ZIP64 の形式で書く（確認用） */
  zip64?: boolean;
}

const MAX16 = 0xffff;
const MAX32 = 0xffffffff;

/** MS-DOS 形式の日時（ZIP のヘッダに入れる、ローカル時刻・2 秒単位） */
function dosDateTime(d: Date): { time: number; date: number } {
  const year = Math.min(2107, Math.max(1980, d.getFullYear()));
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((year - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

/** ZIP64 の追加フィールド（ID 0x0001）。values は 8 バイトの値（大きさ・位置）を並べたもの */
function zip64Extra(values: number[]): Uint8Array<ArrayBuffer> {
  const b = new Uint8Array(4 + values.length * 8);
  const v = new DataView(b.buffer);
  v.setUint16(0, 0x0001, true);
  v.setUint16(2, values.length * 8, true);
  values.forEach((x, i) => v.setBigUint64(4 + i * 8, BigInt(x), true));
  return b;
}

/** entries を順に詰めた ZIP を作る（中身の Blob はコピーせずにつなぐ） */
export function createZip(entries: ZipEntry[], opts: ZipOptions = {}): Blob {
  const enc = new TextEncoder();
  const { time, date } = dosDateTime(opts.date ?? new Date());
  const parts: BlobPart[] = [];
  const central: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;

  for (const e of entries) {
    const name = enc.encode(e.name);
    const size = e.data.size;
    const zip64 = !!opts.zip64 || size >= MAX32 || offset >= MAX32;
    const version = zip64 ? 45 : 20;

    // ローカルファイルヘッダ（ZIP64 では元の大きさと圧縮後の大きさを両方入れる）
    const localExtra = zip64 ? zip64Extra([size, size]) : new Uint8Array(0);
    const local = new Uint8Array(30 + name.length + localExtra.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, version, true);
    lv.setUint16(6, 0x0800, true); // ファイル名は UTF-8
    lv.setUint16(8, 0, true); // 無圧縮
    lv.setUint16(10, time, true);
    lv.setUint16(12, date, true);
    lv.setUint32(14, e.crc >>> 0, true);
    lv.setUint32(18, zip64 ? MAX32 : size, true);
    lv.setUint32(22, zip64 ? MAX32 : size, true);
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, localExtra.length, true);
    local.set(name, 30);
    local.set(localExtra, 30 + name.length);

    // セントラルディレクトリのヘッダ（ZIP64 では大きさ 2 つと位置を追加フィールドに入れる）
    const cdExtra = zip64 ? zip64Extra([size, size, offset]) : new Uint8Array(0);
    const cd = new Uint8Array(46 + name.length + cdExtra.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, version, true); // 作成したバージョン（MS-DOS 互換）
    cv.setUint16(6, version, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, time, true);
    cv.setUint16(14, date, true);
    cv.setUint32(16, e.crc >>> 0, true);
    cv.setUint32(20, zip64 ? MAX32 : size, true);
    cv.setUint32(24, zip64 ? MAX32 : size, true);
    cv.setUint16(28, name.length, true);
    cv.setUint16(30, cdExtra.length, true);
    cv.setUint16(32, 0, true); // コメントの長さ
    cv.setUint16(34, 0, true); // ディスク番号
    cv.setUint16(36, 0, true); // 内部属性
    cv.setUint32(38, 0, true); // 外部属性
    cv.setUint32(42, zip64 ? MAX32 : offset, true);
    cd.set(name, 46);
    cd.set(cdExtra, 46 + name.length);

    parts.push(local, e.data);
    central.push(cd);
    offset += local.length + size;
  }

  const cdOffset = offset;
  const cdSize = central.reduce((a, c) => a + c.length, 0);
  parts.push(...central);
  const count = entries.length;
  const zip64End = !!opts.zip64 || count >= MAX16 || cdOffset >= MAX32 || cdSize >= MAX32;
  if (zip64End) {
    // ZIP64 の終端レコードとその位置
    const rec = new Uint8Array(56 + 20);
    const rv = new DataView(rec.buffer);
    rv.setUint32(0, 0x06064b50, true);
    rv.setBigUint64(4, 44n, true);
    rv.setUint16(12, 45, true);
    rv.setUint16(14, 45, true);
    rv.setUint32(16, 0, true);
    rv.setUint32(20, 0, true);
    rv.setBigUint64(24, BigInt(count), true);
    rv.setBigUint64(32, BigInt(count), true);
    rv.setBigUint64(40, BigInt(cdSize), true);
    rv.setBigUint64(48, BigInt(cdOffset), true);
    rv.setUint32(56, 0x07064b50, true);
    rv.setUint32(60, 0, true);
    rv.setBigUint64(64, BigInt(cdOffset + cdSize), true);
    rv.setUint32(72, 1, true);
    parts.push(rec);
  }
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, zip64End ? MAX16 : count, true);
  ev.setUint16(10, zip64End ? MAX16 : count, true);
  ev.setUint32(12, zip64End ? MAX32 : cdSize, true);
  ev.setUint32(16, zip64End ? MAX32 : cdOffset, true);
  ev.setUint16(20, 0, true);
  parts.push(end);
  return new Blob(parts, { type: 'application/zip' });
}
