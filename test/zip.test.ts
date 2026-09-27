import { describe, expect, it } from 'vitest';
import { crc32, createZip } from '../src/core/zip';

interface ReadEntry {
  name: string;
  utf8: boolean;
  crc: number;
  data: Uint8Array;
  zip64: boolean;
  time: number;
  date: number;
}

/** 検証用の小さな ZIP の読み取り（終端レコード → セントラルディレクトリ → ローカルヘッダの順にたどる） */
function readZip(zip: Uint8Array): ReadEntry[] {
  const v = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const end = zip.length - 22;
  expect(v.getUint32(end, true)).toBe(0x06054b50);
  let count = v.getUint16(end + 10, true);
  let cdOffset = v.getUint32(end + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff) {
    const loc = end - 20;
    expect(v.getUint32(loc, true)).toBe(0x07064b50);
    const rec = Number(v.getBigUint64(loc + 8, true));
    expect(v.getUint32(rec, true)).toBe(0x06064b50);
    count = Number(v.getBigUint64(rec + 32, true));
    cdOffset = Number(v.getBigUint64(rec + 48, true));
  }
  const out: ReadEntry[] = [];
  let p = cdOffset;
  for (let k = 0; k < count; k++) {
    expect(v.getUint32(p, true)).toBe(0x02014b50);
    const flags = v.getUint16(p + 8, true);
    expect(v.getUint16(p + 10, true)).toBe(0); // 無圧縮
    const time = v.getUint16(p + 12, true);
    const date = v.getUint16(p + 14, true);
    const crc = v.getUint32(p + 16, true);
    let size = v.getUint32(p + 24, true);
    const nameLen = v.getUint16(p + 28, true);
    const extraLen = v.getUint16(p + 30, true);
    let offset = v.getUint32(p + 42, true);
    const name = new TextDecoder().decode(zip.subarray(p + 46, p + 46 + nameLen));
    let zip64 = false;
    // ZIP64 の追加フィールド: 0xFFFFFFFF になっている欄の値が順に入る
    for (let q = p + 46 + nameLen; q < p + 46 + nameLen + extraLen; ) {
      const id = v.getUint16(q, true);
      const len = v.getUint16(q + 2, true);
      if (id === 0x0001) {
        zip64 = true;
        let r = q + 4;
        if (v.getUint32(p + 24, true) === 0xffffffff) {
          size = Number(v.getBigUint64(r, true));
          r += 8;
        }
        if (v.getUint32(p + 20, true) === 0xffffffff) r += 8;
        if (v.getUint32(p + 42, true) === 0xffffffff) offset = Number(v.getBigUint64(r, true));
      }
      q += 4 + len;
    }
    expect(v.getUint32(offset, true)).toBe(0x04034b50);
    const start = offset + 30 + v.getUint16(offset + 26, true) + v.getUint16(offset + 28, true);
    out.push({ name, utf8: (flags & 0x0800) !== 0, crc, data: zip.slice(start, start + size), zip64, time, date });
    p += 46 + nameLen + extraLen + v.getUint16(p + 32, true);
  }
  return out;
}

const bytes = (s: string) => new TextEncoder().encode(s);

describe('ZIP', () => {
  it('CRC-32 が既知の値と一致し、続きから計算できる', () => {
    expect(crc32(new Uint8Array(0))).toBe(0);
    expect(crc32(bytes('123456789'))).toBe(0xcbf43926);
    expect(crc32(bytes('The quick brown fox jumps over the lazy dog'))).toBe(0x414fa339);
    expect(crc32(bytes('56789'), crc32(bytes('1234')))).toBe(0xcbf43926);
  });

  for (const zip64 of [false, true]) {
    it(`作った ZIP を読み戻せる（日本語のファイル名・空のファイルを含む${zip64 ? '、ZIP64 の形式' : ''}）`, async () => {
      let seed = 11;
      const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      const files = [
        { name: 'IMG_0001_M10.jpg', data: Uint8Array.from({ length: 70001 }, () => Math.floor(rnd() * 256)) },
        { name: '夕焼けの海_M10.png', data: bytes('PNG のかわり') },
        { name: 'empty.tif', data: new Uint8Array(0) },
      ];
      const when = new Date(2026, 8, 27, 14, 30, 58);
      const blob = createZip(
        files.map((f) => ({ name: f.name, data: new Blob([f.data]), crc: crc32(f.data) })),
        { date: when, zip64 },
      );
      expect(blob.type).toBe('application/zip');
      const zip = new Uint8Array(await blob.arrayBuffer());
      const read = readZip(zip);
      expect(read.map((e) => e.name)).toEqual(files.map((f) => f.name));
      read.forEach((e, i) => {
        expect(e.utf8).toBe(true);
        expect(e.zip64).toBe(zip64);
        expect(Array.from(e.data)).toEqual(Array.from(files[i].data));
        expect(e.crc).toBe(crc32(e.data));
        // 2026-09-27 14:30:58（MS-DOS 形式は 2 秒単位）
        expect(e.date).toBe(((2026 - 1980) << 9) | (9 << 5) | 27);
        expect(e.time).toBe((14 << 11) | (30 << 5) | 29);
      });
    });
  }

  it('空の ZIP も正しい形になる', async () => {
    const zip = new Uint8Array(await createZip([]).arrayBuffer());
    expect(zip.length).toBe(22);
    expect(readZip(zip)).toEqual([]);
  });
});
