import { describe, expect, it } from 'vitest';
import { RawDecoder, withRawDecoder } from '../src/raw';
import { baseName, exposureParts, exposureText, formatEv, isImageFile, WorkerClient } from '../src/ui';

type ToW = { type: 'job'; reqId: number; n: number } | { type: 'note'; text: string };
type FromW =
  | { type: 'progress'; reqId: number; label: string; fraction: number }
  | { type: 'done'; reqId: number; n: number }
  | { type: 'error'; reqId: number; message: string }
  | { type: 'hello'; id: string };

/** postMessage を記録し、返信を onmessage に流せる偽のワーカー */
class FakeWorker {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  sent: Array<{ msg: ToW; transfer: Transferable[] }> = [];
  postMessage(msg: ToW, transfer: Transferable[] = []): void {
    this.sent.push({ msg, transfer });
  }
  reply(m: FromW): void {
    this.onmessage?.({ data: m } as MessageEvent);
  }
}

describe('ワーカーとのやりとり（WorkerClient）', () => {
  it('reqId ごとに返信・進み具合・エラーを振り分け、reqId のないメッセージは onOther に渡す', async () => {
    const w = new FakeWorker();
    const other: FromW[] = [];
    const client = new WorkerClient<ToW, FromW>(w as unknown as Worker, (m) => other.push(m));
    const progress: Array<[string, number]> = [];
    const buf = new ArrayBuffer(8);
    const a = client.request<'done'>((reqId) => ({ type: 'job', reqId, n: 1 }), (label, f) => progress.push([label, f]), [buf]);
    const b = client.request<'done'>((reqId) => ({ type: 'job', reqId, n: 2 }));
    expect(w.sent.map((s) => s.msg)).toEqual([
      { type: 'job', reqId: 1, n: 1 },
      { type: 'job', reqId: 2, n: 2 },
    ]);
    expect(w.sent[0].transfer).toEqual([buf]);

    w.reply({ type: 'progress', reqId: 1, label: '合成中', fraction: 0.5 });
    w.reply({ type: 'hello', id: 'x' });
    w.reply({ type: 'error', reqId: 2, message: 'だめでした' });
    w.reply({ type: 'done', reqId: 1, n: 10 });
    await expect(a).resolves.toEqual({ type: 'done', reqId: 1, n: 10 });
    await expect(b).rejects.toThrow('だめでした');
    expect(progress).toEqual([['合成中', 0.5]]);
    expect(other).toEqual([{ type: 'hello', id: 'x' }]);

    // 返信済みの reqId にもう一度返信が来ても何も起きない
    w.reply({ type: 'done', reqId: 1, n: 99 });
    client.post({ type: 'note', text: 'hi' });
    expect(w.sent[2].msg).toEqual({ type: 'note', text: 'hi' });
  });
});

describe('RAW の順番待ち（withRawDecoder）', () => {
  it('1 つずつ順番に実行し、待ちがなくなったらデコーダを作り直す。失敗しても後ろは止まらない', async () => {
    const log: string[] = [];
    const decoders: RawDecoder[] = [];
    let running = 0;
    const task = (name: string, fail = false) =>
      withRawDecoder(async (d) => {
        running++;
        expect(running).toBe(1);
        decoders.push(d);
        log.push(`${name}:start`);
        await new Promise((r) => setTimeout(r, 5));
        log.push(`${name}:end`);
        running--;
        if (fail) throw new Error(name);
        return name;
      });
    const results = await Promise.allSettled([task('a'), task('b', true), task('c')]);
    expect(results.map((r) => r.status)).toEqual(['fulfilled', 'rejected', 'fulfilled']);
    expect(log).toEqual(['a:start', 'a:end', 'b:start', 'b:end', 'c:start', 'c:end']);
    // 続けて並んだ処理は同じデコーダを使う
    expect(decoders[1]).toBe(decoders[0]);
    expect(decoders[2]).toBe(decoders[0]);
    // 待ちがなくなった後は新しいデコーダ（前のものは破棄済み）
    await new Promise((r) => setTimeout(r, 0));
    await task('d');
    expect(decoders[3]).not.toBe(decoders[0]);
    expect(decoders[3]).toBeInstanceOf(RawDecoder);
  });
});

describe('表示用の小さな道具', () => {
  it('露出の表示', () => {
    expect(formatEv(0.02)).toBe('±0EV');
    expect(formatEv(1.26)).toBe('+1.3EV');
    expect(formatEv(-2)).toBe('−2.0EV');
    expect(exposureParts({ exposureTime: 1 / 125, fNumber: 2.8, iso: 400 })).toEqual(['1/125', 'f/2.8', 'ISO400']);
    expect(exposureParts({ exposureBias: -1 })).toEqual(['補正 −1.0EV']);
    expect(exposureParts({ make: 'Canon' })).toEqual([]);
    expect(exposureText({ make: 'Canon' })).toBe('露出情報なし');
    expect(exposureText(undefined)).toBe('');
  });

  it('ファイル名と形式', () => {
    expect(baseName('IMG_0001.CR3')).toBe('IMG_0001');
    expect(baseName('a.b.jpg')).toBe('a.b');
    expect(baseName('noext')).toBe('noext');
    expect(isImageFile(new File([], 'IMG_0001.CR3'))).toBe(true);
    expect(isImageFile(new File([], 'photo.JPG'))).toBe(true);
    expect(isImageFile(new File([], 'x', { type: 'image/webp' }))).toBe(true);
    expect(isImageFile(new File([], 'memo.txt', { type: 'text/plain' }))).toBe(false);
  });
});
