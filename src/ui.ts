// 2 つのタブ（「AEB 合成」と「Leica M10 の色」）で共通の画面部品と小さな道具。

import { formatShutter, type ExposureInfo } from './core/exif';
import { isRawFile, RAW_ACCEPT } from './raw';

export const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// ファイル

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|gif|bmp|tiff?|heic|heif|jxl)$/i;

/** ファイル選択の accept 属性 */
export const FILE_ACCEPT = `image/*,${RAW_ACCEPT}`;

/** 読み込みを試す画像ファイルか（RAW か、ブラウザで開けそうな画像） */
export function isImageFile(f: File): boolean {
  return isRawFile(f) || f.type.startsWith('image/') || IMAGE_EXT.test(f.name);
}

/** 拡張子を除いたファイル名 */
export function baseName(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export function formatBytes(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`;
}

export function formatEv(ev: number): string {
  const r = Math.round(ev * 10) / 10;
  if (Math.abs(r) < 0.05) return '±0EV';
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}EV`;
}

/** シャッター速度・絞り・ISO（なければ露出補正）の表示 */
export function exposureParts(e: ExposureInfo): string[] {
  const parts: string[] = [];
  if (e.exposureTime) parts.push(formatShutter(e.exposureTime));
  if (e.fNumber) parts.push(`f/${Math.round(e.fNumber * 10) / 10}`);
  if (e.iso) parts.push(`ISO${e.iso}`);
  if (!parts.length && e.exposureBias !== undefined) parts.push(`補正 ${formatEv(e.exposureBias)}`);
  return parts;
}

export function exposureText(e?: ExposureInfo): string {
  if (!e) return '';
  return exposureParts(e).join(' · ') || '露出情報なし';
}

/**
 * 画像の一覧の 1 行（サムネイル・名前・説明・右端の表示と × ボタン）を li の中に作る。
 * サムネイルがまだないときは placeholder の文字（形式など）を出す。
 * onSelect を渡すと、サムネイルと名前の部分がその行を選ぶボタンになる
 */
export function fillFileRow(
  li: HTMLLIElement,
  row: {
    thumb?: string;
    placeholder?: string;
    name: string;
    sub: string;
    side?: HTMLElement[];
    removeLabel: string;
    onRemove: () => void;
    onSelect?: () => void;
    selected?: boolean;
  },
): void {
  li.replaceChildren();
  let img: HTMLElement;
  if (row.thumb || !row.placeholder) {
    const im = document.createElement('img');
    im.className = 'thumb';
    im.alt = '';
    if (row.thumb) im.src = row.thumb;
    img = im;
  } else {
    img = document.createElement('div');
    img.className = 'thumb thumb-empty';
    img.textContent = row.placeholder;
  }
  const meta = document.createElement('div');
  meta.className = 'file-meta';
  const name = document.createElement('div');
  name.className = 'file-name';
  name.textContent = row.name;
  name.title = row.name;
  const sub = document.createElement('div');
  sub.className = 'file-sub';
  sub.textContent = row.sub;
  meta.append(name, sub);
  const side = document.createElement('div');
  side.className = 'file-side';
  const rm = document.createElement('button');
  rm.type = 'button';
  rm.className = 'remove';
  rm.textContent = '×';
  rm.title = row.removeLabel;
  rm.setAttribute('aria-label', `${row.name} を${row.removeLabel}`);
  rm.onclick = row.onRemove;
  side.append(...(row.side ?? []), rm);
  if (row.onSelect) {
    const pick = document.createElement('button');
    pick.type = 'button';
    pick.className = 'file-select';
    pick.setAttribute('aria-pressed', String(!!row.selected));
    pick.onclick = row.onSelect;
    pick.append(img, meta);
    li.append(pick, side);
  } else {
    li.append(img, meta, side);
  }
}

// ---------------------------------------------------------------------------
// 表示

let toastTimer = 0;
export function toast(message: string, error = false): void {
  document.querySelector('.toast')?.remove();
  const t = document.createElement('div');
  t.className = `toast ${error ? 'error' : ''}`;
  t.setAttribute('role', 'status');
  t.textContent = message;
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => t.remove(), error ? 8000 : 4000);
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function drawRGBA(canvas: HTMLCanvasElement, rgba: Uint8ClampedArray, w: number, h: number): void {
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d')!.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
}

/** プレビューの上に重ねる「処理中」の表示（中の .busy-label と .bar-fill を使う） */
export class BusyOverlay {
  private readonly label: HTMLElement;
  private readonly bar: HTMLElement;

  constructor(private readonly root: HTMLElement) {
    this.label = root.querySelector<HTMLElement>('.busy-label')!;
    this.bar = root.querySelector<HTMLElement>('.bar-fill')!;
  }

  show(label: string, fraction: number): void {
    this.root.hidden = false;
    this.label.textContent = label;
    this.bar.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
  }

  hide(): void {
    this.root.hidden = true;
  }
}

/** 比較表示: 比べる画像（before）を結果の上に重ね、境界線より左だけを見せる */
export class CompareView {
  private pos = 0.5;

  constructor(
    private readonly el: {
      toggle: HTMLButtonElement;
      result: HTMLCanvasElement;
      before: HTMLCanvasElement;
      handle: HTMLElement;
      labels: HTMLElement;
    },
  ) {
    el.toggle.addEventListener('click', () => this.set(!this.on));
  }

  get on(): boolean {
    return !this.el.before.hidden;
  }

  set(on: boolean): void {
    this.el.toggle.setAttribute('aria-pressed', String(on));
    this.el.before.hidden = !on;
    this.el.handle.hidden = !on;
    this.el.labels.hidden = !on;
    this.update();
  }

  /** 境界線をポインタの位置へ動かす */
  moveTo(ev: PointerEvent): void {
    const r = this.el.result.getBoundingClientRect();
    this.pos = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    this.update();
  }

  private update(): void {
    const pct = this.pos * 100;
    this.el.before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    this.el.handle.style.left = `${pct}%`;
  }
}

/**
 * スライダーの値を横の <output> に表示し、動かしたら onInput を呼ぶ。ダブルクリックで初期値に戻す。
 * 戻り値は表示を今の値に合わせ直す関数（値をプログラムから変えたときに呼ぶ）
 */
export function bindSliders<K extends string>(
  sliders: Record<K, HTMLInputElement>,
  formats: Partial<Record<K, (v: number) => string>>,
  onInput: (key: K) => void,
): () => void {
  const keys = Object.keys(sliders) as K[];
  const update = () => {
    for (const key of keys) {
      const out = sliders[key].parentElement?.querySelector('output');
      if (out) out.textContent = (formats[key] ?? String)(Number(sliders[key].value));
    }
  };
  for (const key of keys) {
    const input = sliders[key];
    input.addEventListener('input', () => {
      update();
      onInput(key);
    });
    input.addEventListener('dblclick', () => {
      input.value = input.defaultValue;
      input.dispatchEvent(new Event('input'));
    });
  }
  update();
  return update;
}

// ---------------------------------------------------------------------------
// ワーカーとのやりとり

type Pending<M> = {
  resolve: (m: M) => void;
  reject: (e: Error) => void;
  onProgress?: (label: string, fraction: number) => void;
};

/**
 * ワーカーへの依頼と返信を reqId で対応づける。返信のうち progress は進み具合、error は失敗として扱い、
 * それ以外を依頼の結果として返す。reqId のないメッセージは onOther に渡す
 */
export class WorkerClient<ToW, FromW extends { type: string }> {
  private seq = 0;
  private readonly pending = new Map<number, Pending<FromW>>();

  constructor(
    private readonly worker: Worker,
    onOther?: (m: FromW) => void,
  ) {
    worker.onmessage = (ev: MessageEvent<FromW>) => {
      const m = ev.data as FromW & { reqId?: number; label?: string; fraction?: number; message?: string };
      if (m.reqId === undefined) {
        onOther?.(m);
        return;
      }
      const p = this.pending.get(m.reqId);
      if (m.type === 'progress') {
        p?.onProgress?.(m.label ?? '', m.fraction ?? 0);
        return;
      }
      this.pending.delete(m.reqId);
      if (m.type === 'error') p?.reject(new Error(m.message));
      else p?.resolve(m);
    };
  }

  post(msg: ToW, transfer: Transferable[] = []): void {
    this.worker.postMessage(msg, transfer);
  }

  request<T extends FromW['type']>(
    build: (reqId: number) => ToW,
    onProgress?: (label: string, fraction: number) => void,
    transfer: Transferable[] = [],
  ): Promise<Extract<FromW, { type: T }>> {
    const reqId = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(reqId, { resolve: resolve as (m: FromW) => void, reject, onProgress });
      this.post(build(reqId), transfer);
    });
  }
}
