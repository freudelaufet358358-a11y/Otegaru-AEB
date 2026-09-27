import './style.css';
import { ensureCrossOriginIsolation } from './coi';
import { readExif, formatShutter, type ExposureInfo } from './core/exif';
import { DEFAULT_FUSION_WEIGHTS } from './core/fusion';
import type { LookId } from './core/look';
import { isRawFile, makeThumbnail, RAW_ACCEPT, RawDecoder } from './raw';
import type { ExportFormat, FromWorker, LoupeMode, ManualAdjust, Mode, PreparedInfo, RenderParams, ToWorker } from './worker/protocol';

/** プレビューの長辺 (px) */
const PREVIEW_SIDE = 1600;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const el = {
  banner: $('banner'),
  stage: $('stage'),
  dropzone: $('dropzone'),
  viewer: $('viewer'),
  canvasWrap: $('canvas-wrap'),
  result: $<HTMLCanvasElement>('canvas-result'),
  before: $<HTMLCanvasElement>('canvas-before'),
  handle: $('compare-handle'),
  labels: $('compare-labels'),
  busy: $('busy'),
  busyLabel: $('busy-label'),
  busyBar: $('busy-bar'),
  toolbar: $('stage-toolbar'),
  compare: $<HTMLButtonElement>('toggle-compare'),
  info: $('stage-info'),
  miniSpinner: $('mini-spinner'),
  fileList: $<HTMLUListElement>('file-list'),
  fileCount: $('file-count'),
  fileInput: $<HTMLInputElement>('file-input'),
  rawSize: $<HTMLSelectElement>('raw-size'),
  align: $<HTMLInputElement>('align'),
  modeHint: $('mode-hint'),
  exportBtn: $<HTMLButtonElement>('export'),
  exportFormat: $<HTMLSelectElement>('export-format'),
  exportSize: $<HTMLSelectElement>('export-size'),
  exportQuality: $<HTMLInputElement>('export-quality'),
  qualityRow: $('quality-row'),
  exportHint: $('export-hint'),
  help: $<HTMLDialogElement>('help'),
  alignSummary: $('align-summary'),
  manualAlign: $<HTMLDetailsElement>('manual-align'),
  frameChips: $('frame-chips'),
  loupe: $<HTMLCanvasElement>('loupe'),
  loupeMarker: $('loupe-marker'),
  loupeOverlay: $('loupe-overlay'),
  nudgeStep: $<HTMLSelectElement>('nudge-step'),
  manualReadout: $('manual-readout'),
  lookParams: $('look-params'),
  lookHint: $('look-hint'),
  lookTone: $<HTMLInputElement>('look-tone'),
};

const MODE_HINTS: Record<Mode, string> = {
  learned:
    '約 3,500 シーンの写真（RAW と仕上がった写真の組）で学習したモデルが、明るさ・メリハリ・色のバランスを写真らしく整えます。迷ったらこちら。',
  fusion: '各写真の「ちょうど良く写っている部分」をそのままつなぎ合わせます。',
  hdr: '露出の違いから広い明暗差を再現し、暗部と明部をしっかり起こした HDR 調に仕上げます（RAW 向け）。',
};

const MODE_NAMES: Record<Mode, string> = { learned: 'おまかせ', fusion: 'ナチュラル', hdr: 'HDR' };

const LOOK_NAMES: Record<LookId, string> = { none: '', 'leica-m10': 'Leica M10' };

// ---------------------------------------------------------------------------
// 合成ワーカーとの通信

const worker = new Worker(new URL('./worker/process.worker.ts', import.meta.url), { type: 'module' });

type Pending = {
  resolve: (m: FromWorker) => void;
  reject: (e: Error) => void;
  onProgress?: (label: string, fraction: number) => void;
};
let reqSeq = 0;
const pending = new Map<number, Pending>();
const addWaiters = new Map<string, { resolve: (m: Extract<FromWorker, { type: 'added' }>) => void; reject: (e: Error) => void }>();

worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const m = ev.data;
  switch (m.type) {
    case 'added': {
      addWaiters.get(m.id)?.resolve(m);
      addWaiters.delete(m.id);
      break;
    }
    case 'addFailed': {
      addWaiters.get(m.id)?.reject(new Error(m.message));
      addWaiters.delete(m.id);
      break;
    }
    case 'progress':
      pending.get(m.reqId)?.onProgress?.(m.label, m.fraction);
      break;
    case 'error': {
      pending.get(m.reqId)?.reject(new Error(m.message));
      pending.delete(m.reqId);
      break;
    }
    default: {
      pending.get(m.reqId)?.resolve(m);
      pending.delete(m.reqId);
    }
  }
};
worker.onerror = (e) => {
  console.error(e);
  toast('処理中に問題が発生しました。メモリ不足の可能性があります（RAW を 1/2 サイズにすると軽くなります）', true);
};

function send(msg: ToWorker, transfer: Transferable[] = []): void {
  worker.postMessage(msg, transfer);
}

function request<T extends FromWorker['type']>(
  build: (reqId: number) => ToWorker,
  onProgress?: (label: string, fraction: number) => void,
): Promise<Extract<FromWorker, { type: T }>> {
  const reqId = ++reqSeq;
  return new Promise((resolve, reject) => {
    pending.set(reqId, { resolve: resolve as (m: FromWorker) => void, reject, onProgress });
    send(build(reqId));
  });
}

function addToWorker(msg: ToWorker & { id: string }, transfer: Transferable[] = []) {
  return new Promise<Extract<FromWorker, { type: 'added' }>>((resolve, reject) => {
    addWaiters.set(msg.id, { resolve, reject });
    send(msg, transfer);
  });
}

// ---------------------------------------------------------------------------
// 画像の管理

type Status = 'loading' | 'ready' | 'error';

interface Item {
  id: string;
  file: File;
  raw: boolean;
  status: Status;
  message: string;
  exif?: ExposureInfo;
  thumb?: string;
  /** 読み込み直すたびに増える。古い読み込み結果を捨てるために使う */
  version: number;
  el: HTMLLIElement;
}

const items = new Map<string, Item>();
let idSeq = 0;
let generation = 0;
let prepared: PreparedInfo | null = null;

const IMAGE_EXT = /\.(jpe?g|png|webp|avif|gif|bmp|tiff?|heic|heif|jxl)$/i;

function addFiles(files: File[]): void {
  const accepted = files.filter((f) => isRawFile(f) || f.type.startsWith('image/') || IMAGE_EXT.test(f.name));
  const skipped = files.length - accepted.length;
  if (skipped > 0) toast(`${skipped} 件のファイルは画像ではないため読み込みませんでした`);
  for (const file of accepted) {
    const item: Item = {
      id: `f${++idSeq}`,
      file,
      raw: isRawFile(file),
      status: 'loading',
      message: '待機中…',
      version: 0,
      el: document.createElement('li'),
    };
    items.set(item.id, item);
    renderItem(item);
    el.fileList.append(item.el);
    if (item.raw) queueRaw(item);
    else void loadImage(item);
  }
  if (accepted.length) onItemsChanged();
}

let rawChain: Promise<void> = Promise.resolve();
let rawQueued = 0;
let decoder: RawDecoder | null = null;

function queueRaw(item: Item): void {
  rawQueued++;
  const version = ++item.version;
  rawChain = rawChain
    .then(() => loadRaw(item, version))
    .finally(() => {
      if (--rawQueued === 0) {
        // WASM のメモリを解放する
        decoder?.dispose();
        decoder = null;
      }
    });
}

async function loadRaw(item: Item, version: number): Promise<void> {
  // 削除された、または読み込み直しが予約された場合は結果を使わない
  const stale = () => items.get(item.id) !== item || item.version !== version;
  if (stale()) return;
  try {
    decoder ??= new RawDecoder();
    setStatus(item, 'loading', 'RAW を開いています…');
    const meta = await decoder.open(item.file, el.rawSize.value === 'half');
    item.exif = meta.exif;
    if (meta.preview && !item.thumb) item.thumb = await makeThumbnail(meta.preview, meta.flip).catch(() => undefined);
    setStatus(item, 'loading', '現像中…');
    const img = await decoder.develop();
    if (stale()) return;
    await addToWorker(
      { type: 'addRaw', id: item.id, name: item.file.name, width: img.width, height: img.height, data: img.data, exif: meta.exif, color: meta.color },
      [img.data.buffer],
    );
    if (stale()) return;
    setStatus(item, 'ready', '');
  } catch (e) {
    if (stale()) return;
    setStatus(item, 'error', errorMessage(e));
  }
  onItemsChanged();
}

async function loadImage(item: Item): Promise<void> {
  try {
    setStatus(item, 'loading', '読み込み中…');
    item.exif = readExif(await item.file.slice(0, 512 * 1024).arrayBuffer());
    item.thumb = await makeThumbnail(item.file).catch(() => undefined);
    renderItem(item);
    const res = await addToWorker({ type: 'addFile', id: item.id, name: item.file.name, file: item.file });
    item.exif = { ...res.exif, ...item.exif };
    setStatus(item, 'ready', '');
  } catch (e) {
    setStatus(item, 'error', errorMessage(e));
  }
  onItemsChanged();
}

function removeItem(id: string): void {
  const item = items.get(id);
  if (!item) return;
  items.delete(id);
  item.el.remove();
  if (item.thumb) URL.revokeObjectURL(item.thumb);
  send({ type: 'remove', id });
  onItemsChanged();
}

function setStatus(item: Item, status: Status, message: string): void {
  item.status = status;
  item.message = message;
  renderItem(item);
  updateLoadingOverlay();
}

function renderItem(item: Item): void {
  const li = item.el;
  li.className = `file ${item.status === 'error' ? 'error' : ''} ${prepared?.referenceId === item.id ? 'reference' : ''}`;
  li.replaceChildren();
  const img = document.createElement('img');
  img.className = 'thumb';
  img.alt = '';
  if (item.thumb) img.src = item.thumb;
  const meta = document.createElement('div');
  meta.className = 'file-meta';
  const name = document.createElement('div');
  name.className = 'file-name';
  name.textContent = item.file.name;
  name.title = item.file.name;
  const sub = document.createElement('div');
  sub.className = 'file-sub';
  sub.textContent = item.status === 'ready' ? exposureText(item.exif) : item.message;
  meta.append(name, sub);
  const side = document.createElement('div');
  side.className = 'file-side';
  if (item.status === 'loading') {
    const dot = document.createElement('span');
    dot.className = 'loading-dot';
    side.append(dot);
  }
  const ev = prepared?.relativeEv[item.id];
  if (item.status === 'ready' && ev !== undefined) {
    const badge = document.createElement('span');
    const isRef = prepared!.referenceId === item.id;
    badge.className = `ev ${isRef ? 'ref' : ''}`;
    badge.textContent = isRef ? '基準' : formatEv(ev);
    badge.title = isRef ? '合成の基準（中間の露出）' : '基準との露出差（画像から推定）';
    side.append(badge);
  }
  const rm = document.createElement('button');
  rm.type = 'button';
  rm.className = 'remove';
  rm.textContent = '×';
  rm.title = '削除';
  rm.setAttribute('aria-label', `${item.file.name} を削除`);
  rm.onclick = () => removeItem(item.id);
  side.append(rm);
  li.append(img, meta, side);
}

function exposureText(e?: ExposureInfo): string {
  if (!e) return '';
  const parts: string[] = [];
  if (e.exposureTime) parts.push(formatShutter(e.exposureTime));
  if (e.fNumber) parts.push(`f/${Math.round(e.fNumber * 10) / 10}`);
  if (e.iso) parts.push(`ISO${e.iso}`);
  if (!parts.length && e.exposureBias !== undefined) parts.push(`補正 ${formatEv(e.exposureBias)}`);
  return parts.join(' · ') || '露出情報なし';
}

function formatEv(ev: number): string {
  const r = Math.round(ev * 10) / 10;
  if (Math.abs(r) < 0.05) return '±0EV';
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}EV`;
}

function readyItems(): Item[] {
  return [...items.values()].filter((i) => i.status === 'ready');
}

function loadingItems(): Item[] {
  return [...items.values()].filter((i) => i.status === 'loading');
}

function onItemsChanged(): void {
  generation++;
  prepared = null;
  updateAlignmentUI();
  const all = [...items.values()];
  el.fileCount.textContent = all.length ? `${all.length} 枚` : '';
  for (const item of all) renderItem(item);
  const hasItems = all.length > 0;
  el.dropzone.hidden = hasItems && readyItems().length >= 2;
  updateLoadingOverlay();
  updateExportState();
  if (!loadingItems().length) {
    if (readyItems().length >= 2) scheduleProcess();
    else {
      showViewer(false);
      if (readyItems().length === 1) toast('露出の違う写真をもう 1 枚以上追加してください');
    }
  }
}

function updateLoadingOverlay(): void {
  const all = [...items.values()];
  const loading = loadingItems();
  if (loading.length) {
    const done = all.length - loading.length;
    const current = loading.find((i) => i.message && i.message !== '待機中…');
    showBusy(`画像を読み込み中 ${done + 1}/${all.length}${current ? `（${current.message.replace('…', '')}）` : ''}`, done / all.length);
  } else if (!processing) {
    hideBusy();
  }
}

// ---------------------------------------------------------------------------
// 合成とプレビュー

let processTimer = 0;
let processing = false;

function scheduleProcess(): void {
  // 読み込み完了から合成開始までの間もオーバーレイを出したままにする
  processing = true;
  showBusy('準備中…', 0);
  clearTimeout(processTimer);
  processTimer = window.setTimeout(() => void runProcess(), 120);
}

async function runProcess(): Promise<void> {
  if (loadingItems().length || readyItems().length < 2) {
    processing = false;
    updateLoadingOverlay();
    return;
  }
  const gen = generation;
  processing = true;
  showBusy('準備中…', 0);
  try {
    const res = await request<'prepared'>(
      (reqId) => ({ type: 'prepare', reqId, align: el.align.checked, previewSide: PREVIEW_SIDE }),
      (label, f) => showBusy(label, f),
    );
    if (gen !== generation) return;
    manualState.clear();
    applyLayout(res.info, res.reference);
    if (alignmentStats(res.info).shift > Math.max(res.info.width, res.info.height) * 0.015) {
      toast('位置のずれが大きい画像があります。同じ構図で撮ったブラケット写真か確認してください');
    }
    // 暗い→明るい順に並べ替える
    for (const id of res.info.order) {
      const item = items.get(id);
      if (item) el.fileList.append(item.el);
    }
    for (const item of items.values()) {
      if (item.status !== 'ready') el.fileList.append(item.el);
      renderItem(item);
    }
    showBusy('合成中…', 0);
    await renderOnce(gen, true);
  } catch (e) {
    if (gen === generation) {
      toast(errorMessage(e), true);
      showViewer(false);
      el.dropzone.hidden = false;
    }
  } finally {
    processing = false;
    if (!loadingItems().length) hideBusy();
    updateExportState();
  }
}

let renderInFlight = false;
let renderWanted = false;

function requestRender(): void {
  if (!prepared) return;
  renderWanted = true;
  if (!renderInFlight) void renderLoop();
}

async function renderLoop(): Promise<void> {
  renderInFlight = true;
  el.miniSpinner.hidden = false;
  while (renderWanted && prepared) {
    renderWanted = false;
    await renderOnce(generation, false);
  }
  el.miniSpinner.hidden = true;
  renderInFlight = false;
}

async function renderOnce(gen: number, showProgress: boolean): Promise<void> {
  const params = currentParams();
  try {
    const res = await request<'rendered'>(
      (reqId) => ({ type: 'render', reqId, params }),
      showProgress ? (label, f) => showBusy(label + '…', f) : undefined,
    );
    if (gen !== generation || !prepared) return;
    if (res.layout) applyLayout(res.layout.info, res.layout.reference);
    drawRGBA(el.result, res.rgba, res.width, res.height);
    showViewer(true);
    const { shift, rotation } = alignmentStats(prepared);
    el.info.textContent = [
      `${prepared.width}×${prepared.height}`,
      `${prepared.order.length}枚`,
      MODE_NAMES[params.mode],
      ...(params.look.id !== 'none' && params.look.amount > 0 ? [LOOK_NAMES[params.look.id]] : []),
      !prepared.aligned ? '位置合わせオフ' : shift < 0.05 && rotation < 0.005 ? 'ずれなし' : `位置補正 ${shift.toFixed(1)}px・${rotation.toFixed(2)}°`,
    ].join(' · ');
  } catch (e) {
    if (gen === generation) toast(errorMessage(e), true);
  }
}

/** 準備結果（切り抜き範囲・位置合わせ）を反映する */
function applyLayout(info: PreparedInfo, reference: Uint8ClampedArray): void {
  prepared = info;
  drawRGBA(el.before, reference, info.previewWidth, info.previewHeight);
  updateAlignmentUI();
  updateLookHint();
}

/** 自動位置合わせで補正した最大の移動量 [px] と回転 [度] */
function alignmentStats(info: PreparedInfo): { shift: number; rotation: number; partial: boolean } {
  const autos = Object.entries(info.alignment)
    .filter(([id]) => id !== info.referenceId)
    .map(([, a]) => a.auto);
  return {
    shift: Math.max(0, ...autos.map((a) => Math.hypot(a.dx, a.dy))),
    rotation: Math.max(0, ...autos.map((a) => Math.abs(a.rotation))),
    partial: autos.some((a) => !a.precise),
  };
}

function drawRGBA(canvas: HTMLCanvasElement, rgba: Uint8ClampedArray, w: number, h: number): void {
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d')!.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
}

function showViewer(on: boolean): void {
  el.viewer.hidden = !on;
  el.toolbar.hidden = !on;
  if (on) el.dropzone.hidden = true;
  updateLoupeMarker();
}

function showBusy(label: string, fraction: number): void {
  el.busy.hidden = false;
  el.busyLabel.textContent = label;
  el.busyBar.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
}

function hideBusy(): void {
  el.busy.hidden = true;
}

// ---------------------------------------------------------------------------
// パラメータ

let mode: Mode = 'learned';
let look: LookId = 'none';

const sliders = {
  amount: $<HTMLInputElement>('amount'),
  learnedExposure: $<HTMLInputElement>('learned-exposure'),
  fusionDetail: $<HTMLInputElement>('fusion-detail'),
  toneStrength: $<HTMLInputElement>('tone-strength'),
  toneDetail: $<HTMLInputElement>('tone-detail'),
  wCenter: $<HTMLInputElement>('w-center'),
  wContrast: $<HTMLInputElement>('w-contrast'),
  wSaturation: $<HTMLInputElement>('w-saturation'),
  wExposure: $<HTMLInputElement>('w-exposure'),
  brightness: $<HTMLInputElement>('adj-brightness'),
  contrast: $<HTMLInputElement>('adj-contrast'),
  saturation: $<HTMLInputElement>('adj-saturation'),
  lookAmount: $<HTMLInputElement>('look-amount'),
  quality: el.exportQuality,
};

const formats: Partial<Record<keyof typeof sliders, (v: number) => string>> = {
  amount: (v) => `${v}%`,
  learnedExposure: (v) => formatEv(v / 10),
  fusionDetail: (v) => `${(v / 100).toFixed(2)}×`,
  toneDetail: (v) => `${(v / 100).toFixed(2)}×`,
  wCenter: (v) => (v / 100).toFixed(2),
  wContrast: (v) => (v / 100).toFixed(2),
  wSaturation: (v) => (v / 100).toFixed(2),
  wExposure: (v) => (v / 100).toFixed(2),
  brightness: (v) => (v > 0 ? `+${v}` : `${v}`),
  contrast: (v) => (v > 0 ? `+${v}` : `${v}`),
  saturation: (v) => (v > 0 ? `+${v}` : `${v}`),
  lookAmount: (v) => `${v}%`,
};

function num(input: HTMLInputElement): number {
  return Number(input.value);
}

function currentParams(): RenderParams {
  return {
    mode,
    amount: num(sliders.amount) / 100,
    fusion: {
      ...DEFAULT_FUSION_WEIGHTS,
      center: num(sliders.wCenter) / 100,
      contrast: num(sliders.wContrast) / 100,
      saturation: num(sliders.wSaturation) / 100,
      exposure: num(sliders.wExposure) / 100,
      detail: num(sliders.fusionDetail) / 100,
    },
    tone: { strength: num(sliders.toneStrength) / 100, detail: num(sliders.toneDetail) / 100 },
    learned: { exposure: num(sliders.learnedExposure) / 10 },
    look: { id: look, amount: num(sliders.lookAmount) / 100, tone: el.lookTone.checked },
    adjust: {
      brightness: num(sliders.brightness),
      contrast: num(sliders.contrast),
      saturation: num(sliders.saturation),
    },
  };
}

function updateOutputs(): void {
  for (const [key, input] of Object.entries(sliders) as Array<[keyof typeof sliders, HTMLInputElement]>) {
    const out = input.parentElement?.querySelector('output');
    if (out) out.textContent = (formats[key] ?? String)(num(input));
  }
}

function setMode(m: Mode): void {
  mode = m;
  for (const b of document.querySelectorAll<HTMLButtonElement>('.segmented [data-mode]')) {
    b.setAttribute('aria-checked', String(b.dataset.mode === m));
  }
  for (const p of document.querySelectorAll<HTMLElement>('.mode-params')) p.hidden = p.dataset.for !== m;
  el.modeHint.textContent = MODE_HINTS[m];
  requestRender();
}

function setLook(id: LookId): void {
  look = id;
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-look]')) {
    b.setAttribute('aria-checked', String(b.dataset.look === id));
  }
  el.lookParams.hidden = id === 'none';
  updateLookHint();
  requestRender();
}

/** 色の傾向の説明。Leica M10 のときは、センサーの違いをどう変換したか（機種・光源）も添える */
function updateLookHint(): void {
  if (look === 'none') {
    el.lookHint.textContent = 'RAW はカメラの色を正確に再現した癖の少ない色、JPEG は撮ったときの色のままです。';
    return;
  }
  const parts = [
    'Leica M10 のセンサーの色の出方と、Leica のカメラ内 JPEG の色づくり（Canon の JPEG より彩度は控えめ、肌は赤み寄り、黄緑は黄み寄り）を再現します。',
    el.lookTone.checked ? '階調も Leica のカメラ内 JPEG のトーンカーブにします。' : '明るさは合成結果のまま残します。',
  ];
  const info = prepared?.look;
  if (info) {
    if (info.encoding === 'srgb') parts.push('JPEG は Canon のピクチャースタイル「スタンダード」の色を打ち消してから変換します。');
    if (info.cameraMatched) {
      parts.push(`この機種の分光感度から変換しています${info.cct ? `（光源の色温度 約 ${Math.round(info.cct / 100) * 100}K）` : ''}。`);
    } else {
      parts.push('この機種の分光感度データがないため、色を正確に写すカメラとして変換しています。');
    }
  }
  el.lookHint.textContent = parts.join('');
}

// ---------------------------------------------------------------------------
// 比較表示

let comparePos = 0.5;

function setCompare(on: boolean): void {
  el.compare.setAttribute('aria-pressed', String(on));
  el.before.hidden = !on;
  el.handle.hidden = !on;
  el.labels.hidden = !on;
  updateCompare();
}

function updateCompare(): void {
  const pct = comparePos * 100;
  el.before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  el.handle.style.left = `${pct}%`;
}

function onComparePointer(ev: PointerEvent): void {
  if (el.before.hidden) return;
  const r = el.result.getBoundingClientRect();
  comparePos = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
  updateCompare();
}

// ---------------------------------------------------------------------------
// 位置の手動調整

const LOUPE_SIZE = 160;
let selectedFrame: string | null = null;
/** 手動調整の値（ワーカーからの返信で古い値に戻らないよう、画面側を正とする） */
const manualState = new Map<string, ManualAdjust>();
let loupePos = { x: 0.5, y: 0.5 };
let loupeMode: LoupeMode = 'blend';

function adjustableIds(): string[] {
  const info = prepared;
  return info ? info.order.filter((id) => id !== info.referenceId) : [];
}

function updateAlignmentUI(): void {
  const info = prepared;
  if (!info) {
    el.alignSummary.textContent = '';
    el.frameChips.replaceChildren();
    el.manualReadout.textContent = '';
    updateLoupeMarker();
    return;
  }
  const { shift, rotation, partial } = alignmentStats(info);
  el.alignSummary.textContent = !info.aligned
    ? '自動位置合わせはオフです（手動での調整はできます）'
    : shift < 0.05 && rotation < 0.005
      ? 'ずれは見つかりませんでした'
      : `ずれを補正しました（最大 ${shift.toFixed(1)}px・回転 ${rotation.toFixed(2)}°）${partial ? '。一部の写真は平行移動のみ補正しています' : ''}`;

  const ids = adjustableIds();
  if (!selectedFrame || !ids.includes(selectedFrame)) selectedFrame = ids[0] ?? null;
  el.frameChips.replaceChildren(
    ...ids.map((id) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', String(id === selectedFrame));
      const ev = info.relativeEv[id];
      b.textContent = ev !== undefined ? formatEv(ev) : (items.get(id)?.file.name ?? id);
      b.title = items.get(id)?.file.name ?? '';
      const m = manualOf(id);
      if (m.x || m.y || m.rotation) b.classList.add('adjusted');
      b.onclick = () => {
        selectedFrame = id;
        updateAlignmentUI();
        requestLoupe();
      };
      return b;
    }),
  );
  const a = selectedFrame ? info.alignment[selectedFrame] : undefined;
  const m = selectedFrame ? manualOf(selectedFrame) : undefined;
  const sign = (v: number, digits: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`;
  el.manualReadout.textContent =
    a && m
      ? `自動: 横 ${sign(a.auto.dx, 1)}px 縦 ${sign(a.auto.dy, 1)}px 回転 ${sign(a.auto.rotation, 2)}° ／ ` +
        `手動: 横 ${sign(m.x, 2)}px 縦 ${sign(m.y, 2)}px 回転 ${sign(m.rotation, 3)}°`
      : '';
  updateLoupeMarker();
}

function manualOf(id: string): ManualAdjust {
  return manualState.get(id) ?? { x: 0, y: 0, rotation: 0 };
}

function nudge(action: string): void {
  if (!prepared || !selectedFrame) return;
  const step = Number(el.nudgeStep.value);
  const m: ManualAdjust = { ...manualOf(selectedFrame) };
  switch (action) {
    case 'left':
      m.x -= step;
      break;
    case 'right':
      m.x += step;
      break;
    case 'up':
      m.y -= step;
      break;
    case 'down':
      m.y += step;
      break;
    case 'rotl':
      m.rotation -= step * 0.05;
      break;
    case 'rotr':
      m.rotation += step * 0.05;
      break;
    case 'reset':
      m.x = m.y = m.rotation = 0;
      break;
    default:
      return;
  }
  const round = (v: number) => Math.round(v * 10000) / 10000;
  m.x = round(m.x);
  m.y = round(m.y);
  m.rotation = round(m.rotation);
  manualState.set(selectedFrame, m);
  send({ type: 'setManual', id: selectedFrame, manual: m });
  updateAlignmentUI();
  requestLoupe();
  requestRender();
}

let loupeInFlight = false;
let loupeWanted = false;

function requestLoupe(): void {
  if (!prepared || !selectedFrame || !el.manualAlign.open) return;
  loupeWanted = true;
  if (!loupeInFlight) void loupeLoop();
}

async function loupeLoop(): Promise<void> {
  loupeInFlight = true;
  while (loupeWanted && prepared && selectedFrame) {
    loupeWanted = false;
    const gen = generation;
    const id = selectedFrame;
    try {
      const res = await request<'loupe'>((reqId) => ({
        type: 'loupe',
        reqId,
        id,
        cx: loupePos.x,
        cy: loupePos.y,
        size: LOUPE_SIZE,
        mode: loupeMode,
      }));
      if (gen === generation) drawRGBA(el.loupe, res.rgba, res.size, res.size);
    } catch (e) {
      console.warn(e);
    }
  }
  loupeInFlight = false;
}

function updateLoupeMarker(): void {
  const on = !!(prepared && el.manualAlign.open && selectedFrame && !el.viewer.hidden);
  el.loupeMarker.hidden = !on;
  el.loupeOverlay.hidden = !on;
  el.loupeMarker.style.left = `${loupePos.x * 100}%`;
  el.loupeMarker.style.top = `${loupePos.y * 100}%`;
}

function setLoupeFromPointer(ev: PointerEvent): void {
  const r = el.result.getBoundingClientRect();
  loupePos = {
    x: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
    y: Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height)),
  };
  updateLoupeMarker();
  requestLoupe();
}

// ---------------------------------------------------------------------------
// 書き出し

function updateExportState(): void {
  el.exportBtn.disabled = !prepared;
  el.qualityRow.hidden = el.exportFormat.value !== 'jpeg';
}

async function doExport(): Promise<void> {
  if (!prepared) return;
  const gen = generation;
  const format = el.exportFormat.value as ExportFormat;
  const options = { format, quality: num(el.exportQuality) / 100, maxSide: Number(el.exportSize.value) };
  const params = currentParams();
  el.exportBtn.disabled = true;
  el.exportHint.classList.remove('error');
  el.exportHint.textContent = '';
  showBusy('書き出し中…', 0);
  try {
    const res = await request<'exported'>(
      (reqId) => ({ type: 'export', reqId, params, options }),
      (label, f) => showBusy(`${label}…`, f),
    );
    if (gen !== generation) return;
    const ref = items.get(prepared.referenceId);
    const base = (ref?.file.name ?? 'image').replace(/\.[^.]+$/, '');
    const ext = format === 'jpeg' ? 'jpg' : format === 'png' ? 'png' : 'tif';
    const suffix = params.look.id === 'leica-m10' && params.look.amount > 0 ? '_M10' : '';
    download(res.blob, `${base}_${params.mode === 'hdr' ? 'HDR' : 'AEB'}${suffix}.${ext}`);
    el.exportHint.textContent = `${res.width}×${res.height}（${formatBytes(res.blob.size)}）を保存しました · ${(res.elapsed / 1000).toFixed(1)} 秒`;
  } catch (e) {
    el.exportHint.classList.add('error');
    el.exportHint.textContent = errorMessage(e);
  } finally {
    hideBusy();
    updateExportState();
  }
}

function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function formatBytes(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`;
}

// ---------------------------------------------------------------------------
// その他

let toastTimer = 0;
function toast(message: string, error = false): void {
  document.querySelector('.toast')?.remove();
  const t = document.createElement('div');
  t.className = `toast ${error ? 'error' : ''}`;
  t.setAttribute('role', 'status');
  t.textContent = message;
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => t.remove(), error ? 8000 : 4000);
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function bindEvents(): void {
  el.fileInput.accept = `image/*,${RAW_ACCEPT}`;
  $('pick-files').onclick = () => el.fileInput.click();
  $('add-files').onclick = () => el.fileInput.click();
  $('clear-files').onclick = () => {
    for (const id of [...items.keys()]) removeItem(id);
  };
  el.fileInput.onchange = () => {
    addFiles([...(el.fileInput.files ?? [])]);
    el.fileInput.value = '';
  };

  // ドラッグ＆ドロップ（ページ全体で受け付ける）
  let depth = 0;
  window.addEventListener('dragenter', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    depth++;
    document.body.classList.add('dragging');
  });
  window.addEventListener('dragover', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', () => {
    if (--depth <= 0) {
      depth = 0;
      document.body.classList.remove('dragging');
    }
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    document.body.classList.remove('dragging');
    addFiles([...(e.dataTransfer?.files ?? [])]);
  });

  el.rawSize.onchange = () => {
    const raws = [...items.values()].filter((i) => i.raw);
    for (const item of raws) {
      send({ type: 'remove', id: item.id });
      setStatus(item, 'loading', '待機中…');
      queueRaw(item);
    }
    if (raws.length) onItemsChanged();
  };
  el.align.onchange = () => {
    if (readyItems().length >= 2 && !loadingItems().length) scheduleProcess();
  };

  for (const b of document.querySelectorAll<HTMLButtonElement>('.segmented [data-mode]')) {
    b.onclick = () => setMode(b.dataset.mode as Mode);
  }
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-look]')) {
    b.onclick = () => setLook(b.dataset.look as LookId);
  }
  el.lookTone.onchange = () => {
    updateLookHint();
    requestRender();
  };
  for (const input of Object.values(sliders)) {
    input.addEventListener('input', () => {
      updateOutputs();
      if (input !== sliders.quality) requestRender();
    });
    // ダブルクリックで初期値に戻す
    input.addEventListener('dblclick', () => {
      input.value = input.defaultValue;
      input.dispatchEvent(new Event('input'));
    });
  }
  $('reset-adjust').onclick = () => {
    for (const s of [sliders.brightness, sliders.contrast, sliders.saturation]) s.value = s.defaultValue;
    updateOutputs();
    requestRender();
  };

  el.compare.onclick = () => setCompare(el.compare.getAttribute('aria-pressed') !== 'true');
  // プレビュー上のドラッグ: 比較中は境界線、手動調整中はルーペの位置
  let dragging: 'compare' | 'loupe' | null = null;
  el.canvasWrap.addEventListener('pointerdown', (e) => {
    if (!el.before.hidden) dragging = 'compare';
    else if (el.manualAlign.open && prepared) dragging = 'loupe';
    else return;
    el.canvasWrap.setPointerCapture(e.pointerId);
    if (dragging === 'compare') onComparePointer(e);
    else setLoupeFromPointer(e);
  });
  el.canvasWrap.addEventListener('pointermove', (e) => {
    if (dragging === 'compare') onComparePointer(e);
    else if (dragging === 'loupe') setLoupeFromPointer(e);
  });
  el.canvasWrap.addEventListener('pointerup', () => (dragging = null));
  el.canvasWrap.addEventListener('pointercancel', () => (dragging = null));

  el.manualAlign.addEventListener('toggle', () => {
    updateLoupeMarker();
    requestLoupe();
  });
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-nudge]')) {
    b.onclick = () => nudge(b.dataset.nudge!);
  }
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-loupe]')) {
    b.onclick = () => {
      loupeMode = b.dataset.loupe as LoupeMode;
      for (const o of document.querySelectorAll<HTMLButtonElement>('[data-loupe]')) {
        o.setAttribute('aria-checked', String(o === b));
      }
      requestLoupe();
    };
  }
  // 手動調整中はキーボードでも動かせる（矢印キーで移動、[ ] で回転）
  window.addEventListener('keydown', (e) => {
    if (!el.manualAlign.open || !prepared) return;
    const target = e.target as HTMLElement;
    if (target.closest('input, select, textarea')) return;
    const map: Record<string, string> = {
      ArrowLeft: 'left',
      ArrowRight: 'right',
      ArrowUp: 'up',
      ArrowDown: 'down',
      '[': 'rotl',
      ']': 'rotr',
    };
    const action = map[e.key];
    if (!action) return;
    e.preventDefault();
    nudge(action);
  });

  el.exportFormat.onchange = updateExportState;
  el.exportBtn.onclick = () => void doExport();
  $('open-help').onclick = () => el.help.showModal();
}

async function init(): Promise<void> {
  bindEvents();
  updateOutputs();
  setMode('learned');
  setLook('none');
  updateExportState();
  const isolated = await ensureCrossOriginIsolation();
  if (!isolated) {
    el.banner.hidden = false;
    el.banner.textContent =
      'このブラウザ環境では RAW の読み込み機能が使えません（JPEG / PNG は使えます）。通常ウィンドウの最新の Chrome / Edge / Firefox / Safari でお試しください。';
  }
}

void init();
