// 「Leica M10 の色」タブ: 写真（RAW / JPEG など。何枚でも）や「AEB 合成」の結果に Leica M10 の色を掛けて保存する。
// 一覧で選んだ写真をプレビューし、まとめて保存では全部を変換して 1 つの ZIP にする。
// 写真の設定（効き・階調・仕上げ）はすべての写真で共通、AEB の合成結果は 1 枚ずつ設定を持つ。

import type { Adjustments } from './core/adjust';
import { readExif, type ExposureInfo } from './core/exif';
import { createZip, type ZipEntry } from './core/zip';
import { isRawFile, makeThumbnail, withRawDecoder } from './raw';
import {
  $,
  baseName,
  bindSliders,
  BusyOverlay,
  CompareView,
  download,
  drawRGBA,
  errorMessage,
  exposureParts,
  FILE_ACCEPT,
  fillFileRow,
  formatBytes,
  isImageFile,
  toast,
  WorkerClient,
} from './ui';
import type {
  ExportFormat,
  ExportOptions,
  FromLeicaWorker,
  LeicaParams,
  LeicaSourceInfo,
  MergedImage,
  ToLeicaWorker,
} from './worker/protocol';

/** プレビューの長辺 (px) */
const PREVIEW_SIDE = 1600;

/** 「AEB 合成」タブから合成結果を受け取るときの依頼 */
export interface MergedHandoff {
  /** 保存するファイル名のもと（拡張子なし）。同じ名前の合成結果はあとから送ったもので置き換える */
  name: string;
  /** 一覧に出す説明（「3 枚・おまかせ」など） */
  detail: string;
  /** 「AEB 合成」タブの仕上げ調整（引き継ぐ） */
  adjust: Adjustments;
  /** フル解像度の合成結果を作る。progress で進み具合を知らせる */
  load: (progress: (label: string, fraction: number) => void) => Promise<MergedImage>;
}

/** photo: 写真（RAW / JPEG など）、merged: 「AEB 合成」の結果 */
type Kind = 'photo' | 'merged';

interface Settings {
  /** Leica の効き 0..1 */
  amount: number;
  /** 階調も Leica のカメラ内 JPEG に合わせる */
  tone: boolean;
  adjust: Adjustments;
}

interface Item {
  id: string;
  kind: Kind;
  /** 写真のファイル（kind = 'photo'） */
  file?: File;
  /** 保存するファイル名のもと（拡張子なし） */
  name: string;
  /** 一覧に出す名前と説明 */
  title: string;
  detail: string;
  /** 形式の表示（CR3・JPG・AEB 合成 など） */
  format: string;
  thumb?: string;
  /** ワーカーで開いたときの情報 */
  info?: LeicaSourceInfo;
  /** ワーカーが今この写真を持っているか / プレビューも用意してあるか */
  stored: boolean;
  previewReady: boolean;
  /** AEB の合成結果を受け取っている途中 */
  pending?: boolean;
  /** AEB の合成結果の設定（写真はすべて photoSettings を使う） */
  settings?: Settings;
  /** まとめて保存での状態 */
  batch?: 'running' | 'done' | 'error';
  error?: string;
  el: HTMLLIElement;
}

type Progress = (label: string, fraction: number) => void;

const el = {
  panel: $('leica-panel'),
  dropzone: $('leica-dropzone'),
  viewer: $('leica-viewer'),
  canvasWrap: $('leica-canvas-wrap'),
  result: $<HTMLCanvasElement>('leica-result'),
  before: $<HTMLCanvasElement>('leica-before'),
  toolbar: $('leica-toolbar'),
  info: $('leica-info'),
  spinner: $('leica-spinner'),
  cancel: $<HTMLButtonElement>('leica-cancel'),
  list: $<HTMLUListElement>('leica-source'),
  count: $('leica-count'),
  fileInput: $<HTMLInputElement>('leica-file-input'),
  tone: $<HTMLInputElement>('leica-tone'),
  hint: $('leica-hint'),
  exportBtn: $<HTMLButtonElement>('leica-export'),
  exportAll: $<HTMLButtonElement>('leica-export-all'),
  exportFormat: $<HTMLSelectElement>('leica-export-format'),
  exportSize: $<HTMLSelectElement>('leica-export-size'),
  exportQuality: $<HTMLInputElement>('leica-export-quality'),
  qualityRow: $('leica-quality-row'),
  batchHint: $('leica-batch-hint'),
  exportHint: $('leica-export-hint'),
};

const busy = new BusyOverlay($('leica-busy'));
const compare = new CompareView({
  toggle: $<HTMLButtonElement>('leica-compare'),
  result: el.result,
  before: el.before,
  handle: $('leica-compare-handle'),
  labels: $('leica-compare-labels'),
});

const sliders = {
  amount: $<HTMLInputElement>('leica-amount'),
  brightness: $<HTMLInputElement>('leica-brightness'),
  contrast: $<HTMLInputElement>('leica-contrast'),
  saturation: $<HTMLInputElement>('leica-saturation'),
  quality: el.exportQuality,
};

const signed = (v: number) => (v > 0 ? `+${v}` : `${v}`);
const formats: Partial<Record<keyof typeof sliders, (v: number) => string>> = {
  amount: (v) => `${v}%`,
  brightness: signed,
  contrast: signed,
  saturation: signed,
};

/** スライダーの値の表示を今の値に合わせる（bindEvents で用意する） */
let updateOutputs = (): void => {};

/** すべての写真で共通の設定。写真は階調まで Leica のカメラ内 JPEG にするのが既定 */
const photoSettings: Settings = { amount: 1, tone: true, adjust: { brightness: 0, contrast: 0, saturation: 0 } };
/**
 * AEB の合成結果を受け取ったときの効き・階調（最後に選んだものを覚えておく）。
 * HDR 合成で起こした暗部がつぶれないよう、階調は合わせない（明るさを残す）のが既定
 */
const mergedDefaults = { amount: 1, tone: false };

const items: Item[] = [];
let selected: Item | null = null;
/** 表示する写真を選び直すたびに増える。古い読み込みやプレビューの結果を捨てるのに使う */
let viewSeq = 0;
/** 表示する写真を読み込んでいる途中か */
let viewBusy = false;
let idSeq = 0;
/** まとめて保存の途中なら、その状態（中止の合図） */
let batch: { cancel: boolean } | null = null;
/** JPEG などのサムネイルは 1 枚ずつ作る（一度に大量に開くとメモリが足りなくなる） */
let thumbChain: Promise<void> = Promise.resolve();

// ---------------------------------------------------------------------------
// ワーカーとの通信（初めて写真を開くときに起動する）

let client: WorkerClient<ToLeicaWorker, FromLeicaWorker> | null = null;

function worker(): WorkerClient<ToLeicaWorker, FromLeicaWorker> {
  if (!client) {
    const w = new Worker(new URL('./worker/leica.worker.ts', import.meta.url), { type: 'module' });
    w.onerror = (e) => {
      console.error(e);
      toast('処理中に問題が発生しました。メモリ不足の可能性があります（保存サイズを小さくすると軽くなります）', true);
    };
    client = new WorkerClient(w);
  }
  return client;
}

/** ワーカーから写真を手放す */
function release(item: Item): void {
  if (!item.stored) return;
  item.stored = false;
  item.previewReady = false;
  client?.post({ type: 'release', id: item.id });
}

// ---------------------------------------------------------------------------
// 写真の一覧

function addFiles(files: File[]): void {
  if (batch) {
    toast('まとめて保存が終わってから追加してください');
    return;
  }
  const accepted = files.filter(isImageFile);
  const skipped = files.length - accepted.length;
  if (skipped > 0) toast(`${skipped} 件のファイルは画像ではないため読み込みませんでした`);
  if (!accepted.length) return;
  const added = accepted.map(addPhoto);
  updateList();
  void select(added[0]);
}

function newItem(kind: Kind, id: string, fields: Pick<Item, 'name' | 'title' | 'detail' | 'format'> & { file?: File }): Item {
  const item: Item = { id, kind, stored: false, previewReady: false, el: document.createElement('li'), ...fields };
  items.push(item);
  el.list.append(item.el);
  return item;
}

function addPhoto(file: File): Item {
  const raw = isRawFile(file);
  const item = newItem('photo', `p${++idSeq}`, {
    file,
    name: baseName(file.name),
    title: file.name,
    detail: raw ? `RAW · ${formatBytes(file.size)}` : formatBytes(file.size),
    format: file.name.split('.').pop()?.toUpperCase() ?? '',
  });
  // RAW のサムネイルは開いたときに、埋め込みのプレビューから作る
  if (!raw) thumbChain = thumbChain.then(() => describeImage(item));
  return item;
}

/** 機種と露出の説明 */
function describeExif(e: ExposureInfo): string {
  return [e.model, ...exposureParts(e)].filter(Boolean).join(' · ');
}

/** JPEG などの写真のサムネイルと撮影情報 */
async function describeImage(item: Item): Promise<void> {
  const { file } = item;
  if (!file || !items.includes(item)) return;
  try {
    const exif = readExif(await file.slice(0, 512 * 1024).arrayBuffer());
    if (!item.info) item.detail = describeExif(exif) || item.detail;
    if (!item.thumb) setThumb(item, await makeThumbnail(file));
  } catch {
    // サムネイルがなくても変換はできる
  }
  if (items.includes(item)) renderItem(item);
}

function setThumb(item: Item, url: string | undefined): void {
  if (!url) return;
  if (items.includes(item) && !item.thumb) item.thumb = url;
  else if (url.startsWith('blob:')) URL.revokeObjectURL(url);
}

function revokeThumb(item: Item): void {
  if (item.thumb?.startsWith('blob:')) URL.revokeObjectURL(item.thumb);
  item.thumb = undefined;
}

function removeItem(item: Item): void {
  if (batch) return;
  const i = items.indexOf(item);
  if (i < 0) return;
  items.splice(i, 1);
  release(item);
  revokeThumb(item);
  item.el.remove();
  if (selected === item) {
    selected = null;
    viewSeq++;
    viewBusy = false;
    busy.hide();
    const next = [...items.slice(i), ...items.slice(0, i).reverse()].find((it) => !it.pending);
    if (next) void select(next);
    else showNothing();
  }
  updateList();
}

function clearAll(): void {
  if (batch) return;
  for (const item of items) {
    release(item);
    revokeThumb(item);
    item.el.remove();
  }
  items.length = 0;
  selected = null;
  viewSeq++;
  viewBusy = false;
  busy.hide();
  el.exportHint.textContent = '';
  showNothing();
  updateList();
}

/** 表示する写真がないとき */
function showNothing(): void {
  showViewer(false);
  writeControls(photoSettings);
  updateHint();
  updateExportState();
}

function updateList(): void {
  for (const item of items) renderItem(item);
  el.count.textContent = items.length ? `${items.length} 枚` : '';
  updateHint();
  updateExportState();
}

function renderItem(item: Item): void {
  const on = item === selected;
  item.el.className = `file ${on ? 'selected' : ''} ${item.error ? 'error' : ''}`;
  const side: HTMLElement[] = [];
  if (item.pending || item.batch === 'running') {
    const dot = document.createElement('span');
    dot.className = 'loading-dot';
    side.push(dot);
  } else if (item.batch === 'done') {
    const ok = document.createElement('span');
    ok.className = 'ev';
    ok.textContent = '✓';
    ok.title = 'まとめて保存しました';
    side.push(ok);
  }
  fillFileRow(item.el, {
    thumb: item.thumb,
    placeholder: item.format,
    name: item.title,
    sub: item.error ?? item.detail,
    side,
    removeLabel: '削除',
    onRemove: () => removeItem(item),
    onSelect: item.pending ? undefined : () => void select(item),
    selected: on,
  });
}

/** 一覧のサムネイル（プレビューを縮小した JPEG の data URL） */
function thumbnail(canvas: HTMLCanvasElement): string {
  const s = Math.min(1, 96 / Math.min(canvas.width, canvas.height));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(canvas.width * s));
  c.height = Math.max(1, Math.round(canvas.height * s));
  c.getContext('2d')!.drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

// ---------------------------------------------------------------------------
// 写真を開く

/**
 * 写真をワーカーで開く。previewSide が 0 なら書き出しだけに使う（プレビューを作らない）。
 * stale が true を返したら（ほかの写真を選び直したなど）、重い受け渡しをせずにやめる
 */
async function openItem(item: Item, previewSide: number, progress: Progress, stale: () => boolean = () => false): Promise<void> {
  const { file } = item;
  if (!file) throw new Error('この写真はもう開けません');
  let res: Extract<FromLeicaWorker, { type: 'opened' }>;
  if (isRawFile(file)) {
    const raw = await withRawDecoder(async (decoder) => {
      if (stale()) return null;
      progress('RAW を開いています…', 0.1);
      const meta = await decoder.open(file, false);
      if (stale()) return null;
      progress('現像中…', 0.3);
      return { meta, image: await decoder.develop() };
    });
    if (!raw || stale()) return;
    const { meta, image } = raw;
    if (!item.thumb && meta.preview) setThumb(item, await makeThumbnail(meta.preview, meta.flip).catch(() => undefined));
    progress('準備中…', 0.9);
    res = await worker().request<'opened'>(
      (reqId) => ({
        type: 'openRaw',
        reqId,
        id: item.id,
        width: image.width,
        height: image.height,
        data: image.data,
        exif: meta.exif,
        color: meta.color,
        previewSide,
      }),
      undefined,
      [image.data.buffer],
    );
  } else {
    progress('読み込み中…', 0.3);
    res = await worker().request<'opened'>((reqId) => ({ type: 'openFile', reqId, id: item.id, file, previewSide }));
  }
  item.stored = true;
  item.previewReady = previewSide > 0;
  item.info = res.info;
  item.error = undefined;
  item.detail = describeExif(res.info.exif) || item.detail;
  // 開いている間に一覧から削除された
  if (!items.includes(item)) release(item);
  else renderItem(item);
}

/** 一覧で選んだ写真をプレビューする（写真はそのとき開き、前に表示していた写真は手放す） */
async function select(item: Item): Promise<void> {
  if (batch || item.pending || !items.includes(item)) return;
  if (item === selected && item.previewReady) return;
  const prev = selected;
  selected = item;
  const seq = ++viewSeq;
  if (prev && prev !== item && prev.kind === 'photo') release(prev);
  writeControls(settingsOf(item));
  updateList();
  if (item.previewReady) {
    requestRender();
    return;
  }
  const stale = () => seq !== viewSeq;
  viewBusy = true;
  updateExportState();
  busy.show('読み込み中…', 0);
  try {
    await openItem(item, PREVIEW_SIDE, (label, f) => !stale() && busy.show(label, f), stale);
    if (stale()) {
      // 読み込んでいる間にほかの写真を選んだ
      if (item !== selected && item.kind === 'photo') release(item);
      return;
    }
    busy.show('Leica M10 の色に変換中…', 1);
    if ((await renderOnce(seq)) && !item.thumb) {
      item.thumb = thumbnail(el.before);
      renderItem(item);
    }
    updateHint();
  } catch (e) {
    item.error = errorMessage(e);
    renderItem(item);
    if (!stale()) {
      showViewer(false);
      toast(item.error, true);
    }
  } finally {
    if (!stale()) {
      viewBusy = false;
      busy.hide();
    }
    updateExportState();
  }
}

/** 「AEB 合成」タブの合成結果を一覧に加えて表示する（同じ写真の組の結果は置き換える） */
async function openMerged(job: MergedHandoff): Promise<void> {
  if (batch) {
    toast('まとめて保存が終わってから送ってください', true);
    return;
  }
  let item = items.find((it) => it.kind === 'merged' && it.name === job.name);
  if (item) {
    release(item);
    revokeThumb(item);
  } else {
    item = newItem('merged', `m${++idSeq}`, { name: job.name, title: 'AEB 合成の結果', detail: job.detail, format: 'AEB 合成' });
  }
  const target = item;
  target.pending = true;
  target.info = undefined;
  target.error = undefined;
  target.batch = undefined;
  target.detail = job.detail;
  target.settings = { amount: mergedDefaults.amount, tone: mergedDefaults.tone, adjust: { ...job.adjust } };
  const prev = selected;
  selected = target;
  const seq = ++viewSeq;
  const stale = () => seq !== viewSeq;
  if (prev && prev !== target && prev.kind === 'photo') release(prev);
  writeControls(target.settings);
  el.exportHint.textContent = '';
  viewBusy = true;
  updateList();
  busy.show('合成結果を用意しています…', 0);
  try {
    const image = await job.load((label, f) => !stale() && busy.show(`${label}…`, f));
    // 待っている間に一覧から削除された
    if (!items.includes(target)) return;
    if (!stale()) busy.show('プレビューを準備中…', 1);
    const { info } = await worker().request<'opened'>(
      (reqId) => ({ type: 'openMerged', reqId, id: target.id, image, previewSide: PREVIEW_SIDE }),
      undefined,
      [image.data.buffer],
    );
    target.pending = false;
    target.stored = true;
    target.previewReady = true;
    target.info = info;
    target.detail = [job.detail, info.exif.model].filter(Boolean).join(' · ');
    if (!items.includes(target)) {
      release(target);
      return;
    }
    renderItem(target);
    if (stale()) return;
    busy.show('Leica M10 の色に変換中…', 1);
    if (await renderOnce(seq)) {
      target.thumb = thumbnail(el.before);
      renderItem(target);
    }
  } catch (e) {
    // 合成結果を受け取れなかったものは一覧から外す
    if (target.pending) {
      target.pending = false;
      removeItem(target);
    }
    toast(errorMessage(e), true);
  } finally {
    if (!stale()) {
      viewBusy = false;
      busy.hide();
    }
    updateList();
  }
}

// ---------------------------------------------------------------------------
// プレビュー

let renderInFlight = false;
let renderWanted = false;

function requestRender(): void {
  if (!selected?.previewReady) return;
  renderWanted = true;
  if (!renderInFlight) void renderLoop();
}

async function renderLoop(): Promise<void> {
  renderInFlight = true;
  el.spinner.hidden = false;
  while (renderWanted && selected?.previewReady) {
    renderWanted = false;
    await renderOnce(viewSeq);
  }
  el.spinner.hidden = true;
  renderInFlight = false;
}

/** 選んでいる写真のプレビューを作って表示する。表示できたら true */
async function renderOnce(seq: number): Promise<boolean> {
  const item = selected;
  if (!item?.previewReady) return false;
  const params = paramsOf(settingsOf(item));
  try {
    const res = await worker().request<'rendered'>((reqId) => ({ type: 'render', reqId, id: item.id, params }));
    if (seq !== viewSeq || selected !== item) return false;
    drawRGBA(el.result, res.rgba, res.width, res.height);
    drawRGBA(el.before, res.before, res.width, res.height);
    showViewer(true);
    updateInfo(item, params);
    return true;
  } catch (e) {
    if (seq === viewSeq) toast(errorMessage(e), true);
    return false;
  }
}

function showViewer(on: boolean): void {
  el.viewer.hidden = !on;
  el.toolbar.hidden = !on;
  el.dropzone.hidden = on;
}

function updateInfo(item: Item, params: LeicaParams): void {
  const { info } = item;
  if (!info) return;
  const { amount, tone } = params.look;
  const look = amount > 0 ? `Leica M10 ${Math.round(amount * 100)}%${tone ? '・階調も' : ''}` : '標準の色';
  el.info.textContent = [`${info.width}×${info.height}`, item.format, look].join(' · ');
}

// ---------------------------------------------------------------------------
// 設定

/** 写真はすべての写真で共通の設定、AEB の合成結果はそれぞれの設定 */
function settingsOf(item: Item | null): Settings {
  return item?.kind === 'merged' && item.settings ? item.settings : photoSettings;
}

function paramsOf(s: Settings): LeicaParams {
  return { look: { amount: s.amount, tone: s.tone }, adjust: { ...s.adjust } };
}

/** 設定を画面の操作部品に映す */
function writeControls(s: Settings): void {
  sliders.amount.value = String(Math.round(s.amount * 100));
  el.tone.checked = s.tone;
  sliders.brightness.value = String(s.adjust.brightness);
  sliders.contrast.value = String(s.adjust.contrast);
  sliders.saturation.value = String(s.adjust.saturation);
  updateOutputs();
}

/** 操作部品の値を、表示中の写真の設定（写真なら全写真で共通の設定）に書き込む */
function onSettingsInput(): void {
  const s = settingsOf(selected);
  s.amount = Number(sliders.amount.value) / 100;
  s.tone = el.tone.checked;
  s.adjust = {
    brightness: Number(sliders.brightness.value),
    contrast: Number(sliders.contrast.value),
    saturation: Number(sliders.saturation.value),
  };
  if (selected?.kind === 'merged') {
    mergedDefaults.amount = s.amount;
    mergedDefaults.tone = s.tone;
  }
  updateHint();
  requestRender();
}

/** Leica M10 の色の説明。写真を開いていれば、センサーの違いをどう変換するか（機種・光源）も添える */
function updateHint(): void {
  const item = selected;
  const kind = item?.kind ?? 'photo';
  const parts = [
    'Leica M10 のセンサーの色の出方と、Leica のカメラ内 JPEG の色づくり（Canon の JPEG より彩度は控えめ、肌は赤み寄り、黄緑は黄み寄り）を再現します。',
    settingsOf(item).tone
      ? '階調も Leica のカメラ内 JPEG のトーンカーブにします。'
      : kind === 'merged'
        ? '明るさは合成結果のまま残します。'
        : '明るさは元の写真のまま残し、色だけを変えます。',
  ];
  const info = item?.info;
  if (info) {
    if (info.jpeg) parts.push('JPEG は Canon のピクチャースタイル「スタンダード」の色を打ち消してから変換します。');
    if (info.cameraMatched) {
      parts.push(`この機種の分光感度から変換しています${info.cct ? `（光源の色温度 約 ${Math.round(info.cct / 100) * 100}K）` : ''}。`);
    } else if (info.exif.make || info.exif.model) {
      parts.push('この機種の分光感度データがないため、色を正確に写すカメラとして変換しています。');
    } else {
      parts.push('撮影したカメラがわからないため、色を正確に写すカメラとして変換しています。');
    }
  }
  const photos = items.filter((it) => it.kind === 'photo').length;
  if (kind === 'photo' && photos > 1) parts.push(`設定は読み込んだ写真（${photos} 枚）すべてに共通です。`);
  if (kind === 'merged' && items.length > 1) parts.push('この設定は、この合成結果だけに使います。');
  el.hint.textContent = parts.join('');
}

// ---------------------------------------------------------------------------
// 書き出し

function exportOptions(): ExportOptions {
  return {
    format: el.exportFormat.value as ExportFormat,
    quality: Number(el.exportQuality.value) / 100,
    maxSide: Number(el.exportSize.value),
  };
}

function outputName(item: Item, params: LeicaParams, format: ExportFormat): string {
  const ext = format === 'jpeg' ? 'jpg' : format === 'png' ? 'png' : 'tif';
  return `${item.name}${params.look.amount > 0 ? '_M10' : ''}.${ext}`;
}

/** ZIP の中で名前が重ならないよう、同じ名前の 2 つ目からは _2, _3 … を付ける */
function uniqueName(name: string, used: Set<string>): string {
  let out = name;
  for (let k = 2; used.has(out.toLowerCase()); k++) out = name.replace(/(\.[^.]+)?$/, `_${k}$1`);
  used.add(out.toLowerCase());
  return out;
}

function zipName(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `Leica_M10_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.zip`;
}

function updateExportState(): void {
  const ready = items.filter((it) => !it.pending).length;
  const many = ready > 1;
  el.exportBtn.disabled = !!batch || viewBusy || !selected?.stored;
  el.exportBtn.textContent = many ? '表示中の写真だけ保存' : '画像を保存';
  el.exportBtn.classList.toggle('primary', !many);
  el.exportAll.hidden = !many;
  el.exportAll.disabled = !!batch || viewBusy;
  el.exportAll.textContent = `まとめて保存（${ready} 枚・ZIP）`;
  el.qualityRow.hidden = el.exportFormat.value !== 'jpeg';
  el.batchHint.hidden = !many;
  el.batchHint.textContent =
    '読み込んだ写真をすべて Leica M10 の色に変換し、1 つの ZIP ファイルにまとめて保存します。' +
    (el.exportFormat.value === 'tiff' ? 'TIFF（16bit）は 1 枚あたり 2000 万画素で約 120MB になるため、枚数が多いときは JPEG がおすすめです。' : '');
}

/** 表示中の写真を 1 枚保存する */
async function doExport(): Promise<void> {
  const item = selected;
  if (!item?.stored || batch) return;
  const seq = viewSeq;
  const options = exportOptions();
  const params = paramsOf(settingsOf(item));
  el.exportBtn.disabled = true;
  el.exportHint.classList.remove('error');
  el.exportHint.textContent = '';
  busy.show('書き出し中…', 0);
  try {
    const res = await worker().request<'exported'>(
      (reqId) => ({ type: 'export', reqId, id: item.id, params, options }),
      (label, f) => seq === viewSeq && busy.show(`${label}…`, f),
    );
    // 保存中にほかの写真を選んでも、押した保存はそのまま届ける
    download(res.blob, outputName(item, params, options.format));
    if (seq !== viewSeq) return;
    el.exportHint.textContent = `${res.width}×${res.height}（${formatBytes(res.blob.size)}）を保存しました · ${(res.elapsed / 1000).toFixed(1)} 秒`;
  } catch (e) {
    if (seq !== viewSeq) return;
    el.exportHint.classList.add('error');
    el.exportHint.textContent = errorMessage(e);
  } finally {
    if (seq === viewSeq) busy.hide();
    updateExportState();
  }
}

/**
 * 1 枚を書き出す。ワーカーが持っていなければ書き出しのためだけに開き、終わったら手放す
 * （表示中の写真と AEB の合成結果は持ったまま）
 */
async function exportItem(
  item: Item,
  params: LeicaParams,
  options: ExportOptions,
  progress: Progress,
): Promise<Extract<FromLeicaWorker, { type: 'exported' }>> {
  const opened = item.stored;
  if (!opened) await openItem(item, 0, (label, f) => progress(label, f * 0.6));
  try {
    return await worker().request<'exported'>(
      (reqId) => ({ type: 'export', reqId, id: item.id, params, options, crc: true }),
      (label, f) => progress(label, 0.6 + f * 0.4),
    );
  } finally {
    if (!opened) release(item);
  }
}

/** 一覧の写真をすべて変換して、1 つの ZIP にまとめて保存する */
async function exportAll(): Promise<void> {
  const list = items.filter((it) => !it.pending);
  if (batch || viewBusy || list.length < 2) return;
  const run = { cancel: false };
  batch = run;
  const options = exportOptions();
  const t0 = performance.now();
  const entries: ZipEntry[] = [];
  const used = new Set<string>();
  let failed = 0;
  el.panel.inert = true;
  el.cancel.hidden = false;
  el.exportHint.classList.remove('error');
  el.exportHint.textContent = '';
  for (const item of list) item.batch = undefined;
  updateList();
  try {
    for (let k = 0; k < list.length && !run.cancel; k++) {
      const item = list[k];
      const head = `まとめて変換中 ${k + 1}/${list.length}`;
      const at = (f: number) => (k + Math.min(1, Math.max(0, f))) / list.length;
      busy.show(`${head}（${item.title}）`, at(0));
      item.batch = 'running';
      renderItem(item);
      const params = paramsOf(settingsOf(item));
      try {
        const res = await exportItem(item, params, options, (label, f) => busy.show(`${head}（${label.replace(/…$/, '')}）`, at(f)));
        entries.push({ name: uniqueName(outputName(item, params, options.format), used), data: res.blob, crc: res.crc ?? 0 });
        item.batch = 'done';
        item.error = undefined;
      } catch (e) {
        item.batch = 'error';
        item.error = errorMessage(e);
        failed++;
      }
      renderItem(item);
    }
    if (run.cancel) {
      // ZIP は保存していないので、変換済みの印も消す
      for (const item of list) if (item.batch !== 'error') item.batch = undefined;
      el.exportHint.textContent = 'まとめて保存を中止しました';
      return;
    }
    if (!entries.length) throw new Error('変換できた写真がありませんでした');
    busy.show('ZIP にまとめています…', 1);
    const zip = createZip(entries);
    download(zip, zipName());
    const took = ((performance.now() - t0) / 1000).toFixed(1);
    el.exportHint.textContent =
      `${entries.length} 枚を ZIP（${formatBytes(zip.size)}）にまとめて保存しました · ${took} 秒` +
      (failed ? `。${failed} 枚は変換できませんでした（一覧に理由を表示しています）` : '');
    el.exportHint.classList.toggle('error', failed > 0);
  } catch (e) {
    el.exportHint.classList.add('error');
    el.exportHint.textContent = errorMessage(e);
  } finally {
    batch = null;
    el.panel.inert = false;
    el.cancel.hidden = true;
    busy.hide();
    updateList();
  }
}

// ---------------------------------------------------------------------------

function bindEvents(): void {
  el.fileInput.accept = FILE_ACCEPT;
  const pick = () => el.fileInput.click();
  $('leica-pick').onclick = pick;
  $('leica-add').onclick = pick;
  $('leica-clear').onclick = clearAll;
  el.fileInput.onchange = () => {
    addFiles([...(el.fileInput.files ?? [])]);
    el.fileInput.value = '';
  };

  updateOutputs = bindSliders(sliders, formats, (key) => {
    if (key !== 'quality') onSettingsInput();
  });
  el.tone.onchange = onSettingsInput;
  $('leica-reset-adjust').onclick = () => {
    for (const s of [sliders.brightness, sliders.contrast, sliders.saturation]) s.value = s.defaultValue;
    updateOutputs();
    onSettingsInput();
  };

  // 比較中はプレビューのドラッグで境界線を動かす
  let dragging = false;
  el.canvasWrap.addEventListener('pointerdown', (e) => {
    if (!compare.on) return;
    dragging = true;
    el.canvasWrap.setPointerCapture(e.pointerId);
    compare.moveTo(e);
  });
  el.canvasWrap.addEventListener('pointermove', (e) => {
    if (dragging) compare.moveTo(e);
  });
  el.canvasWrap.addEventListener('pointerup', () => (dragging = false));
  el.canvasWrap.addEventListener('pointercancel', () => (dragging = false));

  el.exportFormat.onchange = updateExportState;
  el.exportBtn.onclick = () => void doExport();
  el.exportAll.onclick = () => void exportAll();
  el.cancel.onclick = () => {
    if (!batch) return;
    batch.cancel = true;
    el.cancel.hidden = true;
    busy.show('中止しています（変換中の 1 枚が終わるまでお待ちください）…', 1);
  };
}

/** タブを用意する。ドロップされたファイルは addFiles、「AEB 合成」の結果は openMerged で受け取る */
export function initLeica(): { addFiles: (files: File[]) => void; openMerged: (job: MergedHandoff) => void } {
  bindEvents();
  writeControls(photoSettings);
  updateHint();
  updateExportState();
  return { addFiles, openMerged: (job) => void openMerged(job) };
}
