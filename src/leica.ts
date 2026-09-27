// 「Leica M10 の色」タブ: 写真 1 枚（RAW / JPEG など）か「AEB 合成」の結果に Leica M10 の色を掛けて保存する。

import type { Adjustments } from './core/adjust';
import { isRawFile, withRawDecoder } from './raw';
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
import type { ExportFormat, FromLeicaWorker, LeicaParams, LeicaSourceInfo, MergedImage, ToLeicaWorker } from './worker/protocol';

/** プレビューの長辺 (px) */
const PREVIEW_SIDE = 1600;

/** 「AEB 合成」タブから合成結果を受け取るときの依頼 */
export interface MergedHandoff {
  /** 保存するファイル名のもと（拡張子なし） */
  name: string;
  /** 写真の欄に出す説明（「3 枚・おまかせ」など） */
  detail: string;
  /** 「AEB 合成」タブの仕上げ調整（引き継ぐ） */
  adjust: Adjustments;
  /** フル解像度の合成結果を作る。progress で進み具合を知らせる */
  load: (progress: (label: string, fraction: number) => void) => Promise<MergedImage>;
}

/** photo: 1 枚の写真（RAW / JPEG など）、merged: 「AEB 合成」の結果 */
type Kind = 'photo' | 'merged';

interface Current {
  kind: Kind;
  /** 保存するファイル名のもと（拡張子なし） */
  name: string;
  /** 写真の欄の名前と説明 */
  title: string;
  detail: string;
  /** 形式の表示（CR3・JPG・AEB 合成 など） */
  format: string;
  info: LeicaSourceInfo;
  thumb?: string;
}

const el = {
  dropzone: $('leica-dropzone'),
  viewer: $('leica-viewer'),
  canvasWrap: $('leica-canvas-wrap'),
  result: $<HTMLCanvasElement>('leica-result'),
  before: $<HTMLCanvasElement>('leica-before'),
  toolbar: $('leica-toolbar'),
  info: $('leica-info'),
  spinner: $('leica-spinner'),
  source: $<HTMLUListElement>('leica-source'),
  fileInput: $<HTMLInputElement>('leica-file-input'),
  tone: $<HTMLInputElement>('leica-tone'),
  hint: $('leica-hint'),
  exportBtn: $<HTMLButtonElement>('leica-export'),
  exportFormat: $<HTMLSelectElement>('leica-export-format'),
  exportSize: $<HTMLSelectElement>('leica-export-size'),
  exportQuality: $<HTMLInputElement>('leica-export-quality'),
  qualityRow: $('leica-quality-row'),
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

/**
 * 「階調も Leica のカメラ内 JPEG に合わせる」の設定。1 枚の写真は階調まで Leica にするのが既定、
 * AEB の合成結果は HDR 合成で起こした暗部がつぶれないよう明るさを残すのが既定（変えた値はそれぞれ覚えておく）
 */
const toneByKind: Record<Kind, boolean> = { photo: true, merged: false };

let current: Current | null = null;
/** 読み込み中の写真（写真の欄に出す） */
let loading: { kind: Kind; title: string } | null = null;
/** 写真を開くたびに増える。古い読み込みやプレビューの結果を捨てるのに使う */
let openSeq = 0;

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

// ---------------------------------------------------------------------------
// 写真を開く

function addFiles(files: File[]): void {
  const accepted = files.filter(isImageFile);
  if (!accepted.length) {
    if (files.length) toast('画像ではないファイルは開けません');
    return;
  }
  if (accepted.length > 1) {
    toast(`${accepted.length} 枚のうち最初の 1 枚を開きました（露出違いの写真を合成するときは「AEB 合成」タブへ）`);
  }
  void openPhoto(accepted[0]);
}

/** 新しく開く準備: 今の写真を閉じてメモリを空け、古い結果を捨てるための番号を進める */
function beginOpen(kind: Kind, title: string): number {
  current = null;
  loading = { kind, title };
  client?.post({ type: 'close' });
  el.tone.checked = toneByKind[kind];
  el.exportHint.textContent = '';
  renderSource();
  updateHint();
  updateExportState();
  return ++openSeq;
}

async function openPhoto(file: File): Promise<void> {
  const seq = beginOpen('photo', file.name);
  const stale = () => seq !== openSeq;
  busy.show('読み込み中…', 0);
  try {
    let res: Extract<FromLeicaWorker, { type: 'opened' }>;
    if (isRawFile(file)) {
      const raw = await withRawDecoder(async (decoder) => {
        if (stale()) return null;
        busy.show('RAW を開いています…', 0.1);
        const meta = await decoder.open(file, false);
        if (stale()) return null;
        busy.show('現像中…', 0.3);
        return { meta, image: await decoder.develop() };
      });
      if (!raw || stale()) return;
      busy.show('プレビューを準備中…', 0.9);
      const { meta, image } = raw;
      res = await worker().request<'opened'>(
        (reqId) => ({
          type: 'openRaw',
          reqId,
          width: image.width,
          height: image.height,
          data: image.data,
          exif: meta.exif,
          color: meta.color,
          previewSide: PREVIEW_SIDE,
        }),
        undefined,
        [image.data.buffer],
      );
    } else {
      busy.show('読み込み中…', 0.3);
      res = await worker().request<'opened'>((reqId) => ({ type: 'openFile', reqId, file, previewSide: PREVIEW_SIDE }));
    }
    if (stale()) return;
    const { info } = res;
    const detail = [info.exif.model, ...exposureParts(info.exif)].filter(Boolean).join(' · ') || '撮影情報なし';
    const format = file.name.split('.').pop()?.toUpperCase() ?? '';
    await showSource(seq, { kind: 'photo', name: baseName(file.name), title: file.name, detail, format, info });
  } catch (e) {
    if (!stale()) failOpen(e);
  } finally {
    if (!stale()) busy.hide();
  }
}

/** 「AEB 合成」タブの合成結果を開く（仕上げ調整の値も引き継ぐ） */
async function openMerged(job: MergedHandoff): Promise<void> {
  const seq = beginOpen('merged', 'AEB 合成の結果');
  const stale = () => seq !== openSeq;
  busy.show('合成結果を用意しています…', 0);
  try {
    const image = await job.load((label, f) => {
      if (!stale()) busy.show(`${label}…`, f);
    });
    if (stale()) return;
    busy.show('プレビューを準備中…', 1);
    const { info } = await worker().request<'opened'>(
      (reqId) => ({ type: 'openMerged', reqId, image, previewSide: PREVIEW_SIDE }),
      undefined,
      [image.data.buffer],
    );
    if (stale()) return;
    setAdjust(job.adjust);
    const detail = [job.detail, info.exif.model].filter(Boolean).join(' · ');
    await showSource(seq, { kind: 'merged', name: job.name, title: 'AEB 合成の結果', detail, format: 'AEB 合成', info });
  } catch (e) {
    if (!stale()) failOpen(e);
  } finally {
    if (!stale()) busy.hide();
  }
}

async function showSource(seq: number, c: Current): Promise<void> {
  current = c;
  loading = null;
  renderSource();
  updateHint();
  busy.show('Leica M10 の色に変換中…', 1);
  if (await renderOnce(seq)) {
    c.thumb = thumbnail(el.before);
    renderSource();
  }
  updateExportState();
}

function failOpen(e: unknown): void {
  current = null;
  loading = null;
  renderSource();
  showViewer(false);
  updateExportState();
  toast(errorMessage(e), true);
}

/** 開いている（読み込み中の）写真を閉じる */
function closeSource(): void {
  openSeq++;
  current = null;
  loading = null;
  client?.post({ type: 'close' });
  busy.hide();
  el.exportHint.textContent = '';
  renderSource();
  showViewer(false);
  updateHint();
  updateExportState();
}

function renderSource(): void {
  const title = current?.title ?? loading?.title;
  if (!title) {
    el.source.replaceChildren();
    return;
  }
  const li = document.createElement('li');
  li.className = 'file';
  const side: HTMLElement[] = [];
  if (!current) {
    const dot = document.createElement('span');
    dot.className = 'loading-dot';
    side.push(dot);
  }
  fillFileRow(li, {
    thumb: current?.thumb,
    name: title,
    sub: current?.detail ?? '読み込み中…',
    side,
    removeLabel: '閉じる',
    onRemove: closeSource,
  });
  el.source.replaceChildren(li);
}

/** 写真の欄のサムネイル（プレビューを縮小した JPEG の data URL） */
function thumbnail(canvas: HTMLCanvasElement): string {
  const s = Math.min(1, 96 / Math.min(canvas.width, canvas.height));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(canvas.width * s));
  c.height = Math.max(1, Math.round(canvas.height * s));
  c.getContext('2d')!.drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

// ---------------------------------------------------------------------------
// プレビュー

let renderInFlight = false;
let renderWanted = false;

function requestRender(): void {
  if (!current) return;
  renderWanted = true;
  if (!renderInFlight) void renderLoop();
}

async function renderLoop(): Promise<void> {
  renderInFlight = true;
  el.spinner.hidden = false;
  while (renderWanted && current) {
    renderWanted = false;
    await renderOnce(openSeq);
  }
  el.spinner.hidden = true;
  renderInFlight = false;
}

/** プレビューを作って表示する。表示できたら true */
async function renderOnce(seq: number): Promise<boolean> {
  const params = currentParams();
  try {
    const res = await worker().request<'rendered'>((reqId) => ({ type: 'render', reqId, params }));
    if (seq !== openSeq || !current) return false;
    drawRGBA(el.result, res.rgba, res.width, res.height);
    drawRGBA(el.before, res.before, res.width, res.height);
    showViewer(true);
    updateInfo(params);
    return true;
  } catch (e) {
    if (seq === openSeq) toast(errorMessage(e), true);
    return false;
  }
}

function showViewer(on: boolean): void {
  el.viewer.hidden = !on;
  el.toolbar.hidden = !on;
  el.dropzone.hidden = on;
}

function updateInfo(params: LeicaParams): void {
  const c = current;
  if (!c) return;
  const { amount, tone } = params.look;
  const look = amount > 0 ? `Leica M10 ${Math.round(amount * 100)}%${tone ? '・階調も' : ''}` : '標準の色';
  el.info.textContent = [`${c.info.width}×${c.info.height}`, c.format, look].join(' · ');
}

// ---------------------------------------------------------------------------
// パラメータ

function currentParams(): LeicaParams {
  return {
    look: { amount: Number(sliders.amount.value) / 100, tone: el.tone.checked },
    adjust: {
      brightness: Number(sliders.brightness.value),
      contrast: Number(sliders.contrast.value),
      saturation: Number(sliders.saturation.value),
    },
  };
}

function setAdjust(a: Adjustments): void {
  sliders.brightness.value = String(a.brightness);
  sliders.contrast.value = String(a.contrast);
  sliders.saturation.value = String(a.saturation);
  updateOutputs();
}

/** Leica M10 の色の説明。写真を開いていれば、センサーの違いをどう変換するか（機種・光源）も添える */
function updateHint(): void {
  const kind = (current ?? loading)?.kind ?? 'photo';
  const parts = [
    'Leica M10 のセンサーの色の出方と、Leica のカメラ内 JPEG の色づくり（Canon の JPEG より彩度は控えめ、肌は赤み寄り、黄緑は黄み寄り）を再現します。',
    el.tone.checked
      ? '階調も Leica のカメラ内 JPEG のトーンカーブにします。'
      : kind === 'merged'
        ? '明るさは合成結果のまま残します。'
        : '明るさは元の写真のまま残し、色だけを変えます。',
  ];
  const info = current?.info;
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
  el.hint.textContent = parts.join('');
}

// ---------------------------------------------------------------------------
// 書き出し

function updateExportState(): void {
  el.exportBtn.disabled = !current;
  el.qualityRow.hidden = el.exportFormat.value !== 'jpeg';
}

async function doExport(): Promise<void> {
  const c = current;
  if (!c) return;
  const seq = openSeq;
  const format = el.exportFormat.value as ExportFormat;
  const options = { format, quality: Number(el.exportQuality.value) / 100, maxSide: Number(el.exportSize.value) };
  const params = currentParams();
  el.exportBtn.disabled = true;
  el.exportHint.classList.remove('error');
  el.exportHint.textContent = '';
  busy.show('書き出し中…', 0);
  try {
    const res = await worker().request<'exported'>(
      (reqId) => ({ type: 'export', reqId, params, options }),
      (label, f) => {
        if (seq === openSeq) busy.show(`${label}…`, f);
      },
    );
    // 保存中に別の写真を開いても、押した保存はそのまま届ける
    const ext = format === 'jpeg' ? 'jpg' : format === 'png' ? 'png' : 'tif';
    download(res.blob, `${c.name}${params.look.amount > 0 ? '_M10' : ''}.${ext}`);
    if (seq !== openSeq) return;
    el.exportHint.textContent = `${res.width}×${res.height}（${formatBytes(res.blob.size)}）を保存しました · ${(res.elapsed / 1000).toFixed(1)} 秒`;
  } catch (e) {
    if (seq !== openSeq) return;
    el.exportHint.classList.add('error');
    el.exportHint.textContent = errorMessage(e);
  } finally {
    if (seq === openSeq) busy.hide();
    updateExportState();
  }
}

// ---------------------------------------------------------------------------

function bindEvents(): void {
  el.fileInput.accept = FILE_ACCEPT;
  const pick = () => el.fileInput.click();
  $('leica-pick').onclick = pick;
  $('leica-open').onclick = pick;
  el.fileInput.onchange = () => {
    addFiles([...(el.fileInput.files ?? [])]);
    el.fileInput.value = '';
  };

  updateOutputs = bindSliders(sliders, formats, (key) => {
    if (key !== 'quality') requestRender();
  });
  el.tone.onchange = () => {
    toneByKind[(current ?? loading)?.kind ?? 'photo'] = el.tone.checked;
    updateHint();
    requestRender();
  };
  $('leica-reset-adjust').onclick = () => {
    for (const s of [sliders.brightness, sliders.contrast, sliders.saturation]) s.value = s.defaultValue;
    updateOutputs();
    requestRender();
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
}

/** タブを用意する。ドロップされたファイルは addFiles、「AEB 合成」の結果は openMerged で受け取る */
export function initLeica(): { addFiles: (files: File[]) => void; openMerged: (job: MergedHandoff) => void } {
  bindEvents();
  updateHint();
  updateExportState();
  return { addFiles, openMerged: (job) => void openMerged(job) };
}
