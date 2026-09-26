// メインスレッド ⇔ 合成ワーカー間のメッセージ定義

import type { Adjustments } from '../core/adjust';
import type { ExposureInfo } from '../core/exif';
import type { FusionWeights } from '../core/fusion';
import type { ToneParams } from '../core/hdr';

export type Mode = 'fusion' | 'hdr';

export interface RenderParams {
  mode: Mode;
  fusion: FusionWeights;
  tone: ToneParams;
  adjust: Adjustments;
}

export type ExportFormat = 'jpeg' | 'png' | 'tiff';

export interface ExportOptions {
  format: ExportFormat;
  /** JPEG 品質 0..1 */
  quality: number;
  /** 長辺の上限 (px)。0 なら元のサイズ */
  maxSide: number;
}

export interface PreparedInfo {
  /** 暗い→明るい順の ID */
  order: string[];
  referenceId: string;
  /** 合成結果（切り抜き後）の実寸 */
  width: number;
  height: number;
  previewWidth: number;
  previewHeight: number;
  /** ID → [dx, dy] */
  shifts: Record<string, [number, number]>;
  /** ID → 画像から推定した基準比の露出 (EV) */
  relativeEv: Record<string, number>;
  aligned: boolean;
}

export type ToWorker =
  | { type: 'addRaw'; id: string; name: string; width: number; height: number; data: Uint16Array; exif: ExposureInfo }
  | { type: 'addFile'; id: string; name: string; file: File }
  | { type: 'remove'; id: string }
  | { type: 'prepare'; reqId: number; align: boolean; previewSide: number }
  | { type: 'render'; reqId: number; params: RenderParams }
  | { type: 'export'; reqId: number; params: RenderParams; options: ExportOptions };

export type FromWorker =
  | { type: 'added'; id: string; width: number; height: number; exif: ExposureInfo }
  | { type: 'addFailed'; id: string; message: string }
  | { type: 'progress'; reqId: number; label: string; fraction: number }
  | { type: 'prepared'; reqId: number; info: PreparedInfo; reference: Uint8ClampedArray }
  | { type: 'rendered'; reqId: number; width: number; height: number; rgba: Uint8ClampedArray; elapsed: number }
  | { type: 'exported'; reqId: number; blob: Blob; width: number; height: number; elapsed: number }
  | { type: 'error'; reqId: number; message: string };
