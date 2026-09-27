// メインスレッド ⇔ 合成ワーカー間のメッセージ定義

import type { Adjustments } from '../core/adjust';
import type { ExposureInfo } from '../core/exif';
import type { FusionWeights } from '../core/fusion';
import type { ToneParams } from '../core/hdr';
import type { LearnedParams } from '../core/learned';
import type { LookParams } from '../core/look';

/** learned: 学習済みモデル（おまかせ）、fusion: 露出フュージョン（ナチュラル）、hdr: トーンマッピング */
export type Mode = 'learned' | 'fusion' | 'hdr';

export interface RenderParams {
  mode: Mode;
  /** 効果の強さ 0..1。0 で基準フレームそのまま、1 で合成結果そのまま */
  amount: number;
  fusion: FusionWeights;
  tone: ToneParams;
  learned: LearnedParams;
  /** 色の傾向（Leica M10 など）。合成・効果の強さの後、仕上げ調整の前に掛ける */
  look: LookParams;
  adjust: Adjustments;
}

/** RAW の色の情報（撮影時のホワイトバランスと色行列）。色温度の推定に使う */
export interface RawColor {
  /** カメラのニュートラル（ホワイトバランスの倍率の逆数、G = 1） */
  neutral: [number, number, number];
  /** XYZ → カメラ RGB（D65、行優先 9 要素） */
  camXyz: number[];
}

/** 色の傾向を掛けるときの前提（基準の写真から決まる） */
export interface LookInfo {
  /** 基準の写真が RAW（linear）か JPEG など（srgb）か */
  encoding: 'linear' | 'srgb';
  /** 機種の分光感度データでセンサーの違いを変換したか */
  cameraMatched: boolean;
  /** 撮影時の色温度の推定値 [K] */
  cct?: number;
}

export type ExportFormat = 'jpeg' | 'png' | 'tiff';

export interface ExportOptions {
  format: ExportFormat;
  /** JPEG 品質 0..1 */
  quality: number;
  /** 長辺の上限 (px)。0 なら元のサイズ */
  maxSide: number;
}

/** 手動の位置調整: 写っている内容を右・下へ動かす量 [px]（フル解像度）と時計回りの回転 [度] */
export interface ManualAdjust {
  x: number;
  y: number;
  rotation: number;
}

export interface FrameAlignment {
  /** 自動位置合わせで求めた、基準に対するずれ（画像中心での移動量 [px] と回転 [度]） */
  auto: { dx: number; dy: number; rotation: number; precise: boolean };
  manual: ManualAdjust;
}

export type LoupeMode = 'blend' | 'diff';

export interface PreparedInfo {
  /** 暗い→明るい順の ID */
  order: string[];
  referenceId: string;
  /** 合成結果（切り抜き後）の実寸 */
  width: number;
  height: number;
  previewWidth: number;
  previewHeight: number;
  alignment: Record<string, FrameAlignment>;
  /** ID → 画像から推定した基準比の露出 (EV) */
  relativeEv: Record<string, number>;
  aligned: boolean;
  look: LookInfo;
}

export type ToWorker =
  | { type: 'addRaw'; id: string; name: string; width: number; height: number; data: Uint16Array; exif: ExposureInfo; color?: RawColor }
  | { type: 'addFile'; id: string; name: string; file: File }
  | { type: 'remove'; id: string }
  | { type: 'prepare'; reqId: number; align: boolean; previewSide: number }
  | { type: 'render'; reqId: number; params: RenderParams }
  | { type: 'setManual'; id: string; manual: ManualAdjust }
  | { type: 'loupe'; reqId: number; id: string; cx: number; cy: number; size: number; mode: LoupeMode }
  | { type: 'export'; reqId: number; params: RenderParams; options: ExportOptions };

export type FromWorker =
  | { type: 'added'; id: string; width: number; height: number; exif: ExposureInfo }
  | { type: 'addFailed'; id: string; message: string }
  | { type: 'progress'; reqId: number; label: string; fraction: number }
  | { type: 'prepared'; reqId: number; info: PreparedInfo; reference: Uint8ClampedArray }
  | {
      type: 'rendered';
      reqId: number;
      width: number;
      height: number;
      rgba: Uint8ClampedArray;
      elapsed: number;
      /** 手動調整などで切り抜き範囲が変わったとき、または色の傾向が変わって比較用の画像を作り直したときだけ入る */
      layout?: { info: PreparedInfo; reference: Uint8ClampedArray };
    }
  | { type: 'loupe'; reqId: number; size: number; rgba: Uint8ClampedArray }
  | { type: 'exported'; reqId: number; blob: Blob; width: number; height: number; elapsed: number }
  | { type: 'error'; reqId: number; message: string };
