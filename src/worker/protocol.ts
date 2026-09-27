// メインスレッド ⇔ ワーカー間のメッセージ定義（「AEB 合成」の合成ワーカーと「Leica M10 の色」のワーカー）

import type { Adjustments } from '../core/adjust';
import type { ExposureInfo } from '../core/exif';
import type { FusionWeights } from '../core/fusion';
import type { ToneParams } from '../core/hdr';
import type { LearnedParams } from '../core/learned';
import type { LookParams, LookSource } from '../core/look';

/** learned: 学習済みモデル（おまかせ）、fusion: 露出フュージョン（ナチュラル）、hdr: トーンマッピング */
export type Mode = 'learned' | 'fusion' | 'hdr';

export interface RenderParams {
  mode: Mode;
  /** 効果の強さ 0..1。0 で基準フレームそのまま、1 で合成結果そのまま */
  amount: number;
  fusion: FusionWeights;
  tone: ToneParams;
  learned: LearnedParams;
  adjust: Adjustments;
}

/** RAW の色の情報（撮影時のホワイトバランスと色行列）。色温度の推定に使う */
export interface RawColor {
  /** カメラのニュートラル（ホワイトバランスの倍率の逆数、G = 1） */
  neutral: [number, number, number];
  /** XYZ → カメラ RGB（D65、行優先 9 要素） */
  camXyz: number[];
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
}

/** 「AEB 合成」の結果（フル解像度）を「Leica M10 の色」に渡すときの中身 */
export interface MergedImage {
  width: number;
  height: number;
  /** 表示用 16bit RGB（効果の強さまで掛けたもの。仕上げ調整の前） */
  data: Uint16Array;
  /** 基準の写真の撮影情報（書き出す EXIF に使う） */
  exif: ExposureInfo;
  /** 基準の写真の形式・機種・ホワイトバランス（Leica M10 の色の変換を決める） */
  look: LookSource;
  /** LeicaLook.toLinear の jpegBase（おまかせの結果はカメラ内 JPEG 並みのトーンとして戻す） */
  jpegBase: number;
}

export type ToWorker =
  | { type: 'addRaw'; id: string; name: string; width: number; height: number; data: Uint16Array; exif: ExposureInfo; color?: RawColor }
  | { type: 'addFile'; id: string; name: string; file: File }
  | { type: 'remove'; id: string }
  | { type: 'prepare'; reqId: number; align: boolean; previewSide: number }
  | { type: 'render'; reqId: number; params: RenderParams }
  | { type: 'setManual'; id: string; manual: ManualAdjust }
  | { type: 'loupe'; reqId: number; id: string; cx: number; cy: number; size: number; mode: LoupeMode }
  | { type: 'export'; reqId: number; params: RenderParams; options: ExportOptions }
  /** フル解像度の合成結果を作る（「Leica M10 の色」に渡す） */
  | { type: 'merge'; reqId: number; params: RenderParams };

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
      /** 手動調整などで切り抜き範囲が変わったときだけ入る */
      layout?: { info: PreparedInfo; reference: Uint8ClampedArray };
    }
  | { type: 'loupe'; reqId: number; size: number; rgba: Uint8ClampedArray }
  | { type: 'exported'; reqId: number; blob: Blob; width: number; height: number; elapsed: number }
  | { type: 'merged'; reqId: number; image: MergedImage }
  | { type: 'error'; reqId: number; message: string };

// ---------------------------------------------------------------------------
// 「Leica M10 の色」のワーカー

export interface LeicaParams {
  look: LookParams;
  /** 仕上げ調整（Leica M10 の色の後に掛ける） */
  adjust: Adjustments;
}

/** 開いた写真（または AEB の合成結果）の情報 */
export interface LeicaSourceInfo {
  width: number;
  height: number;
  previewWidth: number;
  previewHeight: number;
  exif: ExposureInfo;
  /** JPEG などの写真として変換するか（Canon のピクチャースタイル「スタンダード」を打ち消してから変換する） */
  jpeg: boolean;
  /** 機種の分光感度データでセンサーの違いを変換したか */
  cameraMatched: boolean;
  /** 撮影時の色温度の推定値 [K] */
  cct?: number;
}

export type ToLeicaWorker =
  | {
      type: 'openRaw';
      reqId: number;
      width: number;
      height: number;
      data: Uint16Array;
      exif: ExposureInfo;
      color?: RawColor;
      previewSide: number;
    }
  | { type: 'openFile'; reqId: number; file: File; previewSide: number }
  | { type: 'openMerged'; reqId: number; image: MergedImage; previewSide: number }
  /** 開いている写真を閉じてメモリを解放する */
  | { type: 'close' }
  | { type: 'render'; reqId: number; params: LeicaParams }
  | { type: 'export'; reqId: number; params: LeicaParams; options: ExportOptions };

export type FromLeicaWorker =
  | { type: 'progress'; reqId: number; label: string; fraction: number }
  | { type: 'opened'; reqId: number; info: LeicaSourceInfo }
  | {
      type: 'rendered';
      reqId: number;
      width: number;
      height: number;
      /** Leica M10 の色を掛けたもの */
      rgba: Uint8ClampedArray;
      /** 比較用: Leica M10 の色を掛ける前（仕上げ調整は同じものを掛ける） */
      before: Uint8ClampedArray;
      elapsed: number;
    }
  | { type: 'exported'; reqId: number; blob: Blob; width: number; height: number; elapsed: number }
  | { type: 'error'; reqId: number; message: string };
