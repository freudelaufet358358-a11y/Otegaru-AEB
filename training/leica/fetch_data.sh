#!/usr/bin/env bash
# Leica M10 の色の調査・当てはめに使う公開データを data/ に取ってくる（合計 約 70MB）。
# どれも GitHub から取れるものだけを使う（raw.pixls.us などが使えない環境でも動くように）。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
cd data

# 必要なファイルだけを取り出す浅いクローン
sparse() { # url dir paths...
  local url=$1 dir=$2; shift 2
  if [ ! -d "$dir/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git -c gc.auto=0 clone -q --depth 1 --filter=blob:none --no-checkout "$url" "$dir"
  fi
  git -C "$dir" -c gc.auto=0 sparse-checkout set --no-cone "$@"
  git -C "$dir" -c gc.auto=0 checkout -q
}

# 1) 推定分光感度（1,000 機種以上）と Adobe の色行列
#    Solomatov & Akkaynak, "Spectral Sensitivity Estimation Without a Camera", ICCP 2023
sparse https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation ssf-estimation '/data/predictions/' '/data/color-matrices/' '/data/ground-truths/'

# 2) 肌の分光反射率（ISSA Skin Color Database、15,256 件）
sparse https://github.com/butcherg/ssf-data ssf-data '/Reference_Spectra/'

# 3) Leica M10 の DNG の先頭 100KB。Leica がファイルに埋め込んだカメラプロファイル（色行列など）が読める
sparse https://github.com/SoftCreatR/mime-detector-fixtures mime-fixtures '/fixture-Leica-M10.dng'

# 4) ACR（Lightroom）の既定トーンカーブ（RawTherapee の実装に載っている表）
sparse https://github.com/Beep6581/RawTherapee rawtherapee '/rtengine/dcp.cc'

# 5) RAW + カメラ内 JPEG の実写サンプル（raw.pixls.us 由来、CC0）
BASE=https://github.com/endosome/elodie-test-assets/releases/download/v2
for f in raw-leica-m10-r.dng raw-canon-eos-r6.cr3; do
  [ -f "$f" ] || curl -fsSL -o "$f" "$BASE/$f"
done
sha256sum -c - <<'SUMS'
ba35d25d521c7ddbc4feca69c8cdbe071ac65dcd6cf2c93374391bf6f3662b8d  raw-leica-m10-r.dng
74abb0a113d075ad9887a058082f40dd2a938c4813a08474d82356f11a027778  raw-canon-eos-r6.cr3
SUMS
echo "done: $(pwd)"
