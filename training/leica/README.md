# 「Leica M10」の色の調査と当てはめ

アプリの仕上げの「Leica M10」は、このフォルダのスクリプトで求めたモデル（`src/models/leica-m10.json`、約 22KB）を使います。
調査の結果と考え方は [docs/leica-m10-color.md](../../docs/leica-m10-color.md) にまとめています。ここでは再現の手順だけを書きます。

## 使うデータ

どれも GitHub から取れる公開データです（`fetch_data.sh` が `data/` に取ってきます。合計 約 80MB）。

| データ | 中身 | 出典 |
| --- | --- | --- |
| 推定分光感度 | 1,000 機種以上のカメラの分光感度（色行列から推定）と、Adobe の色行列 | Solomatov & Akkaynak, *Spectral Sensitivity Estimation Without a Camera*, ICCP 2023（[COLOR-Lab-Eilat/Spectral-sensitivity-estimation](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation)） |
| 実測の分光感度 | 推定値の確からしさの確認用（Jiang et al. 2013 ほか） | 同上 |
| 肌の分光反射率 | ISSA Skin Color Database、15,256 件 | [butcherg/ssf-data](https://github.com/butcherg/ssf-data) |
| Leica M10 の DNG の先頭 100KB | Leica がファイルに埋め込んだカメラプロファイル（色行列など） | [SoftCreatR/mime-detector-fixtures](https://github.com/SoftCreatR/mime-detector-fixtures) |
| RAW + カメラ内 JPEG | Leica M10-R の DNG、Canon EOS R6 の CR3（raw.pixls.us 由来、CC0） | [endosome/elodie-test-assets](https://github.com/endosome/elodie-test-assets) のリリース |
| ACR の既定トーンカーブ | 参考（Lightroom の既定の階調との比較） | RawTherapee の `rtengine/dcp.cc` |
| 反射率・照明 | ColorChecker 24 色、CIE 2017 の 99 色（CES）、CIE 標準光源など | [colour-science](https://www.colour-science.org/) |

## 手順

```bash
cd training/leica
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

./fetch_data.sh                 # データを data/ に取ってくる
python sensor.py                # センサー + 色行列の違い（分光シミュレーション）→ out/sensor.json
python engine.py                # カメラ内 JPEG の仕上げ方（Leica / Canon）→ out/engine.json と検証画像
python export_model.py ../../src/models/leica-m10.json --fixture ../../test/fixtures/leica.json
cd ../.. && npm test            # TypeScript 実装（src/core/look.ts）が Python と一致するか確認

cd training/leica
python report.py                # ドキュメントの表 → out/report.md
python sheet.py ../../docs      # ドキュメントの比較画像
python ssf_check.py             # 推定分光感度の誤差の確認（実測値のあるカメラで）
```

## ファイル

- `common.py`: 分光シミュレーション、色行列による変換（LibRaw 方式・DNG 方式）、DNG のタグの読み取り、トーンカーブ
- `sensor.py`: Canon の RAW（アプリと同じ LibRaw・Adobe の行列）→「Leica M10 で撮り、Leica の行列で現像した色」の 3×3 行列を、昼光用と電球光用に当てはめる（Canon 19 機種 + 汎用）
- `engine.py`: RAW と同じコマのカメラ内 JPEG の位置を合わせ、`JPEG ≈ T(e·A·x)`（A: 3×3、T: R・G・B に同じトーンカーブ）を当てはめる
- `export_model.py`: アプリ用の JSON と、TypeScript の検証用の基準データを書き出す（`apply_look` は `src/core/look.ts` と 1 対 1）
- `report.py` / `sheet.py` / `ssf_check.py`: ドキュメント用の表・画像、推定分光感度の確認
