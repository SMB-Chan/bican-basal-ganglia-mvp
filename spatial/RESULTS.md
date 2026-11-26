# 段階2　マカクMERSCOPEの実位置を確認する

ABC Atlasの `HMBA-BG-MERSCOPE-Macaque/20261101` を取得し、元の座標、細胞型、領域注釈、標本情報を結合した。対象は公式データのアカゲザル1個体 `QM23.50.001`。大きい4表はHTTP Rangeで先頭8 MiBずつ取得し、行の途中で終わる末尾を除いた。細胞IDで37,728行を照合し、QCと標本の切片面の一致を通過したのは36,857行。使用した入力のハッシュと実行条件は [`preflight.json`](preflight.json) に記録した。 [公式データと利用条件（CC BY-NC 4.0）](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-Spatial-BG.html) · [座標・解剖注釈の公式ノートブック](https://alleninstitute.github.io/abc_atlas_access/notebooks/hmba_bg_spatial_slabs_and_taxonomy.html) · [Hewittらの原著](https://doi.org/10.1016/j.cell.2026.10.023)

## 1切片の予備解析

切片面 `QM23.50.001.CX.44.Plane.05` で、背側線条体D1 SPN 4,277個、D2 SPN 4,039個の実測座標が得られた。領域注釈でSTR、細胞型階層が整合し、サブクラスとグループの両方のブートストラップ確率が0.8以上の細胞を選んだ。各細胞の近傍8個のうち同じD1/D2型の割合を比べた。位置と細胞数を共通に保ち、同じ切片内か、同じ切片かつ0.5 mm格子内で、細胞型ラベルだけを500回入れ替えた。

| 同じD1/D2型の近傍割合 | 割合 |
|---|---:|
| 観測された型と座標 | 50.64% |
| 切片内で型を無作為化 | 50.02% |
| 切片内の0.5 mm格子ごとに型を無作為化 | 50.53% |

同じ切片全体でラベルを入れ替えると観測値との差は0.62パーセントポイント。0.5 mm格子で各場所の型構成を保存すると差は0.11ポイントになり、観測値は並べ替えの中央95%範囲（50.06〜50.96%）に入った。これはデータファイルの一部から得た1切片の座標確認と探索的な比較。位置ごとの構成差と近接セルの細かな混在を扱うには、この二つの対照を区別する必要がある。個体差、全線条体、発火機能への効果は推定していない。

## 3Dの機能比較に残る条件

取得した実座標は **各スラブ面内のx・y（mm）**。この版のマカク公開ファイル一覧には、同じ細胞IDの `ccf_coordinates` が見当たらない。公開論文配布フォルダから標本マニフェストとCX44の変換マニフェストも取得したが、照合できたのは画像・ブロックとスラブの配置。スラブ面から個体MRI、さらに現行モデルのMac25Rhesus RAS mmに至る座標対応は確認できていない。HOMBA原著には個体MRIとテンプレートの非線形登録、および組織スラブを個体MRIへ対応付ける手順が記載されている。テンプレート内で位置と領域を照合できてから、観測座標と同じサンプリング範囲を使った一様配置・ラベルシャッフル・実ラベルの比較に進む。 [公開変換マニフェスト](https://allen-hmba-releases.s3.us-west-2.amazonaws.com/Hewitt_Turner_Cell2026/for_mike/QM23.50.001.CX44.transform_manifest.json) · [標本マニフェスト](https://allen-hmba-releases.s3.us-west-2.amazonaws.com/Hewitt_Turner_Cell2026/for_mike/QM23.50.001_manifest.json) · [HOMBA原著](https://doi.org/10.1016/j.cell.2026.10.033)

## 再現

```powershell
python -m spatial.preflight --fetch --cache merscope_probe_cache --out merscope_probe_cache/spatial_check.json
```

元の公開表は再取得する。結果に含めるのは切片の範囲、細胞数、近傍統計と入力ハッシュ。比較対象の3Dモデルは今回更新していない。
