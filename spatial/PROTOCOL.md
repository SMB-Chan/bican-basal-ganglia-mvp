# 段階2の続き　MERSCOPE全データの組成と局所配置

## 使うデータ

ABC Atlasの `HMBA-BG-MERSCOPE-Macaque/20261101`（確認日 2026-11-26、同日時点でバケット上の最新版）。前回の予備解析は4表の先頭8 MiBだけを読んだが、今回は `cell_metadata`、`slab_plane_coordinates`、`cell_anatomical_annotations`、`mmc_results` の全行と `specimen_metadata`、`donor` を取得した。取得時に公開サーバーのContent-Lengthと一致することを確かめ、SHA-256を [`results/census.json`](results/census.json) に記録した。元データは同梱しない（CC BY-NC 4.0、合計約2 GB）。 [公式データと利用条件](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-Spatial-BG.html) · [Hewittらの原著](https://doi.org/10.1016/j.cell.2026.10.023) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

4表を `cell_label` で結合し、`qc_pass` が真かつ座標表の切片面が標本表の切片面と一致する細胞を使う。座標は各スラブ面内の2次元mm。異なる組織切片の細胞が近傍として混ざらないよう、近傍は1組織切片の中だけで計算する。

## 細胞の分類

MapMyCellsの階層ラベルをそのまま使い、モデルの群に次のように対応付ける。

| 区分 | 条件 |
|---|---|
| 背側SPN | サブクラス `STR D1 SPN` / `STR D2 SPN` で、グループ名が `STRd ` で始まる |
| 腹側SPN | 同じサブクラスで、グループ名が `STRv ` で始まる |
| 偏在SPN（eSPN） | サブクラス `STR Eccentric SPN`。モデルに対応する群がないため、モデル用の比率から除く |
| 線条体介在ニューロン | クラス `CN MGE GABA` または `CN CGE GABA`（PTHLH-PVALB、TAC3、コリン作動性、SST、VIP、LAMP5など） |
| ニューロン | ネイバーフッドが `Nonneuron`・`Adjacent` 以外 |

**背側線条体の範囲。** 解剖注釈STRの細胞を0.25 mm格子に分け、背側SPNが5個以上かつ腹側SPNより多い格子を背側とした。0.5 mm格子でも再計算した。介在ニューロンの割合は、モデルの対応に合わせた「背側SPN＋介在ニューロン」に対する割合と、文献比較用の「全線条体ニューロン（背側・腹側SPN、eSPN、介在ニューロン）」に対する割合の両方を記録する。

**領域間の比。** GPe・GPi・STNは解剖注釈（STNは `STH`）を使う。各組織切片で細胞の存在する格子の面積を合計し、面積当たりのニューロン数を求める。これに同梱MRIアトラスの片側体積を掛けて相対数とする。この推定は、(1) 切片の厚さと検出率が領域間で等しい、(2) 標本に含まれる切片がその領域の平均密度を代表する、(3) 1個体の組織注釈と25個体平均のMRI体積を組み合わせてよい、という仮定に依存する。格子は0.05、0.1、0.2 mmで比較し、モデルには0.1 mmの値を使う。分割したニューロンの等価直径の中央値はGPe 11.4 µm、GPi 11.4 µm、STN 11.0 µmで、セグメンテーションでは領域差が小さかった。ただし10 µm切片内の体積から求めた直径は細胞体全体の大きさを表さないため、Abercrombie補正は行っていない。

## 局所配置の並べ替え検定

1切片に信頼度の高い背側SPN（サブクラス・グループのブートストラップ確率0.8以上、階層整合）が1,000個以上ある切片を使う。各細胞の最近傍8個のうち同じラベルの割合を、2種類のラベルで計算する。

- **D1/D2。** 前回の予備解析と同じ問い。
- **ストリオソーム/マトリックス。** 既知の区画構造で、陽性対照として使う。

座標と細胞数を固定し、ラベルを(1)切片全体で、(2)切片内の0.5 mm格子ごとに500回並べ替える。観測値から並べ替えの平均を引いた値を「超過」として切片ごとに記録する。(2)は数百µm以上の組成の偏りを保存したうえで、それより細かい混ざり方を問う。

## ネットワークへの代入

実測から得た群の大きさを、線条体の合計500細胞と線条体外の合計500細胞を保ったまま代入する。各群は正の偶数に丸める。

| 組成 | D1 | D2 | SIn | GPe | STN | GPi |
|---|---:|---:|---:|---:|---:|---:|
| 元の設定 | 236 | 236 | 28 | 200 | 120 | 180 |
| 線条体を実測 | 224 | 228 | 48 | 200 | 120 | 180 |
| GPe/STN/GPiを実測 | 236 | 236 | 28 | 310 | 80 | 110 |
| 両方を実測 | 224 | 228 | 48 | 310 | 80 | 110 |
| 線条体を文献値に（感度分析） | 216 | 214 | 70 | 200 | 120 | 180 |

文献値は、対照サルの線条体ニューロンの約14%が介在ニューロンとした免疫組織化学・立体解析による。 [Del Reyら (2022)](https://doi.org/10.1111/nan.12812)

本モデルは各発信細胞の出次数を24に固定している。このため群の大きさを変えると、受け手1細胞当たりの入力数も変わる。組成そのものの効果と入力総量の変化を分けるため、各組成を2通りで評価する。

- **組成のみ（raw）。** 群の大きさだけを変える。GPiの数が変わると発信細胞数と総辺数も変わる（GPi 110個で21,360辺、STN由来の興奮性辺1,920本）。
- **入力総量一致（drive_matched）。** 同じシードの元の組成で実現した「発信群×受け手群」ごとの受け手1細胞当たり平均入力数を基準にする。これに一致するよう、その群対の全辺の重みに同じ倍率を掛ける。元の組成では倍率はすべて1になる。

配線はvalidation段階のE（既知投射＋経路別オッズ）とF（Eに Patch-seq膜時定数）。外部評価は [validation段階](../validation/PROTOCOL.md) と同じ覚醒アカゲザルの定常発火率・CV、およびSTN単発刺激とSTN→GPe遮断。刺激はSTNの半数に与える。STNの数が変わると刺激する細胞数も変わるため、元の設定と同じ60細胞を刺激する場合も補助的に記録する。既存のどの段階とも重複しない24シードを使い、シードを再抽出する5,000回のブートストラップで対応差の95%区間を求めた。組成の推定値は外部評価の値を見ずに決めた。

## 再現

元データの取得には約2 GBの空き容量と接続が必要。解析は1分強、ネットワーク比較は4コアの計算機で約20分。

```powershell
python -m unittest -v
python -m spatial.census --fetch --cache merscope_full_cache --out spatial_reproduced
python -m spatial.composition --census spatial_reproduced/census.json --out spatial_reproduced/composition
python -m spatial.analyze spatial_reproduced
```

前回の1切片面の予備解析は `python -m spatial.preflight --fetch --cache merscope_probe_cache --out merscope_probe_cache/spatial_check.json` で再現でき、結果は [`preflight.json`](preflight.json)。

`census.json` には入力ハッシュ、領域別のサブクラス数、背側線条体の組成、格子別の密度、モデル用の群の大きさが入る。`spn_mixing_by_section.csv` は切片ごとの近傍統計、`composition/composition_metrics.csv` は各シード・組成・配線・入力条件の評価、`composition_paired.csv` は対応差。
