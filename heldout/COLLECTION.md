# 仮説H1〜H3を検証するデータの収集

[ドーパミン枯渇の設定の分解](../depletion/RESULTS.md)で残った3つの仮説を区別できる、まだ使っていないデータを探した。霊長類を優先したが、出力核の膜の性質を測った霊長類の研究はまれなため、齧歯類の相同部位（GPiに相当する脚内核、同じ出力核の黒質網様部）と線条体も対象にした。

| 仮説 | 内容 | 区別に必要なデータ |
|---|---|---|
| H1 | ドーパミン枯渇でGPi自身の興奮性が下がる | 枯渇後の出力核の膜の性質、入力を遮断したときの発火 |
| H2 | GPe→GPiの抑制が強まる | 枯渇後の淡蒼球→出力核のシナプスの強さ、GABA遮断の効果 |
| H3 | 段階4BのD2の設定（静止電位+5.5 mV）が強すぎる | 枯渇後のD1・D2 SPNの興奮性や発火率 |

## 方法と、目にした情報

DANDIのAPIでキーワード（6-OHDA、dopamine depletion、MPTP、parkinsonian、substantia nigra reticulata、entopeduncular、globus pallidus、subthalamic、spiny projection neurons、Parkinson）を検索し、メタデータ（題名、種、手法、説明文の冒頭、ファイル名、ユニット表の列）だけを読んだ。データの値は読んでいない。H1・H2については、DANDIに該当がなかったためウェブ検索を3回行い、要旨レベルの結果を目にした（下の表）。未使用の候補を減らさないよう、それ以上の文献検索は控えた。

## 候補

| 優先 | データ | 内容 | 仮説 | 状態と注意 |
|---|---|---|---|---|
| **1** | **DANDI:001538**（Surmeier研究室、CC-BY-4.0） | 片側6-OHDAマウスの線条体スライスで、dSPN（D1）とiSPN（D2）の電流固定・電圧固定と2光子画像。dSPNは未処置6・6-OHDA 5・L-DOPA慢性投与の休薬時8セッション、iSPNは未処置13・6-OHDA 10・休薬時14セッション（遺伝子改変なし、薬剤なし） | H3：枯渇後のD1・D2 SPN自身の興奮性の変化を、段階4Bの設定（D1 −4 mV、D2 +5.5 mV）と比べる | 値は未読。ただし同じ研究室の先行研究（Fieblinger et al. 2014、枯渇後のdSPN・iSPNの興奮性の変化）の概要は、学習知識として知っている可能性が高い。完全なブラインドではないことを明記して使う。スライスの固有の興奮性は、モデルの「静止電位の変化」（固有の性質と入力の両方を含む）の一部にしか対応しない。マウス |
| 2 | DANDI:001177（CC-BY-4.0） | 両側線条体に生理食塩水または6-OHDAを注入したマウスの線条体のユニット記録。D1-Creマウスの光遺伝学セッションを含む | H3：生体でのD1 SPNの発火率の変化 | 記録の大半がプラミペキソール投与下。薬剤投与前の区間と、光遺伝学によるD1 SPNの同定が取れるかは未確認 |
| 3 | DANDI:001777（CC-BY-4.0） | 黒質の持続的なGABA信号へのドーパミンの作用（電圧固定、マウス） | H2の一部（出力核へのGABAの作用） | 枯渇モデルを含むかは未確認 |
| — | DANDI:001180 | 6-OHDAマウスの線条体ユニット、L-DOPA投与1日目から記録 | H3 | 全個体が6-OHDAで、健常対照との比較ができないため外す |

### 結果を目にした文献（ブラインドな検証には使わない）

| 文献 | 目にした内容 | 仮説との関係 |
|---|---|---|
| Faynveitz et al. 2019（ラット、[PMC6753199](https://pmc.ncbi.nlm.nih.gov/articles/PMC6753199)） | 6-OHDA後に淡蒼球→黒質網様部のGABAシナプスが強まり、抑制性終末が増え、黒質網様部の発火率と規則性が下がった | H2を支持する方向 |
| Galvan et al.（MPTPマカク、GPiのGABA-B拮抗薬） | パーキンソン状態でGABA-B拮抗薬がGPiの発火をより大きく変えた | GPiへのGABAの作用の変化を示唆。H2と整合しうる |
| Atherton & Bevan（黒質網様部スライス） | 正常状態の自発発火は自律的 | 直接は関係しない |
| eNeuro 2021（マウスの運動性視床） | 6-OHDA後に視床ニューロンの固有の興奮性が上がった | 直接は関係しない |

**H1（出力核自身の興奮性の低下）を直接測った、結果を見ていないデータは見つからなかった。** H2には、結果を見た文献からの間接的な支持があるが、検証には数えない。

## 次の段階

候補1（DANDI:001538）でH3を検証する手順を事前登録する。比較の方法（スライスで測った興奮性の変化を、モデルの静止電位の変化にどう対応させるか）を、値を読む前に決める必要がある。具体的には、段階4Bの設定が意味する「dSPNは興奮しにくく、iSPNは興奮しやすくなる」という向きと、閾値電流（rheobase）の相対的な変化の大きさを予測として固定し、dSPN・iSPNそれぞれで6-OHDA群と未処置群を比べる。

## 出典

- [DANDI:001538](https://dandiarchive.org/dandiset/001538) · 関連論文：[bioRxiv 2025.01.02.631090](https://www.biorxiv.org/content/10.1101/2025.01.02.631090v2) · [Science Advances](https://www.science.org/doi/10.1126/sciadv.adv8224)
- [DANDI:001177](https://dandiarchive.org/dandiset/001177) · [DANDI:001777](https://dandiarchive.org/dandiset/001777) · [DANDI:001180](https://dandiarchive.org/dandiset/001180)
- 検索で目にした文献：[Faynveitz et al. 2019](https://www.frontiersin.org/articles/10.3389/fncel.2019.00417/full) · [PMC2864357](https://pmc.ncbi.nlm.nih.gov/articles/PMC2864357/figure/F5) · [PMC6725542](https://pmc.ncbi.nlm.nih.gov/articles/PMC6725542) · [eNeuro 2021](https://www.eneuro.org/content/8/1/ENEURO.0436-20.2021)
