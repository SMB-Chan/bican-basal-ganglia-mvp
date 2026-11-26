# データ出典と利用条件

確認日 2026-11-26（日本時間）。

## 理研発表と原著

- 理研の発表では、ヒト・マカク・マーモセット・マウスを対象とするBICANの大脳基底核アトラスが紹介されている。理研チームは主にマカクとマーモセットのMRI解析を担当した。 [理研発表](https://www.riken.jp/pr/news/2026/20261126_1/index.html) · [BICANのデータ案内](https://brain-bican.org/science/basal-ganglia)
- Dingらの原著は、HOMBA用語体系とヒト・マカク・マーモセットの座標参照アトラスを記載する。今回の数値入力は、この原著が案内するHMBA Adult Rhesus Macaque Brain Atlas（RRID:SCR_029053）。Mac25Rhesusは成体アカゲザル25個体の平均MRIテンプレート。モデルに使ったのは160 µm格子の領域注釈。 [Cell原著](https://doi.org/10.1016/j.cell.2026.10.033) · [原著本文検索表示](https://www.sciencedirect.com/science/article/pii/S0092867426012833) · [配布と説明](https://alleninstitute.github.io/CCF-MAP/descriptions/rhesus_macaque_ccf.html)
- Johansen・Fuらの原著は霊長類の細胞型の分子分類を、Hewittらの原著は空間トランスクリプトームの組織内分布を記載する。Liuらの原著はマカク線条体の細胞型別の形態と電気生理を扱う。原著の書誌・要旨と、取得可能な本文検索表示・配布文書を確認した。Cellの直接の全文ページはアクセス制限があり、全文閲読は行えていない。数値計算に使った実測由来入力は上記MRI注釈のみ。 [Johansen・Fuら](https://pubmed.ncbi.nlm.nih.gov/43020019/) · [Hewittら](https://pubmed.ncbi.nlm.nih.gov/43020020/) · [Liuら](https://pubmed.ncbi.nlm.nih.gov/43020028/)

## 同梱したデータ

帰属表記は **Allen Institute for Brain Science / HMBA-BICAN, HMBA Adult Rhesus Macaque Brain Atlas, 2025, RRID:SCR_029053, Ding et al. (2026)**。配布物の `data_description.json` で、アトラス・注釈セット・用語表のライセンス表示がそれぞれ `CC-BY-4.0` であることを確認した。CC BY 4.0 は出典・ライセンス・変更の表示を条件に再利用や改変を認める。配布元ファイルはバイト単位で同梱し、名前を一部短縮した。モデルはHOMBA子領域をまとめ、RAS xが画像の正中面近似より大きい側から一様にサンプリングし、合成の細胞種とチャネルを付けた。 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Allen Institute利用条件](https://alleninstitute.org/legal/terms-of-use)

| 同梱名 | 使用目的 | 配布元 |
|---|---|---|
| `rhesus_annotations_160.nii.gz` | 整数ラベル付き3D MRI領域注釈、RAS mm座標 | [公式注釈](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-rhesusmacaque-homba-annotation/2025/annotations_compressed_160.nii.gz) |
| `terminology.csv` | HOMBA識別子と子領域のラベル値 | [公式用語表](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/terminologies/hmba-mammalian-homba-terminology/2025/terminology.csv) |
| `volumes.csv` | 全体積のボクセル数を照合 | [公式容積表](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-rhesusmacaque-homba-annotation/2025/parcellation_volumes.csv) |
| `atlas_manifest.json` | 版と関連ファイルの照合 | [公式マニフェスト](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/atlases/hmba-adult-rhesusmacaque-homba-atlas/2025/manifest.json) |
| `atlas_description.json`、`annotation_description.json`、`terminology_description.json` | 由来とCC BY 4.0の表示 | [アトラス](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/atlases/hmba-adult-rhesusmacaque-homba-atlas/2025/data_description.json) · [注釈セット](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/annotation-sets/hmba-adult-rhesusmacaque-homba-annotation/2025/data_description.json) · [用語表](https://allen-atlas-assets.s3.us-west-2.amazonaws.com/terminologies/hmba-mammalian-homba-terminology/2025/data_description.json) |

入力3点のSHA-256と取得URLは `atlas.py` と `results/run_manifest.json` に固定した。再取得するには `python atlas.py --fetch --data fresh_atlas`。同コマンドはチェックサムを照合して、各領域の全体積カウント、採用した半分のカウント、重心、NIfTIアフィンを表示する。

## 確認したその他の公開データ

ABC Atlasはアカウント不要の公開AWSバケット `allen-brain-cell-atlas` で配布している。以下の個別データセットはドキュメントに **CC BY-NC 4.0** と明記されており、帰属と非営利条件がある。今回のファイル同梱と解析の範囲は上表のMRIアトラスに限る。 [ABC Atlasの取得案内](https://alleninstitute.github.io/abc_atlas_access/notebooks/getting_started.html) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

| 公開データ | 固定版・取得の入口 | 将来使える測定 |
|---|---|---|
| HMBA-BG taxonomy | `s3://allen-brain-cell-atlas/metadata/HMBA-BG-taxonomy/20261031/`。 [データセット](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-BG-taxonomy.html) | 細胞ラベルとクラスタ階層 |
| マカクMERSCOPE | `s3://allen-brain-cell-atlas/metadata/HMBA-BG-MERSCOPE-Macaque/20261101/`。公開ノートブックは標本 `QM23.50.001` の `cell_metadata` と `cell_anatomical_annotations` を `cell_label` で対応付ける。 [データセット](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-Spatial-BG.html) · [取得ノートブック](https://alleninstitute.github.io/abc_atlas_access/notebooks/hmba_bg_spatial_slabs_and_taxonomy.html) | 部位別の細胞型やサンプリング位置 |
| マカクPatch-seq | `s3://allen-brain-cell-atlas/metadata/HMBA-Macaque-PatchSeq/20260228/`。 [データセット](https://alleninstitute.github.io/abc_atlas_access/descriptions/HMBA-Macaque-PatchSeq-BG.html) | 形態・膜特性の細胞型別要約 |

これらを別途取得する場合、`aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/metadata/HMBA-Macaque-PatchSeq/20260228/` のように公開プレフィックスを一覧し、該当ファイルを `aws s3 cp --no-sign-request <S3のファイルURL> <保存先>` で取得できる。ABC Atlas公式ノートブックの `AbcProjectCache` も利用できる。空間情報を組み込む場合は、切片ごとの座標系、同一標本対応、QCおよび個体差を確認してから用いる。解剖学的入力が手元になければ、`python run.py --mode synthetic --out synthetic_results` で完全合成座標版を実行できる。

## 配線・課題に用いた文献

- 線条体からGPe・GPiへの投射と軸索側枝は霊長類の単一軸索トレーシングに基づく。D1→GPi、D2→GPeの優先度は作業仮説として採用し、D1→GPeとD2→GPiにも正の重みを置いた。霊長類での細胞型別の数値的な接続オッズは測定値として扱っていない。 [Lévesque & Parent (2005)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1187973/) · [Gerfenら (1990)、齧歯類](https://pubmed.ncbi.nlm.nih.gov/2147780/)
- GPe→STN/GPi/線条体、STN→GPe/GPiは霊長類のトレーシングで報告された方向を採用した。GPe内および線条体内の接続は細胞種を粗くまとめた局所回路の仮定。 [Satoら、GPe](https://pubmed.ncbi.nlm.nih.gov/10660885/) · [Satoら、STN](https://pubmed.ncbi.nlm.nih.gov/10888744/) · [Shinkら、GPe–STN](https://pubmed.ncbi.nlm.nih.gov/8783253/)
- 二択課題では、GPi発火の減少で下流の脱抑制を近似する既存の計算モデルを参考にした。チャネルの割当・同チャネル選好・入力強度・判定窓は本MVPの設計。 [Gurneyら (2001)](https://pubmed.ncbi.nlm.nih.gov/11417052/) · [Girardらの霊長類スパイキングモデル](https://pmc.ncbi.nlm.nih.gov/articles/PMC8246891/)
