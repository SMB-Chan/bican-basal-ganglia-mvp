# BICAN大脳基底核・疑似コネクトームMVP

アカゲザルの公開MRIアトラスを使って合成ニューロン1,000個を配置し、配線の仮定を変えた4条件のスパイキング活動を比較するPythonプロジェクト。アトラスから得たのは領域マスクで、各細胞の位置・種類・シナプス・発火はモデルが生成する。既定の実行には合成座標を用いたDの対照も含む。

- [結果と解釈](RESULTS.md)
- [原著・データ・利用条件](SOURCES.md)
- [設定](config.json)

## 独立した神経記録による機能検証

1,000細胞のまま制約を一つずつ追加する追加実験を収録した。定常発火率・発火間隔のばらつき・視床下核刺激に対する応答を、モデル設定に使わなかった霊長類の記録と比較する。公開Patch-seqによる膜時定数、MRI形状、課題チャネルの同側選好の寄与も個別に評価した。

- [検証結果](validation/RESULTS.md)
- [検証指標・データ出典・再現手順](validation/PROTOCOL.md)
- [細胞ごとの膜特性を加えた結果](heterogeneity/RESULTS.md)
- [同一細胞内の膜特性の組み合わせとドナー単位の検証手順](heterogeneity/PROTOCOL.md)
- [同じ出力を与える配線と、投射仮説を区別できる指標](identifiability/RESULTS.md)
- [配線の識別実験の設定と再現手順](identifiability/PROTOCOL.md)
- [遮断後に残る抑制の原因：細胞の発火の規則性](dynamics/RESULTS.md)
- [細胞動態の比較の設定と再現手順](dynamics/PROTOCOL.md)
- [シナプス減衰・二時間スケールゆらぎ・間接路結合の要因比較](biophysics/RESULTS.md)
- [機構改良実験の設定と再現手順](biophysics/PROTOCOL.md)
- [マカクMERSCOPE実座標の予備解析と、複数スケールの組成](spatial/RESULTS.md)
- [皮質刺激の設定用・評価用の資料](identifiability/CORTICAL_BENCHMARK.md)
- [段階3：皮質（M1）刺激と経路遮断によるホールドアウト検証](cortical/RESULTS.md)
- [段階3皮質刺激ベンチマークの設定と再現手順](cortical/PROTOCOL.md)
- [段階4A：STN脱抑制後リバウンド（Cav3）と淡蒼球―視床下核ルーティングの要因比較](rebound/RESULTS.md)
- [段階4A正常状態要因比較の設定と再現手順](rebound/PROTOCOL.md)
- [段階4B：パーキンソン病（MPTPマカク）8〜15 Hz病態振動・皮質応答とSTN遮断回復の盲検検証](parkinsonian/RESULTS.md)
- [段階4Bパーキンソン病ホールドアウト検証の事前登録手順](parkinsonian/PROTOCOL.md)
- [段階4Aの新しいシードでの確認と、段階4Bの比較の補足](confirmation/RESULTS.md)
- [開ループのGPe–STN回路でPD特異的な振動が出るかの探索](oscillation/RESULTS.md)
- [振動指数の陽性対照：12 Hzのリズムを外から与えた場合](readout/RESULTS.md)
- [外部データの使用台帳（設定・選択・評価の区別）](DATA_LEDGER.md)

## 再現手順

Python 3.12で動作を確認した。ZIPを展開し、このREADMEがあるフォルダで実行する。解析時にネット接続は不要。依存パッケージの初回導入と、同梱アトラスを省いた場合の再取得には接続が必要。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest -v
.venv\Scripts\python run.py --out reproduced
.venv\Scripts\python analyze.py reproduced
```

macOS/Linuxでは `.venv\Scripts\python` を `.venv/bin/python` に置き換える。`--out` には未使用のフォルダ名を指定する。短時間で確認するには `--seeds 11` を追加する。小さくする場合は `config.json` のコピーを作り、各群のニューロン数を正の偶数にし、Dで各発信細胞の許容標的数以上にならないよう `out_degree` を調整して `--config` で指定する。アトラスなしで試すには `python run.py --mode synthetic --out synthetic_results`。合成座標版の出力には `synthetic` と記録される。データ取得方法とライセンスの詳細は `SOURCES.md` を参照。

## 実験設計

母集団は背側線条体のD1/D2投射ニューロンと抑制性介在ニューロン（SIn）、淡蒼球外節（GPe）、視床下核（STN）、淡蒼球内節（GPi）の6群。各群で同数になるよう、個々の細胞をチャネル0/1に無作為割当する。チャネルは課題上のラベルで、組織上の位置や左右半球には対応しない。位置は右側のMRI領域マスクのボクセル中心から領域ごとに重複なしで一様抽出する。細胞型ごとの場所の偏り、細胞密度、実測の細胞型比率は導入していない。

| 条件 | 発信細胞ごとの標的抽出 |
|---|---|
| A ランダム | 許容候補から一様に抽出。A〜C共通の候補は自己接続を除く全細胞。 |
| B 距離 | Aの候補に距離カーネル `exp(-d/3 mm)` を掛ける。 |
| C 距離＋細胞種 | Bに細胞種ペア別の局所親和度を掛ける。明示したペア以外は1。 |
| D 既知投射＋MRIアトラス | Cに文献に基づく方向付き経路のマスクと経路別の仮定の重みを掛ける。一部の経路では同チャネル標的を4倍にする。配置はA〜Cと共通のMRIマスク。 |
| D合成座標 | Dのルールを、設定で定義した合成の領域別ガウス座標に適用する。 |

どの条件もGPiを出力専用の群とし、残り820個の細胞はそれぞれ24個の重複しない標的を持つ。有向接続19,680本、全有向ペアに対して約1.97%、発信可能な細胞のペアに対して約2.40%。各発信細胞からの符号と強度は共通。STN由来の興奮性接続2,880本、それ以外の抑制性接続16,800本。Aの標的選択は距離を参照せず、伝導遅延は全条件で距離を使う。辺ごとに `ceil((1 ms + d / 0.9 mm/ms) / dt) × dt` で遅延を与える。距離は直線距離で、実軸索長と伝導速度を推定した値ではない。接続生成には共通のGumbel乱数を使い、行ごとの重み付き非復元抽出を行う。

発火モデルはリーク付き積分発火（LIF）。刻み1 msで、バイアスによる平衡電位に指数緩和し、独立ガウス雑音と到着したシナプスの符号付き電位ジャンプを加える。閾値を越えるとリセットし、2刻みの不応期に入る。A〜Dで各シードの個体配置、チャネル、試行順、初期電位、雑音を共有する。D合成座標もチャネルと試行入力を共有する。細胞数、電位、時定数、雑音、伝導速度、結合の重みはモデルの設定値。

課題は二択の代用指標。200 msのベースライン後、D1/D2とSTNに300 msの入力を与え、片方の線条体チャネルだけを追加で2 mVまたは5 mV強くする。最後の200 msで平均GPi発火率が低い側を選択し、同値は0.5点とする。各強度・各側4試行、等入力4試行を加えて1グラフ20試行。等入力には正解ラベルがないため、チャネル0の選択割合を診断に用いる。発火率のベースライン窓は刺激前の最後150 ms。集計はまずシードごとに行い、不確かさは6シード間の標準偏差で示す。

## ファイルの読み方

| パス | 内容 |
|---|---|
| `atlas.py`, `atlas_data/` | 固定した実測由来の入力、配布メタデータ、チェックサム照合 |
| `model.py`, `run.py`, `config.json` | 配線、発火、課題の実行入口とパラメータ |
| `test_model.py` | 方向・符号・遅延・不応期・シード再現性・MRI領域・課題の検証 |
| `results/run_manifest.json` | 実行モード、環境、アトラス領域、設定、元結果のSHA-256 |
| `results/graphs.csv`, `edge_matrices.csv` | グラフごとの統計、群対ごとの本数と有向ペア密度 |
| `results/trials.csv`, `rates.csv` | 各試行の選択・GPi発火・各群とチャネルの発火率、無発火細胞率 |
| `results/*summary.csv`, `task_by_seed.csv`, `paired_task_differences.csv` | シード単位の集計と対応差 |
| `results/network_*_seed11.npz` | 各条件の再利用可能な疑似配線、座標、代表1試行のスパイク。`source[k] → target[k]`、遅延ms、重みmV、0始まりの細胞索引。 |
| `analyze.py`, `results/*.png` | 集計・作図入口と出力された図 |
