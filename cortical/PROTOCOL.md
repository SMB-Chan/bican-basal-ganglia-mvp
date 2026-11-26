# 段階3　皮質（M1）単発刺激と経路遮断による盲検ホールドアウト検証

Gatev & Wichmann（定常発火率・CV）とKitaら（STN刺激・GPe遮断）は、段階1〜2および [機構改良実験](../biophysics/PROTOCOL.md) の設計・モデル選択に使用したため、以後の盲検評価には用いない。本実験では、これまでパラメータ選択や構造調整に一切使っていない覚醒マカク一次運動野（M1）刺激に対するGPi三相性応答と薬理遮断データをホールドアウト評価として1回だけ適用する。すべての入力パラメータ・時間窓・対照判定規則は、シミュレーション実行前に [`protocol.json`](protocol.json) と [設定資料](../identifiability/CORTICAL_BENCHMARK.md) に固定した。

## 1. 外部入力の事前設定（GPi出力に対する非調整）

個別の大脳皮質ニューロンは追加せず、既存の1,000細胞・19,680有向辺の基底核回路に対し、時刻300 msのM1単発刺激を模した外部興奮性シナプス入力を各シードで無作為に選んだ半数（前肢領域相当の50%）のSTNおよび背側線条体（D1・D2・SIn）細胞へ与える。

- **ハイパー直接路入力（`M1→STN`）**：太い錐体路側枝の速い伝導（マカクSTNの直行性発火潜時 約5〜6 ms）に基づき、到達遅延4 ms、指数減衰時定数3 ms、時間積分電位10 mVとする。
- **皮質線条体入力（`M1→STRd`）**：細い皮質線条体軸索の遅い伝導（マカク被殻SPNの発火潜時 約10〜12 ms）および静止時SPNが閾値下（平衡電位−55 mV、閾値−50 mV）にあることに基づき、到達遅延9 ms、指数減衰時定数4 ms、時間積分電位18 mVとする。

各シード・各条件で同じ背景雑音と初期電位を共有する無刺激対照（sham）と刺激試行を12反復ずつ走らせ、差分からGPi（およびSTN・GPe・D1・D2）の応答波形を得る。シードは過去すべての実験と重複しない12シード（907〜983）。

## 2. 評価対象モデルと7つの介入・対照条件

評価する動態設定は、(1) 既定の単一区画LIF（Patch-seq膜時定数F、既知投射E配線）と、(2) 皮質データを見る前に [機構改良実験](../biophysics/PROTOCOL.md) で定常活動・STN刺激のみから事前選択・固定されたモデル（`biophysics/results/selection.json`）の2種類のみとする。

| 条件名 | 外部入力と回路操作 | 検証する問い・出典 |
|---|---|---|
| `intact` | `M1→STN` ＋ `M1→STRd`、回路無傷 | 早期興奮（6〜16 ms）・抑制（16〜28 ms）・後期興奮（28〜50 ms）の三相と開始潜時。 [Iwamuroら 表1（MIf：10.2±3.0, 20.1±4.2, 27.8±4.4 ms）](https://onlinelibrary.wiley.com/doi/10.1111/ejn.13738) |
| `hyperdirect_gpe_blocked` | `M1→STN` のみ＋ `STN→GPe` 遮断 | **非シナプス性の細胞動態アーチファクト対照。** 抑制窓（16〜28 ms）で負の応答が出れば細胞自身の位相リセット谷を表す。この絶対値が `intact` の抑制の半分を超えた場合、抑制相を回路経路の再現とは判定しない。 |
| `hyperdirect_only` | `M1→STN` のみ、回路無傷 | `hyperdirect_gpe_blocked` との差により、`M1→STN→GPe→GPi` フィードフォワード抑制の寄与を分離する。 |
| `direct_only` | `M1→STRd` のみ（`M1→STN=0`） | 早期興奮（6〜16 ms）が生じない（+0.25 Hz以下）ことを確認する。 |
| `d1_gpi_blocked` | 本条件入力＋ `D1→GPi` 遮断 | `intact` との差から直接路（`M1→D1→GPi`）による抑制寄与を測る。 |
| `stn_blocked` | 本条件入力＋ `STN` 全出力遮断 | STN機能遮断で早期興奮と後期興奮の両方が減弱するか。 [Nambuら (2000)](https://doi.org/10.1152/jn.2000.84.1.289) |
| `gpi_gaba_blocked` | 本条件入力＋ `GPi` への全抑制辺遮断 | GPi局所GABA遮断で抑制相が消えるか（−0.25 Hz以上になるか）。 [Tachibanaら (2008)](https://pubmed.ncbi.nlm.nih.gov/18093168/) |

## 3. 採点規則と再現手順

各窓の平均発火率差の符号判定閾値は0.25 Hz。開始潜時は3 ms移動平均波形が±1.0 Hzを初めて越える時刻（早期興奮4〜20 ms、抑制12〜36 ms、後期興奮20〜60 ms）として自動抽出する。95%区間は12シードの5,000回ブートストラップで算出する。

```powershell
python -m cortical.run --biophysics biophysics/results/selection.json --out cortical_reproduced
python -m cortical.analyze cortical_reproduced
```
