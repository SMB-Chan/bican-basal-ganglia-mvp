# 段階4B　パーキンソン病（MPTPマカク）病態振動・皮質応答とSTN遮断回復の事前登録手順

段階4A（[`rebound/`](../rebound/)）のモデル選択および病態シミュレーション実行の**前**に、完全に独立な外部ベンチマークとして覚醒MPTPパーキンソン病マカクの2系統の電気生理実験を固定する。モデルの内部パラメータ・結合重み・刺激波形は一切再調整せず、凍結済みの3モデル（既定LIF `baseline_lif`、段階2凍結モデル `selected_biophysics`、段階4A凍結モデル `selected_rebound`）を新しい12シード（1103〜1193）で1回だけ盲検評価する。

## 1. ドーパミン枯渇摂動の定義（非線条体の固有パラメータ不変）

黒質線条体および淡蒼球視床下核路におけるドーパミン欠乏の既知の生理作用のみを与える（[`protocol.json`](protocol.json)）。GPe・STN・GPi・SInの静止バイアス・膜時定数・ノイズ・リバウンド特性は正常状態の凍結値から一切変更しない。

1. **直接路D1-SPNの興奮性低下**（$D_1$ 受容体促通の喪失；Mallet et al., 2006）：静止バイアス `-4.0 mV`、皮質線条体入力ゲイン `0.5×`。
2. **間接路D2-SPNの脱抑制**（$D_2$ 受容体抑制の喪失；Mallet et al., 2006）：静止バイアス `+5.5 mV`、皮質線条体入力ゲイン `1.3×`。
3. **`GPe→STN` 終末のシナプス前 $D_2$ 抑制喪失**（Shen & Johnson, 2000, 2005; Baufreton & Bevan, 2008; Nambu & Tachibana, 2014）：`GPe→STN` シナプス重み `1.5×`。線条体変化のみの寄与を分離する対照条件 `pd_striatal_only`（`GPe→STN` 倍率 `1.0×`）も同時に記録する。

## 2. 外部生体参照と6つの事前登録テスト

### A. 自発発火率・8〜15 Hz（霊長類 low-$\beta$）病態振動と薬理遮断（Tachibana et al., 2011; Nambu & Tachibana, 2014）

6.5秒間（最初の500 msを捨てた6,000 ms）の自発活動を記録し、1,000 ms Hann窓・50%重複（周波数分解能1.0 Hz、11区間）のWelch法で集団発火率と単一細胞のパワースペクトル密度（PSD）を算出する。発火率変化による見かけのパワー増減を排除するため、8〜15 Hz平均PSDを55〜150 Hzポアソン漸近帯域の平均PSDで割った無次元**振動指数**（非振動ポアソン過程では発火率によらず期待値 `1.0`）を主指標とし、8〜15 Hz積分パワー（$\text{Hz}^2$）および3〜50 Hz帯域比を併記する。

1. **PD発火率シフト（`pd_rate_shift_pass`）**：`normal_intact` に対し `pd_intact` でGPe発火率が `1.0 Hz` 以上低下し、かつSTN発火率が `1.0 Hz` 以上上昇する（生体参照：GPe `65.2 → 41.2 Hz`、STN `19.8 → 27.6 Hz`；GPi平均発火率は `67.0 → 63.1 Hz` と有意差がないため、GPi変化量も直接報告する）。
2. **PD 8〜15 Hz病態振動の出現（`pd_beta_emergence_pass`）**：STNとGPiの両方で、`pd_intact` の8〜15 Hz集団振動指数が `1.5` を超え、かつ `normal_intact` の `1.25` 倍以上に増加する。
3. **PD状態でのSTN不活化テスト（`tachibana_stn_inactivation_pass`；Tachibana et al., 2011 図1）**：`pd_stn_blocked`（STN出力全遮断）により、GPiの8〜15 Hzパワーおよび振動指数が低下し、かつGPi発火率が `1.0 Hz` 以上低下する。
4. **PD状態での `GPe→STN` 遮断テスト（`tachibana_gpe_stn_block_pass`；Tachibana et al., 2011 図3）**：`pd_gpe_to_stn_blocked` により、STNの8〜15 Hzパワーおよび振動指数が低下し、かつSTN発火率が `1.0 Hz` 以上上昇する。

### B. PD状態でのM1単発刺激応答の変容とSTN遮断による直接路抑制の回復（Chiken, Takada & Nambu, 2021）

[`cortical/protocol.json`](../cortical/protocol.json) と同一のM1単発刺激・解析窓（早期 `6〜16 ms`、抑制 `16〜28 ms`、後期 `28〜50 ms`、許容幅 `0.25 Hz`）をそのまま用いる。

5. **PD状態での皮質誘発応答の変容（`chiken_pd_cortical_shift_pass`）**：`normal_intact` と比べ `pd_intact` では、GPi抑制窓（`16〜28 ms`）の抑制が減弱（`+0.25 Hz` 以上浅くなる）し、かつGPe後期興奮（`28〜50 ms`）が増強（`+0.25 Hz` 以上増加）する。
6. **PD状態でのSTN遮断によるGPi抑制の顕在化（`chiken_stn_block_rescue_pass`）**：`pd_stn_blocked` により、消失・減弱していたGPi抑制窓応答が負の抑制（`< -0.25 Hz`）として回復・深化（`pd_intact` より `0.25 Hz` 以上負方向へ変化）し、同時にGPiの早期興奮と後期興奮がともに `0.25 Hz` 以上減弱する。

## 3. 再現手順

```powershell
python -m parkinsonian.run --biophysics biophysics/results/selection.json --rebound rebound/results/selection.json --out parkinsonian_reproduced
python -m parkinsonian.analyze parkinsonian_reproduced
```
