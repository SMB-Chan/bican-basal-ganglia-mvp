# シナプス減衰・二時間スケールゆらぎ・間接路結合の要因比較

[細胞動態の追試](../dynamics/RESULTS.md) では、1 ms刻みの白色雑音を10倍にするとLIFの位相リセット谷は消えたが、閾値下の膜電位標準偏差が約31 mVに達してシナプス応答と刺激動員がともに埋没した。また既定設定（×1）でも15〜25 msにはGPe経由の抑制（−3.26 Hz）が到達していたが、瞬間電位ジャンプ（時定数0 ms）による短命さと25〜35 msのリバウンド興奮に打ち消されていた。

ここでは膜特性F（Patch-seq膜時定数）を基礎とし、**発火のゆらぎ（3水準）× シナプス減衰時定数（2水準）× 間接路 `STN→GPe→GPi` 結合（2水準）の 3×2×2＝12条件**、および刺激動員数を一致させた単一群分解対照を、新しい12シード（809〜877）で評価する。設定と選択規則は [`protocol.json`](protocol.json) に固定し、ここで選ばれた1条件のみを凍結して未使用の [段階3 皮質刺激ベンチマーク](../cortical/PROTOCOL.md) へ渡す。

## 1. 比較する3つの要因

1. **細胞ゆらぎ（`fluctuation_regimes`）**
   - `pacemaker`：既定の1 ms白色雑音のみ（倍率1、遅いOUゆらぎ0 mV）。
   - `moderate_ou`：速い雑音を2.2倍（定常 $\sigma_{V,\text{fast}} \approx 6.7\text{ mV}$、位相ロックを崩しつつ閾値幅18 mVより十分小さい範囲）にとどめ、休止・バーストによる緩やかな発火率変動を近似する時定数160 msのOrnstein–Uhlenbeck（OU）電位ゆらぎ（SD 9.0 mV）をGPe・STN・GPiに加える。
   - `high_cv_ou`：速い雑音3.0倍に加え、孤立細胞でGatev & Wichmannの参照CV（GPe 1.83、STN 1.23、GPi 1.07）に届くまで遅いOUゆらぎのSDを二分法で拡大する（GPeで約32 mV）。対称ガウスゆらぎだけでGPeの高いCVを無理に合わせると相動性応答がどう変化するかを検証する。
2. **シナプス時定数（`synapse_modes`）**
   - `delta`：既定の1 ms瞬間電位ジャンプ（時定数0 ms）。
   - `exponential`：時間積分電荷 $(1 - e^{-\Delta t/\tau_c})$ を保存した指数減衰シナプス電流（興奮性 $\tau_{\text{exc}} = 3\text{ ms}$、抑制性 $\tau_{\text{inh}} = 14\text{ ms}$）。
3. **間接路バランス（`pathway_modes`）**
   - `baseline`：既定の投射オッズ（`STN: {GPe: 2, GPi: 2}`、MRI上の距離差によりSTN辺の約30%のみがGPeへ接続）と `GPe jump_mv = 0.65 mV`。
   - `indirect_balanced`：マカク単一軸索トレーシング（Satoら, 2000）および段階1の `toward_GPe` 仮説に合わせてMRI距離バイアスを補正し（`STN: {GPe: 4, GPi: 1}`、`GPe→GPi: 2.4`）、GPe抑制性ジャンプを `1.20 mV` とする。

## 2. 定常発火率の一致較正と刺激動員一致の単一群分解

- **定常発火率の分離較正（テストシード非参照）**：経路強度やゆらぎを変えた際に定常発火率のドリフトが刺激応答と交絡するのを防ぐため、テスト用12シードとは独立な較正シード `800` の定常活動（無刺激2秒）だけを使い、GPe・STN・GPiの定常発火率が基準ネットワーク（`pacemaker + delta + baseline`）の内部発火率と0.15 Hz以内で一致するよう各群のバイアス電位を決定する。較正時にKitaらの刺激・遮断データは一切参照しない。
- **刺激動員数をそろえた単一群分解**：`dynamics/` ではSTNのみを不規則化した際に12 mV刺激に対するSTN直後（`0–2 ms`）の誘発発火が約1/4へ減っていた。そこで `baseline + delta` 上でSTNのみを `moderate_ou` に変える条件について、刺激電圧12 mV据え置き（`stn_only_unmatched`）と、較正シード `800` でSTN直後の誘発発火率が `pacemaker` と一致するよう刺激ジャンプを再較正した条件（`stn_only_matched`）、およびGPiのみ変更（`gpi_only`）の3対照を比較する。

## 3. 段階3へ渡すモデルの事前選択規則

12要因条件のうち、12テストシード平均で厳格適合率 `strict_candidate`（定常発火率誤差5 Hz以下、Kitaらの4符号全一致、かつ遮断後の後期応答 `|blocked_late_hz| <= 0.5 Hz`）が最大の条件を選び、同率なら `tonic_cv_mae + 0.1 × tonic_rate_mae_hz` が最小のものを選んで `results/selection.json` に凍結する。

## 4. 再現手順

```powershell
python -m biophysics.run --out biophysics_reproduced
python -m biophysics.analyze biophysics_reproduced
```
