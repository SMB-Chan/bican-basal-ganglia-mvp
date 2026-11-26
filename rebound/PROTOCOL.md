# 段階4A　STN脱抑制後リバウンド（Cav3 / $I_T$）と淡蒼球―視床下核ルーティングの要因比較手順

段階3（[`cortical/RESULTS.md`](../cortical/RESULTS.md)）の盲検評価で明らかになった課題――受動LIFのSTN＋`GPe→GPi` 偏重配線では、M1刺激後のGPi後期興奮が `M1→D2→GPe⊣GPi` の直接脱抑制で生じてしまい、Nambuら（2000）のSTN遮断で消失しない問題――を、**正常状態の生理指標のみ**を用いて検証・解決し、段階4B（[`parkinsonian/`](../parkinsonian/)）へ渡す単一の凍結モデルを選ぶ。

## 1. 要因設計（2 × 2 × 3 ＝ 12条件、12シード 1009〜1069）

すべて [`biophysics/`](../biophysics/) で確立した二時間スケール膜ゆらぎ（白色雑音 `2.2×` ＋ `τ = 160 ms, 9 mV` OUゆらぎ）の上で、較正用シード `800` のみを用いて各群の定常発火率を既定モデルに合わせた後、互いに素な12シード（1009〜1069）で評価する。

1. **淡蒼球―視床下核ルーティング（2水準）**
   - `gpe_gpi_heavy`：段階2の `indirect_balanced`（`GPe→GPi` 優位、`STN: {GPe: 4.0, GPi: 1.0}`）。
   - `stn_loop_dominant`：霊長類解剖に沿い `GPe→STN` 投射を主軸としつつ `GPe→GPi` フィードフォワード抑制と `STN→GPe/GPi` 双方向出力を保つ配線（`GPe: {STN: 9.0, GPi: 1.5}`, `STN: {GPe: 2.6, GPi: 1.8}`）。
2. **シナプス減衰時定数（2水準）**
   - `delta`：瞬時電位ジャンプ（`exc_tau_ms = 0`, `inh_tau_ms = 0`）。
   - `physiological_37c`：体温（37°C）のAMPA / $\text{GABA}_A$ 受容体時定数（`exc_tau_ms = 2.0 ms`, `inh_tau_ms = 4.0 ms`）。
3. **STN低閾値 $\text{Ca}^{2+}$ リバウンド強度（Integrate-and-Fire-or-Burst, 3水準）**
   - Smith, Cox, Sherman & Rinzel（2000）およびSTNのCav3チャネル特性（Bevan & Wilson, 1999; Hallworth et al., 2003）に基づき、リセット電位より深い過分極時（`v < -68.5 mV`）にのみ不活性化変数 $h$ が回復（$\tau_h^+ = 25\text{ ms}$）し、抑制解除で `v >= -68.0 mV` に戻った直後から低閾値 $\text{Ca}^{2+}$ 脱分極を開放して速やかに不活性化する（$\tau_h^- = 20\text{ ms}$）。
   - `passive`（`0 mV`）、`moderate_ifb`（`75 mV`）、`strong_ifb`（`120 mV`）の3水準を比較し、正常状態ベンチマーク間のトレードオフ（過大リバウンドによるKita後期抑制の相殺）も同時に検証する。

## 2. 正常状態のみの事前選択規則（病態データ非参照）

以下の4系統の**正常状態**指標をすべて同時に満たす条件を `joint_normal_pass = 1` とし、12シードでの合格率が最大の条件を [`rebound/results/selection.json`](results/selection.json) に自動凍結する（同率時は `tonic_cv_mae + 0.1 * tonic_rate_mae_hz + 0.05 * triphasic_latency_mae_ms` が最小のものを選ぶ）：

1. **孤立STN細胞の過分極後リバウンド特性**（Bevan & Wilson, 1999）：20 msの過分極入力解除後（222〜250 ms）にシャム比 `+5.0 Hz` 以上のリバウンド発火を示す。
2. **覚醒マカク定常発火率**（Gatev & Wichmann, 2009）：3核平均絶対誤差 `<= 5.0 Hz`。
3. **STN単発刺激と `STN→GPe` 遮断**（Kita et al., 2005）：4符号すべて一致、かつ遮断後の後期応答 `|blocked_late_hz| <= 0.50 Hz`。
4. **M1単発刺激の三相応答と全因果遮断テスト**（Iwamuro et al., 2017; Nambu et al., 2000; Tachibana et al., 2008）：三相符号一致、非シナプス性アーチファクト50%以下、`M1→STRd` 単独で早期興奮なし、GPi局所GABA遮断で抑制消失、**かつSTN遮断で早期興奮・後期興奮の両方が有意に減弱**（`all_causal_checks == 1`）。

## 3. 再現手順

```powershell
python -m rebound.run --out rebound_reproduced
python -m rebound.analyze rebound_reproduced
```
