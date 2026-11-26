# 遮断後に残る抑制は細胞動態から生じるか

段階1では、STN→GPeを遮断しても後期のGPi抑制が残り、生体の遮断結果を満たす配線が0件だった。ここでは、その残りが配線ではなくLIF細胞の発火の規則性から生じるという仮説を、配線を固定したまま調べる。設定と選択規則は、ネットワークの結果を見る前に [`protocol.json`](protocol.json) に固定した。孤立細胞の予備確認（GPiのみ、1シード相当）はその前に行った。

## 1. 孤立細胞

GPe・STN・GPiの各群について、シナプスなしの細胞3,000個を1,200 ms走らせる。雑音の振幅を既定の1、2、3、4、5、6、8、10倍にし、それぞれでバイアスを二分法で再設定して、孤立時の平均発火率を既定設定（倍率1）の値に一致させる。目標値は外部記録ではなく、モデル自身の既定設定の発火率。全条件で同じ雑音の乱数列を使う。1,000 ms時点で4 mVの電位ジャンプを1回与え、同じ雑音の無刺激対照を差し引いて0〜69 msの応答を得る。指標は発火間隔のCV、直後0〜2 msの応答、3〜53 msの平均、3〜27 msの5 ms区間ごとの最小値（谷）。

## 2. ネットワークでの水準比較

新しい12シード（503〜587）で、既知投射マスクと経路別オッズを使う配線（前回のE、同チャネル選好なし）を生成し、膜特性E（仮定値）とF（Patch-seqの膜時定数）を適用する。配線は全水準で共通。GPe・STN・GPiに1の水準を同時に入れる8条件に加え、倍率5の水準をGPi・STN・GPeの一群だけに入れる分解条件3つを評価する。定常活動と、Kitaらに基づくSTN単発刺激（遮断あり・なし）の手順・時間窓は [前回のプロトコル](../validation/PROTOCOL.md) と同じ。刺激時にはGPiに加えSTNの応答も記録する。

**選択規則。** 膜特性Eで、GPe・STN・GPiの定常発火率の平均絶対誤差（Gatev & Wichmannの3個体平均との差）が5 Hz以下の水準のうち、CVの平均絶対誤差が最小のものを選ぶ。選択には定常の発火率とCVの集計だけを使い、刺激応答と遮断の結果は読まない（`select_regime` とそのテスト）。CVの参照値に合わせて選ぶため、この段階でGatev & WichmannのCVは評価用ではなく設定用データになる。

## 3. 段階1の配線群の再評価

選んだ水準で、段階1と同じ192グラフ（4配置×16乱数×3投射仮説）を膜特性E・Fの384条件で再評価する。グラフ・乱数・判定は段階1と同じで、変えるのはGPe・STN・GPiのバイアスと雑音の倍率だけ。出力は `results/ensemble/`。[構造とノイズの解析](../identifiability/structure.py) も同じ出力に適用する。

## 解釈の範囲

雑音を増やしてバイアスを下げる操作は、平均電位で駆動される発火から、ゆらぎで駆動される発火への移行を単一区画のLIFで近似したもの。生体のGPe・STN・GPiの自発発火やバースト・休止を生む電流は表していない。探索した範囲では最大の倍率10でもネットワークのCVは1.06〜1.18で、Gatev & Wichmannの参照値（GPe 1.83、STN 1.23、GPi 1.07）のうちGPeには届かない。区間は12シードの再抽出によるもので、個体間の変動は含まない。

## 再現手順

```powershell
python -m unittest -v
python -m dynamics.run --out dynamics_reproduced
python -m dynamics.analyze dynamics_reproduced
python -m identifiability.analyze dynamics_reproduced/ensemble
python -m identifiability.structure dynamics_reproduced/ensemble
```

`--seeds 503 --skip-ensemble` で短時間の確認ができる。外部データの取得は不要。`isolated.csv` と `isolated_curves.csv` は孤立細胞、`network.csv`・`network_tonic.csv`・`network_pulse_curves.csv` はネットワーク、`network_summary.csv` はシード集計と95%区間、`selection.json` は選ばれた水準と規則、`manifest.json` は設定と出力のハッシュ。
