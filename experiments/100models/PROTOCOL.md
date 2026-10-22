# 100モデルでの尤度法とグラフ法の対照実験

## 目的と再現の範囲

Oyama, Yamagiwa, Takase, Shimodaira (ACL 2025), “Mapping 1,000+ Language Models via the Log-Likelihood Vector” の尤度座標・クリッピング・可視化を実装し、同じ100モデル・同じ300テキストでFisherグラフ法と比較する。原論文の1,018モデル・The Pile 10,000テキストの完全再現ではなく、共通コーパスにおける手法の追試である。

- 論文: https://aclanthology.org/2025.acl-long.1584/
- 著者実装: https://github.com/shimo-lab/modelmap
- 参照commit: `194c5436ed01d16381b1b903d3e2feec916a534c`
- 尤度実装: `1000models/codes/src/calculator/likelihood.py`
- コーパス: 既存実験の15分野×20テキスト。全100トークナイザに対して512 tokens以内の共通文字列prefixを固定。以前の15モデル実験から変化したテキスト数を記録。
- モデル: `configs/models_100.json`。100個の異なるrepository ID。ベース・指示調整・コード・数学モデルを含む便宜的標本であり、LLM全体の無作為標本ではない。モデルrevisionとテキストSHA256を保存。

## 比較条件

### 数値エラーからの再開

初回のFP16実行では66モデルが完了し、25モデルで非有限の尤度、9モデルで1トークンの入力を拒否するチェックの失敗が発生した。確認したPythia-70mでは重みに非有限値はなく、FP16推論でNaN、FP32推論で有限値となった。非有限値で失敗した25モデルは、グラフ特徴と両方の尤度をFP32で計算し直す。途中までのFP16特徴は別ディレクトリに保存し、FP32特徴と混ぜない。完了済みの66モデルは保持する。

したがって、以降の100モデル結果は「既定FP16、一部モデルはFP32で安定化」の条件であり、全モデルが原論文と同じFP16精度という意味での再現ではない。モデルごとの実際のdtypeを`complete.json`、例外一覧を`config.json`の`dtype_overrides`と`numerical_recovery.json`に記録する。精度による交絡を確認する際は、FP16で完了したモデルのみの部分集合も区別する。

既定トークナイズで1トークンになる入力の著者方式の対数尤度は、次トークン項がないため空和の0となる。この場合も入力テキストを変更せず、著者コードの定義に従う。手動BOS条件ではBOSからその1トークンを予測するため、両条件は区別される。

| 条件 | 特徴・距離 | 目的 |
|---|---|---|
| Shimodaira L raw | 既定トークナイズによる文書の対数尤度ベクトル | 平均尤度も含む対照 |
| Shimodaira L clipped | 全モデル×全文書の下位2パーセンタイルで下側クリップ | 原論文の主図に対応 |
| Shimodaira Q raw | 生Lを行・列方向に二重中心化 | クリップの効果の対照 |
| Shimodaira Q clipped | クリップ後のLを二重中心化 | KL近似に対応する主な尤度比較 |
| Uniform-BOS likelihood | 全モデルにBOSまたはEOSを明示して尤度計算 | 以前の実験・特殊トークン差の対照 |
| Hidden / eta | CKA由来のchord距離、eta近傍不一致 | グラフ化前の対照 |
| Euclidean graph | union kNNのユークリッド辺重み | Fisher導入の対照 |
| Fisher graph | 同一union kNNに局所逆共分散で辺重み | 提案する構成 |

`L[i,s] = log p_i(x_s)`、`Q = L - rowmean(L) - colmean(L) + mean(L)`。
`D_Q[i,j] = ||Q_i-Q_j||/sqrt(N)`、`KL_approx = D_Q**2/2`を別ファイルで保存する。二乗KL近似自体を距離として次元削減には渡さない。Qの距離はモデル行の平均を引いた対数尤度差の標準偏差に一致する。

著者コードに合わせ、尤度はfloat16モデルの`log_softmax`をPython float相当のfloat64で累積する。`tokenizer.encode(..., add_special_tokens=True)`を用い、手動BOSは追加しない。グラフ入力は前実験と同じ明示BOS条件とし、この差をuniform-BOS尤度条件で監査する。既定のモデル推論dtypeはfloat16とし、上記の数値エラーが出た25モデルでは両手法をfloat32にそろえる。以前のbfloat16結果は混ぜない。

重みがPyTorch形式のみのモデルも含むため、100モデル実験にはPyTorch 2.6.0の専用環境を用いる。既存の15モデル実験の環境は変更しない。実行時の依存ライブラリのバージョンは`execution_environment.json`に保存する。

グラフはPCA 16次元、正則化係数`alpha=1e-3`、テキスト近傍数`k=20`を開始値とし、全モデルが連結になる共通kを選ぶ。グラフ間比較は平均最短経路距離で正規化したFrobenius、非正規化Frobenius、順位、近傍不一致、NetLSD-style。全300ノードでの近似GWは計算量が大きいため主解析と別の追加解析に分ける。主解析の完了をGWの完了とは報告しない。

## 実験1: 距離関数を比較

全条件で同じモデル集合とテキストを用いる。地図はmetric MDSに固定し、seed 0–9、`n_init=4`。次元削減前の距離行列でも系列対応・最近傍・公理の数値監査を評価する。

1. モデル内グラフの同分野近傍率と分野シルエット。
2. モデル間距離の系列シルエット・同系列最近傍率。Llama 3と3.1等の細かな系列と、より広い系列で解釈が変わりうるためラベル依存性を明示する。
3. 手法間の距離順位相関。
4. 同一の分野層化80%部分集合による安定性。100回の距離部分行列対照と20回のグラフ再構築。尤度法でも同じ20個のテキスト部分集合を使い、行中心化を再計算。クリップ閾値は全データで固定。

PCAと局所計量は部分集合実験で固定される。完全な再推定bootstrapではない。距離行列の4,950要素を独立標本とみなした検定はしない。

## 実験2: 次元削減を比較

保存済みの同一モデル間距離を用い、MDS、Isomap (近傍10/30)、UMAP (近傍5/15/30)、t-SNE (perplexity 30)を比較。確率的手法は10 seed。非連結Isomapは補完せず除外し記録する。

- 距離スケールを最小二乗で合わせたストレス（小さいほど良い）
- モデル間距離のSpearman相関
- モデル近傍3件の保存率
- seed間の指標変動

主比較のt-SNEは距離行列入力・random初期化。他に原論文に対応するclipped Lのfeature入力・PCA初期化・perplexity 30を独立して出力し、clipped Q入力も補足する。全手法の指標が同じ意味になるよう、評価距離を必ず記録する。

## 出力と再開

各モデルのグラフ画像と整列座標、比較法別の距離行列・ヒートマップ・地図・CSV、尤度L/QとKL近似、評価表を保存する。回転のみと鏡映を含む直交整列を両方保存する。整列は表示用であり、距離計算には使わない。

100モデルの地図とヒートマップはモデル番号で表示し、`evaluation/model_key.csv`でモデルIDに対応づける。モデル内グラフの一覧画像は10モデルずつ分割する。解析後に`RESULTS.md`を自動生成し、`deliver.py`で生の特徴抽出チェックポイントを除いた成果物を手元へコピーできる。

`prepare.py`はメタデータと共通テキストを固定する。`run.py`は空いているGPUにモデルを順に割り当て、重みの不足分を取得し、文書ごとのチェックポイントから再開する。失敗モデルは記録して残りを進め、100個すべての抽出が完了するまで100モデルの集計を開始しない。`master.py`は3モデルのsmoke成功後に主実験へ進む。

著者の尤度コードは再配布せず、固定commitから取得して検証時に使う。3モデル各1テキストでの検証結果は`docs/experiments/smoke100_validation.json`に保存している。これは100モデル全体の計算完了を意味しない。

再実行の例（`shared_texts.json`は`id`・`text`・`category`を持つ300行のJSON配列）:

```bash
python experiments/100models/prepare.py --output artifacts100 \
  --manifest configs/models_100.json --source shared_texts.json --cache /path/to/hf/hub
python experiments/100models/audit_weights.py --output artifacts100
curl -L --fail \
  https://raw.githubusercontent.com/shimo-lab/modelmap/194c5436ed01d16381b1b903d3e2feec916a534c/1000models/codes/src/calculator/likelihood.py \
  -o experiments/100models/official_likelihood.py
python experiments/100models/master.py --output artifacts100 \
  --runner experiments/100models/run.py --cache /path/to/hf/hub
```

### 注意

モデル間距離は選んだコーパスと観測表現に依存する。KLの近似には近傍分布の仮定があり、離れた全モデル対でのKL精度を保証しない。NetLSD-styleはスペクトル要約によりノード対応を捨てる。系列分離・低ストレス・蒸留やモデルマージへの有用性はそれぞれ別の評価対象である。
