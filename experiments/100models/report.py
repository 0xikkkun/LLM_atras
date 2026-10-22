"""Create a source-linked numerical readout after analysis, without inferring superiority."""
import argparse,csv,json,statistics as st
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output/'main/analysis';ev=out/'evaluation'
def rows(name):return list(csv.DictReader((ev/name).open()))
config=json.loads((a.output/'config.json').read_text());done=json.loads((out/'complete.json').read_text())
lines=['# 100モデル対照実験の結果','',f"モデル {done['models']}、共通テキスト {done['texts']}。距離 {len(done['methods'])} 条件。次元削減 {done['map_runs']} 回。",'', '共通300テキストでの手法追試。原論文の1,018モデル・10,000テキストの完全再現ではない。', '',f"以前の15モデル実験からprefixが変わったテキスト: {config['texts_changed_from_15_model_run']} 件。推論dtype: {config['dtype']}。",'','## モデル内グラフ','', '| 重み | 同分野の近傍10件の割合 | 分野シルエット |','|---|---:|---:|']
g=rows('internal_graph_quality.csv')
for w in ['euclidean','fisher']:
 r=[x for x in g if x['weight']==w];lines.append(f"| {w} | {st.mean(float(x['category_knn10_hit']) for x in r):.2%} | {st.mean(float(x['category_silhouette']) for x in r):.4f} |")
e={x['model']:float(x['category_knn10_hit']) for x in g if x['weight']=='euclidean'}
d=[float(x['category_knn10_hit'])-e[x['model']] for x in g if x['weight']=='fisher'];lines+=['',f'Fisherで近傍割合が増えたモデル: {sum(v>0 for v in d)}/{len(d)}。平均差: {st.mean(d)*100:.2f}ポイント。','', '## 距離関数と系列','', '| 条件 | 同系列最近傍割合 | 系列シルエット |','|---|---:|---:|']
for r in rows('family_association.csv'):lines.append(f"| {r['method']} | {float(r['nearest_same_family']):.1%} | {float(r['family_silhouette']):.4f} |")
lines+=['','系列ラベルの粒度とモデル構成に依存する指標である。','', '## 同じ距離行列に対する次元削減','', '| 距離 | 次元削減 | 近傍数等 | ストレス平均 | 距離順位相関平均 | 近傍保存率平均 |','|---|---|---:|---:|---:|']
for r in rows('reduction_summary.csv'):
 if r['method'] in ['shimodaira_Q_clipped','shimodaira_L_clipped','euclidean_frob','fisher_frob']:
  lines.append(f"| {r['method']} | {r['reducer']} | {r['neighbors']} | {float(r['stress_scale_fitted_mean']):.4f} | {float(r['spearman_mean']):.4f} | {float(r['model_knn_overlap_mean']):.1%} |")
lines+=['','## テキスト部分集合への安定性','', '| 条件 | 距離順位相関 平均±標準偏差 | 近傍保存率平均 |','|---|---:|---:|']
r=rows('rebuilt_graph_stability.csv')
for method in sorted({x['method'] for x in r}):
 s=[x for x in r if x['method']==method];v=[float(x['spearman']) for x in s];lines.append(f"| {method} | {st.mean(v):.4f} ± {st.stdev(v):.4f} | {st.mean(float(x['model_knn_overlap']) for x in s):.1%} |")
lines+=['','PCAと局所計量は固定。尤度側は同じ部分集合を使い、全体で決めたクリップ閾値を固定して再中心化している。','', '## 出典','', '- 設定: `config.json` / `main/analysis/provenance.json`','- 評価: `main/analysis/evaluation/`','- 地図: `main/analysis/maps/`','- モデル内グラフ: `main/analysis/graphs/`','- 尤度法: https://aclanthology.org/2025.acl-long.1584/','', '## 解釈の範囲','', '系列対応・入力分野の近傍保存・地図の距離保存は異なる評価である。低ストレスだけを距離関数の有用性とみなさない。異なるモデル対の距離を独立標本とみなした有意差検定はしていない。蒸留・マージ性能は未検証。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('REPORT_SAVED',a.output/'RESULTS.md',flush=True)
