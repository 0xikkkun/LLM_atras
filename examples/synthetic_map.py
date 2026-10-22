"""Small CPU example: no model download, CUDA, or corpus required."""
from pathlib import Path
import numpy as np
from llm_atras import align, classical, graph_distances, normalize, pairwise_from_features

rng = np.random.default_rng(7)
base = rng.normal(size=(60, 4))
matrices = []
output = Path("artifacts/synthetic")
output.mkdir(parents=True, exist_ok=True)
for i, noise in enumerate([0.0, 0.1, 0.6]):
    X = base + noise * rng.normal(size=base.shape)
    metric = np.broadcast_to(np.eye(4), (len(X), 4, 4)).copy()
    graph, components = graph_distances(X, metric, k=10)
    assert components == 1
    S = normalize(graph["fisher"]["S"])
    matrices.append(S)
    Y = classical(S)
    if i == 0:
        reference = Y.copy()
    Y, rotation, residual = align(Y, reference)
    np.savetxt(output / f"graph_{i}.csv", Y, delimiter=",", header="x,y", comments="")
D = pairwise_from_features([S[np.triu_indices(len(S), 1)] for S in matrices])
Y = classical(D)
np.savetxt(output / "model_distances.csv", D, delimiter=",")
np.savetxt(output / "model_map.csv", Y, delimiter=",", header="x,y", comments="")
print(f"Saved graph coordinates, model distances and a model map to {output}")
