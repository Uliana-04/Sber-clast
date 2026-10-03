"""Calculate all six ICVI on identical monthly partitions and preserve evidence.

AVI and AVU: Shalileh et al., equations 18--21 (2025), using edge weights.
MQ: Chen et al. (2023), equations 13--15, weighted density modularization quality.
Newman modularity Q is stored separately and is not relabelled as MQ.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from s_dbw import S_Dbw
from threadpoolctl import threadpool_limits
from .dynamics import make_features, cosine_knn_affinity
from .config import CONFIG, project_path

OUT = project_path(CONFIG["output_dir"]) / "icvi"


def network_indices(graph, labels):
    """Return weighted AVI/AVU/MQ and Newman Q from a loop-free graph.

    Symmetric internal edges occur twice in B; external edges once per block.
    AVU zero/zero contributes zero. Reject clusters with no degree volume.
    """
    unique, inv = np.unique(labels, return_inverse=True)
    k = len(unique)
    sizes = np.bincount(inv)
    membership = csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), k))
    block = (membership.T @ graph @ membership).toarray()
    inside = np.diag(block)
    volume = block.sum(axis=1)
    outside = volume - inside
    if np.any(volume <= 0):
        raise ValueError("Zero-volume cluster: AVI undefined")
    avi = np.mean(inside / volume)
    denominator = outside[:, None] + outside[None, :] - block
    mask = ~np.eye(k, dtype=bool)
    ratio = np.divide(block, denominator, out=np.zeros_like(block), where=(denominator > 0) & mask)
    # Undefined 0/0 for two isolated clusters is defined as no merge tendency.
    avu = ratio[mask].sum() / k
    cohesion = inside / (sizes * sizes)
    coupling = block / (2 * sizes[:, None] * sizes[None, :])
    np.fill_diagonal(coupling, 0)
    mq = cohesion.mean() - coupling.sum() / (k * (k - 1) / 2) if k > 1 else cohesion[0]
    q = np.sum(inside / block.sum() - (volume / block.sum()) ** 2)
    return (
        {"AVI": float(avi), "AVU": float(avu), "MQ": float(mq), "Q_Newman": float(q)},
        block,
        sizes,
    )


def independent_check(g, labels, actual):
    """Check the vectorized indices by independent sparse graph slicing."""
    keys = np.unique(labels)
    k = len(keys)
    blocks = np.array([[g[labels == i][:, labels == j].sum() for j in keys] for i in keys])
    sizes = np.array([(labels == i).sum() for i in keys])
    outs = np.array([blocks[i].sum() - blocks[i, i] for i in range(k)])
    avi = sum(blocks[i, i] / blocks[i].sum() for i in range(k)) / k
    avu = (
        sum(
            blocks[i, j] / (outs[i] + outs[j] - blocks[i, j])
            if outs[i] + outs[j] - blocks[i, j] > 0
            else 0
            for i in range(k)
            for j in range(k)
            if i != j
        )
        / k
    )
    mq = sum(blocks[i, i] / sizes[i] ** 2 for i in range(k)) / k - sum(
        blocks[i, j] / (2 * sizes[i] * sizes[j]) for i in range(k) for j in range(k) if i != j
    ) / (k * (k - 1) / 2)
    assert np.allclose([avi, avu, mq], [actual["AVI"], actual["AVU"], actual["MQ"]], atol=1e-12)


def run():
    """Compare three methods per month and save full-panel metrics and evidence."""
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(project_path(CONFIG["data"]["consumption"]))
    assigned = pd.read_parquet(
        project_path(CONFIG["output_dir"]) / "stage5/monthly_clusters.parquet"
    )
    rows = []
    all_labels = []
    evidence = []
    for month in sorted(raw.date.unique()):
        features = make_features(raw[raw.date.eq(month)])
        x = StandardScaler().fit_transform(features)
        g = cosine_knn_affinity(pd.DataFrame(x, index=features.index))
        g.setdiag(0)
        g.eliminate_zeros()
        y0 = (
            assigned[assigned.date.eq(month)]
            .set_index("territory_id")
            .loc[features.index, "cluster"]
            .to_numpy()
        )
        partitions = [
            ("Spectral", y0),
            (
                "KMeans",
                KMeans(
                    n_clusters=CONFIG["clustering"]["n_clusters"],
                    n_init=CONFIG["clustering"]["kmeans_n_init"],
                    random_state=CONFIG["seed"],
                ).fit_predict(x),
            ),
            (
                "Ward",
                AgglomerativeClustering(
                    n_clusters=CONFIG["clustering"]["n_clusters"], linkage="ward"
                ).fit_predict(x),
            ),
        ]
        for method, y in partitions:
            net, b, n = network_indices(g, y)
            independent_check(g, y, net)
            sw = silhouette_score(x, y)
            ch = calinski_harabasz_score(x, y)
            centers = np.array([x[y == i].mean(axis=0) for i in np.unique(y)])
            within = sum(((x[y == i] - centers[i]) ** 2).sum() for i in range(len(n)))
            between = sum(n[i] * ((centers[i] - x.mean(axis=0)) ** 2).sum() for i in range(len(n)))
            assert np.isclose(ch, (between / (len(n) - 1)) / (within / (len(x) - len(n))))
            rows.append(
                {
                    "date": month,
                    "method": method,
                    "n_municipalities": len(x),
                    "SW": sw,
                    "CH": ch,
                    "S_Dbw": S_Dbw(
                        x,
                        y,
                        method=CONFIG["quality"]["s_dbw_method"],
                        nearest_centr=CONFIG["quality"]["nearest_centr"],
                    ),
                    **net,
                }
            )
            all_labels.extend(
                {"date": month, "method": method, "territory_id": int(t), "cluster": int(c)}
                for t, c in zip(features.index, y)
            )
            evidence.append(
                {
                    "date": month,
                    "method": method,
                    "block_weight_sums": b.tolist(),
                    "cluster_sizes": n.tolist(),
                }
            )
        print(month, "ICVI verified", flush=True)
    results = pd.DataFrame(rows)
    assert results[["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]].notna().all().all()
    assert (
        results.SW.between(-1, 1).all()
        and results.AVI.between(0, 1).all()
        and results.AVU.between(0, 1 + 1e-10).all()
    )
    results.to_csv(OUT / "icvi_by_month_method.csv", index=False)
    results.groupby("method")[["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q_Newman"]].agg(
        ["mean", "min", "max"]
    ).to_csv(OUT / "icvi_summary.csv")
    pd.DataFrame(all_labels).to_parquet(OUT / "comparison_partitions.parquet", index=False)
    definitions = {
        "SW": "Exact full-month mean silhouette on Euclidean standardized log1p shares; no sampling.",
        "CH": "Calinski-Harabasz on complete monthly attributes.",
        "S_Dbw": f"s-dbw 0.4.0; {CONFIG['quality']}; complete monthly attributes.",
        "AVI": "B=S.T A S; I_k=B_kk; T_k=sum_l B_kl; AVI=mean_k(I_k/T_k). Weighted extension of Eq.18-19.",
        "AVU": "E_k=T_k-I_k; AVU=(1/K)*sum_{k!=l} B_kl/(E_k+E_l-B_kl). Eq.20-21. Zero/zero contributes 0 (no merging edges).",
        "MQ": "Weighted modularization quality (Chen et al. Eq.13-15): mean_k(B_kk/n_k^2) - sum_{k!=l}(B_kl/(2*n_k*n_l))/(K*(K-1)/2). Not Newman Q; definition chosen explicitly because task does not specify MQ variant.",
        "network": f"Symmetric clipped cosine kNN{CONFIG['network']['n_neighbors']} of monthly z-scored log1p shares; diagonal removed.",
        "direction": {
            "SW": "maximize",
            "CH": "maximize",
            "S_Dbw": "minimize",
            "AVI": "maximize",
            "AVU": "minimize",
            "MQ": "maximize",
        },
        "sources": [
            "https://link.springer.com/article/10.1134/S1064562425700589",
            "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/sfw2.12075",
            "https://github.com/alashkov83/S_Dbw",
            "https://scikit-learn.org/stable/modules/clustering.html",
        ],
        "checks": "Independent block slicing verifies AVI, AVU, MQ for all 72 partitions; direct ANOVA sums verify CH; bounds and finiteness verified.",
        "seed": CONFIG["seed"],
        "months": int(raw.date.nunique()),
        "methods": ["Spectral", "KMeans", "Ward"],
        "K": CONFIG["clustering"]["n_clusters"],
        "input_sha256": hashlib.sha256(
            Path(project_path(CONFIG["data"]["consumption"])).read_bytes()
        ).hexdigest(),
    }
    (OUT / "definitions_and_validation.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2)
    )
    (OUT / "network_block_evidence.json").write_text(json.dumps(evidence))
    print(results.groupby("method")[["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]].mean().to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=CONFIG["threads"]):
        run()
