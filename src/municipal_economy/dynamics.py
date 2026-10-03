"""Stage 5: dynamic clustering and cluster-transition analysis.

Input: SberIndex consumption panel in long Parquet format.
Output: reproducible monthly network clusters, aligned labels, transitions,
stability statistics, and a machine-readable validation report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


from .config import CONFIG, project_path

SEED = CONFIG["seed"]
N_CLUSTERS = CONFIG["clustering"]["n_clusters"]
N_NEIGHBORS = CONFIG["network"]["n_neighbors"]
INPUT = project_path(CONFIG["data"]["consumption"])
OUT = project_path(CONFIG["output_dir"]) / "stage5"
TOTAL_CATEGORY = CONFIG["features"]["total_category"]
PROFILE_CATEGORIES = CONFIG["features"]["categories"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build scale-free monthly spending profiles for municipalities."""
    wide = frame.pivot(index="territory_id", columns="category", values="value")
    wide = wide.reindex(columns=[TOTAL_CATEGORY, *PROFILE_CATEGORIES])
    wide = wide.dropna()
    wide = wide.loc[wide[TOTAL_CATEGORY] > 0]
    profiles = wide[PROFILE_CATEGORIES].div(wide[TOTAL_CATEGORY], axis=0)
    # Log1p compresses skewed shares; this is not a compositional log-ratio.
    return (
        np.log1p(profiles * CONFIG["features"]["share_scale"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def cosine_knn_affinity(features: pd.DataFrame) -> csr_matrix:
    """Create a symmetric weighted k-nearest-neighbour economic network."""
    distances = kneighbors_graph(
        features.to_numpy(),
        n_neighbors=min(N_NEIGHBORS, len(features) - 1),
        metric="cosine",
        mode="distance",
        include_self=False,
    )
    similarity = distances.copy()
    similarity.data = np.clip(1 - similarity.data, 0, 1)
    similarity = similarity.maximum(similarity.T)
    similarity.setdiag(1.0)
    similarity.eliminate_zeros()
    return similarity.tocsr()


def cluster_month(features: pd.DataFrame) -> np.ndarray:
    """Cluster the monthly attributed network with spectral graph partitioning."""
    standardized = StandardScaler().fit_transform(features)
    affinity = cosine_knn_affinity(pd.DataFrame(standardized, index=features.index))
    model = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity="precomputed",
        assign_labels=CONFIG["clustering"]["assign_labels"],
        eigen_solver=CONFIG["clustering"]["eigen_solver"],
        random_state=SEED,
        n_init=10,
    )
    return model.fit_predict(affinity)


def align_to_previous(
    previous: pd.Series, current_raw: pd.Series
) -> tuple[pd.Series, dict[int, int], int]:
    """Align current cluster IDs to prior month through maximum overlap."""
    common = previous.index.intersection(current_raw.index)
    if common.empty:
        raise ValueError("Adjacent months have no municipalities in common.")
    matrix = pd.crosstab(current_raw.loc[common], previous.loc[common])
    matrix = matrix.reindex(index=range(N_CLUSTERS), columns=range(N_CLUSTERS), fill_value=0)
    row_idx, col_idx = linear_sum_assignment(-matrix.to_numpy())
    mapping = {int(raw): int(aligned) for raw, aligned in zip(row_idx, col_idx)}
    aligned = current_raw.map(mapping).astype("int16")
    overlap = int(matrix.to_numpy()[row_idx, col_idx].sum())
    return aligned, mapping, overlap


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INPUT)
    required_columns = {"date", "territory_id", "category", "value"}
    if not required_columns.issubset(raw.columns):
        raise ValueError("Missing consumption columns")
    if raw.duplicated(["territory_id", "date", "category"]).any():
        raise ValueError("Duplicate municipality-month-category observations")
    if not np.isfinite(raw["value"]).all() or raw["value"].lt(0).any():
        raise ValueError("Spending must be finite and nonnegative")
    raw["date"] = pd.PeriodIndex(raw["date"], freq="M")
    dates = sorted(raw["date"].unique())
    if len(dates) < 2:
        raise ValueError("At least two months are required for dynamics")
    required = {TOTAL_CATEGORY, *PROFILE_CATEGORIES}
    missing = required.difference(raw["category"].unique())
    if missing:
        raise ValueError(f"Missing required categories: {sorted(missing)}")

    raw_clusters: dict[pd.Period, pd.Series] = {}
    monthly_rows = []
    for date in dates:
        features = make_features(raw.loc[raw["date"] == date])
        if len(features) <= N_CLUSTERS:
            raise ValueError(f"{date}: insufficient complete observations for selected K")
        labels = cluster_month(features)
        raw_clusters[date] = pd.Series(labels, index=features.index, name="cluster_raw")
        monthly_rows.append(
            pd.DataFrame({"territory_id": features.index, "date": str(date), "cluster_raw": labels})
        )

    aligned_parts = []
    mappings = []
    first = dates[0]
    previous = raw_clusters[first].astype("int16")
    aligned_parts.append(
        pd.DataFrame(
            {"territory_id": previous.index, "date": str(first), "cluster": previous.values}
        )
    )
    mappings.append(
        {
            "date": str(first),
            "mapping": {str(i): int(i) for i in range(N_CLUSTERS)},
            "matched_overlap": None,
        }
    )

    for date in dates[1:]:
        current, mapping, overlap = align_to_previous(previous, raw_clusters[date])
        aligned_parts.append(
            pd.DataFrame(
                {"territory_id": current.index, "date": str(date), "cluster": current.values}
            )
        )
        mappings.append(
            {
                "date": str(date),
                "mapping": {str(k): v for k, v in mapping.items()},
                "matched_overlap": overlap,
            }
        )
        previous = current

    clusters = pd.concat(aligned_parts, ignore_index=True).sort_values(["territory_id", "date"])
    clusters["cluster"] = clusters["cluster"].astype("int16")
    clusters.to_parquet(OUT / "monthly_clusters.parquet", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(
        OUT / "monthly_clusters_raw.parquet", index=False
    )

    pairs = []
    date_strings = [str(d) for d in dates]
    for start, end in zip(date_strings[:-1], date_strings[1:]):
        if pd.Period(end, freq="M").ordinal - pd.Period(start, freq="M").ordinal != 1:
            continue  # Entirely missing months do not count as adjacent transitions.
        left = clusters.loc[clusters["date"] == start, ["territory_id", "cluster"]].rename(
            columns={"cluster": "from_cluster"}
        )
        right = clusters.loc[clusters["date"] == end, ["territory_id", "cluster"]].rename(
            columns={"cluster": "to_cluster"}
        )
        joined = left.merge(right, on="territory_id", how="inner")
        joined["from_date"] = start
        joined["to_date"] = end
        pairs.append(joined)
    if not pairs:
        raise ValueError("No adjacent observed months available")
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_parquet(OUT / "adjacent_pairs.parquet", index=False)

    transition_counts = (
        pairs.groupby(["from_cluster", "to_cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_transitions"})
    )
    transition_counts["probability"] = transition_counts[
        "n_transitions"
    ] / transition_counts.groupby("from_cluster")["n_transitions"].transform("sum")
    transition_counts.to_csv(OUT / "transition_matrix_long.csv", index=False)

    size_by_month = (
        clusters.groupby(["date", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_municipalities"})
    )
    size_by_month.to_csv(OUT / "cluster_sizes_by_month.csv", index=False)

    stability = (
        pairs.assign(changed=lambda x: x["from_cluster"].ne(x["to_cluster"]).astype(int))
        .groupby("territory_id", as_index=False)
        .agg(n_adjacent_transitions=("changed", "size"), n_cluster_changes=("changed", "sum"))
    )
    observed = (
        clusters.groupby("territory_id", as_index=False)
        .size()
        .rename(columns={"size": "n_observed_months"})
    )
    stability = observed.merge(stability, on="territory_id", how="left").fillna(
        {"n_adjacent_transitions": 0, "n_cluster_changes": 0}
    )
    stability["stability"] = np.where(
        stability["n_adjacent_transitions"] > 0,
        1 - stability["n_cluster_changes"] / stability["n_adjacent_transitions"],
        np.nan,
    )
    stability.to_csv(OUT / "municipality_stability.csv", index=False)

    # Independent validation from saved outputs rather than in-memory intermediates.
    reloaded = pd.read_parquet(OUT / "monthly_clusters.parquet")
    reloaded_pairs = pd.read_parquet(OUT / "adjacent_pairs.parquet")
    row_probability_sums = transition_counts.groupby("from_cluster")["probability"].sum()
    validation = {
        "n_months": len(dates),
        "n_municipalities_any_month": int(reloaded["territory_id"].nunique()),
        "n_cluster_assignments": int(len(reloaded)),
        "cluster_labels_valid": bool(reloaded["cluster"].between(0, N_CLUSTERS - 1).all()),
        "alignment_mappings_bijective": bool(
            all(len(set(item["mapping"].values())) == N_CLUSTERS for item in mappings)
        ),
        "pair_rows_match_transition_total": bool(
            len(reloaded_pairs) == int(transition_counts["n_transitions"].sum())
        ),
        "transition_rows_sum_to_one": bool(
            np.allclose(row_probability_sums.to_numpy(), 1.0, atol=1e-12)
        ),
        "stability_in_unit_interval": bool(stability["stability"].dropna().between(0, 1).all()),
        "median_stability": float(stability["stability"].median()),
        "mean_stability": float(stability["stability"].mean()),
    }
    with (OUT / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
    with (OUT / "label_mappings.json").open("w", encoding="utf-8") as handle:
        json.dump(mappings, handle, ensure_ascii=False, indent=2)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
