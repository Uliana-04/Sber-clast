"""Stage 6: economic interpretation of dynamic municipal clusters."""

from __future__ import annotations

import json

import pandas as pd


from .config import CONFIG, project_path

INPUT_CONSUMPTION = project_path(CONFIG["data"]["consumption"])
INPUT_ACCESS = project_path(CONFIG["data"]["market_access"])
INPUT_CLUSTERS = project_path(CONFIG["output_dir"]) / "stage5/monthly_clusters.parquet"
INPUT_STABILITY = project_path(CONFIG["output_dir"]) / "stage5/municipality_stability.csv"
OUT = project_path(CONFIG["output_dir"]) / "stage6"
TOTAL = CONFIG["features"]["total_category"]
CATEGORIES = CONFIG["features"]["categories"]
# Neutral names remain valid when K, features, or seed changes.
NAMES = {i: f"Потребительский профиль {i + 1}" for i in range(CONFIG["clustering"]["n_clusters"])}


def build_panel() -> pd.DataFrame:
    consumption = pd.read_parquet(INPUT_CONSUMPTION)
    clusters = pd.read_parquet(INPUT_CLUSTERS)
    access = pd.read_parquet(INPUT_ACCESS)
    wide = consumption.pivot(
        index=["date", "territory_id"], columns="category", values="value"
    ).reset_index()
    wide = wide.dropna(subset=[TOTAL, *CATEGORIES])
    wide = wide.loc[wide[TOTAL] > 0].copy()
    for category in CATEGORIES:
        wide[f"share_{category}"] = 100 * wide[category] / wide[TOTAL]
    panel = wide.merge(clusters, on=["date", "territory_id"], how="inner", validate="one_to_one")
    panel = panel.merge(access, on="territory_id", how="left", validate="many_to_one")
    panel["cluster_name"] = panel["cluster"].map(NAMES)
    return panel


def write_report(summary: pd.DataFrame, evidence: pd.DataFrame) -> None:
    lines = [
        "# Экономическая интерпретация кластеров",
        "",
        "Названия типов основаны на отличиях от среднего профиля расходов, а не на номере кластера.",
        "Доли — категория / все категории в оценке средних безналичных расходов жителей. Это не совокупный оборот МО. Производственная специализация не измерялась.",
        "",
    ]
    for row in summary.sort_values("cluster").itertuples(index=False):
        strongest = (
            evidence.loc[evidence["cluster"].eq(row.cluster)]
            .sort_values("lift", ascending=False)
            .head(2)
        )
        strongest_text = "; ".join(
            f"{x.category}: {x.lift:.2f}× среднего" for x in strongest.itertuples(index=False)
        )
        lines.extend(
            [
                f"## {row.cluster}. {row.cluster_name}",
                "",
                f"Средний профиль: {strongest_text}.",
                f"Медианная оценка средних безналичных расходов жителей: {row.median_total_spend:,.0f} руб.; "
                f"медианный индекс доступности рынков: {row.median_market_access:.1f}.",
                f"Средняя устойчивость принадлежности к типу: {row.mean_stability:.3f}.",
                "",
            ]
        )
    (OUT / "cluster_interpretation_ru.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    shares = [f"share_{category}" for category in CATEGORIES]
    profile = panel.groupby(["cluster", "cluster_name"], as_index=False)[shares].mean()
    overall = panel[shares].mean()
    lift = profile.copy()
    for category in CATEGORIES:
        lift[f"lift_{category}"] = profile[f"share_{category}"] / overall[f"share_{category}"]
    evidence = lift.melt(
        id_vars=["cluster", "cluster_name"],
        value_vars=[f"lift_{category}" for category in CATEGORIES],
        var_name="category",
        value_name="lift",
    )
    evidence["category"] = evidence["category"].str.removeprefix("lift_")
    evidence.to_csv(OUT / "cluster_feature_lifts.csv", index=False)
    profile.to_csv(OUT / "cluster_profiles_percent.csv", index=False)

    static_cluster = (
        panel.groupby("territory_id")
        .agg(cluster=("cluster", lambda x: x.mode().iat[0]))
        .reset_index()
    )
    stability = pd.read_csv(INPUT_STABILITY)
    stability_by_cluster = (
        static_cluster.merge(stability, on="territory_id", how="left")
        .groupby("cluster", as_index=False)
        .agg(
            mean_stability=("stability", "mean"),
            median_stability=("stability", "median"),
            n_municipalities=("territory_id", "nunique"),
        )
    )
    context = panel.groupby(["cluster", "cluster_name"], as_index=False).agg(
        median_total_spend=(TOTAL, "median"),
        mean_total_spend=(TOTAL, "mean"),
        median_market_access=("market_access", "median"),
        mean_market_access=("market_access", "mean"),
        observations=("territory_id", "size"),
    )
    summary = context.merge(stability_by_cluster, on="cluster", how="left").sort_values("cluster")
    summary.to_csv(OUT / "cluster_interpretation_summary.csv", index=False)

    monthly = panel.groupby(["date", "cluster", "cluster_name"], as_index=False).agg(
        n_municipalities=("territory_id", "nunique"),
        median_total_spend=(TOTAL, "median"),
    )
    monthly.to_csv(OUT / "cluster_context_by_month.csv", index=False)
    write_report(summary, evidence)

    checks = {
        "all_cluster_names_assigned": bool(panel["cluster_name"].notna().all()),
        "configured_clusters_profiled": bool(
            profile["cluster"].nunique() == CONFIG["clustering"]["n_clusters"]
        ),
        "all_profile_shares_nonnegative": bool((profile[shares] >= 0).all().all()),
        "market_access_present_for_majority": bool(panel["market_access"].notna().mean() > 0.95),
        "interpretation_table_complete": bool(summary["mean_stability"].notna().all()),
    }
    checks["all_checks_pass"] = all(checks.values())
    (OUT / "validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
