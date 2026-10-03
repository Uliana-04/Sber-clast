"""Command-line entry point: load configuration before importing analysis."""

import argparse
import json
import os
from pathlib import Path
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description="Динамическая экономическая сеть МО")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stage", choices=["all", "dynamics", "quality", "interpretation"], default="all"
    )
    args = parser.parse_args()
    os.environ["MUNICIPAL_CONFIG"] = str(Path(args.config).resolve())
    from .config import CONFIG, project_path
    from . import dynamics, quality, interpretation

    out = project_path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    saved_config = out / "resolved_config.json"
    if args.stage in {"quality", "interpretation"}:
        if not saved_config.exists():
            parser.error("Сначала выполните dynamics с этой конфигурацией")
        if json.loads(saved_config.read_text(encoding="utf-8")) != CONFIG:
            parser.error("Параметры отличаются от dynamics; используйте новый output_dir")
    (out / "resolved_config.json").write_text(
        json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with threadpool_limits(limits=CONFIG["threads"]):
        for name, module in [
            ("dynamics", dynamics),
            ("quality", quality),
            ("interpretation", interpretation),
        ]:
            if args.stage in {"all", name}:
                print(f"Running {name}", flush=True)
                module.run()


if __name__ == "__main__":
    main()
