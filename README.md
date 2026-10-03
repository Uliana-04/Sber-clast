# Экономические сети российских муниципалитетов

Воспроизводимая типология структуры безналичных расходов жителей МО по данным СберИндекса за 2023–2024 годы. Узлы — МО, атрибуты — пять долей расходов, рёбра — положительное косинусное сходство в симметричной kNN-сети. Построение сети и кластеризация выполняются отдельно для каждого месяца.

## Быстрый запуск

Рекомендуется Python 3.12. Все команды выполнять из корня проекта.

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`.
macOS/Linux: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python run.py --config configs/default.yaml
```

Исходные `consumption.parquet` и `market_access.parquet` уже лежат в `data/raw/`. Интернет нужен для установки зависимостей, но не для анализа. `connection.parquet` не включён: расстояния не использовались в основной модели.

JSON-конфигурация эквивалентна YAML:

```bash
python run.py --config configs/default.json
```

Альтернативный запуск после `python -m pip install -e .`: `municipal-network --config configs/default.yaml`.

## Настройка гиперпараметров

| Поле | Назначение | Значение для сдачи |
|---|---|---|
| `seed` | Случайное начальное состояние | 42 |
| `threads` | Потоки вычислений | 2 |
| `features.categories` | Категории атрибутов | пять категорий |
| `features.share_scale` | Масштаб перед log1p | 100 |
| `network.n_neighbors` | Соседи до симметризации | 15 |
| `clustering.n_clusters` | Число кластеров | 5 |
| `clustering.assign_labels` | Присвоение меток Spectral | cluster_qr |
| `clustering.eigen_solver` | Решатель | arpack |
| `clustering.kmeans_n_init` | Перезапуски K-Means | 10 |
| `quality.s_dbw_method` | Вариант S_Dbw | Halkidi |
| `quality.nearest_centr` | Выбор центра S_Dbw | false |
| `output_dir` | Каталог новых результатов | outputs |

Для эксперимента скопируйте конфигурацию, измените параметры и `output_dir`. Значения K=5 и k=15 — рабочий выбор, после проведения анализа. Названия профилей в генерируемом отчёте нейтральные: при изменении параметров экономическую интерпретацию требуется повторить.

## Отдельные этапы

```bash
python run.py --config configs/default.yaml --stage dynamics
python run.py --config configs/default.yaml --stage quality
python run.py --config configs/default.yaml --stage interpretation
```

`quality` и `interpretation` требуют предварительно рассчитанного `dynamics` с той же конфигурацией. В `outputs/resolved_config.json` сохраняются параметры текущего запуска; не смешивайте разные эксперименты в одном каталоге.

## Результаты

- `outputs/stage5/`: месячные исходные и выровненные метки, переходы, устойчивость, размеры кластеров и проверки.
- `outputs/icvi/`: 72 сравнения (24 месяца × 3 метода), SW, CH, S_Dbw, AVI, AVU, MQ; контрольные суммы, определения и блоки сетевых весов.
- `outputs/stage6/`: доли категорий, lift, контекст доступности рынков и краткая интерпретация на русском.
- `reference/`: ранее проверенные результаты для сверки.
- `presentation/economic_municipalities.html`: готовый автономный интерактивный лендинг. Откройте двойным щелчком. Его данные соответствуют стандартной конфигурации; лендинг не меняется автоматически при новых экспериментах.

## Методология и воспроизводимость

Описание формул и ограничений: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
Проверки: `python -m unittest discover -s tests -v`.
Исходные файлы: описание и лицензия в [data/raw/README.md](data/raw/README.md).
Численные показатели могут незначительно различаться между платформами; номера кластеров сами по себе не имеют экономического смысла.

