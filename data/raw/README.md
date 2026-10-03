# Исходные данные

Источник: СберИндекс, набор муниципальных данных, предоставленный пользователем.
Описание: https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim

`consumption.parquet`: date, territory_id, category, value; 303126 строк, 2190 МО, 24 месяца. value — оценка средних безналичных расходов жителей в рублях.
`market_access.parquet`: статический индекс доступности рынков за 2024 год; не временной признак.

Исходные данные и производные таблицы: CC BY-SA 4.0 с указанием СберИндекса, согласно приложенному лицензионному PDF. Лицензия кода оформлена отдельно. Файл connection.parquet не нужен для текущих вычислений и не включён.
