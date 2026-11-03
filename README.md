# Сервис прогнозирования оттока клиентов

Учебный сервис на FastAPI обучает классификатор на табличных данных и возвращает прогноз оттока и вероятности классов. Метка `churn = 1` означает, что клиент ушёл в следующем месяце; `churn = 0` — что остался.

## Данные

В `data/churn_dataset.csv` находятся признаки клиента и целевая колонка:

- Числовые признаки: `monthly_fee`, `usage_hours`, `support_requests`, `account_age_months`, `failed_payments`, `autopay_enabled` (значение `0` или `1`).
- Категориальные признаки: `region` (`europe`, `asia`, `america`, `africa`), `device_type` (`mobile`, `desktop`, `tablet`), `payment_method` (`card`, `paypal`, `crypto`).
- Целевая колонка: `churn` (`0` или `1`). Она используется для обучения и не передаётся в `/predict`.

## Запуск локально

Для локального запуска используйте Python 3.12, как и в Docker-образе. Из корня проекта создайте виртуальное окружение, установите зависимости и запустите API:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Интерактивная документация доступна по адресу `http://127.0.0.1:8000/docs`, проверка состояния — `http://127.0.0.1:8000/health`. При первом запуске обученной модели ещё нет; вызовите `POST /model/train`.

### Обучение модели

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"logreg","hyperparameters":{"C":1.0,"max_iter":1000,"solver":"lbfgs"}}'
```

Допустимые модели: `logreg` и `random_forest`. Для обучения на случайном лесе отправьте, например, `{"model_type":"random_forest","hyperparameters":{"n_estimators":100}}`. Ответ содержит признак доступности модели, время обучения и метрики на отложенной выборке. Модель и история обучения сохраняются в `models/`.

### Прогноз

После обучения отправьте один объект со всеми обязательными признаками:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"monthly_fee":79.99,"usage_hours":4.5,"support_requests":3,"account_age_months":2,"failed_payments":2,"region":"europe","device_type":"mobile","payment_method":"crypto","autopay_enabled":0}'
```

Ответ имеет вид:

```json
{
  "predicted_class": 1,
  "class_probabilities": {"0": 0.12, "1": 0.88}
}
```

`/predict` также принимает массив таких объектов и возвращает массив результатов в том же порядке. Дополнительные сведения доступны через `GET /model/status`, `GET /model/metrics`, `GET /dataset/info` и `GET /model/schema`.

## Запуск в Docker

```bash
docker build -t churn-api .
docker run --name churn-api -p 8000:8000 -v churn-models:/app/models churn-api
```

Первая команда собирает образ по `Dockerfile`, вторая запускает контейнер и сохраняет модель с историей в Docker-томе `churn-models`. Сервис будет доступен на `http://localhost:8000`; документация — `/docs`, состояние — `/health`. Для остановки нажмите `Ctrl+C`. Чтобы запустить контейнер повторно после остановки, выполните `docker start -ai churn-api`. Используйте те же запросы `curl` для обучения и прогнозов.

## Тесты

Установив зависимости в виртуальное окружение, запустите:

```bash
.venv/bin/python -m pytest
```
