# Transcription API

REST API для асинхронной транскрипции разговоров оператора с клиентом с использованием Faster-Whisper и Pyannote Audio.

Проект предназначен для:

- транскрипции аудиозаписей;
- диаризации (разделения речи по спикерам);
- идентификации операторов по голосовым эмбеддингам.

## Стек технологий

- Python 3.12
- FastAPI
- PostgreSQL + pgvector
- Redis
- Celery
- Faster-Whisper
- Pyannote Audio
- SpeechBrain
- Docker
- Docker Compose

---

## Архитектура

Проект построен с разделением ответственности между слоями приложения.

```
api/
├── core/             # Конфигурация приложения
├── engines/          # ML-модели
├── models/           # SQLAlchemy модели
├── orchestrators/    # Оркестрация бизнес-процессов
├── processors/       # Обработка результатов моделей
├── repositories/     # Работа с БД
├── routers/          # REST API
├── schemas/          # Pydantic схемы
├── services/         # Бизнес-логика
└── tasks/            # Celery задачи
```

---

## Требования

Для работы проекта необходимы:

- Docker
- Docker Compose
- NVIDIA Container Toolkit (при использовании GPU)

---

## Hugging Face

Для работы проекта необходимо получить токен Hugging Face.

1. Зарегистрируйтесь на https://huggingface.co
2. Создайте токен доступа (Read).
3. Примите условия использования моделей:
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
4. Создайте файл `.env` на основе `.env.example`
5. Добавьте токен в `.env`:

```env
HF_TOKEN=your_huggingface_token
```

---

## Запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/samson205/transcription-api
cd transcription-api
```

### 2. Заполнить файл `.env`

Пример:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=transcription

HF_TOKEN=your_huggingface_token
...
```

При необходимости заполните остальные параметры конфигурации.

---

### 3. Собрать проект

```bash
docker compose build
```

Первая сборка может занимать продолжительное время.

> **Важно**
>
> Проект использует PyTorch, Faster-Whisper, Pyannote Audio и CUDA-библиотеки. Во время первой сборки Docker может потребоваться **до 30 ГБ свободного места** для хранения промежуточных слоев и зависимостей. После завершения сборки неиспользуемые слои можно удалить командой:
>
> ```bash
> docker builder prune
> ```

---

### 4. Запустить контейнеры

```bash
docker compose up
```

Будут запущены:

- API
- Celery Worker
- PostgreSQL
- Redis

После успешного запуска документация OpenAPI будет доступна по адресу:

**http://localhost:8000/docs**

---

## Используемые модели

Проект автоматически загружает необходимые модели при первом запуске.

Используются:

- Faster-Whisper
- Pyannote Audio
- SpeechBrain ECAPA

Модели сохраняются локально и повторно не скачиваются.

---

## Возможности

- Асинхронная транскрипция аудио
- Диаризация двух спикеров
- Очередь задач через Celery
- Хранение голосовых эмбеддингов операторов
- Поиск наиболее похожего оператора с использованием pgvector
- REST API для управления задачами и операторами

---

## Статус проекта

Проект находится в активной разработке.

Планируемые улучшения:

- добавление cli интерфейса;
- покрытие тестами;
- оптимизация Docker-образов;
- расширение возможностей идентификации спикеров.
