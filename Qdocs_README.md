# Qdocs

Qdocs — форк [Docling](https://github.com/docling-project/docling) для преобразования документов в структурированные данные. Проект предоставляет Python SDK и CLI для обработки PDF, сканов, изображений, офисных файлов и других форматов с экспортом в Markdown, JSON и HTML.

Основные изменения Qdocs касаются OCR и потребления памяти:

- Поддержка полного пайплайна PaddleOCR-VL-1.6 с анализом макета PP-DocLayoutV3, автоматическим определением языка и пакетной обработкой страниц.
- Снижение копирования изображений в EasyOCR и обработка распознавания с `batch_size=1`, `workers=0`.
- Явное и автоматическое освобождение кешированных пайплайнов, рабочих процессов моделей и неиспользуемой памяти ускорителей.

Qdocs сохраняет пространство имён `docling`, CLI-команды `docling` и `docling-tools`, а также модель документа `DoclingDocument`. Изменения форка устанавливаются из [этого репозитория](https://github.com/KrickmanC/Qdocs).

## Возможности

- Локальная конвертация файлов, каталогов и документов по HTTP/HTTPS-ссылкам.
- Извлечение текста, структуры таблиц, порядка чтения и макета PDF.
- OCR сканов и изображений с выбором движка.
- Работа с PDF, DOCX, PPTX, XLSX, HTML, Markdown и изображениями в стандартной конфигурации; дополнительные форматы подключаются через extras.
- Экспорт в Markdown, JSON, HTML и другие форматы Docling.
- Удалённая конвертация через совместимый сервер `docling-serve`.

Полный перечень входных и выходных форматов приведён в [документации](docs/usage/supported_formats.md).

## Установка из исходников

Требуются Python 3.10+, Git и uv. Выполните:

```bash
git clone https://github.com/KrickmanC/Qdocs.git
cd Qdocs
uv sync --frozen --extra standard --no-default-groups
```

Команда устанавливает код Qdocs из текущего checkout и стандартные зависимости конвертации. Дополнительные OCR-движки устанавливаются отдельно. Команды ниже выполняются из корня репозитория; `uv run --no-sync` использует уже подготовленное окружение, включая отдельно установленный runtime OCR.

Для локальной обработки PDF и OCR необходимые модели могут загружаться при первом использовании. Настройка локальных артефактов описана в [расширенных опциях](docs/usage/advanced_options.md).

## Быстрый старт

### CLI

Преобразовать PDF в Markdown и JSON:

```bash
uv run --no-sync docling convert --to md --to json --output ./output document.pdf
```

Результаты появятся в каталоге `output`. Для конвертации каталога можно передать его путь вместо имени файла:

```bash
uv run --no-sync docling convert --from pdf --output ./output ./documents
```

Справка по командам и настройкам:

```bash
uv run --no-sync docling --help
uv run --no-sync docling convert --help
```

### Python

```python
from pathlib import Path

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
try:
    result = converter.convert(Path("document.pdf"))
    print(result.document.export_to_markdown())
finally:
    converter.release_resources()
```

Запускайте свои скрипты в окружении проекта через `uv run --no-sync python script.py`.

## PaddleOCR-VL-1.6

Qdocs запускает PaddleOCR-VL-1.6 вместе с PP-DocLayoutV3 в отдельном рабочем процессе. Движок сам определяет поддерживаемые языки документа; параметр `--ocr-lang` для него не используется.

Установите extra `feat-ocr-paddleocr` и совместимый runtime отдельно. Пример для x64 CPU с PaddlePaddle:

```bash
uv sync --frozen --extra standard --extra feat-ocr-paddleocr --no-default-groups
uv pip install "paddlepaddle>=3.2.1,<4" --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
uv run --no-sync docling convert --device cpu --ocr-engine paddleocr-vl-1.6 --output ./output document.pdf
```

Для GPU установите runtime, соответствующий ускорителю, и выберите устройство через `--device`. Подробности приведены в [инструкции по установке](docs/getting_started/installation.md) и [настройке GPU](docs/usage/gpu.md).

Настройка CPU-конвертации в Python:

```python
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaddleOcrVlOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    accelerator_options=AcceleratorOptions(device="cpu"),
    ocr_options=PaddleOcrVlOptions(),
)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
try:
    result = converter.convert(Path("document.pdf"))
    print(result.document.export_to_markdown())
finally:
    converter.release_resources()
```

Основные параметры `PaddleOcrVlOptions`:

| Параметр | По умолчанию | Назначение |
| --- | --- | --- |
| `scale` | `1.0` | Масштаб растеризации PDF: 72 DPI. Увеличьте для мелкого текста. |
| `page_batch_size` | `4` | Число страниц в одном вызове распознавания. Уменьшите при нехватке памяти. |
| `use_queues` | `True` | Асинхронные очереди анализа макета и VLM-распознавания. |
| `layout_detection_model_dir` | `None` | Каталог локальных весов PP-DocLayoutV3. |
| `vl_rec_model_dir` | `None` | Каталог локальных весов PaddleOCR-VL-1.6. |

Веса PaddleOCR загружаются при первом использовании, если локальные каталоги не заданы. Для подключения VLM-сервера задайте `vl_rec_backend`, `vl_rec_server_url` и при необходимости `vl_rec_max_concurrency`; пример есть в [документации GPU](docs/usage/gpu.md).

## EasyOCR

Для EasyOCR установите соответствующий extra:

```bash
uv sync --frozen --extra standard --extra feat-ocr-easyocr --no-default-groups
uv run --no-sync docling convert --ocr-engine easyocr --ocr-lang ru,en --output ./output document.pdf
```

В Qdocs распознавание использует по одному изображению в батче и выполняется без дополнительных процессов EasyOCR. Это уменьшает пиковое потребление памяти при обработке больших сканов.

## Управление памятью

Конвертер кеширует модели для повторного использования. После завершения всех конвертаций вызовите `converter.release_resources()`, как в примерах Python выше.

Чтобы автоматически освобождать ресурсы после завершения последней активной конвертации на конвертере, включите настройку до начала обработки:

```python
from docling.datamodel.settings import settings

settings.resources.release_after_conversion = True
```

Для CLI можно задать переменную окружения. Пример для PowerShell:

```powershell
$env:DOCLING_RESOURCES_RELEASE_AFTER_CONVERSION = "true"
uv run --no-sync docling convert --output ./output document.pdf
```

Для Bash:

```bash
DOCLING_RESOURCES_RELEASE_AFTER_CONVERSION=true uv run --no-sync docling convert --output ./output document.pdf
```

По умолчанию автоматическое освобождение выключено. Очистка закрывает рабочие процессы моделей, удаляет кеш пайплайнов, запускает сборку мусора и освобождает неиспользуемые кеши аллокаторов ускорителей. Следующая конвертация заново инициализирует модели. Контексты ускорителей и объекты, на которые ещё ссылается приложение, могут сохранять часть памяти.

Вызов `release_resources()` во время активной конвертации завершается `RuntimeError`. Настройки для сервера описаны в [руководстве по развёртыванию](docs/usage/api_server/deployment.md#release-models-while-the-server-is-idle).

## Удалённая конвертация

Для обработки на уже запущенном сервере `docling-serve` используйте `convert-remote`:

```bash
uv run --no-sync docling convert-remote --service-url http://localhost:5001 --to md --to json --output ./output document.pdf
```

Локальные файлы передаются серверу, а результаты сохраняются в указанный каталог на клиенте. Команда также принимает каталоги и HTTP/HTTPS-ссылки.

Адрес сервера можно задать через `DOCLING_SERVICE_URL`, а ключ API — через `DOCLING_SERVICE_API_KEY` или `--api-key`. Приоритет настроек: параметры CLI, переменные окружения, файл `.env` в рабочем каталоге.

```bash
uv run --no-sync docling convert-remote --help
```

Для собственного сервера и Python-клиента см. [развёртывание API-сервера](docs/usage/api_server/deployment.md) и [REST API](docs/usage/api_server/rest_api.md).

## Разработка

Основные команды из корня проекта:

```bash
make setup          # Установить окружение разработки из uv.lock
make hooks-install  # Установить Git-хуки
make test           # Запустить тесты
make check          # Выполнить проверки без изменения исходников
make validate       # Запустить хуки для изменённых и новых файлов
```

`Makefile` использует POSIX shell; на Windows запускайте эти команды из Git Bash. Подготовить окружение напрямую через uv можно так:

```bash
uv sync --frozen --group dev --all-extras --no-group docs --no-group examples
```

Для изменений OCR и управления ресурсами запускайте соответствующие тесты:

```bash
uv run --no-sync pytest tests/test_easyocr_model.py tests/test_paddleocr_vl_model.py tests/test_document_converter_resources.py tests/test_settings_load.py
```

Дополнительные правила работы с унаследованным кодом приведены в [CONTRIBUTING.md](CONTRIBUTING.md).

### Структура репозитория

```text
docling/                 # Python SDK, CLI, модели и пайплайны
packages/docling/        # Метапакет с совместимыми extras
packages/docling-slim/   # README модульного пакета
tests/                   # Тесты и эталонные документы
docs/                    # Документация и примеры
scripts/                 # Скрипты проверок и сборки документации
```

## Документация

- [Установка и OCR-движки](docs/getting_started/installation.md)
- [Параметры пайплайнов](docs/reference/pipeline_options.md)
- [GPU и настройка производительности](docs/usage/gpu.md)
- [Модель документа](docs/concepts/docling_document.md)
- [Расширенные опции и локальные артефакты](docs/usage/advanced_options.md)
- [API-сервер](docs/usage/api_server/index.md)

Документация в `docs/` включает материалы исходного Docling и настройки, используемые в Qdocs.

## Происхождение и лицензия

Qdocs основан на [Docling](https://github.com/docling-project/docling). Авторские уведомления исходного проекта сохранены в [LICENSE](LICENSE). Код распространяется по лицензии MIT; используемые модели имеют собственные лицензии.

Сведения для цитирования исходного Docling находятся в [CITATION.cff](CITATION.cff).
