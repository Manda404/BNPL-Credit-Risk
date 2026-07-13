FROM python:3.11-slim

ENV POETRY_VERSION=2.1.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./
RUN poetry install --only main

ENTRYPOINT ["poetry", "run", "bnpl-risk"]
CMD ["--help"]
