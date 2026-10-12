FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

WORKDIR /code

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Install the same spaCy model used locally
RUN uv pip install --python /code/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

COPY app ./app
COPY helper_lib ./helper_lib
COPY models/assignment2_cnn.pth ./models/assignment2_cnn.pth

EXPOSE 8000

CMD ["/code/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]