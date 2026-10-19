FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY translator_svc ./translator_svc
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "translator_svc.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
