FROM python:3.12-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1
ENV DEVICE=auto
ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 8 -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "-m", "scripts.entrypoint"]
