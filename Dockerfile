FROM python:3.12-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1
ENV DEVICE=auto

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "scripts/entrypoint.py"]
