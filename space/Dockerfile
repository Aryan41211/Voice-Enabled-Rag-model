# HF Space image: clone the public repo, bake the index at build time so
# boot is fast and demo-reliable. Corpus + model downloads happen here.
FROM python:3.12-slim

WORKDIR /repo

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Aryan41211/Voice-Enabled-Rag-model.git .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV STT_PROVIDER=sarvam
ENV LLM_PROVIDER=extractive
ENV DATA_STRATEGY=metadata

RUN python -m app.ingestion.build_index --lang hi --strategies metadata \
    --corpus-queries 1000 --eval-queries 50

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
