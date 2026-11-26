# Reproducible end-to-end run: download -> cohort -> train -> tests
# Override with --build-arg BASE_IMAGE=... to use a registry mirror
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY README.md .

# Outputs land in /app/reports and /app/models; mount them to keep results:
#   docker run --rm -v "$PWD/reports:/app/reports" -v "$PWD/models:/app/models" mimic-icu-risk
CMD ["sh", "-c", "set -e; \
python src/download_data.py && \
python src/cohort.py && \
python src/train.py && \
python -m pytest -v"]
