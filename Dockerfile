FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code only (no data/, notebooks/, tests/).
COPY src/ ./src/
COPY api/ ./api/

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/Titanic_Survival_prediction/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=2142f82b212c257e5619ae088d31af5149988ad6a032ee64cf434af75328bfdf
RUN mkdir -p models && \
    curl -fsSL -o models/titanic_pipeline.joblib \
      "https://github.com/Bakr1m/Titanic_Survival_prediction/releases/download/${MODEL_TAG}/titanic_pipeline.joblib" && \
    echo "${MODEL_SHA256}  models/titanic_pipeline.joblib" | sha256sum -c - && \
    python -c "import joblib; m=joblib.load('models/titanic_pipeline.joblib'); print('artifact OK:', type(m).__name__)"

EXPOSE 8000

CMD ["python", "api/main.py"]
