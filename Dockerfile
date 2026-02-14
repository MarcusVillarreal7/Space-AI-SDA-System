# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/src/dashboard
COPY src/dashboard/package.json src/dashboard/package-lock.json ./
RUN npm ci
COPY src/dashboard/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install system deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to keep image small)
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    -r requirements-docker.txt

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY checkpoints/ ./checkpoints/
COPY data/processed/ml_train_1k/ ./data/processed/ml_train_1k/

# Copy built frontend
COPY --from=frontend-build /app/src/dashboard/dist ./src/dashboard/dist

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
