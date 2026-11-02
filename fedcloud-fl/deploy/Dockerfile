# One image for every role (server / client, sync / async). ~250 MB, no GPU needed.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fedcloud/ fedcloud/
COPY scripts/ scripts/
COPY configs/ configs/
ENV PYTHONUNBUFFERED=1
# Default: synchronous Flower server. Override CMD per role (see docker-compose.yml).
EXPOSE 8080 8081
CMD ["python", "-m", "fedcloud.sync_server", "--address", "0.0.0.0:8080", "--test", "/data/server_test.npz", "--out", "/results/sync.json"]
