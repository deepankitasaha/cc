# Hugging Face Docker Space: FedAsync aggregation server on the single public port 7860.
# scripts/deploy_hf.py uploads this file with app.py, fedcloud/ and server_test.npz to the Space root.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV PYTHONUNBUFFERED=1 ROUNDS=30 CLIENTS=5
EXPOSE 7860
CMD ["python", "app.py"]
