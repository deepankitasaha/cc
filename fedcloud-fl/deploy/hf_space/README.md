---
title: FedCloud FL Server
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# FedCloud-FL aggregation server (FedAsync)

The aggregation server for the ICT-4441 Cloud Computing mini project *Federated Learning Across Cloud
Nodes*. Client nodes on Colab or Kaggle pull the global model, train on their own private slice of the
Credit Card Fraud dataset, and push back only model weights. The server mixes each update in as it
arrives, discounted by staleness.

The app page is a live dashboard. Clients connect with
`python -m fedcloud.async_client --server https://<this-space>.hf.space --cid K --data client_K.npz --no-emu`.
