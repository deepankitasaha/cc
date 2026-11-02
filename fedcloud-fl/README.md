# FedCloud-FL: federated fraud detection across cloud nodes

This repository holds the code for the ICT-4441 Cloud Computing mini project, *Federated Learning Across Cloud Nodes:
Accuracy–Communication Trade-offs Under Non-IID Data*.

Five client nodes each keep a private slice of the Credit Card Fraud dataset (284,807
transactions, 0.17% fraud). An aggregation server trains a shared MLP with them and never sees
their raw data. The pipeline compares four things:

| Component | Implementation |
|---|---|
| Synchronous FedAvg | Flower 1.18 (`fedcloud/sync_server.py`, `fedcloud/sync_client.py`) over gRPC |
| Synchronous FedProx | same Flower server; the proximal term (μ) is applied on the clients |
| Asynchronous FedAsync | staleness-discounted mixing over HTTP (`fedcloud/async_server.py`, `async_client.py`) |
| Centralised baseline | `scripts/run_centralized.py`, which pools all the data |

The model is a 30-64-32-1 MLP written in pure NumPy, with 4,097 parameters and a 17.2 kB payload (17,156 bytes).
Because it is pure NumPy, any free-tier node can install it in seconds.

Non-IID data comes from Dirichlet label skew (`dir1.0`, `dir0.5`, `dir0.1`), with an `iid` control.
Heterogeneity is reported as Hellinger distance and the fraud-count coefficient of variation.

Each run logs AUPRC, F1, ROC-AUC, recall, precision and accuracy, plus wall-clock time, bytes
up and down per round, rounds-to-target, MB-to-target, async staleness and per-client participation.

## Layout
```
fedcloud/            core library
  data.py            loading, scaling, IID/Dirichlet partitioning, heterogeneity metrics
  model.py           NumPy MLP, Adam, class-weighted BCE, FedProx term
  metrics.py         imbalance-aware evaluation (AUPRC is the primary metric)
  netemu.py          per-client WAN latency/bandwidth + straggler emulator
  serialization.py   numpy.save tensor encoding (same as Flower), so byte counts are comparable
  sync_server.py / sync_client.py      Flower FedAvg/FedProx
  async_server.py / async_client.py    FedAsync
scripts/
  prepare_data.py    builds data/parts_<scheme>/client_k.npz + server_test.npz
  show_partitions.py live proof of the IID / Dirichlet splits (+ --plot)
  run_centralized.py centralised baseline
  run_experiment.py  launches 1 server + N clients as separate processes over real sockets
  run_all.sh         the full experiment matrix used in the paper (3 seeds)
  make_figures.py    results/*.json -> paper figures, LaTeX tables, results/summary.csv
configs/profiles.json  per-node RTT / bandwidth / compute-speed profiles
deploy/              Dockerfile, docker-compose.yml, Colab/Kaggle client notebook,
                     CLOUD_DEPLOYMENT.md (real multi-cloud set-up)
results/             JSON logs of every run used in the paper + summary.csv
tests/test_smoke.py  fast end-to-end sanity test
```

## Quick start (single machine, about 2 minutes)
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --scheme dir0.5          # downloads the CSV if data/creditcard.csv is missing
python scripts/run_experiment.py --mode sync  --strategy fedavg --scheme dir0.5 --rounds 10
python scripts/run_experiment.py --mode async --scheme dir0.5 --rounds 10
python scripts/make_figures.py
```
Logs for every process land in `results/logs/`, and each run writes a JSON file to `results/`.

To start nodes by hand in separate terminals:
```bash
python -m fedcloud.sync_server --address 0.0.0.0:8080 --clients 5 --rounds 30 --test data/parts_dir0.5/server_test.npz
python -m fedcloud.sync_client --server 127.0.0.1:8080 --cid 0 --data data/parts_dir0.5/client_0.npz   # repeat for 1..4
```

## Live demo helpers
```bash
python scripts/show_partitions.py --plot     # proves the IID / Dirichlet splits from the files on disk
```
It prints rows, frauds and each client's share of all frauds per split, then checks that every
transaction went to exactly one client, rebuilds the split from `creditcard.csv`, and confirms
the server's test set was never given to a client. Note that the raw CSV contains 1,081
duplicate transactions, so identical-looking rows across clients are expected.

When a client starts, it prints its own data (rows, frauds). Every round it prints the
hyperparameters it received from the server (`epochs`, `lr`, `prox_mu`, `seed`). Change them on
the server only (for example `--lr 0.005 --epochs 2 --strategy fedprox --prox-mu 0.1`) and watch
the clients pick up the new values.

## Reproduce the paper
```bash
./scripts/run_all.sh          # about 1 hour on 2 vCPUs; already-finished runs are skipped
python scripts/make_figures.py --paper ../paper
```

## Real cross-cloud deployment
See `deploy/CLOUD_DEPLOYMENT.md`. In short, the server runs on a free-tier VM, the clients run in
Colab/Kaggle notebooks (`deploy/client_node.ipynb`) with `--no-emu`, and the result JSONs are
saved as `results/cloud_*.json`. `make_figures.py` then adds them to the paper tables.

## Dataset
ULB Machine Learning Group, *Credit Card Fraud Detection*: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
(Database Contents License). `prepare_data.py` falls back to a public GitHub mirror of the same file.
The global test set is a fixed stratified 20% hold-out (56,962 rows, 98 frauds) that only the server holds.

## Team
Pranav Kasliwal (server and orchestration) · Nandini Neelesh (client training) ·
Harshini Rebala (non-IID partitioning and evaluation) · Deepankita Saha (communication benchmarking)
