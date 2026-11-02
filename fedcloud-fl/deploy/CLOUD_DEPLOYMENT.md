# Deploying across real, independent cloud nodes

This takes the same code that the single-host harness runs and spreads it over separate
free-tier cloud accounts. Only **one** node needs a public IP, and that is the aggregation server.
Every client makes only outbound connections, so Colab and Kaggle notebooks work as clients.

```
                +----------------------------------------+
                |  Aggregation server (public IP)        |
                |  GCP e2-micro / Oracle Free VM / any VM|
                |  :8080 Flower gRPC (sync FedAvg/Prox)  |
                |  :8081 HTTP (async FedAsync)           |
                +----------------^-----------------------+
     outbound gRPC/HTTP  |     |     |     |     |
                   node-A    node-B  node-C  node-D  node-E
                  (Kaggle)  (Colab) (Colab) (Kaggle) (any VM / laptop)
                  client_0  client_1 ...                client_4   <- each holds ONLY its own .npz
```

## 1. Prepare the partitions (once, on any machine)
```bash
pip install -r requirements.txt
python scripts/prepare_data.py --scheme dir0.5      # or iid / dir1.0 / dir0.1
# -> data/parts_dir0.5/client_{0..4}.npz, server_test.npz, partition_report.json
```
Upload `server_test.npz` to the server only, and `client_k.npz` to node *k* only.

## 2. Aggregation server (free-tier VM)
Any VM with a public IP works. Two options:
- **GCP:** Compute Engine, then e2-micro (always-free regions: us-west1, us-central1, us-east1), Debian 12.
- **Oracle Cloud:** Always Free, VM.Standard.E2.1.Micro or Ampere A1.

Open TCP **8080** (sync) and **8081** (async) in the firewall:
- **GCP:** VPC network, then Firewall, then add an ingress rule for tcp:8080,8081 from `0.0.0.0/0`. Lock this down to client IPs when you can.
- **Oracle:** Subnet, then Security List, then add the ingress rules. Also run `sudo iptables -I INPUT -p tcp --dport 8080:8081 -j ACCEPT`.

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git
python3 -m venv fl && . fl/bin/activate
# copy the code zip to the VM (scp / gcloud compute scp / git clone), then:
pip install -r requirements.txt
# synchronous FedAvg (or --strategy fedprox --prox-mu 0.1)
python -m fedcloud.sync_server --address 0.0.0.0:8080 --clients 5 --rounds 30 \
       --test server_test.npz --out results/cloud_sync_fedavg.json
# asynchronous FedAsync
python -m fedcloud.async_server --port 8081 --clients 5 --rounds 30 \
       --test server_test.npz --out results/cloud_async.json
```
Docker alternative: `docker build -f deploy/Dockerfile -t fedcloud .`, then
`docker run -p 8080:8080 -v $PWD:/data -v $PWD/results:/results fedcloud`.

## 3. Client nodes (Colab / Kaggle / another VM)
Open `deploy/client_node.ipynb` in Colab or Kaggle. On Kaggle, go to Settings and turn **Internet on**,
which needs a phone-verified account. Upload the code zip and that node's `client_k.npz`, set
`SERVER_IP` and `CID`, and run all cells. In a shell, this is the equivalent:
```bash
pip install -r requirements.txt
python -m fedcloud.sync_client  --server <SERVER_IP>:8080 --cid 0 --data client_0.npz --no-emu
python -m fedcloud.async_client --server http://<SERVER_IP>:8081 --cid 0 --data client_0.npz --no-emu
```
`--no-emu` switches off the WAN emulator, so the measured latency and bandwidth come from the real cloud links.

## 4. Collect and plot
Copy the server's `results/cloud_*.json` into `results/` and run `python scripts/make_figures.py`.
Any file whose name starts with `cloud_` is added to the paper tables as a **real cross-cloud** row.

## Tips and gotchas
- The sync server waits until all `--clients` nodes have connected, so start the server first.
- Colab disconnects idle sessions. Keep the tab open; 30 rounds take about 2–5 minutes.
- Measure RTT from each client with `ping <SERVER_IP>` (or `curl -w "%{time_connect}"`) and report it.
- To reproduce the emulated straggler on a real node, drop `--no-emu` and pass `--profile-offset`.
