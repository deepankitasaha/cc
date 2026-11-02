#!/usr/bin/env bash
# Full experiment matrix used in the paper (single host, WAN emulation on). ~1 h on 2 vCPUs.
set -u
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2}"
R="${ROUNDS:-30}"
for s in iid dir1.0 dir0.5 dir0.1; do [ -f data/parts_$s/client_0.npz ] || python3 scripts/prepare_data.py --scheme $s; done
run() { local out="results/$1.json"; shift; [ -f "$out" ] && { echo "skip $out"; return; }; python3 scripts/run_experiment.py "$@" --out "$out"; }
for seed in $SEEDS; do
  [ -f results/centralized_s$seed.json ] || python3 scripts/run_centralized.py --parts data/parts_iid --epochs $R --seed $seed
  for sc in iid dir1.0 dir0.5 dir0.1; do
    run sync_fedavg_${sc}_E1_s$seed   --mode sync  --strategy fedavg  --scheme $sc --rounds $R --seed $seed
    run async_fedasync_${sc}_E1_s$seed --mode async --scheme $sc --rounds $R --seed $seed
  done
  for sc in dir0.5 dir0.1; do
    run sync_fedprox_${sc}_E1_s$seed  --mode sync --strategy fedprox --prox-mu 0.1 --scheme $sc --rounds $R --seed $seed
  done
  # straggler-placement ablation: fraud-rich client 0 becomes the slowest node
  run async_fedasync_dir0.1_E1_s${seed}_po4 --mode async --scheme dir0.1 --rounds $R --seed $seed --profile-offset 4
  # local-epochs vs communication trade-off
  for E in 2 5; do
    run sync_fedavg_dir0.5_E${E}_s$seed --mode sync --strategy fedavg --scheme dir0.5 --rounds $R --epochs $E --seed $seed
  done
done
echo ALL_DONE
