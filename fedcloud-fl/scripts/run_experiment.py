"""Launch one complete FL experiment as SEPARATE OS PROCESSES (1 server + N clients) talking
over real sockets (gRPC for Flower/sync, HTTP for FedAsync), with per-client WAN emulation.

    python scripts/run_experiment.py --mode sync  --strategy fedavg  --scheme dir0.1 --seed 0
    python scripts/run_experiment.py --mode sync  --strategy fedprox --scheme dir0.1 --seed 0
    python scripts/run_experiment.py --mode async --scheme dir0.1 --seed 0

For a real multi-cloud deployment run the server/client modules directly on each node
(see deploy/CLOUD_DEPLOYMENT.md); this script is the single-host reproducible harness.
"""
import argparse
import zlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--scheme", default="iid")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--prox-mu", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--profile-offset", type=int, default=0,
                    help="rotate WAN/compute profiles: client k gets profile (k+offset) mod 5")
    ap.add_argument("--no-emu", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    parts = os.path.join(ROOT, "data", f"parts_{a.scheme}")
    if not os.path.exists(os.path.join(parts, "client_0.npz")):
        subprocess.check_call([sys.executable, "scripts/prepare_data.py", "--scheme", a.scheme,
                               "--clients", str(a.clients)], cwd=ROOT)
    name = f"{a.mode}_{a.strategy if a.mode == 'sync' else 'fedasync'}_{a.scheme}_E{a.epochs}_s{a.seed}" + (f"_po{a.profile_offset}" if a.profile_offset else "")
    out = a.out or os.path.join(ROOT, "results", name + ".json")
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    port = a.port or (18000 + (zlib.crc32(name.encode()) % 2000))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    py = sys.executable
    test = os.path.join(parts, "server_test.npz")
    emu = ["--no-emu"] if a.no_emu else ["--profile-offset", str(a.profile_offset)]

    if a.mode == "sync":
        srv = [py, "-m", "fedcloud.sync_server", "--address", f"127.0.0.1:{port}", "--rounds", str(a.rounds),
               "--clients", str(a.clients), "--strategy", a.strategy, "--prox-mu", str(a.prox_mu),
               "--epochs", str(a.epochs), "--test", test, "--seed", str(a.seed), "--out", out,
               "--tag", a.scheme]
        cli = lambda k: [py, "-m", "fedcloud.sync_client", "--server", f"127.0.0.1:{port}", "--cid", str(k),
                         "--data", os.path.join(parts, f"client_{k}.npz")] + emu
    else:
        srv = [py, "-m", "fedcloud.async_server", "--host", "127.0.0.1", "--port", str(port),
               "--rounds", str(a.rounds), "--clients", str(a.clients), "--alpha", str(a.alpha),
               "--epochs", str(a.epochs), "--test", test, "--seed", str(a.seed), "--out", out, "--tag", a.scheme]
        cli = lambda k: [py, "-m", "fedcloud.async_client", "--server", f"http://127.0.0.1:{port}",
                         "--cid", str(k), "--data", os.path.join(parts, f"client_{k}.npz"),
                         "--epochs", str(a.epochs), "--seed", str(a.seed)] + emu

    logdir = os.path.join(ROOT, "results", "logs")
    print(f"=== {name} (port {port})", flush=True)
    t0 = time.time()
    s = subprocess.Popen(srv, cwd=ROOT, env=env, stdout=open(os.path.join(logdir, name + "_server.log"), "w"),
                         stderr=subprocess.STDOUT)
    time.sleep(3)
    cs = [subprocess.Popen(cli(k), cwd=ROOT, env=env, stderr=subprocess.STDOUT,
                           stdout=open(os.path.join(logdir, f"{name}_client{k}.log"), "w"))
          for k in range(a.clients)]
    rc = s.wait()
    for c in cs:
        try:
            c.wait(timeout=30)
        except subprocess.TimeoutExpired:
            c.kill()
    print(f"=== {name} done rc={rc} in {time.time()-t0:.1f}s -> {out}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
