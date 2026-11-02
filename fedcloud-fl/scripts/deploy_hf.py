"""Publish the FedAsync server to a Hugging Face Docker Space and the client partitions to a dataset repo.

    pip install huggingface_hub
    HF_TOKEN=hf_... python scripts/deploy_hf.py --user <hf-username> --scheme dir0.5

Space   <user>/fedcloud-server       Dockerfile + app.py + fedcloud/ + server_test.npz (the server only)
Dataset <user>/fedcloud-client-data  client_k.npz, one file per client node
Re-running updates both repos in place; the Space rebuilds automatically.
"""
from __future__ import annotations

import argparse
import glob
import os

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--scheme", default="dir0.5", help="partition folder data/parts_<scheme>/")
    ap.add_argument("--space", default="fedcloud-server")
    ap.add_argument("--dataset", default="fedcloud-client-data")
    a = ap.parse_args(argv)
    api = HfApi(token=os.environ["HF_TOKEN"])
    parts = os.path.join(ROOT, "data", f"parts_{a.scheme}")

    space = f"{a.user}/{a.space}"
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    ops = [CommitOperationAdd(os.path.basename(f), f) for f in glob.glob(os.path.join(ROOT, "deploy", "hf_space", "*"))]
    ops += [CommitOperationAdd(f"fedcloud/{os.path.basename(f)}", f) for f in glob.glob(os.path.join(ROOT, "fedcloud", "*.py"))]
    ops.append(CommitOperationAdd("server_test.npz", os.path.join(parts, "server_test.npz")))
    api.create_commit(space, ops, commit_message=f"Deploy FedAsync server ({a.scheme})", repo_type="space")
    print(f"space   https://huggingface.co/spaces/{space}")
    print(f"server  https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space")

    ds = f"{a.user}/{a.dataset}"
    api.create_repo(ds, repo_type="dataset", exist_ok=True)
    ops = [CommitOperationAdd(os.path.basename(f), f) for f in sorted(glob.glob(os.path.join(parts, "client_*.npz")))]
    ops.append(CommitOperationAdd("partition_report.json", os.path.join(parts, "partition_report.json")))
    api.create_commit(ds, ops, commit_message=f"Client partitions ({a.scheme})", repo_type="dataset")
    print(f"dataset https://huggingface.co/datasets/{ds}")


if __name__ == "__main__":
    main()
