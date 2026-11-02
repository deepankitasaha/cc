# Demo video: narration script (about 4 minutes)

`demo.mp4` is a captioned walkthrough of a real captured run, and it has no voice track. If your
evaluator expects narration, either read this script over the video or record your own screen
while running the same commands. Each of the four presenters takes about one minute.

| Time | On screen | Speaker | Narration |
|---|---|---|---|
| 0:00 | Title | Pranav | "We are presenting *Federated Learning Across Cloud Nodes*. We built an FL pipeline in which five cloud nodes train one fraud-detection model without ever sharing their raw transaction data." |
| 0:06 | Problem & architecture | Pranav | "Two problems appear in real deployments. Each bank's data is different, which is called non-IID data, and cloud links are slow and uneven. Our system has one aggregation server and five client nodes. Only model weights, about 17 kilobytes, cross the network." |
| 0:20 | Terminal: `prepare_data.py` | Harshini | "We split the Kaggle credit-card fraud dataset over five clients using a Dirichlet distribution. Here, at alpha 0.1, client 0 receives 393 of the 394 frauds and client 3 receives 80 percent of the data but no fraud at all. Each node gets only its own file." |
| 0:35 | Partition chart | Harshini | "We measure the skew with the Hellinger distance, as recommended by the non-IID survey we reviewed." |
| 0:47 | Six-pane live sync run | Nandini | "This is synchronous FedAvg on Flower. The server on the left sends the model, all five clients train locally, and the server averages. Watch node-E, our slow VM: every round waits for it. The yellow curve is the global model's AUPRC on the server's test set." |
| 1:15 | Six-pane live async run | Deepankita | "This is asynchronous FedAsync. Nobody waits. Each update is merged as soon as it arrives and discounted by its staleness. The fast nodes contribute many more updates than node-E." |
| 1:35 | Results: non-IID | Harshini | "Results: with IID or moderate skew, federated learning matches centralised training at an AUPRC of about 0.83. Under extreme skew, FedAvg collapses to near-zero AUPRC, yet its accuracy is still 99.8 percent. That is why we never report accuracy alone. FedProx does not fix it." |
| 2:05 | Results: sync vs async | Deepankita | "Async finishes the same number of updates about 2.8 times faster while sending the same number of megabytes. The bottom row shows that stale updates are worth less per megabyte." |
| 2:30 | Results: straggler placement | Deepankita | "Our key finding: in the extreme case, async recovers a usable model when the fraud-rich node is fast. If we make that same node the straggler, accuracy collapses. Data skew and system speed have to be studied together." |
| 2:55 | Results: local epochs | Nandini | "More local epochs mean fewer messages to reach the target, but longer rounds, because the straggler's compute time grows." |
| 3:15 | Deployment terminal | Pranav | "The same code deploys to real clouds. The server runs on a free-tier VM with ports 8080 and 8081 open, and each client runs our Colab or Kaggle notebook with `--no-emu`. Docker Compose starts the whole cluster with one command." |
| 3:35 | Thank you | Pranav | "Code, logs and the LaTeX paper are in our submission. Thank you." |

**Recording your own:** run `python scripts/run_experiment.py --mode sync --scheme dir0.5 --rounds 10`
with the six log files in `results/logs/` open in a split terminal (`tail -f`), or use
`python scripts/demo_capture.py` to save a timestamped capture.
