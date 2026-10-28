# Running BlockSwarm on a GCP VM

A runbook for running episodes on a private Google Cloud VM reached over IAP SSH. Fill in
your own values once per terminal (or keep a filled-in copy as `docs/vm.local.md`, which is
gitignored):

```bash
export PROJECT=your-gcp-project
export ZONE=us-central1-a
export VM=your-vm-name
```

| Setting | Assumed setup |
| --- | --- |
| OS / Python | Ubuntu with Python 3.10+ (stdlib only, no installs needed) |
| Access | Private VM, IAP SSH only, outbound internet via Cloud NAT |
| Models | OpenRouter, key in the host's `.env` (`OPENROUTER_API_KEY`), `chmod 600` |

**Sizing.** BlockSwarm needs very little of the machine. Each episode is a small Python
process (megabytes); the agents are API calls. The limit is OpenRouter rate limits and
spend, not CPU or RAM. A small machine such as `n2-standard-8` is enough for development
runs; a 64-vCPU machine can run hundreds of episodes in parallel. A larger VM would only
matter for hosting open-weight models locally, which needs GPUs more than RAM.

**Stop the VM when you're done.** Closing SSH or detaching tmux leaves it running and
billing.

## 1. Start and connect (from your machine)

```bash
gcloud compute instances describe $VM --project=$PROJECT --zone=$ZONE --format='value(status)'
gcloud compute instances start $VM --project=$PROJECT --zone=$ZONE
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --tunnel-through-iap
```

## 2. Get the code onto the VM

Clone from GitHub if the VM can reach it, or send a git bundle over IAP (from your
machine, in the repo folder):

```bash
git bundle create /tmp/blockswarm.bundle main
gcloud compute scp /tmp/blockswarm.bundle $VM:~/ --project=$PROJECT --zone=$ZONE --tunnel-through-iap
```

Then on the VM:

```bash
git clone ~/blockswarm.bundle blockswarm && cd blockswarm
python3 -m unittest discover -s tests
python3 scripts/run_demo.py
```

Later updates: create a new bundle, copy it, then `git pull ~/blockswarm.bundle main` on the VM.
Never copy your local `.env`; create it on the VM by hand.

## 3. Long runs

Use tmux so a run survives disconnecting:

```bash
tmux new -s blockswarm
```

Detach with Ctrl-B then D; reattach with `tmux attach -t blockswarm`.

## 4. Watch a run live in 3D (through a private tunnel)

The VM has no public address. A tunnel over IAP forwards the viewer's port to your
machine; only you can open it.

**On the VM** (in a second tmux window, next to the running episode), serve the viewer and
follow the episode's log. The log file may not exist yet; the viewer waits for it:

```bash
python3 scripts/serve_viewer.py runs/EPISODE.jsonl --follow --no-open --port 8765
```

**On your machine**, open the tunnel and leave this terminal open while you watch:

```bash
gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --tunnel-through-iap -- -N -L 8765:localhost:8765
```

Then open the URL the VM printed (it starts with `http://localhost:8765/viewer/index.html`)
in your browser. A **LIVE** badge shows while the episode runs. You can scrub back to
earlier moments and click "Back to live" to return. If the VM printed a port other than
8765, use that port in both places.

## 5. Bring logs back and replay them

```bash
gcloud compute scp --recurse $VM:~/blockswarm/runs ./runs-vm --project=$PROJECT --zone=$ZONE --tunnel-through-iap
```

```bash
python3 scripts/serve_viewer.py runs-vm/demo/open_house_exact.jsonl
```

## 6. Stop the VM

```bash
gcloud compute instances stop $VM --project=$PROJECT --zone=$ZONE
gcloud compute instances describe $VM --project=$PROJECT --zone=$ZONE --format='value(status)'
```

Confirm it says `TERMINATED`. If the VM is shared, check first that no one else is using it.
