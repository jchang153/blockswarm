# Deploying the BlockSwarm replay site

`scripts/export_site.py` turns episode logs into a plain static folder (relative paths, no
server code). The pages read `site-config.json` at load; its `data_base_url` says where
`episodes.json` and the `<id>.jsonl.gz` logs live (default `./data/`). This file is not copied
into the build.

## Layout A: pages and data together (simplest)

```bash
python3 scripts/export_site.py runs/<batch>/*.jsonl --out site
python3 -m http.server -d site 8000        # preview at http://localhost:8000/
```

Upload the whole `site/` folder to any static host (Cloudflare Pages, a Hugging Face *static*
Space, GitHub Pages, S3, ...). Result:

```
site/
  index.html  episode.html  about.html  site-config.json   ({"data_base_url": "./data/", ...})
  css/  js/  img/
  data/episodes.json  data/<id>.jsonl.gz
```

## Layout B: pages on one host, data in a public Hugging Face dataset

```bash
# 1. data only: episodes.json + logs at the folder root
python3 scripts/export_site.py runs/<batch>/*.jsonl --data-only --out site_data
#    upload the contents of site_data/ to the dataset repo, e.g.
#    huggingface-cli upload <user>/<dataset> site_data . --repo-type dataset

# 2. pages only, pointing at the dataset
python3 scripts/export_site.py --site-only \
  --data-base-url https://huggingface.co/datasets/<user>/<dataset>/resolve/main/ --out site
#    deploy site/ to the static host
```

Notes:
- `resolve/main/...` redirects to a CDN; `fetch()` follows redirects and the responses allow
  cross-origin reads, so no proxy is needed. If the data sits in a subfolder of the dataset, add it
  to the URL (`.../resolve/main/blockswarm/`).
- The gallery downloads only `episodes.json` (which includes each run's final blocks for the
  thumbnails); a replay page downloads one `.jsonl.gz`.
- Logs are decompressed in the browser with `DecompressionStream('gzip')`, after checking the gzip
  magic bytes, so it does not matter whether the host also sends `Content-Encoding: gzip`.
- To update the data without redeploying the pages, re-run step 1 and upload again: pages read
  `episodes.json` on every visit. Some CDNs cache for a few minutes.
- To pin a version, use a commit hash instead of `main` in the base URL.

## Before publishing

- The default export drops `model_call` records (raw model inputs/outputs) and any unknown record
  types; `--keep-model-calls` keeps them. `--drop-notes` also removes agents' private note text.
  Chat, including private whispers, is always published.
- Check the licence and the model providers' terms for publishing model outputs, and fill in the
  TODOs on `about.html`.
- Per-run titles/models/notes: put `<log>.meta.json` next to a log, e.g.
  `{"title": "Three agents, scarce blocks", "model": "provider/model-name", "notes": "..."}`;
  `{"hidden": true}` leaves a run out.
