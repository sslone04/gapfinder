#!/usr/bin/env python3
"""Run the v3 extraction over the full corpus.

Thin wrapper around probe_v3: reuses its schema, prompt, cache, rate limiter, retry
ladder and stop-on-failure behaviour, but reads corpus.csv instead of the probe CSVs
and streams results to extractions.jsonl.

  GEMINI_MODEL=gemini-3.5-flash-lite python run_extraction.py          # confirms first
  ... python run_extraction.py --yes                                   # no prompt
  ... python run_extraction.py --limit 20 --rate-gap 6                 # smaller/faster
"""

import json, os, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_v3 as v3          # schema, prompt, gemini_json, cache, retry ladder

OUT = HERE / os.environ.get("EXTRACT_OUT", "extractions.jsonl")
PROGRESS_EVERY = 25

def load_env(path=HERE / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, val = line.split("=", 1)
            os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))

def non_human(cell):
    return [s.strip() for s in str(cell or "").split(";")
            if s.strip() and not s.strip().lower().startswith("human")]

def eligible():
    """Rows passing the standard filter, from the frozen corpus plus anything the agent
    has ingested since (corpus_live.csv). No re-derivation -- flags only."""
    src = HERE / "corpus.csv"
    if not src.exists(): sys.exit("error: no corpus.csv -- run build_corpus.py first")
    frames = [pd.read_csv(src, dtype=str).fillna("")]
    live = HERE / "corpus_live.csv"
    if live.exists():
        frames.append(pd.read_csv(live, dtype=str).fillna(""))
    d = pd.concat(frames, ignore_index=True)
    keep = d[(d["is_primary_research"] == "True")
             & (d["mesh_species"].map(lambda c: bool(non_human(c))))
             & (d["abstract"].str.strip() != "")]
    return keep.drop_duplicates("openalex_id").reset_index(drop=True)

def arg(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

def main():
    load_env()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("error: set GEMINI_API_KEY (or GOOGLE_API_KEY), e.g. in .env")
    v3.RATE_GAP = arg("--rate-gap", float, v3.RATE_GAP)

    df = eligible()
    if (lim := arg("--limit", int, 0)):
        df = df.head(lim)
    prompts = [v3.EXTRACT_PROMPT.format(abstract=a) for a in df["abstract"]]
    live = sum(1 for q in prompts if not v3.cache_path("v3", q).exists())
    secs = live * v3.RATE_GAP

    print(f"model          : {v3.MODEL}")
    print(f"schema version : {v3.SCHEMA_VERSION}")
    print(f"papers         : {len(df)} passing the standard filter")
    print(f"live API calls : {live}  ({len(prompts) - live} already cached)")
    print(f"rate limit     : {v3.RATE_GAP:.0f}s between live calls")
    print(f"estimated time : {v3.fmt(secs)}  (excludes retry backoff)")
    print(f"output         : {OUT.name}")
    if not live:
        print("\nnothing to fetch -- every paper is cached")
    if "--yes" not in sys.argv:
        try:
            if input("\nproceed? [y/N] ").strip().lower() not in ("y", "yes"):
                sys.exit("aborted -- nothing spent")
        except EOFError:
            sys.exit("aborted: no tty for confirmation; pass --yes to run non-interactively")

    v3._expected = live
    t0, rows = time.time(), []
    with OUT.open("w") as fh:
        for i, ((_, r), q) in enumerate(zip(df.iterrows(), prompts), 1):
            d = v3.gemini_json("v3", q, v3.ExtractionV3)
            if d is None:                      # retries exhausted: stop, never emit a blank
                print(f"\n!! ABORTED after {i - 1}/{len(df)} papers -- call gave up "
                      f"(see the error above for the attempt count)")
                print(f"   failed PMID: {r['pmid'] or '(no pmid)'}  "
                      f"openalex={r['openalex_id']}")
                print(f"   {len(rows)} completed papers are written to {OUT.name} and "
                      f"cached; rerun to resume")
                fh.flush()
                sys.exit(1)
            rec = v3.normalise(d)
            rec = {"pmid": r["pmid"], "openalex_id": r["openalex_id"],
                   "title": r["title"], "publication_year": r["publication_year"], **rec}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            rows.append(rec)
            if i % PROGRESS_EVERY == 0 or i == len(df):
                el = time.time() - t0
                left = (el / i) * (len(df) - i)
                print(f"[{i}/{len(df)}] {v3.fmt(el)} elapsed, ~{v3.fmt(left)} left",
                      flush=True)

    n = len(rows) or 1
    print(f"\n=== per-field fill rate (n={len(rows)}) ===")
    for f in v3.FIELDS:
        k = sum(1 for rec in rows if v3.filled(f, rec))
        print(f"  {f:<16} {k:>4}/{len(rows)} ({100.0 * k / n:>5.1f}%)")
    print(f"\nwrote {OUT.name}")

if __name__ == "__main__":
    main()
