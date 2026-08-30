#!/usr/bin/env python3
"""The daily cycle. One invocation = one cycle. Runs locally now, Cloud Run job later.

  ingest -> extract -> rebuild grid -> diff vs previous state -> filter -> emit

  python agent_loop.py --replay-batch 10     # demo: replay real held-out papers
  python agent_loop.py                       # live: OpenAlex works updated since last run
  python agent_loop.py --replay-batch 10 --dry-run

State lives in agent_state.json (cycle number, last run, replay cursor). The previous
grid is copied into state_history/ before being overwritten.
"""

import json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_v3 as v3
import build_grid as bg
import grid_config as cfg
import diff_engine as de

STATE = HERE / "agent_state.json"
WATCH = HERE / "watch_config.json"
GRID = HERE / "grid.json"
HISTORY = HERE / "state_history"
EVENTS_LOG = HERE / "events_log.jsonl"
ALERT = HERE / "latest_alert.md"
EXTRACTIONS = HERE / "extractions.jsonl"
CORPUS_LIVE = HERE / "corpus_live.csv"
FUTURE = HERE / "corpus_future.csv"

# Files that constitute cycle state. Everything else in the image is immutable input.
def state_files():
    return [STATE, GRID, CORPUS_LIVE, EXTRACTIONS, EVENTS_LOG, WATCH, ALERT]

def gcs_bucket():
    """Bucket handle when STATE_BUCKET is set, else None -- local paths, unchanged."""
    name = os.environ.get("STATE_BUCKET")
    if not name:
        return None
    from google.cloud import storage
    return storage.Client().bucket(name)

def state_pull(bucket):
    """Cycle N must start from cycle N-1's state, which lives in the bucket, not the
    image. Missing blobs are fine on the very first run."""
    if bucket is None:
        return
    got = []
    for p in state_files():
        blob = bucket.blob(f"state/{p.name}")
        if blob.exists():
            blob.download_to_filename(str(p))
            got.append(p.name)
    print(f"  gcs: pulled {len(got)} state files from gs://{bucket.name}/state/"
          + (f" ({', '.join(got)})" if got else " (bucket empty -- first run)"))

def state_push(bucket, history_files):
    """Push cycle state plus this cycle's history entries. Last step of the cycle."""
    if bucket is None:
        return
    n = 0
    for p in state_files():
        if p.exists():
            bucket.blob(f"state/{p.name}").upload_from_filename(str(p))
            n += 1
    for p in history_files:
        if p.exists():
            bucket.blob(f"state_history/{p.name}").upload_from_filename(str(p))
            n += 1
    print(f"  gcs: pushed {n} files to gs://{bucket.name}/")

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load_env(path=HERE / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, val = line.split("=", 1)
            os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cycle": 0, "last_run": None, "replay_cursor": 0}

def load_watch(grid):
    """Watched cells + contradiction keys. Defaults to every no_evidence_found cell
    and the current contradiction set, then is hand-editable."""
    if WATCH.exists():
        return json.loads(WATCH.read_text())
    w = {
        "note": "Hand-editable. cells are [species, model, outcome_domain] triples.",
        "generated": now_iso(),
        "alert_event_types": [t for t, r in de.SEVERITY.items() if r <= 4],
        "gemini_usd_per_call": None,
        "cells": [[c["species"], c["model"], c["outcome_domain"]]
                  for c in grid["cells"] if c["status"] == "no_evidence_found"],
        "contradictions": [[c["model"], c["outcome_domain"], c["measure"]]
                           for c in grid.get("contradictions", [])
                           if c.get("kind") == "cross_paper"],
    }
    WATCH.write_text(json.dumps(w, indent=2))
    print(f"  wrote default {WATCH.name}: {len(w['cells'])} watched cells, "
          f"{len(w['contradictions'])} watched contradictions")
    return w

def standard_filter(df):
    def nonhuman(cell):
        return any(s.strip() and not s.strip().lower().startswith("human")
                   for s in str(cell or "").split(";"))
    return df[(df["is_primary_research"] == "True")
              & (df["mesh_species"].map(nonhuman))
              & (df["abstract"].str.strip() != "")]

def already_extracted():
    if not EXTRACTIONS.exists():
        return set()
    return {json.loads(l)["pmid"] for l in EXTRACTIONS.read_text().splitlines() if l.strip()}

def ingest_replay(n, cursor):
    """Next n eligible papers from the held-out set, in publication order.
    Real papers, replayed -- the corpus_future.csv pull is frozen the same way."""
    if not FUTURE.exists(): sys.exit(f"error: no {FUTURE.name} -- run build_corpus.py")
    df = pd.read_csv(FUTURE, dtype=str).fillna("")
    elig = standard_filter(df).sort_values(["publication_year", "openalex_id"])
    elig = elig.reset_index(drop=True)
    batch = elig.iloc[cursor:cursor + n]
    return batch, len(elig)

def ingest_live(since):
    """OpenAlex works updated since the last cycle, using the frozen corpus query."""
    import pyalex
    from pyalex import Works
    conf = json.loads((HERE / "corpus_config.json").read_text())
    pyalex.config.email = "gapfinder@example.com"
    seen, rows = set(), []
    for term in conf["terms"]:
        q = Works().search_filter(title_and_abstract=term).filter(
            from_publication_date=conf["date_range"]["from"])
        if since:
            q = q.filter(from_updated_date=since[:10])
        for page in q.paginate(per_page=200, n_max=conf["limits"]["max_works_per_term"]):
            for w in page:
                if w.get("id") and w["id"] not in seen:
                    seen.add(w["id"])
                    rows.append(dict(w))
    print(f"  live ingest: {len(rows)} works updated since {since or 'the beginning'}")
    print("  (live path needs build_corpus.parse_pubmed to fill the MeSH layer; "
          "the demo runs --replay-batch)")
    return pd.DataFrame(), 0

def extract(batch, cycle):
    """Frozen v3 schema, same cache and retry ladder. Returns (records, live_calls).
    Each record is stamped with the cycle that produced it, so downstream consumers do
    not have to infer recency from the event log."""
    done, records, live = already_extracted(), [], 0
    for _, r in batch.iterrows():
        if r["pmid"] in done:
            continue
        prompt = v3.EXTRACT_PROMPT.format(abstract=r["abstract"])
        was_cached = v3.cache_path("v3", prompt).exists()
        d = v3.gemini_json("v3", prompt, v3.ExtractionV3)
        if not was_cached:
            live += 1
        if d is None:
            print(f"  ! extraction gave up on {r['pmid']} -- stopping this cycle",
                  file=sys.stderr)
            break
        rec = v3.normalise(d)
        records.append({"pmid": r["pmid"], "openalex_id": r["openalex_id"],
                        "title": r["title"], "publication_year": r["publication_year"],
                        "cycle": cycle, **rec})
    return records, live

def rebuild(extra_corpus):
    papers = bg.load(extra_corpus=extra_corpus)
    cells, _ = bg.build(papers)
    return {"axes": {"species": cfg.SPECIES_AXIS, "model": cfg.MODEL_AXIS,
                     "outcome_domain": cfg.OUTCOME_AXIS},
            "papers_total": len(papers), "min_screened": cfg.MIN_SCREENED,
            "cells": cells, "adjacent_gaps": bg.adjacency(cells, papers),
            "contradictions": bg.contradictions(papers)}, papers

def select(events, watch):
    """Keep the high-severity events plus anything touching a watched cell."""
    types = set(watch.get("alert_event_types") or [])
    cells = {tuple(c) for c in watch.get("cells", [])}
    contras = {tuple(c) for c in watch.get("contradictions", [])}
    out = []
    for e in events:
        k = (e.get("species"), e["model"], e["outcome_domain"])
        why = []
        if e["event"] in types:
            why.append("severity")
        if k in cells:
            why.append("watched_cell")
        if "measure" in e and (e["model"], e["outcome_domain"], e["measure"]) in contras:
            why.append("watched_contradiction")
        if why:
            out.append({**e, "matched": why})
    return out

def title_map():
    """PMID -> title across every corpus the loop knows about. Events routinely name
    papers ingested in earlier cycles, so the current batch alone is not enough."""
    out = {}
    for path in (HERE / "corpus.csv", CORPUS_LIVE):
        if not Path(path).exists():
            continue
        d = pd.read_csv(path, dtype=str).fillna("")
        for _, r in d.iterrows():
            if r.get("pmid") and str(r.get("title", "")).strip():
                out.setdefault(r["pmid"], r["title"])
    return out

def render_alert(events, cycle, ingested, titles, ts=None):
    ts = ts or now_iso()
    if not events:
        return (f"# Cycle {cycle} — no material change\n\n"
                f"{ts} · {ingested} papers ingested · 0 events\n")
    by = {}
    for e in events:
        by.setdefault(e["event"], []).append(e)
    lines = [f"# Cycle {cycle} — {len(events)} events", "",
             f"{ts} · {ingested} papers ingested", ""]
    for t in sorted(by, key=lambda t: de.SEVERITY[t]):
        lines.append(f"## {t} ({len(by[t])})")
        for e in by[t]:
            loc = f"**{e.get('species')} / {e['model']} / {e['outcome_domain']}**"
            if "measure" in e:
                loc += f" / _{e['measure']}_"
            lines.append(f"- {loc}")
            lines.append(f"  - {e['note']}")
            lines.append(f"  - matched: {', '.join(e['matched'])}")
            for p in e["pmids"][:5]:
                t = titles.get(p) or "(title not in corpus)"
                lines.append(f"  - `{p}` {t[:88]}")
        lines.append("")
    return "\n".join(lines) + "\n"

def rerender(cycle):
    """Rebuild a past cycle's alert from events_log.jsonl. Read-only."""
    if not EVENTS_LOG.exists(): sys.exit("error: no events_log.jsonl")
    ev = [json.loads(l) for l in EVENTS_LOG.read_text().splitlines() if l.strip()]
    mine = [e for e in ev if e.get("cycle") == cycle]
    if not mine: sys.exit(f"error: no events logged for cycle {cycle}")
    ts = mine[0].get("ts")
    print(render_alert([{k: v for k, v in e.items() if k not in ("cycle", "ts")}
                        for e in mine], cycle, "?", title_map(), ts=ts))

def main():
    load_env()
    if "--rerender" in sys.argv:
        return rerender(int(sys.argv[sys.argv.index("--rerender") + 1]))
    dry = "--dry-run" in sys.argv
    n = int(sys.argv[sys.argv.index("--replay-batch") + 1]) if "--replay-batch" in sys.argv else 0
    bucket = gcs_bucket()
    state_pull(bucket)
    if not GRID.exists(): sys.exit("error: no grid.json -- run build_grid.py first")

    state = load_state()
    cycle = state["cycle"] + 1
    old_grid = json.loads(GRID.read_text())
    watch = load_watch(old_grid)
    t0 = time.time()
    print(f"=== cycle {cycle} ({'dry run' if dry else 'live'}) {now_iso()} ===")

    if n:
        batch, total = ingest_replay(n, state["replay_cursor"])
        print(f"  replay: papers {state['replay_cursor']}-"
              f"{state['replay_cursor']+len(batch)} of {total} eligible held-out")
    else:
        batch, total = ingest_live(state["last_run"])
    if len(batch) == 0:
        print("  nothing new to ingest")

    records, live_calls = extract(batch, cycle) if len(batch) else ([], 0)
    print(f"  extracted {len(records)} papers ({live_calls} live Gemini calls)")

    # stage the new rows so the grid can see them, then roll back if this is a dry run
    ex_backup = EXTRACTIONS.read_text() if EXTRACTIONS.exists() else ""
    live_backup = CORPUS_LIVE.read_text() if CORPUS_LIVE.exists() else None
    if records:
        with EXTRACTIONS.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        cols = pd.read_csv(HERE / "corpus.csv", nrows=0).columns
        add = batch[batch["pmid"].isin({r["pmid"] for r in records})]
        add = add.reindex(columns=cols, fill_value="")
        add.to_csv(CORPUS_LIVE, mode="a", header=not CORPUS_LIVE.exists(), index=False)

    titles = title_map()
    titles.update({r["pmid"]: r["title"] for r in records if r.get("title")})
    new_grid, papers = rebuild([CORPUS_LIVE])
    all_events, _, _ = de.diff(old_grid, new_grid)
    kept = select(all_events, watch)
    print(f"  grid: {old_grid['papers_total']} -> {new_grid['papers_total']} papers; "
          f"{len(all_events)} raw events, {len(kept)} after filtering")

    rate = watch.get("gemini_usd_per_call")
    cost = f"${live_calls * rate:.4f}" if rate else f"{live_calls} calls (no rate set)"

    if dry:
        if ex_backup or EXTRACTIONS.exists():
            EXTRACTIONS.write_text(ex_backup)
        if live_backup is None:
            CORPUS_LIVE.unlink(missing_ok=True)
        else:
            CORPUS_LIVE.write_text(live_backup)
        print(f"  dry run: state untouched (gemini_cache keeps any new responses)")
    else:
        HISTORY.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy(GRID, HISTORY / f"grid_{stamp}_cycle{cycle - 1}.json")
        GRID.write_text(json.dumps(new_grid, indent=2))
        with EVENTS_LOG.open("a") as fh:
            for e in kept:
                fh.write(json.dumps({"cycle": cycle, "ts": now_iso(), **e}) + "\n")
        md = render_alert(kept, cycle, len(records), titles)
        (HISTORY / f"alert_{stamp}.md").write_text(md)   # pairs with the grid snapshot
        ALERT.write_text(md)                             # pointer at the current cycle
        STATE.write_text(json.dumps({"cycle": cycle, "last_run": now_iso(),
                                     "replay_cursor": state["replay_cursor"] + len(batch)},
                                    indent=2))
        state_push(bucket, [HISTORY / f"grid_{stamp}_cycle{cycle - 1}.json",
                            HISTORY / f"alert_{stamp}.md"])

    types = {}
    for e in kept:
        types[e["event"]] = types.get(e["event"], 0) + 1
    detail = ", ".join(f"{k} {v}" for k, v in sorted(types.items())) or "none"
    print(f"CYCLE {cycle}: ingested {len(records)} papers | {len(kept)} events "
          f"({detail}) | gemini {cost} | {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
