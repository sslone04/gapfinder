#!/usr/bin/env python3
"""Read-only dashboard over the gapfinder state bucket.

Serves the current grid, the event/alert feed, and a co-occurrence network built from
extractions.jsonl. There are no write endpoints: the agent job owns all state.

  STATE_BUCKET=gapfinder-state uvicorn main:app --port 8080
  STATE_DIR=/path/to/local/state uvicorn main:app --port 8080   # offline dev
"""

import csv, io, json, os, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid_config as cfg

BUCKET = os.environ.get("STATE_BUCKET")
STATE_DIR = os.environ.get("STATE_DIR")
CACHE_TTL = 60
STALE_AFTER_H = 26

STATE_FILES = ["agent_state.json", "grid.json", "events_log.jsonl",
               "latest_alert.md", "extractions.jsonl", "watch_config.json"]

app = FastAPI(title="gapfinder dashboard", docs_url=None, redoc_url=None)
_cache = {}          # name -> (fetched_at, text)
_client = None

def client():
    """On Cloud Run the metadata server supplies ADC. Locally, point
    GOOGLE_APPLICATION_CREDENTIALS at a key file, or use STATE_DIR for offline dev."""
    global _client
    if _client is None:
        from google.cloud import storage
        try:
            _client = storage.Client()
        except Exception as e:
            raise RuntimeError(
                "no Google credentials. On Cloud Run this resolves automatically; "
                "locally set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json, or "
                f"run against a local copy with STATE_DIR=/path/to/state. ({e})") from e
    return _client

def read(name, missing_ok=True):
    """Fetch one state file, cached for CACHE_TTL seconds."""
    hit = _cache.get(name)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    text = None
    try:
        if STATE_DIR:
            p = Path(STATE_DIR) / name
            text = p.read_text() if p.exists() else None
        elif BUCKET:
            blob = client().bucket(BUCKET).blob(f"state/{name}")
            text = blob.download_as_text() if blob.exists() else None
        else:
            raise RuntimeError("set STATE_BUCKET or STATE_DIR")
    except Exception as e:
        if not missing_ok:
            raise HTTPException(502, f"cannot read {name}: {e}")
        print(f"  ! read {name}: {e}", file=sys.stderr)
    _cache[name] = (time.time(), text)
    return text

def read_json(name):
    t = read(name)
    return json.loads(t) if t else None

def read_jsonl(name):
    t = read(name)
    return [json.loads(l) for l in (t or "").splitlines() if l.strip()]

_corpus = {"at": 0.0, "rows": None}

def corpus_index():
    """pmid -> corpus row. corpus.csv is frozen and baked into the image; corpus_live.csv
    accumulates papers the agent has ingested since and lives in the state bucket."""
    if _corpus["rows"] is not None and time.time() - _corpus["at"] < CACHE_TTL:
        return _corpus["rows"]
    rows = {}
    local = HERE / "corpus.csv"
    if local.exists():
        with local.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("pmid"):
                    rows[r["pmid"]] = r
    live = read("corpus_live.csv")
    if live:
        for r in csv.DictReader(io.StringIO(live)):
            if r.get("pmid"):
                rows[r["pmid"]] = r          # live wins: it is the newer ingest
    _corpus.update({"at": time.time(), "rows": rows})
    return rows

def species_of(mesh_cell):
    low = norm(mesh_cell)
    for name, keys in cfg.SPECIES_RULES:
        if any(hit(k, low) for k in keys):
            return name
    return cfg.SPECIES_FALLBACK

# --- canonical model, lifted from build_grid so the dashboard needs no pandas -----
_WORDLIKE = re.compile(r"^[a-z0-9].*[a-z0-9]$|^[a-z0-9]$")

def norm(s):
    t = re.sub(r"\s+", " ", str(s or "").lower())
    return re.sub(r"\s*([/\-])\s*", r"\1", t).strip()

def hit(pattern, blob):
    pat = norm(pattern)
    if _WORDLIKE.match(pat) and not re.search(r"[^a-z0-9 ]", pat.replace("-", "")):
        return re.search(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", blob) is not None
    return pat in blob

def model_of(dm):
    low = norm(" | ".join(dm.get("components") or []))
    for name, rule in cfg.MODEL_RULES:
        if "any" in rule and any(hit(k, low) for k in rule["any"]):
            return name
        if "all" in rule and all(any(hit(k, low) for k in g) for g in rule["all"]):
            return name
    if dm.get("model_type") == "genetic_strain" or any(hit(h, low) for h in cfg.GENETIC_HINTS):
        return cfg.MODEL_FALLBACK_GENETIC
    return cfg.MODEL_FALLBACK_OTHER

# --- minimal markdown -> html (no dependency; the alert format is known) ----------
def md_to_html(md):
    out, in_list = [], False
    for line in (md or "").splitlines():
        s = line.rstrip()
        esc = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
        esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"_([^_]+)_", r"<em>\1</em>", esc)
        stripped = esc.strip()
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if stripped.startswith("## "):
            out.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("# "):
            out.append(f"<h2>{stripped[2:]}</h2>")
        elif stripped:
            out.append(f"<p>{stripped}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)

# --- endpoints --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()

@app.get("/api/status")
def status():
    st = read_json("agent_state.json") or {}
    grid = read_json("grid.json") or {}
    age_h, stale = None, None
    last = st.get("last_run")
    if last:
        try:
            from datetime import datetime, timezone
            t = datetime.fromisoformat(last)
            age_h = (datetime.now(timezone.utc) - t).total_seconds() / 3600
            stale = age_h > STALE_AFTER_H
        except Exception:
            pass
    return {"cycle": st.get("cycle"), "last_run": last,
            "replay_cursor": st.get("replay_cursor"),
            "papers_total": grid.get("papers_total"),
            "age_hours": round(age_h, 1) if age_h is not None else None,
            "stale": stale, "stale_after_hours": STALE_AFTER_H,
            "source": f"gs://{BUCKET}/state/" if BUCKET else f"file://{STATE_DIR}"}

@app.get("/api/grid")
def grid():
    g = read_json("grid.json")
    if not g:
        raise HTTPException(404, "grid.json not in state")
    return g

@app.get("/api/events")
def events(limit: int = Query(500, ge=1, le=5000)):
    """Events, newest first, enriched with both sides of the story so the browser never
    has to fetch per event: the contradiction's two camps, the papers whose silence
    defined a filled gap, and the characterisation papers a first intervention joins."""
    ev = read_jsonl("events_log.jsonl")
    ev.reverse()
    ev = ev[:limit]
    grid = read_json("grid.json") or {}
    corpus = corpus_index()
    recs = {r["pmid"]: r for r in read_jsonl("extractions.jsonl") if r.get("pmid")}
    cells = {(c["species"], c["model"], c["outcome_domain"]): c for c in grid.get("cells", [])}
    contra = {(c["model"], c["outcome_domain"], c["measure"]): c
              for c in grid.get("contradictions", [])}
    pair = defaultdict(set)                        # species x model -> every pmid there
    for c in grid.get("cells", []):
        for p in c.get("pmids") or []:
            pair[(c["species"], c["model"])].add(p)

    # A paper is NEW to an event if its cycle stamp says so; records written before
    # stamping fall back to "first cycle this pmid was ever mentioned in the log".
    first_seen = {}
    for e in sorted(ev, key=lambda e: e.get("cycle", 0)):
        for p in e.get("pmids") or []:
            first_seen.setdefault(p, e.get("cycle"))

    refs, out = set(), []
    for e in ev:
        pm = e.get("pmids") or []
        cyc = e.get("cycle")
        new = [p for p in pm
               if (recs[p].get("cycle") == cyc if p in recs and recs[p].get("cycle") is not None
                   else first_seen.get(p) == cyc)]
        ctx = {}
        ekey = (e.get("species"), e.get("model"), e.get("outcome_domain"))
        if e["event"] in ("contradiction_new", "contradiction_resolved"):
            c = contra.get((e.get("model"), e.get("outcome_domain"), e.get("measure")))
            if c:
                ctx["camps"] = {"improved": c.get("improved_pmids", []),
                                "worsened": c.get("worsened_pmids", []),
                                "both_ways": c.get("both_ways_pmids", []),
                                "kind": c.get("kind")}
        elif e["event"] == "gap_filled":
            cell = cells.get(ekey) or {}
            others = sorted(pair.get((e.get("species"), e.get("model")), set()) - set(pm))
            ctx["screened_context"] = others[:60]      # the silence that made it a gap
            ctx["screened_k"] = cell.get("screened_k")
            ctx["screened_total"] = cell.get("screened_total")
        elif e["event"] == "first_interventional":
            cell = cells.get(ekey) or {}
            prior = [p for p in (cell.get("pmids") or []) if p not in pm]
            ctx["characterization_pmids"] = [p for p in prior if not (recs.get(p) or {}).get("arms")]
            ctx["other_interventional_pmids"] = [p for p in prior if (recs.get(p) or {}).get("arms")]
        for v in ctx.values():
            if isinstance(v, list):
                refs.update(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, list):
                        refs.update(vv)
        refs.update(pm)
        out.append({**e, "new_pmids": new, "context": ctx})

    titles = {p: ((corpus.get(p) or {}).get("title") or (recs.get(p) or {}).get("title") or "")
              for p in refs}
    return {"n_total": len(read_jsonl("events_log.jsonl")), "events": out, "titles": titles}

@app.get("/api/alert")
def alert():
    md = read("latest_alert.md") or "# no alert yet"
    return {"markdown": md, "html": md_to_html(md)}

@app.get("/api/paper/{pmid}")
def paper(pmid: str):
    """Everything we hold on one paper. We serve the abstract we ingested; full text is
    a link out, because most of it is paywalled and was never in our data."""
    row = corpus_index().get(pmid)
    extraction = next((r for r in read_jsonl("extractions.jsonl") if r.get("pmid") == pmid), None)
    if row is None and extraction is None:
        raise HTTPException(404, f"no paper {pmid} in corpus or extractions")
    row = row or {}
    grid = read_json("grid.json") or {}
    cells = [{"species": c["species"], "model": c["model"],
              "outcome_domain": c["outcome_domain"], "status": c["status"],
              "denominator": c["denominator"]}
             for c in grid.get("cells", []) if pmid in (c.get("pmids") or [])]
    doi = (row.get("doi") or "").strip()
    return {
        "pmid": pmid,
        "title": row.get("title") or (extraction or {}).get("title") or "",
        "year": row.get("publication_year") or (extraction or {}).get("publication_year"),
        # Journal is not in corpus.csv -- build_corpus never captured it. Explicitly null
        # rather than guessed, so the UI can omit the field instead of inventing one.
        "journal": None,
        "abstract": row.get("abstract") or "",
        "abstract_source": row.get("abstract_source") or "",
        "abstract_chars": row.get("abstract_chars") or "",
        "species_mesh": row.get("species_mesh") or "",
        "doi": doi,
        "doi_url": doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else ""),
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "extraction": extraction,
        "grid_cells": cells,
    }

def parse_weights(spec):
    w = {"model": 2.0, "entity": 1.0, "domain": 1.0, "mechanism": 1.0, "time": 0.5}
    for part in (spec or "").split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            k = k.strip()
            if k in w:
                try:
                    w[k] = float(v)
                except ValueError:
                    pass
    return w

@app.get("/api/network")
def network(weights: str = Query(None, description="model:2,entity:1,domain:1,..."),
            max_edges: int = Query(4000, ge=100, le=20000),
            min_shared: int = Query(2, ge=1)):
    """Nodes = papers. Edges = shared model / entities / outcome domains / mechanisms /
    time-point phases. Per-type shared counts are returned so the client can re-weight
    sliders without another round trip."""
    w = parse_weights(weights)
    recs = read_jsonl("extractions.jsonl")
    if not recs:
        raise HTTPException(404, "extractions.jsonl not in state")

    ev = read_jsonl("events_log.jsonl")
    st = read_json("agent_state.json") or {}
    latest_cycle = max([st.get("cycle") or 0] + [e.get("cycle", 0) for e in ev] +
                       [r.get("cycle", 0) or 0 for r in recs])
    # Records written since the cycle stamp landed carry it; older ones do not, so for
    # those we fall back to "named in one of the latest cycle's events".
    inferred = {p for e in ev if e.get("cycle") == latest_cycle for p in e.get("pmids", [])}
    n_stamped = sum(1 for r in recs if r.get("cycle") is not None)

    def is_new(r, pid):
        if r.get("cycle") is not None:
            return r["cycle"] == latest_cycle
        return pid in inferred

    corpus = corpus_index()
    nodes, feats = [], {}
    for r in recs:
        pid = r.get("pmid") or r.get("openalex_id")
        dm = r.get("disease_model") or {}
        f = {
            "model": {model_of(dm)},
            "entity": {norm(e).upper() for e in (r.get("entities") or []) if norm(e)},
            "domain": {o.get("domain") for o in (r.get("outcomes") or []) if o.get("domain")},
            "mechanism": {norm(m) for m in (r.get("mechanisms") or []) if norm(m)},
            "time": {t.get("phase") for t in (r.get("time_points") or []) if t.get("phase")},
        }
        feats[pid] = f
        doms = [o.get("domain") for o in (r.get("outcomes") or []) if o.get("domain")]
        dominant = Counter(doms).most_common(1)[0][0] if doms else "none"
        nodes.append({"id": pid, "title": (r.get("title") or "")[:140],
                      "year": r.get("publication_year"), "model": model_of(dm),
                      "hf_class": cfg.HF_CLASS.get(model_of(dm), "unclassified"),
                      "species": species_of((corpus.get(pid) or {}).get("species_mesh", "")),
                      "dominant_domain": dominant,
                      "n_outcomes": len(r.get("outcomes") or []),
                      "interventional": bool(r.get("arms")),
                      "cycle": r.get("cycle"),
                      "new_this_cycle": is_new(r, pid)})

    # invert each feature type, then only score pairs that actually co-occur
    pair = defaultdict(lambda: defaultdict(int))
    for ftype in ("model", "entity", "domain", "mechanism", "time"):
        buckets = defaultdict(list)
        for pid, f in feats.items():
            for v in f[ftype]:
                buckets[v].append(pid)
        for v, members in buckets.items():
            if len(members) > 120:      # ubiquitous terms ("cardiac") say nothing
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    pair[(a, b)][ftype] += 1

    edges = []
    for (a, b), shared in pair.items():
        total = sum(shared.values())
        if total < min_shared:
            continue
        edges.append({"source": a, "target": b, "shared": dict(shared),
                      "total_shared": total,
                      "weight": round(sum(w[k] * v for k, v in shared.items()), 3)})
    edges.sort(key=lambda e: -e["total_shared"])
    truncated = len(edges) > max_edges
    edges = edges[:max_edges]

    return JSONResponse({
        "weights": w, "latest_cycle": latest_cycle,
        "n_nodes": len(nodes), "n_edges": len(edges),
        "n_new_this_cycle": sum(1 for n in nodes if n["new_this_cycle"]),
        "truncated": truncated, "min_shared": min_shared,
        "n_stamped": n_stamped, "n_unstamped": len(recs) - n_stamped,
        "note": ("new_this_cycle uses each record's cycle stamp; records written before "
                 "stamping fall back to 'named in a latest-cycle event'"),
        "nodes": nodes, "edges": edges})
