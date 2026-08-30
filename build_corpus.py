#!/usr/bin/env python3
"""Production HFpEF corpus ingestion.

OpenAlex (frozen query in corpus_config.json) -> PubMed XML -> corpus.csv.
Writes every work it finds and flags them; no filtering happens here.

Parsing logic is lifted, not imported, from probe_ingest.py / probe_mesh.py: those
are throwaway probes and this must not break when they are edited or deleted.

  python build_corpus.py            # main + held-out future pull
  python build_corpus.py --main-only
"""

import json, os, re, sys, time
from collections import Counter
from hashlib import md5
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import pyalex
from pyalex import Works
from Bio import Entrez

HERE = Path(__file__).resolve().parent
CACHE = HERE / "raw_cache"
CACHE.mkdir(exist_ok=True)
CONFIG = HERE / "corpus_config.json"
EMAIL = "gapfinder@example.com"

DEFAULT_CONFIG = {
    "version": "1.0",
    "description": "Frozen HFpEF corpus query. Edit deliberately; changing this "
                   "changes what the corpus means.",
    "terms": [
        "HFpEF animal model",
        "HFpEF mouse model",
        "HFpEF rat model",
        "heart failure with preserved ejection fraction mouse",
        "heart failure with preserved ejection fraction rat",
        "heart failure preserved ejection fraction animal model",
    ],
    "merge_order": "terms in listed order; dedupe by OpenAlex ID then normalised DOI; "
                   "first occurrence wins",
    "date_range": {"from": "2015-01-01", "to": "2025-08-31",
                   "note": "ends a year back on purpose -- later papers are held out"},
    "future_range": {"from": "2025-09-01", "to": "2026-08-24",
                     "note": "held-out replay set for the time-slice demo"},
    "openalex_filter": {"field": "title_and_abstract", "search_type": "search_filter"},
    "standard_downstream_filter": "is_primary_research AND >=1 non-Human species MeSH "
                                  "AND non-empty abstract",
    "limits": {"max_works_per_term": 1500, "per_page": 200},
    "pubmed": {"requests_per_sec": 10, "sleep_seconds": 0.11,
               "api_key_env": "NCBI_API_KEY"},
}

# ---- lifted from probe_mesh.py -------------------------------------------------
NON_PRIMARY = {"Review", "Systematic Review", "Meta-Analysis", "Editorial",
               "Comment", "Letter", "Case Reports"}
SPECIES_STEMS = ["mice", "rats", "swine", "dogs", "rabbits", "humans", "guinea pigs",
                 "sheep", "cats", "cricetinae", "hamsters", "zebrafish", "macaca",
                 "primates", "goats", "ferrets", "chickens", "cattle", "horses"]
MODEL_TERMS = ["Diet, High-Fat", "Diet, Sodium-Restricted", "Sodium Chloride, Dietary",
               "Desoxycorticosterone", "Angiotensin II", "Streptozocin", "Ovariectomy",
               "Nephrectomy", "Aortic Constriction", "Disease Models, Animal",
               "Diet, Western"]
COLS = ["openalex_id", "doi", "pmid", "title", "publication_year", "abstract",
        "abstract_chars", "abstract_source", "abstract_word_count", "concepts",
        "topics", "mesh_headings", "mesh_species", "pub_types", "is_primary_research",
        "species_mesh", "chemicals", "model_mesh", "qualifiers", "mesh_major"]

def load_env(path=HERE / ".env"):
    """Minimal .env reader -- KEY=value, ignoring blanks and #comments."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def load_config():
    """Read the frozen query config, writing the default on first run."""
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"wrote default {CONFIG.name} -- the query is now frozen there")
        return DEFAULT_CONFIG
    cfg = json.loads(CONFIG.read_text())
    print(f"using frozen query from {CONFIG.name} (version {cfg.get('version', '?')})")
    return cfg

def text(el):
    return (el.text or "").strip() if el is not None else ""

def is_species(d):
    low = d.lower()
    return any(low == s or low.startswith(s + ",") or low.startswith(s + " ")
               for s in SPECIES_STEMS)

def is_model_term(d):
    return any(d == t or d.startswith(t + " ") or d.startswith(t + ",")
               for t in MODEL_TERMS)

def parse_pubmed(root):
    """Cached PubMed XML root -> abstract + every Layer 1 MeSH field."""
    if root is None:
        return {}
    parts = []
    for a in root.iter("AbstractText"):
        body = " ".join("".join(a.itertext()).split())
        if body:
            label = (a.get("Label") or "").strip()
            parts.append(f"{label}: {body}" if label else body)
    chemicals = []
    for c in root.iter("Chemical"):
        name, reg = text(c.find("NameOfSubstance")), text(c.find("RegistryNumber"))
        if name:
            chemicals.append(f"{name} ({reg})" if reg and reg != "0" else name)
    headings, species, model, quals, major = [], [], [], [], []
    for h in root.iter("MeshHeading"):
        dn = h.find("DescriptorName")
        d = text(dn)
        if not d:
            continue
        headings.append(d)
        if dn.get("MajorTopicYN") == "Y":
            major.append(d)
        if is_species(d):
            species.append(d)
        if is_model_term(d):
            model.append(d)
        for q in h.findall("QualifierName"):
            qt = text(q)
            if qt:
                quals.append(f"{d}/{qt}")
                if q.get("MajorTopicYN") == "Y":
                    major.append(f"{d}/{qt}")
    pub_types = [text(p) for p in root.iter("PublicationType") if text(p)]
    return {"abstract": " ".join(parts), "pub_types": pub_types,
            "is_primary_research": not any(p in NON_PRIMARY for p in pub_types),
            "mesh_headings": headings, "species_mesh": species, "chemicals": chemicals,
            "model_mesh": model, "qualifiers": quals, "mesh_major": major}

# ---- lifted from probe_ingest.py -----------------------------------------------
def openalex_abstract(work):
    idx = work.get("abstract_inverted_index") or {}
    slots = [(pos, word) for word, ps in idx.items() for pos in (ps or [])]
    return " ".join(w for _, w in sorted(slots))

def scored(items):
    return "; ".join(f"{i.get('display_name', '?')}:{round(i.get('score') or 0, 4)}"
                     for i in (items or []))

def pmid_of(work):
    m = re.search(r"(\d+)\s*$", str((work.get("ids") or {}).get("pmid") or ""))
    return m.group(1) if m else ""

def norm_doi(work):
    return (work.get("doi") or "").strip().lower().replace("https://doi.org/", "")

# ---- fetching -------------------------------------------------------------------
def openalex_pull(term, date_from, date_to, cap, per_page):
    """All works for one term in one date window, cached. Returns (works, cap_hit)."""
    key = md5(f"{term}|{date_from}|{date_to}|{cap}".encode()).hexdigest()[:12]
    p = CACHE / f"openalex_corpus_{key}.json"
    if p.exists():
        try:
            works = json.loads(p.read_text())
            return works, len(works) >= cap
        except ValueError:
            pass
    works = []
    for page in (Works().search_filter(title_and_abstract=term)
                 .filter(from_publication_date=date_from, to_publication_date=date_to)
                 .paginate(per_page=per_page, n_max=cap)):
        works.extend(dict(w) for w in page)
    p.write_text(json.dumps(works))
    return works, len(works) >= cap

def pubmed_root(pmid, sleep_s):
    """Cached PubMed XML for one PMID. Network only on a cache miss."""
    p = CACHE / f"pubmed_{pmid}.xml"
    if not p.exists():
        time.sleep(sleep_s)
        try:
            with Entrez.efetch(db="pubmed", id=pmid, retmode="xml") as h:
                raw = h.read()
        except Exception as e:
            print(f"  ! efetch failed for PMID {pmid}: {e}", file=sys.stderr)
            return None
        p.write_text(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
    try:
        return ET.fromstring(p.read_text())
    except (ET.ParseError, OSError) as e:
        print(f"  ! unparseable {p.name}: {e}", file=sys.stderr)
        return None

def build(cfg, date_from, date_to, label):
    """Pull one date window end to end and return its rows."""
    cap = cfg["limits"]["max_works_per_term"]
    per_page = cfg["limits"]["per_page"]
    sleep_s = cfg["pubmed"]["sleep_seconds"]
    print(f"\n=== {label}: {date_from} .. {date_to} ===")

    merged, seen_id, seen_doi, capped = [], set(), set(), []
    for term in cfg["terms"]:
        works, cap_hit = openalex_pull(term, date_from, date_to, cap, per_page)
        if cap_hit:
            capped.append(term)
        added = 0
        for w in works:
            wid, doi = w.get("id"), norm_doi(w)
            if not wid or wid in seen_id or (doi and doi in seen_doi):
                continue
            seen_id.add(wid)
            if doi:
                seen_doi.add(doi)
            merged.append(w)
            added += 1
        print(f"  {len(works):>5} found  {added:>5} new   {term}"
              + ("   <== CAP HIT" if cap_hit else ""))
    print(f"  {len(merged)} unique after dedupe (by OpenAlex ID + DOI)")
    if capped:
        print(f"  !! per-term cap of {cap} hit for: {', '.join(capped)} -- "
              f"results for those terms are truncated", file=sys.stderr)

    pmids = [p for p in (pmid_of(w) for w in merged) if p]
    todo = sum(1 for p in pmids if not (CACHE / f"pubmed_{p}.xml").exists())
    print(f"  {len(pmids)} works have a PMID; {todo} need fetching "
          f"(~{todo * sleep_s / 60:.1f} min), {len(pmids) - todo} already cached")

    rows = []
    for i, w in enumerate(merged, 1):
        pmid = pmid_of(w)
        pm = parse_pubmed(pubmed_root(pmid, sleep_s)) if pmid else {}
        oa_abs = openalex_abstract(w)
        abstract = pm.get("abstract") or oa_abs
        source = "pubmed" if pm.get("abstract") else ("openalex" if oa_abs else "")
        if i % 100 == 0 or i == len(merged):
            print(f"    [{i}/{len(merged)}] built", flush=True)
        species = pm.get("species_mesh", [])
        rows.append({
            "openalex_id": w.get("id") or "", "doi": w.get("doi") or "", "pmid": pmid,
            "title": w.get("title") or w.get("display_name") or "",
            "publication_year": w.get("publication_year") or "",
            "abstract": abstract, "abstract_chars": len(abstract),
            "abstract_source": source, "abstract_word_count": len(abstract.split()),
            "concepts": scored(w.get("concepts")), "topics": scored(w.get("topics")),
            "mesh_headings": "; ".join(pm.get("mesh_headings", [])),
            "mesh_species": "; ".join(species),        # probe_output.csv name
            "species_mesh": "; ".join(species),        # probe_mesh.py name (same data)
            "pub_types": "; ".join(pm.get("pub_types", [])),
            "is_primary_research": pm.get("is_primary_research", ""),
            "chemicals": "; ".join(pm.get("chemicals", [])),
            "model_mesh": "; ".join(pm.get("model_mesh", [])),
            "qualifiers": "; ".join(pm.get("qualifiers", [])),
            "mesh_major": "; ".join(pm.get("mesh_major", [])),
        })
    return rows

def non_human(cell):
    return [s.strip() for s in str(cell or "").split(";")
            if s.strip() and not s.strip().lower().startswith("human")]

def report(rows, name):
    n = len(rows) or 1
    pct = lambda k: f"{100.0 * k / n:.1f}%"
    pmid = sum(1 for r in rows if r["pmid"])
    prim = sum(1 for r in rows if r["is_primary_research"] is True)
    spec = sum(1 for r in rows if non_human(r["mesh_species"]))
    src = Counter(r["abstract_source"] or "none" for r in rows)
    keep = sum(1 for r in rows if r["is_primary_research"] is True
               and non_human(r["mesh_species"]) and r["abstract"].strip())
    print(f"\n--- {name} ---")
    print(f"  total works              : {len(rows)}")
    print(f"  with PMID                : {pmid} ({pct(pmid)})")
    print(f"  primary research         : {prim} ({pct(prim)})")
    for s in ("pubmed", "openalex", "none"):
        print(f"  abstract from {s:<10}: {src[s]} ({pct(src[s])})")
    print(f"  >=1 non-Human species    : {spec} ({pct(spec)})")
    print(f"  passes standard filter   : {keep} ({pct(keep)})")
    return keep

def main():
    load_env()
    cfg = load_config()
    pyalex.config.email = EMAIL
    Entrez.email = EMAIL
    key = os.environ.get(cfg["pubmed"]["api_key_env"])
    if key:
        Entrez.api_key = key
        print(f"NCBI API key loaded -- {cfg['pubmed']['requests_per_sec']} req/sec")
    else:
        print("!! no NCBI_API_KEY; staying at 3 req/sec", file=sys.stderr)
        cfg["pubmed"]["sleep_seconds"] = 0.4

    d = cfg["date_range"]
    rows = build(cfg, d["from"], d["to"], "main corpus")
    pd.DataFrame(rows, columns=COLS).to_csv(HERE / "corpus.csv", index=False)
    report(rows, "corpus.csv")

    if "--main-only" not in sys.argv:
        f = cfg["future_range"]
        frows = build(cfg, f["from"], f["to"], "held-out future set")
        pd.DataFrame(frows, columns=COLS).to_csv(HERE / "corpus_future.csv", index=False)
        report(frows, "corpus_future.csv (held out -- replay set)")

if __name__ == "__main__":
    main()
