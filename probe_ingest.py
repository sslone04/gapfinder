#!/usr/bin/env python3
"""Probe: do HFpEF animal-model abstracts carry enough structured signal to fill a
species x intervention x readout grid? Throwaway de-risk -- no LLM, no grid."""

import json, re, socket, statistics, sys, time
from collections import Counter
from hashlib import md5
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd, pyalex
from pyalex import Works
from Bio import Entrez

EMAIL = "probe@example.com"          # placeholder -- polite pool for both APIs
pyalex.config.email = EMAIL
Entrez.email = EMAIL
socket.setdefaulttimeout(30)               # never let one efetch stall the run

HERE = Path(__file__).resolve().parent
CACHE = HERE / "raw_cache"
CACHE.mkdir(exist_ok=True)
LIMIT, PER_QUERY, DATE_FROM, DATE_TO = 100, 50, "2015-01-01", "2024-12-31"

QUERIES = [
    "HFpEF animal model", "HFpEF mouse model", "HFpEF rat model",
    "heart failure with preserved ejection fraction mouse",
    "heart failure with preserved ejection fraction rat",
    "heart failure preserved ejection fraction animal model",
]
# MeSH organism-branch (B01) stems; matched whole-word or as "Stem, qualifier"
SPECIES_STEMS = [
    "mice", "rats", "swine", "dogs", "rabbits", "humans", "guinea pigs", "sheep",
    "cats", "cricetinae", "hamsters", "zebrafish", "macaca", "primates", "goats",
]

def cached_json(key, fetch):
    p = CACHE / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    data = fetch()
    p.write_text(json.dumps(data))
    return data

def openalex_query(q):
    def go():
        pages = (Works().search_filter(title_and_abstract=q)
                 .filter(from_publication_date=DATE_FROM, to_publication_date=DATE_TO)
                 .paginate(per_page=50, n_max=PER_QUERY))
        return [dict(w) for page in pages for w in page]
    return cached_json("openalex_" + md5(q.encode()).hexdigest()[:12], go)

def abstract_of(work):
    idx = work.get("abstract_inverted_index") or {}
    slots = [(pos, word) for word, ps in idx.items() for pos in (ps or [])]
    return " ".join(w for _, w in sorted(slots))

def scored(items):
    return "; ".join(f"{i.get('display_name', '?')}:{round(i.get('score') or 0, 4)}"
                     for i in (items or []))

def pmid_of(work):
    m = re.search(r"(\d+)\s*$", str((work.get("ids") or {}).get("pmid") or ""))
    return m.group(1) if m else ""

def pubmed_xml(pmid):
    p = CACHE / f"pubmed_{pmid}.xml"
    if p.exists():
        return p.read_text()
    time.sleep(0.4)                                   # 3 req/sec cap, no API key
    try:
        with Entrez.efetch(db="pubmed", id=pmid, retmode="xml") as h:
            raw = h.read()
    except Exception as e:                            # network, 429, bad id
        print(f"  ! efetch failed for PMID {pmid}: {e}", file=sys.stderr)
        return ""
    xml = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    p.write_text(xml)
    return xml

def root_of(xml):
    if not xml:
        return None
    try:
        return ET.fromstring(xml)
    except ET.ParseError:
        return None

def mesh_of(root):
    if root is None:
        return []
    return [(h.findtext("DescriptorName") or "").strip() for h in root.iter("MeshHeading")
            if (h.findtext("DescriptorName") or "").strip()]

def pubmed_abstract(root):
    """Full <AbstractText> in document order, labelled sections as "LABEL: text".
    Authoritative: OpenAlex's inverted-index reconstruction drops text mid-abstract."""
    if root is None:
        return ""
    parts = []
    for a in root.iter("AbstractText"):
        body = " ".join("".join(a.itertext()).split())
        if not body:
            continue
        label = (a.get("Label") or "").strip()
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts)

def species_terms(mesh):
    return [t for t in mesh if any(t.lower() == s or t.lower().startswith(s + ",")
                                   or t.lower().startswith(s + " ") for s in SPECIES_STEMS)]

def collect_works():
    """Round-robin across queries so every phrasing contributes to the sample."""
    pools = [openalex_query(q) for q in QUERIES]
    works, seen = [], set()
    for rank in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            w = pool[rank] if rank < len(pool) else None
            if w and w.get("id") and w["id"] not in seen:
                seen.add(w["id"])
                works.append(w)
                if len(works) >= LIMIT:
                    return works
    return works

def main():
    works = collect_works()
    print(f"OpenAlex: {len(works)} unique works from {len(QUERIES)} queries\n")
    rows = []
    for i, w in enumerate(works, 1):
        pmid = pmid_of(w)
        root = root_of(pubmed_xml(pmid)) if pmid else None
        mesh = mesh_of(root)
        abstract = pubmed_abstract(root)
        source = "pubmed" if abstract else ("openalex" if abstract_of(w) else "")
        if not abstract:
            abstract = abstract_of(w)
        print(f"[{i}/{len(works)}] {(w.get('id') or '')[-12:]} pmid={pmid or '-':>9} "
              f"abs={len(abstract):>5}c {source or '-':<8} mesh={len(mesh)}")
        rows.append({
            "openalex_id": w.get("id") or "", "doi": w.get("doi") or "", "pmid": pmid,
            "title": w.get("title") or w.get("display_name") or "",
            "publication_year": w.get("publication_year") or "",
            "abstract": abstract, "abstract_chars": len(abstract),
            "abstract_source": source, "abstract_word_count": len(abstract.split()),
            "concepts": scored(w.get("concepts")), "topics": scored(w.get("topics")),
            "mesh_headings": "; ".join(mesh),
            "mesh_species": "; ".join(species_terms(mesh)),
        })

    out = HERE / "probe_output.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    n = len(rows) or 1
    pct = lambda k: 100.0 * sum(1 for r in rows if r[k]) / n
    wc = [r["abstract_word_count"] for r in rows if r["abstract_word_count"]]
    mesh_counts = Counter(m for r in rows for m in r["mesh_headings"].split("; ") if m)
    print(f"\n=== coverage report ({out.name}) ===")
    print(f"total works                : {len(rows)}")
    print(f"non-empty abstract         : {pct('abstract'):.1f}%")
    print(f"has PMID                   : {pct('pmid'):.1f}%")
    print(f">=1 species MeSH term      : {pct('mesh_species'):.1f}%")
    print(f"median abstract word count : {statistics.median(wc) if wc else 0}")
    src = Counter(r["abstract_source"] for r in rows if r["abstract_source"])
    print(f"abstract source            : " + ", ".join(f"{k} {v}" for k, v in src.most_common()))
    print(f"\ntop 25 MeSH headings ({len(mesh_counts)} distinct):")
    for term, c in mesh_counts.most_common(25):
        print(f"  {c:>4}  {term}")

if __name__ == "__main__":
    main()
