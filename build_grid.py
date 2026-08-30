#!/usr/bin/env python3
"""Build the species x model x outcome_domain evidence grid.

Deterministic: no network, no LLM. Reads extractions.jsonl + corpus.csv, canonicalises
the axes using the hand-editable tables in grid_config.py, and writes grid.json.

Every cell carries its own denominator, and empty cells are split into
"no_evidence_found" (enough papers sat at that species x model pair to have reported it)
and "unscreened" (too few papers to conclude anything).
"""

import json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import grid_config as cfg

HERE = Path(__file__).resolve().parent
DIRECTIONS = ["improved", "worsened", "no_change", "mixed", "not_stated"]

def norm(s):
    """Lowercase, fold unicode dashes, collapse whitespace, and close the gaps around
    separators so "Db/ db", "db / db" and "db/db" all normalise to the same string."""
    t = unicodedata.normalize("NFKC", str(s or "")).lower()
    t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*([/\-])\s*", r"\1", t)
    return t.strip()

_WORDLIKE = re.compile(r"^[a-z0-9].*[a-z0-9]$|^[a-z0-9]$")

def hit(pattern, blob):
    """Word-boundary match for word-like patterns, substring for punctuated ones.
    Keeps "aged" out of "packaged" while letting "db/db" and "(tac)" match."""
    pat = norm(pattern)
    if _WORDLIKE.match(pat) and not re.search(r"[^a-z0-9 ]", pat.replace("-", "")):
        return re.search(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", blob) is not None
    return pat in blob

def species_candidates(mesh_cell):
    """All canonical species a paper is tagged with, in SPECIES_RULES order.
    Multi-species papers are common ("Mice; Rats, Zucker"), so this keeps every option."""
    low = norm(mesh_cell)
    found = [name for name, keys in cfg.SPECIES_RULES if any(hit(k, low) for k in keys)]
    return found or [cfg.SPECIES_FALLBACK]

def species_of(mesh_cell, model):
    """Pick the species that is actually compatible with the model. A ZSF1 paper tagged
    "Mice; Rats, Zucker" is a rat study; taking the first tag alone would file it under
    an impossible Mouse x ZSF1 pair."""
    cands = species_candidates(mesh_cell)
    for c in cands:
        if (c, model) not in cfg.NOT_APPLICABLE:
            return c, (c != cands[0])
    return cands[0], False

def model_of(disease_model):
    """disease_model -> one canonical model. First matching rule wins; see grid_config."""
    comps = " | ".join(disease_model.get("components") or [])
    low = norm(comps)
    for name, rule in cfg.MODEL_RULES:
        if "any" in rule and any(hit(k, low) for k in rule["any"]):
            return name
        if "all" in rule and all(any(hit(k, low) for k in grp) for grp in rule["all"]):
            return name
    if (disease_model.get("model_type") == "genetic_strain"
            or any(hit(h, low) for h in cfg.GENETIC_HINTS)):
        return cfg.MODEL_FALLBACK_GENETIC
    return cfg.MODEL_FALLBACK_OTHER

def load(extra_corpus=None, extractions=None):
    """Papers ready for gridding. extra_corpus lets the agent loop add rows ingested
    after corpus.csv was frozen, without mutating the frozen corpus itself."""
    ex = Path(extractions) if extractions else HERE / "extractions.jsonl"
    cp = HERE / "corpus.csv"
    for f in (ex, cp):
        if not f.exists(): sys.exit(f"error: no {f.name} -- run the earlier steps first")
    species = {}
    for path in [cp] + [Path(x) for x in (extra_corpus or [])]:
        if not Path(path).exists():
            continue
        for _, r in pd.read_csv(path, dtype=str).fillna("").iterrows():
            if r["pmid"]:
                species[r["pmid"]] = r["species_mesh"]
    papers = []
    for line in ex.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        mesh = species.get(r["pmid"], "")
        model = model_of(r.get("disease_model") or {})
        sp, moved = species_of(mesh, model)
        papers.append({
            "pmid": r["pmid"], "title": r["title"], "year": r["publication_year"],
            "species": sp, "species_reassigned": moved, "species_mesh": mesh,
            "model": model,
            "components": (r.get("disease_model") or {}).get("components") or [],
            "outcomes": r.get("outcomes") or [], "arms": r.get("arms") or [],
        })
    return papers

def build(papers):
    """Return (cells, screened) keyed by tuple. screened[(species,model)] = paper count."""
    screened = Counter((p["species"], p["model"]) for p in papers)
    hits = defaultdict(lambda: {"pmids": set(), "arms": set(), "dirs": Counter(),
                                "measures": 0})
    for p in papers:
        for o in p["outcomes"]:
            dom = o.get("domain") or "other"
            if dom not in cfg.OUTCOME_AXIS:
                dom = "other"
            c = hits[(p["species"], p["model"], dom)]
            c["pmids"].add(p["pmid"])
            c["dirs"][o.get("direction") or "not_stated"] += 1
            c["measures"] += 1
            if p["arms"]:
                c["arms"].add(p["pmid"])

    total = len(papers)
    cells = []
    for s in cfg.SPECIES_AXIS:
        for m in cfg.MODEL_AXIS:
            k = screened[(s, m)]
            for d in cfg.OUTCOME_AXIS:
                h = hits.get((s, m, d))
                n = len(h["pmids"]) if h else 0
                na = (s, m) in cfg.NOT_APPLICABLE
                if n:
                    # papers in an "impossible" pair means a mapping bug, not evidence
                    status = "evidence"
                elif na:
                    status = "not_applicable"
                elif k >= cfg.MIN_SCREENED:
                    status = "no_evidence_found"
                else:
                    status = "unscreened"
                cells.append({
                    "species": s, "model": m, "outcome_domain": d, "status": status,
                    "hf_class": cfg.HF_CLASS.get(m, "unclassified"),
                    "not_applicable_pair": na,
                    "n_papers": n, "n_with_arms": len(h["arms"]) if h else 0,
                    "n_outcome_rows": h["measures"] if h else 0,
                    "pmids": sorted(h["pmids"]) if h else [],
                    "directions": {x: (h["dirs"][x] if h else 0) for x in DIRECTIONS},
                    "screened_k": k, "screened_total": total,
                    "denominator": f"{k} of {total} screened",
                })
    return cells, screened

def adjacency(cells, papers):
    """For empty cells: how populated are the three neighbouring axis pairs?"""
    sm, sd, md = Counter(), Counter(), Counter()
    for c in cells:
        if c["n_papers"] and not c["not_applicable_pair"]:
            sm[(c["species"], c["model"])] += c["n_papers"]
            sd[(c["species"], c["outcome_domain"])] += c["n_papers"]
            md[(c["model"], c["outcome_domain"])] += c["n_papers"]
    out = []
    for c in cells:
        if c["n_papers"] or c["not_applicable_pair"]:
            continue                        # impossible pairs are never gaps
        a = (sm[(c["species"], c["model"])], sd[(c["species"], c["outcome_domain"])],
             md[(c["model"], c["outcome_domain"])])
        if min(a) > 0:                      # every neighbouring pair is populated
            out.append({**c, "adj_species_model": a[0], "adj_species_domain": a[1],
                        "adj_model_domain": a[2], "adjacency": min(a), "adj_sum": sum(a)})
    return sorted(out, key=lambda c: (-c["adjacency"], -c["adj_sum"]))

def contradictions(papers):
    """Same model + domain + measure, interventional only, reported both ways."""
    groups = defaultdict(lambda: defaultdict(set))
    for p in papers:
        if not p["arms"]:
            continue
        for o in p["outcomes"]:
            key = (p["model"], o.get("domain") or "other", norm(o.get("measure")))
            if key[2]:
                groups[key][o.get("direction") or "not_stated"].add(p["pmid"])
    out = []
    for key, by_dir in groups.items():
        imp, wor = by_dir.get("improved", set()), by_dir.get("worsened", set())
        if not (imp and wor):
            continue
        only_i, only_w, both = imp - wor, wor - imp, imp & wor
        # A real between-study disagreement needs a paper on each side. When the same
        # PMID appears both ways it is one paper describing the model AND the treatment.
        kind = "cross_paper" if (only_i and only_w) else "within_paper"
        out.append({"kind": kind, "model": key[0], "outcome_domain": key[1],
                    "measure": key[2], "improved_pmids": sorted(imp),
                    "worsened_pmids": sorted(wor), "both_ways_pmids": sorted(both),
                    "n_papers": len(imp | wor)})
    return sorted(out, key=lambda c: (c["kind"] != "cross_paper", -c["n_papers"]))

def main():
    papers = load()
    cells, screened = build(papers)
    gaps = adjacency(cells, papers)
    contra = contradictions(papers)

    occupied = [c for c in cells if c["n_papers"]]
    status_counts = Counter(c["status"] for c in cells)
    n = len(cells)
    out = {
        "axes": {"species": cfg.SPECIES_AXIS, "model": cfg.MODEL_AXIS,
                 "outcome_domain": cfg.OUTCOME_AXIS},
        "papers_total": len(papers), "min_screened": cfg.MIN_SCREENED,
        "cells": cells, "adjacent_gaps": gaps, "contradictions": contra,
    }
    (HERE / "grid.json").write_text(json.dumps(out, indent=2))

    print(f"=== grid {len(cfg.SPECIES_AXIS)} species x {len(cfg.MODEL_AXIS)} models "
          f"x {len(cfg.OUTCOME_AXIS)} domains ===")
    print(f"papers placed        : {len(papers)}")
    print(f"total cells          : {n}")
    print(f"occupied             : {len(occupied)} ({100.0*len(occupied)/n:.1f}%)")
    print(f"empty                : {n-len(occupied)} ({100.0*(n-len(occupied))/n:.1f}%)")
    for s in ("evidence", "no_evidence_found", "unscreened", "not_applicable"):
        print(f"  {s:<20}: {status_counts[s]:>4} ({100.0*status_counts[s]/n:>5.1f}%)")

    moved = [p for p in papers if p["species_reassigned"]]
    if moved:
        print(f"\n  {len(moved)} multi-species papers were re-filed off an impossible "
              f"pair onto a compatible species:")
        for p in moved[:6]:
            print(f"     {p['pmid']} -> {p['species']} x {p['model']}   "
                  f"({p['species_mesh'][:52]})")

    bad = [c for c in cells if c["not_applicable_pair"] and c["n_papers"]]
    if bad:
        pairs = sorted({(c["species"], c["model"]) for c in bad})
        print(f"\n  !! {len(bad)} cells sit on an impossible species x model pair yet "
              f"hold papers -- a mapping bug, not evidence:")
        for sp, mo in pairs:
            pm = sorted({p for c in bad if (c["species"], c["model"]) == (sp, mo)
                         for p in c["pmids"]})
            print(f"     {sp} x {mo}: {len(pm)} papers e.g. {', '.join(pm[:4])}")

    print(f"\naxis marginals:")
    for label, key in (("species", "species"), ("model", "model")):
        c = Counter(p[key] for p in papers)
        print(f"  {label:<8} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))

    hf = Counter(cfg.HF_CLASS.get(p["model"], "unclassified") for p in papers)
    print(f"\npapers per hf_class ({len(papers)} total):")
    for k in ("hfpef", "hfref", "mixed", "unclassified"):
        models = sorted(m for m, v in cfg.HF_CLASS.items() if v == k)
        print(f"  {k:<13} {hf[k]:>4} ({100.0*hf[k]/len(papers):>5.1f}%)  "
              f"{', '.join(models)[:78]}")

    for bucket in (cfg.MODEL_FALLBACK_OTHER, cfg.MODEL_FALLBACK_GENETIC):
        comps = Counter(norm(c) for p in papers if p["model"] == bucket
                        for c in p["components"])
        if comps:
            n_p = sum(1 for p in papers if p["model"] == bucket)
            print(f"\nunresolved -> {bucket} ({n_p} papers). Top components to add "
                  f"rules for in grid_config.py:")
            for k, v in comps.most_common(8):
                print(f"  {v:>3}  {k[:64]}")

    print(f"\n=== 15 most-populated cells ===")
    print(f"  {'n':>3} {'arms':>5}  {'species':<7} {'model':<22} {'domain':<19} denominator")
    for c in sorted(occupied, key=lambda c: -c["n_papers"])[:15]:
        print(f"  {c['n_papers']:>3} {c['n_with_arms']:>5}  {c['species']:<7} "
              f"{c['model']:<22} {c['outcome_domain']:<19} {c['denominator']}")

    print(f"\n=== 15 highest-adjacency empty cells (the interesting gaps) ===")
    print(f"  {'adj':>4}  {'species':<7} {'model':<22} {'domain':<19} "
          f"{'s*m':>4} {'s*d':>4} {'m*d':>4}  status")
    for c in gaps[:15]:
        print(f"  {c['adjacency']:>4}  {c['species']:<7} {c['model']:<22} "
              f"{c['outcome_domain']:<19} {c['adj_species_model']:>4} "
              f"{c['adj_species_domain']:>4} {c['adj_model_domain']:>4}  {c['status']}")

    cross = [c for c in contra if c["kind"] == "cross_paper"]
    within = [c for c in contra if c["kind"] == "within_paper"]
    print(f"\n=== contradiction candidates: {len(cross)} cross-paper "
          f"({len(within)} same-paper-both-ways, listed in grid.json but not here -- "
          f"those are one paper describing model AND treatment, not a disagreement) ===")
    if not cross:
        print("  none")
    for c in cross[:20]:
        print(f"  {c['model']} / {c['outcome_domain']} / \"{c['measure'][:52]}\"")
        print(f"      improved: {', '.join(c['improved_pmids'][:6])}")
        print(f"      worsened: {', '.join(c['worsened_pmids'][:6])}")
        if c["both_ways_pmids"]:
            print(f"      both ways in one paper: {', '.join(c['both_ways_pmids'][:6])}")
    print(f"\nwrote grid.json")

if __name__ == "__main__":
    main()
