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
            "sexes_studied": r.get("sexes_studied") or [],
            "genotypes_studied": r.get("genotypes_studied") or [],
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

AGENT_ALIASES = {          # hand-editable: brand/salt/abbreviation variants -> one key
    "empa": "empagliflozin", "jardiance": "empagliflozin",
    "dapa": "dapagliflozin", "farxiga": "dapagliflozin",
    "cana": "canagliflozin", "sema": "semaglutide", "lira": "liraglutide",
    "lcz696": "sacubitril/valsartan", "entresto": "sacubitril/valsartan",
    "sac/val": "sacubitril/valsartan", "spiro": "spironolactone",
    "met": "metformin", "sita": "sitagliptin",
}
_AGENT_STRIP = re.compile(
    r"\b(treatment|therapy|administration|supplementation|infusion|injection|"
    r"hydrochloride|hcl|sodium|potassium|sulfate|citrate|maleate|mesylate|"
    r"tartrate|acetate|dihydrate|monohydrate)\b")

def measure_key(measure):
    """Canonical measure for grouping. Verbatim text is kept for display; two papers
    saying "E/e' ratio" and "diastolic dysfunction" must land on one key or they can
    never be compared."""
    return cfg.MEASURE_CANON.get(norm(measure), norm(measure))

def agent_key(name):
    """Fold an agent name to a comparison key. 'Empagliflozin (EMPA) treatment' and
    'EMPA' must land on the same key, or a drug will appear to contradict itself."""
    t = norm(name)
    t = re.sub(r"\(([^)]*)\)", " ", t)
    t = _AGENT_STRIP.sub(" ", t)
    t = re.sub(r"[^a-z0-9/+\- ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return AGENT_ALIASES.get(t, t)

def contradictions(papers):
    """A contradiction needs the same disease model, the same measure, the same KIND of
    comparison, and -- for treatment results -- the same agent.

    Two papers reporting opposite directions conflict only if they asked the same
    question. A drug improving a measure against vehicle does not contradict the model
    worsening that measure against healthy animals, nor a different drug doing the
    opposite. Sex and genotype travel with each paper for display; they never merge or
    split a group.
    """
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    variants = defaultdict(set)      # canonical key -> the verbatim phrases behind it
    meta = {}
    for p in papers:
        meta[p["pmid"]] = {"sexes_studied": p.get("sexes_studied") or [],
                           "genotypes_studied": p.get("genotypes_studied") or [],
                           "arms": p.get("arms") or []}
        for o in p["outcomes"]:
            raw_m = norm(o.get("measure"))
            if not raw_m:
                continue
            canon = measure_key(raw_m)
            # coarse buckets still count as evidence in the grid; they just cannot form
            # a contradiction key, because they lump unrelated molecules together
            if canon in getattr(cfg, "MEASURE_EXCLUDE_FROM_CONTRADICTION", set()):
                continue
            key = (p["model"], o.get("domain") or "other", canon)
            variants[key].add(raw_m)
            comp = o.get("comparator") or "not_stated"
            if comp == "vs_untreated_disease":
                sub = agent_key(o.get("agent") or "") or "(agent not stated)"
            elif comp == "vs_other_subgroup":
                sub = o.get("subgroup_axis") or "other"
            else:
                sub = ""
            groups[key][comp][sub][o.get("direction") or "not_stated"].add(p["pmid"])

    def entry(key, kind, comp, sub, dirs_map, extra=None):
        pmids = sorted({x for ps in dirs_map.values() for x in ps})
        e = {"kind": kind, "model": key[0], "outcome_domain": key[1], "measure": key[2],
             "measure_variants": sorted(variants.get(key, [])),   # verbatim, for display
             "comparator": comp,
             "agent": sub if comp == "vs_untreated_disease" and sub else None,
             "subgroup_axis": sub if comp == "vs_other_subgroup" and sub else None,
             "by_direction": {d: sorted(ps) for d, ps in dirs_map.items() if ps},
             "improved_pmids": sorted(dirs_map.get("improved", set())),
             "worsened_pmids": sorted(dirs_map.get("worsened", set())),
             "both_ways_pmids": sorted(dirs_map.get("improved", set())
                                       & dirs_map.get("worsened", set())),
             "paper_context": {x: {"sexes_studied": meta.get(x, {}).get("sexes_studied", []),
                                   "genotypes_studied": meta.get(x, {}).get("genotypes_studied", []),
                                   "partition": ("interventional"
                                                 if meta.get(x, {}).get("arms") else "characterization")}
                               for x in pmids},
             "n_papers": len(pmids)}
        if extra:
            e.update(extra)
        return e

    out = []
    for key, by_comp in groups.items():
        for comp, by_sub in by_comp.items():
            for sub, dirs in by_sub.items():
                imp, wor = dirs.get("improved", set()), dirs.get("worsened", set())
                if not (imp and wor):
                    continue
                kind = "contradiction" if (imp - wor and wor - imp) else "within_paper"
                out.append(entry(key, kind, comp, sub, dirs))
            subs = list(by_sub)
            if comp in ("vs_untreated_disease", "vs_other_subgroup") and len(subs) > 1:
                for i in range(len(subs)):
                    for jx in range(i + 1, len(subs)):
                        a, b = subs[i], subs[jx]
                        if ((by_sub[a].get("improved") and by_sub[b].get("worsened")) or
                                (by_sub[a].get("worsened") and by_sub[b].get("improved"))):
                            merged = defaultdict(set)
                            for src in (a, b):
                                for d, ps in by_sub[src].items():
                                    merged[d] |= ps
                            label = "agent" if comp == "vs_untreated_disease" else "subgroup_axis"
                            out.append(entry(key, "divergent_context", comp, "", merged,
                                             {f"{label}_sides": [a, b]}))
        comps = list(by_comp)
        for i in range(len(comps)):
            for jx in range(i + 1, len(comps)):
                ca, cb = comps[i], comps[jx]
                fa = {d: {x for sub in by_comp[ca].values() for x in sub.get(d, set())}
                      for d in ("improved", "worsened")}
                fb = {d: {x for sub in by_comp[cb].values() for x in sub.get(d, set())}
                      for d in ("improved", "worsened")}
                if (fa["improved"] and fb["worsened"]) or (fa["worsened"] and fb["improved"]):
                    merged = defaultdict(set)
                    for f in (fa, fb):
                        for d, ps in f.items():
                            merged[d] |= ps
                    out.append(entry(key, "divergent_context", f"{ca} | {cb}", "", merged,
                                     {"comparator_sides": [ca, cb]}))
    order = {"contradiction": 0, "divergent_context": 1, "within_paper": 2}
    return sorted(out, key=lambda c: (order.get(c["kind"], 9), -c["n_papers"]))

def main():
    # corpus_live.csv holds papers the agent ingested after corpus.csv was frozen; without
    # it their species MeSH is missing and every one of them falls back to "Other".
    live = HERE / "corpus_live.csv"
    papers = load(extra_corpus=[live] if live.exists() else None)
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

    seen = Counter()
    for p in papers:
        for o in p["outcomes"]:
            m = norm(o.get("measure"))
            if m and m not in cfg.MEASURE_CANON:
                seen[m] += 1
    if seen:
        print(f"\nmeasures with no entry in grid_config.MEASURE_GROUPS ({len(seen)} distinct, "
              f"{sum(seen.values())} outcomes) -- they group on their verbatim text:")
        for k, v in seen.most_common(8):
            print(f"  {v:>3}  {k[:64]}")
    near = defaultdict(list)
    for canon in cfg.MEASURE_GROUPS:
        near[canon.replace("cardiac_", "").replace("_", "")].append(canon)
    dupes = {k: v for k, v in near.items() if len(v) > 1}
    if dupes:
        print(f"\ncanonical names that may be duplicates of each other "
              f"({len(dupes)} pairs) -- worth merging by hand in grid_config.py:")
        for k, v in list(dupes.items())[:8]:
            print(f"  {' / '.join(v)}")

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

    cross = [c for c in contra if c["kind"] == "contradiction"]
    diverg = [c for c in contra if c["kind"] == "divergent_context"]
    within = [c for c in contra if c["kind"] == "within_paper"]
    print(f"\n=== contradictions: {len(cross)} (same comparison type, opposite directions) "
          f"| {len(diverg)} divergent_context (opposite only across interventional vs "
          f"characterization -- expected, never alerts) | {len(within)} same-paper ===")
    if not cross:
        print("  none")
    for c in cross[:20]:
        tag = c["comparator"] + (f" · {c['agent']}" if c.get("agent") else "") + \
              (f" · {c['subgroup_axis']}" if c.get("subgroup_axis") else "")
        print(f"  [{tag}] {c['model']} / {c['outcome_domain']} / \"{c['measure'][:44]}\"")
        print(f"      improved: {', '.join(c['improved_pmids'][:6])}")
        print(f"      worsened: {', '.join(c['worsened_pmids'][:6])}")
    print(f"\nwrote grid.json")

if __name__ == "__main__":
    main()
