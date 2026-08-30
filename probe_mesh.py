#!/usr/bin/env python3
"""Layer 3 probe: what does MeSH indexing give us for free, and how often does it
catch what the Gemini pass missed? Deterministic -- reads raw_cache/*.xml and
probe_output.csv only. No network, no LLM."""

import re, sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "raw_cache"

NON_PRIMARY = {"Review", "Systematic Review", "Meta-Analysis", "Editorial",
               "Comment", "Letter", "Case Reports"}
# organism branch (B01); prefix matching pulls strain terms in automatically
# ("Rats, Zucker", "Mice, Inbred C57BL", "Mice, Knockout", "Mice, Transgenic")
SPECIES_STEMS = ["mice", "rats", "swine", "dogs", "rabbits", "humans", "guinea pigs",
                 "sheep", "cats", "cricetinae", "hamsters", "zebrafish", "macaca",
                 "primates", "goats", "ferrets", "chickens", "cattle", "horses"]
# fixed induction list; matched exactly or as a family prefix, so bare
# "Desoxycorticosterone" also catches the "Desoxycorticosterone Acetate" descriptor
MODEL_TERMS = ["Diet, High-Fat", "Diet, Sodium-Restricted", "Sodium Chloride, Dietary",
               "Desoxycorticosterone", "Angiotensin II", "Streptozocin", "Ovariectomy",
               "Nephrectomy", "Aortic Constriction", "Disease Models, Animal",
               "Diet, Western"]      # not in the spec'd list; present in this corpus
COLS = ["pmid", "publication_year", "title", "pub_types", "is_primary_research",
        "species_mesh", "chemicals", "model_mesh", "qualifiers", "mesh_major"]

def text(el):
    return (el.text or "").strip() if el is not None else ""

def is_species(d):
    low = d.lower()
    return any(low == s or low.startswith(s + ",") or low.startswith(s + " ")
               for s in SPECIES_STEMS)

def is_model(d):
    return any(d == t or d.startswith(t + " ") or d.startswith(t + ",")
               for t in MODEL_TERMS)

def parse(path):
    """One cached PubMed XML -> dict of MeSH-layer fields. {} if unparseable."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        print(f"  ! unparseable {path.name}: {e}", file=sys.stderr)
        return {}
    pub_types = [text(p) for p in root.iter("PublicationType") if text(p)]
    chemicals = []
    for c in root.iter("Chemical"):
        name, reg = text(c.find("NameOfSubstance")), text(c.find("RegistryNumber"))
        if name:
            chemicals.append(f"{name} ({reg})" if reg and reg != "0" else name)
    species, model, quals, major = [], [], [], []
    for h in root.iter("MeshHeading"):
        d = text(h.find("DescriptorName"))
        if not d:
            continue
        dn = h.find("DescriptorName")
        if dn.get("MajorTopicYN") == "Y":
            major.append(d)
        if is_species(d):
            species.append(d)
        if is_model(d):
            model.append(d)
        for q in h.findall("QualifierName"):
            qt = text(q)
            if qt:
                quals.append(f"{d}/{qt}")
                if q.get("MajorTopicYN") == "Y":
                    major.append(f"{d}/{qt}")
    return {"pub_types": pub_types,
            "is_primary_research": not any(p in NON_PRIMARY for p in pub_types),
            "species_mesh": species, "chemicals": chemicals, "model_mesh": model,
            "qualifiers": quals, "mesh_major": major}

def pct(k, n):
    return f"{100.0 * k / (n or 1):.1f}%"

def top(counter, k, n, label):
    print(f"\ntop {k} {label} ({len(counter)} distinct):")
    for name, c in counter.most_common(k):
        print(f"  {c:>4} ({pct(c, n):>6})  {name}")

def main():
    src = HERE / "probe_output.csv"
    if not src.exists(): sys.exit(f"error: no {src.name} -- run probe_ingest.py first")
    works = pd.read_csv(src, dtype=str).fillna("")
    meta = {r["pmid"]: (r["publication_year"], r["title"])
            for _, r in works.iterrows() if r["pmid"]}

    rows = []
    for p in sorted(CACHE.glob("pubmed_*.xml")):
        pmid = p.stem.replace("pubmed_", "")
        d = parse(p)
        if not d:
            continue
        year, title = meta.get(pmid, ("", ""))
        rows.append({"pmid": pmid, "publication_year": year, "title": title,
                     **{k: ("; ".join(v) if isinstance(v, list) else v)
                        for k, v in d.items()}, "_raw": d})

    out = HERE / "mesh_layer.csv"
    pd.DataFrame([{c: r[c] for c in COLS} for r in rows]).to_csv(out, index=False)
    n = len(rows)
    print(f"parsed {n} cached PubMed XML files -> {out.name}\n")

    prim = sum(1 for r in rows if r["is_primary_research"])
    print(f"primary research      : {prim}/{n} ({pct(prim, n)})")
    for f in ["species_mesh", "chemicals", "model_mesh"]:
        k = sum(1 for r in rows if r["_raw"][f])
        print(f">=1 {f:<18}: {k}/{n} ({pct(k, n)})")

    pts = Counter(t for r in rows for t in r["_raw"]["pub_types"])
    print(f"\npublication type distribution ({len(pts)} distinct):")
    for t, c in pts.most_common():
        flag = "  <- non-primary" if t in NON_PRIMARY else ""
        print(f"  {c:>4} ({pct(c, n):>6})  {t}{flag}")

    top(Counter(c for r in rows for c in r["_raw"]["chemicals"]), 20, n, "chemicals")
    top(Counter(m for r in rows for m in r["_raw"]["model_mesh"]), 20, n,
        "model_mesh descriptors")
    top(Counter(m for r in rows for m in r["_raw"]["mesh_major"]), 20, n,
        "major-topic descriptors")

    vr = HERE / "vocab_raw.csv"
    if not vr.exists():
        print(f"\n(no {vr.name} -- skipping Layer 3 miss counts)")
        return
    gem = pd.read_csv(vr, dtype=str).fillna("")
    llm = {r["pmid"]: r for _, r in gem.iterrows() if r.get("pmid")}
    joined = [r for r in rows if r["pmid"] in llm]
    print(f"\n=== Layer 3 miss counts (joined on PMID: {len(joined)} of {n} MeSH rows "
          f"x {len(llm)} {vr.name} rows) ===")
    for mesh_f, gem_f in [("chemicals", "intervention"),
                          ("model_mesh", "induction_method")]:
        have = [r for r in joined if r["_raw"][mesh_f]]
        miss = [r for r in have if not llm[r["pmid"]].get(gem_f, "").strip()]
        print(f"\n{mesh_f} non-empty: {len(have)} papers; "
              f"Gemini {gem_f} was null for {len(miss)} of them ({pct(len(miss), len(have))})")
        print(f"  -> Layer 3 miss count = {len(miss)}")
        for r in miss[:5]:
            print(f"     {r['pmid']}  {r['_raw'][mesh_f][0][:60]}")

if __name__ == "__main__":
    main()
