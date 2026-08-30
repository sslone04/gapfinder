#!/usr/bin/env python3
"""Pick a stratified 10-paper gold set to hand-annotate, so later automated
extraction has something to be scored against. Deterministic -- no network, no LLM."""

import random, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42
N_TOTAL = 10
DRUG_QUALS = ("drug therapy", "drug effects", "pharmacology")
STRAINS = ("zucker", "dahl", "c57bl")
ANNOTATION_COLS = ["model_type", "model_components", "arms", "outcomes",
                   "mechanisms", "entities"]

def non_human_species(cell):
    return [s.strip() for s in str(cell or "").split(";")
            if s.strip() and not s.strip().lower().startswith("human")]

def has_drug_qualifier(cell):
    low = str(cell or "").lower()
    return any(f"/{q}" in low for q in DRUG_QUALS)

def has_strain(cell):
    low = str(cell or "").lower()
    return any(s in low for s in STRAINS)

def take(rng, pool, k, used, label, picked):
    """Draw k unused pmids from pool, recording the stratum each came from."""
    avail = sorted(p for p in pool if p not in used)
    n = min(k, len(avail))
    if n < k:
        print(f"  ! stratum '{label}' wanted {k}, only {n} available", file=sys.stderr)
    for pmid in rng.sample(avail, n):
        used.add(pmid)
        picked.append((pmid, label))
    return n

def refresh_abstracts():
    """Re-read abstracts for the EXISTING gold PMIDs from probe_output.csv, keeping the
    sample and any annotations intact. Use after the abstract source changes -- re-running
    the sampler instead would draw a different 10 from a changed eligible pool."""
    gold = HERE / "gold_annotate.csv"
    if not gold.exists(): sys.exit(f"error: no {gold.name} -- run without --refresh first")
    g = pd.read_csv(gold, dtype=str).fillna("")
    src = pd.read_csv(HERE / "probe_output.csv", dtype=str).fillna("")
    fresh = {r["pmid"]: r["abstract"] for _, r in src.iterrows() if r["pmid"]}
    changed = 0
    for i, pmid in enumerate(g["pmid"]):
        new_a = fresh.get(pmid, "")
        if new_a and new_a != g.at[i, "abstract"]:
            print(f"  {pmid}: {len(g.at[i, 'abstract'])}c -> {len(new_a)}c")
            g.at[i, "abstract"] = new_a
            changed += 1
    g.to_csv(gold, index=False)
    print(f"refreshed {changed}/{len(g)} abstracts in {gold.name} (sample unchanged)")

def main():
    if "--refresh-abstracts" in sys.argv:
        return refresh_abstracts()
    works = HERE / "probe_output.csv"
    mesh = HERE / "mesh_layer.csv"
    for f in (works, mesh):
        if not f.exists(): sys.exit(f"error: no {f.name} -- run the earlier probes first")

    w = pd.read_csv(works, dtype=str).fillna("")
    m = pd.read_csv(mesh, dtype=str).fillna("")
    df = w.merge(m[["pmid", "is_primary_research", "species_mesh", "qualifiers"]],
                 on="pmid", suffixes=("", "_mesh"))
    print(f"{len(w)} works x {len(m)} mesh rows -> {len(df)} joined on PMID")

    df = df[(df["is_primary_research"] == "True")
            & (df["species_mesh"].map(lambda c: bool(non_human_species(c))))
            & (df["abstract"].str.strip() != "")]
    df = df.drop_duplicates("pmid").sort_values("pmid").reset_index(drop=True)
    print(f"{len(df)} eligible (primary research + non-Human species MeSH + abstract)\n")
    if df.empty: sys.exit("error: nothing eligible to sample")

    drug = set(df[df["qualifiers"].map(has_drug_qualifier)]["pmid"])
    nodrug = set(df["pmid"]) - drug
    strain = set(df[df["species_mesh"].map(has_strain)]["pmid"])
    print(f"strata sizes: drug-qualifier {len(drug)}, no-drug-qualifier {len(nodrug)}, "
          f"strain-level {len(strain)}")

    rng, used, picked = random.Random(SEED), set(), []
    take(rng, drug, 3, used, "drug_qualifier", picked)
    take(rng, nodrug, 3, used, "no_drug_qualifier", picked)
    take(rng, strain, 2, used, "strain_level", picked)
    take(rng, set(df["pmid"]), N_TOTAL - len(picked), used, "random", picked)
    if len(picked) < N_TOTAL:
        print(f"  ! only {len(picked)}/{N_TOTAL} papers available", file=sys.stderr)

    by_pmid = {r["pmid"]: r for _, r in df.iterrows()}
    rows = []
    print(f"\n=== gold sample: {len(picked)} papers (seed={SEED}) ===")
    for pmid, stratum in picked:
        r = by_pmid[pmid]
        print(f"\n[{stratum}] {pmid}")
        print(f"  {r['title'][:100]}")
        print(f"  species: {'; '.join(non_human_species(r['species_mesh']))}")
        rows.append({"pmid": pmid, "title": r["title"], "abstract": r["abstract"],
                     **{c: "" for c in ANNOTATION_COLS}})

    out = HERE / "gold_annotate.csv"
    pd.DataFrame(rows, columns=["pmid", "title", "abstract"] + ANNOTATION_COLS
                 ).to_csv(out, index=False)
    print(f"\nwrote {out.name} -- {len(rows)} rows, {len(ANNOTATION_COLS)} blank columns")
    print("  model_type      : genetic_strain | diet | surgical | pharmacologic | combination")
    print("  model_components: free text, ;-separated")
    print("  arms            : ;-separated interventions, or NONE")
    print("  outcomes        : ;-separated, each as domain|measure|direction")
    print("  mechanisms      : ;-separated")
    print("  entities        : ;-separated gene/protein symbols")

if __name__ == "__main__":
    main()
