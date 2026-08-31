#!/usr/bin/env python3
"""Diff two grid snapshots and emit the material changes.

Deterministic: no network, no LLM. Compares every cell's status, paper count,
interventional count and direction tallies, plus the cross-paper contradiction set.

  python diff_engine.py --old grid_old.json --new grid.json
  python diff_engine.py --selftest
"""

import copy, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECTIONS = ["improved", "worsened", "no_change", "mixed", "not_stated"]

# gap_filled > contradiction_new > first_interventional > gap_promoted > gap_touched.
# contradiction_resolved is not in the requested ranking; it sits with its twin so the
# two halves of a contradiction change stay adjacent in the output.
SEVERITY = {
    "gap_filled": 1,
    "contradiction_new": 2,
    "contradiction_resolved": 3,
    "first_interventional": 4,
    "gap_promoted": 5,
    "gap_touched": 6,
}

def key(c):
    return (c["species"], c["model"], c["outcome_domain"])

def cellmap(grid):
    return {key(c): c for c in grid.get("cells", [])}

def contradiction_map(grid):
    """Only true contradictions -- opposite directions within the SAME comparison type.
    divergent_context entries are recorded in the grid but never alert: an interventional
    improvement beside a characterisation worsening is expected, not a disagreement.
    Keyed by partition too, so an interventional and a characterisation contradiction on
    the same measure stay distinct."""
    return {(c["model"], c["outcome_domain"], c["measure"], c.get("partition", "")): c
            for c in grid.get("contradictions", []) if c.get("kind") == "contradiction"}

def pair_pmids(cells_by_key, species, model):
    """Every PMID sitting at one species x model pair, across all outcome domains."""
    return {p for k, c in cells_by_key.items()
            if k[0] == species and k[1] == model for p in c.get("pmids", [])}

def diff(old, new):
    o, n = cellmap(old), cellmap(new)
    events = []

    only_new = sorted(set(n) - set(o))
    only_old = sorted(set(o) - set(n))

    for k in sorted(set(o) & set(n)):
        co, cn = o[k], n[k]
        so, sn = co["status"], cn["status"]
        po, pn = set(co.get("pmids", [])), set(cn.get("pmids", []))
        gained = sorted(pn - po)
        base = {"species": k[0], "model": k[1], "outcome_domain": k[2],
                "hf_class": cn.get("hf_class"),
                "old": {"status": so, "n_papers": co["n_papers"],
                        "n_with_arms": co["n_with_arms"],
                        "directions": co.get("directions", {})},
                "new": {"status": sn, "n_papers": cn["n_papers"],
                        "n_with_arms": cn["n_with_arms"],
                        "directions": cn.get("directions", {})}}

        if so == "no_evidence_found" and sn == "evidence":
            events.append({**base, "event": "gap_filled", "pmids": gained,
                           "note": f"cell had {co['screened_k']} papers screened and "
                                   f"nothing reported; now {cn['n_papers']}"})
        elif so == "unscreened" and sn == "evidence":
            events.append({**base, "event": "gap_touched", "pmids": gained,
                           "note": "first paper to land in a previously unscreened cell"})
        elif so == "unscreened" and sn == "no_evidence_found":
            arrived = sorted(pair_pmids(n, k[0], k[1]) - pair_pmids(o, k[0], k[1]))
            events.append({**base, "event": "gap_promoted", "pmids": arrived,
                           "note": f"screening rose {co['screened_k']} -> "
                                   f"{cn['screened_k']}; still nothing reported here"})
        elif sn == "evidence" and cn["n_papers"] > co["n_papers"]:
            events.append({**base, "event": "gap_touched", "pmids": gained,
                           "note": f"paper count {co['n_papers']} -> {cn['n_papers']}"})

        if co["n_with_arms"] == 0 and cn["n_with_arms"] > 0:
            witharms = sorted(pn - po) or sorted(pn)
            events.append({**base, "event": "first_interventional", "pmids": witharms,
                           "note": "cell held only characterisation papers; now has "
                                   "interventional evidence"})

    co_, cn_ = contradiction_map(old), contradiction_map(new)
    for ck in sorted(set(cn_) - set(co_)):
        c = cn_[ck]
        events.append({"species": "*", "model": ck[0], "outcome_domain": ck[1],
                       "measure": ck[2], "partition": ck[3], "event": "contradiction_new",
                       "old": None, "new": {"improved": c["improved_pmids"],
                                            "worsened": c["worsened_pmids"]},
                       "pmids": sorted(set(c["improved_pmids"]) | set(c["worsened_pmids"])),
                       "note": "papers testing the same kind of comparison report opposite directions"})
    for ck in sorted(set(co_) - set(cn_)):
        c = co_[ck]
        events.append({"species": "*", "model": ck[0], "outcome_domain": ck[1],
                       "measure": ck[2], "partition": ck[3], "event": "contradiction_resolved",
                       "old": {"improved": c["improved_pmids"],
                               "worsened": c["worsened_pmids"]}, "new": None,
                       "pmids": sorted(set(c["improved_pmids"]) | set(c["worsened_pmids"])),
                       "note": "cross-paper disagreement no longer present"})

    events.sort(key=lambda e: (SEVERITY[e["event"]], e["model"], e["outcome_domain"],
                               e.get("species") or ""))
    return events, only_new, only_old

def run(old_path, new_path, out_path):
    old = json.loads(Path(old_path).read_text())
    new = json.loads(Path(new_path).read_text())
    if old.get("axes") != new.get("axes"):
        print("!! axes differ between snapshots -- only cells present in both are "
              "compared", file=sys.stderr)
    events, only_new, only_old = diff(old, new)

    if not events:
        payload = {"events": [], "message": "no material change"}
        Path(out_path).write_text(json.dumps(payload, indent=2))
        print("no material change")
        return payload

    counts = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    payload = {"events": events,
               "summary": {"n_events": len(events), "by_type": counts,
                           "cells_only_in_new": len(only_new),
                           "cells_only_in_old": len(only_old),
                           "old": str(old_path), "new": str(new_path)}}
    Path(out_path).write_text(json.dumps(payload, indent=2))

    print(f"=== {len(events)} events ({Path(old_path).name} -> {Path(new_path).name}) ===")
    for t in sorted(counts, key=lambda t: SEVERITY[t]):
        print(f"  {counts[t]:>4}  {t}")
    print()
    for e in events[:25]:
        loc = f"{e.get('species')} / {e['model']} / {e['outcome_domain']}"
        extra = f" / \"{e['measure'][:40]}\"" if "measure" in e else ""
        print(f"[{e['event']}] {loc}{extra}")
        print(f"    {e['note']}")
        if e["pmids"]:
            print(f"    pmids: {', '.join(e['pmids'][:8])}"
                  + (f" (+{len(e['pmids'])-8})" if len(e["pmids"]) > 8 else ""))
    if len(events) > 25:
        print(f"\n... {len(events)-25} more in {Path(out_path).name}")
    print(f"\nwrote {Path(out_path).name}")
    return payload

def selftest():
    """Mutate a real grid, then assert the engine reports exactly what changed."""
    src = HERE / "grid.json"
    if not src.exists(): sys.exit("error: no grid.json -- run build_grid.py first")
    grid = json.loads(src.read_text())
    cells = {key(c): c for c in grid["cells"]}
    ok = fail = 0

    def check(label, cond):
        nonlocal ok, fail
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        ok, fail = ok + bool(cond), fail + (not cond)

    # 1. a grid against itself must be silent
    ev, _, _ = diff(grid, grid)
    check("identical snapshots produce no events", ev == [])

    # 2. remove a single-paper cell that sits above the screening threshold, so the
    #    synthetic "old" grid reads no_evidence_found and the real one reads evidence
    target = next((c for c in grid["cells"]
                   if c["n_papers"] == 1 and c["n_with_arms"] == 1
                   and c["screened_k"] >= grid["min_screened"]
                   and not c["not_applicable_pair"]), None)
    if target is None:
        sys.exit("selftest: no suitable single-paper cell found")
    tk, pmid = key(target), target["pmids"][0]
    print(f"  target cell: {tk}  pmid {pmid}")

    old = copy.deepcopy(grid)
    for c in old["cells"]:
        if key(c) == tk:
            c.update({"n_papers": 0, "n_with_arms": 0, "n_outcome_rows": 0, "pmids": [],
                      "directions": {d: 0 for d in DIRECTIONS},
                      "status": "no_evidence_found"})
    ev, _, _ = diff(old, grid)
    filled = [e for e in ev if e["event"] == "gap_filled" and key(e) == tk]
    check("gap_filled fires on no_evidence_found -> evidence", len(filled) == 1)
    check("gap_filled names the responsible pmid",
          bool(filled) and filled[0]["pmids"] == [pmid])
    fi = [e for e in ev if e["event"] == "first_interventional" and key(e) == tk]
    check("first_interventional fires alongside it", len(fi) == 1)

    # 3. the reverse direction must not claim a gap was filled
    rev, _, _ = diff(grid, old)
    check("reverse diff reports no gap_filled at that cell",
          not [e for e in rev if e["event"] == "gap_filled" and key(e) == tk])

    # 4. unscreened -> no_evidence_found must promote
    old2 = copy.deepcopy(old)
    for c in old2["cells"]:
        if key(c) == tk:
            c.update({"status": "unscreened", "screened_k": 0})
    ev2, _, _ = diff(old2, old)
    check("gap_promoted fires on unscreened -> no_evidence_found",
          any(e["event"] == "gap_promoted" and key(e) == tk for e in ev2))

    # 5. contradictions appear and resolve symmetrically
    if grid.get("contradictions"):
        cross = [c for c in grid["contradictions"] if c["kind"] == "cross_paper"]
        if cross:
            stripped = copy.deepcopy(grid)
            stripped["contradictions"] = [c for c in stripped["contradictions"]
                                          if c is not cross[0] and c != cross[0]]
            a, _, _ = diff(stripped, grid)
            b, _, _ = diff(grid, stripped)
            check("contradiction_new fires when one appears",
                  any(e["event"] == "contradiction_new" for e in a))
            check("contradiction_resolved fires in the reverse direction",
                  any(e["event"] == "contradiction_resolved" for e in b))

    # 6. severity ordering holds in the emitted list
    ranks = [SEVERITY[e["event"]] for e in ev]
    check("events are sorted by severity", ranks == sorted(ranks))

    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)

def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

def main():
    if "--selftest" in sys.argv:
        return selftest()
    old, new = arg("--old", HERE / "grid_old.json"), arg("--new", HERE / "grid.json")
    if not Path(old).exists(): sys.exit(f"error: no {Path(old).name}")
    if not Path(new).exists(): sys.exit(f"error: no {Path(new).name}")
    run(old, new, arg("--out", HERE / "diff.json"))

if __name__ == "__main__":
    main()
