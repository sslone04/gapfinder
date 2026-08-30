#!/usr/bin/env python3
"""Probe 2: can one LLM pass pull verbatim structured fields out of HFpEF animal
abstracts, and do the free-text ones cluster? Input: probe_output.csv."""

import hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel
from google import genai

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5")
SCHEMA_VERSION = "v2-9field"         # part of the cache key: v1 3-field hits won't match
HERE = Path(__file__).resolve().parent
CACHE = HERE / "gemini_cache"
CACHE.mkdir(exist_ok=True)
RATE_GAP, MAX_RETRIES = 13.0, 4      # free-tier friendly: >=13s between live calls
_client = None
_last_call, _t0, _made, _expected = 0.0, None, 0, 0

TEXT_FIELDS = ["induction_method", "intervention", "readout", "mechanism_proposed",
               "hypothesis", "future_directions"]
CLUSTER_FIELDS = TEXT_FIELDS[:4]                      # 1-4 get canonical vocabularies
LIST_FIELDS = ["molecular_entities", "genetic_manipulation"]
FIELDS = TEXT_FIELDS[:4] + ["finding_direction"] + TEXT_FIELDS[4:] + LIST_FIELDS
DIRECTIONS = ["improved", "worsened", "no_change", "mixed", "not_stated"]
ALTERATIONS = ["knockout", "knockin", "transgenic_overexpression", "point_mutation",
               "conditional_knockout", "knockdown", "naturally_occurring_mutation", "other"]
SECTION_RE = re.compile(r"\b(background|methods?|results?|conclusions?)\s*:", re.I)

class GeneticManipulation(BaseModel):
    gene: str
    alteration_type: Literal[tuple(ALTERATIONS)]
    detail: Optional[str] = None

class Extraction(BaseModel):                  # verbatim-or-null, one per abstract
    induction_method: Optional[str] = None
    intervention: Optional[str] = None
    readout: Optional[str] = None
    mechanism_proposed: Optional[str] = None
    finding_direction: Literal[tuple(DIRECTIONS)] = "not_stated"
    hypothesis: Optional[str] = None
    future_directions: Optional[str] = None
    molecular_entities: list[str] = []
    genetic_manipulation: list[GeneticManipulation] = []
    structured_abstract: bool = False

class Mapping(BaseModel):
    raw: str
    canonical: str

class Clustering(BaseModel):
    mappings: list[Mapping]

EXTRACT_PROMPT = """Extract structured fields from this scientific abstract.

- induction_method: how the disease state was CREATED in the animal.
- intervention: what treatment or manipulation was TESTED.
- readout: what was MEASURED as the primary outcome.
- mechanism_proposed: the pathway, molecule, or process the authors claim explains
  their result. Take this from the CONCLUSION/RESULTS, never from the background.
- finding_direction: effect of the intervention on the primary readout. Closed enum,
  choose one: improved | worsened | no_change | mixed | not_stated.
- hypothesis: the stated expectation or aim.
- future_directions: what the authors state is still unknown, unresolved, or needed
  next. This must be the AUTHORS' OWN claim about remaining gaps -- never your
  inference about what would be interesting to do next. Null if they state none.
- molecular_entities: list of gene or protein names mentioned, verbatim as written.
  Do NOT expand abbreviations or normalise casing. Empty list if none.
- genetic_manipulation: list of genetic alterations USED in the study. Each entry:
  gene (symbol as written), alteration_type (one of: knockout | knockin |
  transgenic_overexpression | point_mutation | conditional_knockout | knockdown |
  naturally_occurring_mutation | other), and detail (specific mutation notation if
  stated, e.g. "C340S", "N2B truncation"; null otherwise). Empty list if none used.
- structured_abstract: true if the abstract text contains section labels such as
  "Background:", "Methods:", "Results:", "Conclusions:".

HARD RULES:
- Every text field is copied VERBATIM. Do not paraphrase, normalise, or expand
  abbreviations. finding_direction and structured_abstract are the only exceptions.
- Field not stated -> null (or empty list). Do NOT infer, guess, or generalise.
  Null is a correct answer.
- Do not merge multiple things into one text field; pick the primary one.

ABSTRACT:
{abstract}
"""

CLUSTER_PROMPT = """Deduplicated verbatim "{field}" strings from HFpEF animal-model
abstracts. Cluster into 8-15 canonical categories, mapping EVERY string. Rules:
- Short lowercase snake_case labels (e.g. "aerobic_exercise").
- Every raw string appears exactly once, spelled EXACTLY as given.
- 8-15 distinct labels. No catch-all "other" unless genuinely unclusterable.

RAW STRINGS ({n}):
{items}
"""

ENTITY_PROMPT = """Deduplicated gene/protein names extracted verbatim from HFpEF
abstracts, so the same molecule appears under several aliases and casings.

Map EVERY string to ONE canonical official gene symbol. Rules:
- Collapse alias variants onto a single canonical symbol: "Grk2", "GRK2" and
  "ADRBK1" all map to "GRK2".
- Canonical symbols are UPPERCASE official HGNC symbols where one exists.
- Every raw string appears exactly once, spelled EXACTLY as given.
- This is alias normalisation, NOT categorisation: do not group distinct molecules
  into pathway or family buckets.

RAW STRINGS ({n}):
{items}
"""

def client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key: sys.exit("error: set GEMINI_API_KEY or GOOGLE_API_KEY in the env")
        _client = genai.Client(api_key=key)
    return _client

def cache_path(tag, prompt):
    h = hashlib.md5(f"{MODEL}|{SCHEMA_VERSION}|{tag}|{prompt}".encode()).hexdigest()[:16]
    return CACHE / f"{tag}_{SCHEMA_VERSION}_{h}.json"

def fmt(sec):
    return f"{int(sec) // 60}m{int(sec) % 60:02d}s"

def eta():
    """Running elapsed / remaining over the LIVE (uncached) call budget."""
    if _t0 is None: return ""
    el = time.time() - _t0
    per = el / _made if _made else RATE_GAP
    return f" {fmt(el)} elapsed, ~{fmt(max(0, _expected - _made) * per)} left"

def throttled_429(err):
    """None if not a 429, else seconds to wait: the server's retryDelay + 2s."""
    if not (getattr(err, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(err)
            or "429" in str(err)): return None
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    return (float(m.group(1)) if m else 30.0) + 2

def gemini_json(tag, prompt, schema):
    """One schema-constrained Gemini call. Cache hits return instantly with no sleep;
    live calls are spaced >=RATE_GAP apart and retried on 429. None = gave up."""
    global _last_call, _t0, _made
    p = cache_path(tag, prompt)
    if p.exists():
        try:
            return json.loads(p.read_text())          # cached -- no throttle, no sleep
        except ValueError:
            pass
    if _t0 is None: _t0 = time.time()
    for attempt in range(MAX_RETRIES + 1):
        gap = RATE_GAP - (time.time() - _last_call)   # pace live calls only
        if gap > 0: time.sleep(gap)
        _last_call = time.time()
        try:
            resp = client().models.generate_content(
                model=MODEL, contents=prompt,
                config={"response_mime_type": "application/json",
                        "response_schema": schema, "temperature": 0})
            data = json.loads(resp.text)
        except Exception as e:
            wait = throttled_429(e)
            if wait is not None and attempt < MAX_RETRIES:
                print(f"  ~ 429 on {tag}: retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            _made += 1
            print(f"  ! {tag} failed after {attempt + 1} attempt(s): {e}", file=sys.stderr)
            return None
        _made += 1
        p.write_text(json.dumps(data))
        return data

def is_animal(row):
    sp = [s.strip() for s in str(row.get("mesh_species") or "").split(";")]
    heads = [h.strip() for h in str(row.get("mesh_headings") or "").split(";")]
    return (any(s and not s.lower().startswith("human") for s in sp)
            or "Disease Models, Animal" in heads)

def clean(v):
    v = (v or "").strip() if isinstance(v, str) else ""
    return v if v and v.lower() not in ("null", "none", "n/a", "not stated") else ""

def cluster(tag, prompt, uniq):
    """Returns {canonical: [raw, ...]}. Strings the model drops go to _unmapped."""
    data = gemini_json(tag, prompt, Clustering)
    groups, mapped = defaultdict(list), set()
    for m in (data or {}).get("mappings", []):
        raw, canon = clean(m.get("raw")), clean(m.get("canonical"))
        if raw in uniq and raw not in mapped and canon:
            groups[canon].append(raw)
            mapped.add(raw)
    missed = [u for u in uniq if u not in mapped]
    if missed:                                    # model dropped strings -- don't hide it
        print(f"  ! {tag}: {len(missed)} raw unmapped -> _unmapped", file=sys.stderr)
        groups["_unmapped"] = missed
    return dict(groups)

def rank(groups, weight):
    """Canonical groups sorted by paper count. weight(raw) -> papers mentioning it."""
    return sorted(((c, sorted(m), sum(weight(x) for x in m)) for c, m in groups.items()),
                  key=lambda t: -t[2])

def extract_rows(withabs):
    global _expected
    prompts = [EXTRACT_PROMPT.format(abstract=a) for a in withabs["abstract"]]
    live = sum(1 for q in prompts if not cache_path("extract", q).exists())
    _expected = live + len(CLUSTER_FIELDS) + 1     # + molecular_entities clustering
    print(f"{live} live calls needed ({len(prompts) - live} cached); "
          f"at {RATE_GAP:.0f}s spacing that is ~{fmt(_expected * RATE_GAP)}\n")

    rows, failed = [], 0
    for i, ((_, r), q) in enumerate(zip(withabs.iterrows(), prompts), 1):
        d = gemini_json("extract", q, Extraction)
        failed += d is None
        d = d or {}
        gm = [g for g in (d.get("genetic_manipulation") or []) if clean(g.get("gene"))]
        rec = {"pmid": r.get("pmid", "")}
        rec.update({f: clean(d.get(f)) for f in TEXT_FIELDS})
        rec["finding_direction"] = (d.get("finding_direction") if d.get("finding_direction")
                                    in DIRECTIONS else "not_stated")
        rec["molecular_entities"] = [e for e in
                                     (clean(x) for x in (d.get("molecular_entities") or [])) if e]
        rec["genetic_manipulation"] = gm
        rec["structured_abstract"] = bool(d.get("structured_abstract"))
        rec["_regex_structured"] = bool(SECTION_RE.search(r["abstract"]))
        rows.append(rec)
        got = "FAIL" if not d else "".join(          # one slot per field, in FIELDS order
            "." if not rec[f] or rec[f] == "not_stated" else "#" for f in FIELDS)
        print(f"[{i}/{len(withabs)}] pmid={rec['pmid'] or '-':>9} [{got}]{eta()}")
    return rows, failed

def write_raw(rows, path):
    flat = []
    for r in rows:
        o = {k: r[k] for k in ["pmid"] + TEXT_FIELDS + ["finding_direction"]}
        o["molecular_entities"] = "; ".join(r["molecular_entities"])
        o["genetic_manipulation"] = json.dumps(r["genetic_manipulation"])
        o["structured_abstract"] = r["structured_abstract"]
        flat.append(o)
    pd.DataFrame(flat).to_csv(path, index=False)

def report(rows, vocab):
    n = len(rows) or 1
    print(f"\n=== per-field extraction (n={len(rows)}) ===")
    for f in FIELDS:
        vals = [r[f] for r in rows if r[f]]
        if f in LIST_FIELDS:                       # distinct ITEMS, not distinct lists
            distinct = len({json.dumps(x, sort_keys=True) if isinstance(x, dict) else x
                            for r in rows for x in r[f]})
        else:
            distinct = len(set(vals))
        label = "papers with >=1" if f in LIST_FIELDS else "non-null"
        print(f"  {f:<24} {100.0 * len(vals) / n:>5.1f}% {label:<15} "
              f"{distinct:>4} distinct")

    print(f"\nfinding_direction distribution:")
    dist = Counter(r["finding_direction"] for r in rows)
    for d in DIRECTIONS:
        print(f"  {dist[d]:>4} ({100.0 * dist[d] / n:>5.1f}%)  {d}")

    model_yes = sum(1 for r in rows if r["structured_abstract"])
    rx_yes = sum(1 for r in rows if r["_regex_structured"])
    disagree = sum(1 for r in rows if r["structured_abstract"] != r["_regex_structured"])
    print(f"\nstructured abstracts: {100.0 * model_yes / n:.1f}% per model, "
          f"{100.0 * rx_yes / n:.1f}% per local regex ({disagree} disagreements)")

    ents = [r["molecular_entities"] for r in rows]
    with_ent = sum(1 for e in ents if e)
    print(f"\nmolecular_entities: {with_ent}/{len(rows)} papers with >=1 "
          f"({100.0 * with_ent / n:.1f}%), mean {sum(len(e) for e in ents) / n:.2f} per paper")
    for c, m, tot in vocab.get("molecular_entities", [])[:25]:
        print(f"  {tot:>4} papers  {c:<14} <- {', '.join(m)}")

    gms = [r["genetic_manipulation"] for r in rows]
    with_gm = sum(1 for g in gms if g)
    print(f"\ngenetic_manipulation: {with_gm}/{len(rows)} papers with >=1 "
          f"({100.0 * with_gm / n:.1f}%), {sum(len(g) for g in gms)} total")
    for a, c in Counter(g.get("alteration_type", "other") for g in
                        [x for gg in gms for x in gg]).most_common():
        print(f"  {c:>4}  {a}")

def main():
    src = HERE / "probe_output.csv"
    if not src.exists(): sys.exit(f"error: no {src.name} -- run probe_ingest.py first")
    df = pd.read_csv(src, dtype=str).fillna("")
    animal = df[df.apply(is_animal, axis=1)]
    withabs = animal[animal["abstract"].str.strip() != ""]
    print(f"{len(df)} rows in {src.name}\n{len(animal)} survive the animal-model filter"
          f" (non-Human species MeSH or 'Disease Models, Animal')\n"
          f"{len(withabs)} of those have a non-empty abstract -> extraction set\n")

    rows, failed = extract_rows(withabs)
    raw_out = HERE / "vocab_raw.csv"
    write_raw(rows, raw_out)
    if failed:
        print(f"\n!! {failed}/{len(rows)} abstracts failed after {MAX_RETRIES} retries")

    ranked, vocab = {}, {}
    for f in CLUSTER_FIELDS:                       # fields 1-4: category vocabularies
        vals = [r[f] for r in rows if r[f]]
        uniq, counts = sorted(set(vals)), Counter(vals)
        if not uniq: continue
        groups = cluster(f"cluster_{f}", CLUSTER_PROMPT.format(
            field=f, n=len(uniq), items="\n".join(f"- {u}" for u in uniq)), uniq)
        ranked[f] = rank(groups, lambda x: counts[x])

    papers = Counter()                             # entity alias -> papers mentioning it
    for r in rows:
        papers.update(set(r["molecular_entities"]))
    if papers:                                     # entities: alias -> canonical symbol
        uniq = sorted(papers)
        groups = cluster("cluster_entities", ENTITY_PROMPT.format(
            n=len(uniq), items="\n".join(f"- {u}" for u in uniq)), uniq)
        ranked["molecular_entities"] = rank(groups, lambda x: papers[x])

    for f, rk in ranked.items():
        vocab[f] = [{"canonical": c, "n_papers": t, "members": m} for c, m, t in rk]
    prop = HERE / "vocab_proposed.json"
    prop.write_text(json.dumps(vocab, indent=2))

    report(rows, ranked)
    print(f"\n=== proposed vocabularies ({prop.name}) ===")
    for f in CLUSTER_FIELDS:
        if f not in ranked: continue
        print(f"\n{f}: {len(ranked[f])} canonical categories")
        for c, m, tot in ranked[f]:
            print(f"  {tot:>4} papers  {len(m):>3} raw  {c}")
    print(f"\nwrote {raw_out.name} and {prop.name}")

if __name__ == "__main__":
    main()
