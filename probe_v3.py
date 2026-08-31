#!/usr/bin/env python3
"""Probe v3 (Layer 2): richer per-abstract schema -- disease model, arms, typed
outcomes, mechanisms, entities, time points. probe_vocab.py is left untouched so the
two can be compared. Default run is the full eligible set; --gold-only does the 10
hand-annotated papers from gold_annotate.csv."""

import hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from google import genai

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5")
SCHEMA_VERSION = "v4-comparator"         # cache key: v1/v2 responses can never be reused
HERE = Path(__file__).resolve().parent
CACHE = HERE / "gemini_cache"
CACHE.mkdir(exist_ok=True)
RATE_GAP, MAX_RETRIES = 13.0, 4      # free-tier friendly: >=13s between live calls
_client = None
_last_call, _t0, _made, _expected = 0.0, None, 0, 0

MODEL_TYPES = ["genetic_strain", "diet", "surgical", "pharmacologic", "combination",
               "not_stated"]
DOMAINS = ["cardiac", "vascular", "pulmonary", "skeletal_muscle", "metabolic", "renal",
           "molecular", "histological", "behavioral_exercise", "other"]
DIRECTIONS = ["improved", "worsened", "no_change", "mixed", "not_stated"]
PHASES = ["model_duration", "treatment_duration", "age_at_assessment", "follow_up", "other"]
COMPARATORS = ["vs_healthy_control", "vs_untreated_disease", "vs_other_subgroup",
               "vs_baseline", "not_stated"]
SUBGROUP_AXES = ["sex", "genotype", "age", "other"]
SEXES = ["male", "female", "not_stated"]
FIELDS = ["disease_model", "arms", "outcomes", "mechanisms", "entities", "time_points"]

class DiseaseModel(BaseModel):
    model_type: Literal[tuple(MODEL_TYPES)] = "not_stated"
    components: list[str] = []

class Outcome(BaseModel):
    domain: Literal[tuple(DOMAINS)] = "other"
    measure: str = ""
    direction: Literal[tuple(DIRECTIONS)] = "not_stated"
    comparator: Literal[tuple(COMPARATORS)] = "not_stated"
    comparator_detail: str = ""
    agent: Optional[str] = None                       # vs_untreated_disease only
    subgroup_axis: Optional[Literal[tuple(SUBGROUP_AXES)]] = None   # vs_other_subgroup only

class TimePoint(BaseModel):
    phase: Literal[tuple(PHASES)] = "other"
    value: str = ""

class ExtractionV3(BaseModel):
    disease_model: DiseaseModel = Field(default_factory=DiseaseModel)
    arms: list[str] = []
    outcomes: list[Outcome] = []
    mechanisms: list[str] = []
    entities: list[str] = []
    time_points: list[TimePoint] = []
    sexes_studied: list[Literal[tuple(SEXES)]] = []
    genotypes_studied: list[str] = []

EXTRACT_PROMPT = """Extract a structured record from this HFpEF animal-study abstract.

- disease_model: ONE object describing how the disease state was created.
  - model_type: one of genetic_strain | diet | surgical | pharmacologic |
    combination | not_stated.
  - components: list of the model's parts, VERBATIM, e.g. ["ZSF1 obese rat"] or
    ["high-fat diet", "L-NAME"].
  A NAMED STRAIN IS THE MODEL. If the study uses ZSF1, Dahl salt-sensitive, db/db,
  ZDF, SHR or any other named strain, model_type is genetic_strain (or combination
  if a diet/surgery is added on top) and the strain goes in components. Never leave
  disease_model empty just because the abstract does not use the word "model".
- arms: list of interventions TESTED, verbatim. A characterization study that tests
  no intervention gets an EMPTY list. Do NOT list vehicle, sham, or control arms.
- outcomes: list of objects, one per measured outcome:
  - domain: cardiac | vascular | pulmonary | skeletal_muscle | metabolic | renal |
    molecular | histological | behavioral_exercise | other
  - measure: VERBATIM name of what was measured, e.g. "E/e' ratio".
  - direction: improved | worsened | no_change | mixed | not_stated -- the effect
    reported for that measure.
  - comparator: WHAT WAS COMPARED WITH WHAT. Choose one:
      vs_healthy_control   diseased / model animals against healthy or wild-type animals
      vs_untreated_disease treated animals against vehicle or untreated diseased animals
      vs_other_subgroup    one subgroup against another inside the study
                           (male vs female, one genotype vs another, old vs young)
      vs_baseline          the same animals compared with their own earlier timepoint
      not_stated           the abstract does not say what the comparison was
  - comparator_detail: the VERBATIM phrase that identifies the comparison, e.g.
    "empagliflozin-treated vs vehicle", "female vs male", "db/db+Aldo vs control".
  - agent: for vs_untreated_disease ONLY, the treatment being compared, VERBATIM.
    null for every other comparator.
  - subgroup_axis: for vs_other_subgroup ONLY, one of sex | genotype | age | other.
    null for every other comparator.

  ONE PAPER USUALLY MIXES COMPARATORS. A single abstract that says the model developed
  diastolic dysfunction versus wild-type AND that a drug reversed it must produce at
  least two outcomes with DIFFERENT comparator values -- vs_healthy_control for the
  model finding and vs_untreated_disease for the drug finding. Do not stamp one
  comparator across every outcome in the paper.
- mechanisms: list of pathways or processes the authors CLAIM explain their result.
  Take these from the results/conclusions, never from the background.
- entities: list of gene or protein names, VERBATIM as written. Do not expand
  abbreviations or normalise casing.
- sexes_studied: which sexes the study used -- list from male | female | not_stated.
- genotypes_studied: list of genotypes/strains studied, VERBATIM (e.g. "db/db",
  "SIRT3 KO", "wild-type"). Empty list if none named.
- time_points: list of objects: phase (model_duration | treatment_duration |
  age_at_assessment | follow_up | other) and value VERBATIM, e.g. "12 weeks of
  diet", "20-week-old". Empty list if the abstract states no durations or ages.

HARD RULES:
- All text is copied VERBATIM. Do not paraphrase, normalise, or expand abbreviations.
  The closed enums are the only fields you may choose rather than copy.
- Not stated -> empty list (or not_stated). Do NOT infer, guess, or generalise from
  context. An empty list is a correct answer.

ABSTRACT:
{abstract}
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

def retry_after(err, attempt):
    """Seconds to wait before retrying a transient error, or None if it is terminal.
    429  -> the server's own retryDelay + 2s.
    5xx  -> escalating backoff; "high demand" UNAVAILABLE is transient, not an answer."""
    s, code = str(err), getattr(err, "code", None)
    if code == 429 or "RESOURCE_EXHAUSTED" in s or "429" in s:
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s)
        return (float(m.group(1)) if m else 30.0) + 2
    if code in (500, 502, 503, 504) or any(
            k in s for k in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")):
        return 15.0 * (attempt + 1)
    if isinstance(err, json.JSONDecodeError):      # truncated/garbled body; re-roll
        return 5.0 * (attempt + 1)
    # transport-level faults: the connection died, not the request. Always worth a retry.
    if isinstance(err, (ConnectionError, TimeoutError, OSError)) or any(
            k in s for k in ("Connection reset", "Connection aborted", "Broken pipe",
                             "Remote end closed", "Server disconnected", "timed out",
                             "RemoteProtocolError", "Errno 54", "Errno 32", "Errno 104")):
        return 10.0 * (attempt + 1)
    return None

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
            wait = retry_after(e, attempt)
            if wait is not None and attempt < MAX_RETRIES:
                print(f"  ~ transient error on {tag} ({getattr(e, 'code', '?')}): "
                      f"retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            _made += 1
            print(f"  ! {tag} failed after {attempt + 1} attempt(s): {e}", file=sys.stderr)
            return None
        _made += 1
        p.write_text(json.dumps(data))
        return data

def non_human_species(cell):
    return [s.strip() for s in str(cell or "").split(";")
            if s.strip() and not s.strip().lower().startswith("human")]

def eligible():
    """probe_output.csv x mesh_layer.csv, primary research + animal + has abstract."""
    works, mesh = HERE / "probe_output.csv", HERE / "mesh_layer.csv"
    for f in (works, mesh):
        if not f.exists(): sys.exit(f"error: no {f.name} -- run the earlier probes first")
    w, m = pd.read_csv(works, dtype=str).fillna(""), pd.read_csv(mesh, dtype=str).fillna("")
    df = w.merge(m[["pmid", "is_primary_research", "species_mesh"]], on="pmid")
    df = df[(df["is_primary_research"] == "True")
            & (df["species_mesh"].map(lambda c: bool(non_human_species(c))))
            & (df["abstract"].str.strip() != "")]
    return df.drop_duplicates("pmid").sort_values("pmid").reset_index(drop=True)

def normalise(d):
    """Model output -> validated dict, dropping malformed entries rather than crashing."""
    try:
        return ExtractionV3.model_validate(d or {}).model_dump()
    except Exception as e:
        print(f"  ! schema validation failed, keeping raw: {e}", file=sys.stderr)
        return {f: (d or {}).get(f) for f in FIELDS}

def filled(field, rec):
    if field == "disease_model":
        dm = rec.get("disease_model") or {}
        return bool(dm.get("components")) or dm.get("model_type") not in (None, "not_stated")
    return bool(rec.get(field))

def main():
    global _expected
    gold_only = "--gold-only" in sys.argv
    df = eligible()
    print(f"{len(df)} eligible papers (primary research + non-Human species + abstract)")

    if gold_only:
        gold = HERE / "gold_annotate.csv"
        if not gold.exists(): sys.exit(f"error: no {gold.name} -- run gold_sample.py first")
        want = [p for p in pd.read_csv(gold, dtype=str).fillna("")["pmid"] if p]
        df = df[df["pmid"].isin(want)].reset_index(drop=True)
        missing = sorted(set(want) - set(df["pmid"]))
        print(f"--gold-only: {len(df)} of {len(want)} gold PMIDs are eligible"
              + (f" (missing: {', '.join(missing)})" if missing else ""))
    else:
        print("(full set -- pass --gold-only to run just the 10 annotated papers)")
    if df.empty: sys.exit("error: nothing to extract")

    prompts = [EXTRACT_PROMPT.format(abstract=a) for a in df["abstract"]]
    live = sum(1 for q in prompts if not cache_path("v3", q).exists())
    _expected = live
    print(f"\n{live} live calls needed ({len(prompts) - live} cached); "
          f"at {RATE_GAP:.0f}s spacing that is ~{fmt(live * RATE_GAP)}\n")

    out, failed = {}, []
    for i, ((_, r), q) in enumerate(zip(df.iterrows(), prompts), 1):
        d = gemini_json("v3", q, ExtractionV3)
        if d is None:                      # retries exhausted: stop, never emit a blank
            failed.append(r["pmid"])       # record that would read as "nothing stated"
            print(f"\n!! ABORTED after {i - 1}/{len(df)} papers -- call gave up "
                  f"(see the error above for the attempt count)")
            print(f"   failed PMID(s): {', '.join(failed)}")
            print(f"   no output file written; the {len(out)} completed papers are "
                  f"cached, so a rerun resumes from there")
            sys.exit(1)
        rec = normalise(d)
        rec["_title"] = r["title"]
        out[r["pmid"]] = rec
        mark = "FAIL" if d is None else "".join("#" if filled(f, rec) else "." for f in FIELDS)
        print(f"[{i}/{len(df)}] pmid={r['pmid']:>9} [{mark}]{eta()}")

    dest = HERE / ("v3_gold.json" if gold_only else "v3_full.json")
    dest.write_text(json.dumps(out, indent=2))
    n = len(out) or 1

    print(f"\n=== per-field fill rate (n={len(out)}) ===")
    for f in FIELDS:
        k = sum(1 for rec in out.values() if filled(f, rec))
        print(f"  {f:<16} {k:>3}/{len(out)} ({100.0 * k / n:>5.1f}%)")

    print(f"\n=== full extractions (compare against gold_annotate.csv) ===")
    for pmid, rec in out.items():
        body = {k: v for k, v in rec.items() if not k.startswith("_")}
        print(f"\n--- {pmid}  {rec['_title'][:80]}")
        print(json.dumps(body, indent=2))
    print(f"\nwrote {dest.name}")

if __name__ == "__main__":
    main()
