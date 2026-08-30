"""Hand-editable axis definitions for build_grid.py.

Rules are ORDERED and FIRST MATCH WINS, so specific named models sit above generic
ones. Matching runs on the normalised (lowercased, whitespace- and separator-collapsed)
disease_model.components joined into one string, so "Db/ db", "db / db" and "db/db"
all match the same pattern.

  {"any": [...]}           -> matches if any pattern is present
  {"all": [[...], [...]]}  -> matches only if EACH inner group has a hit
                              (two-hit models such as HFD + L-NAME)

A word-like pattern ("aged") is matched on word boundaries, so it will not fire inside
"packaged" or "damaged". Patterns containing punctuation ("db/db", "(tac)", "-/-") are
matched as plain substrings.
"""

# --- species axis: matched against the species_mesh column of corpus.csv -----------
SPECIES_RULES = [
    ("Mouse",  ["mice", "mus musculus"]),
    ("Rat",    ["rats", "rattus"]),
    ("Pig",    ["swine", "sus scrofa"]),
    ("Dog",    ["dogs", "canis"]),
    ("Rabbit", ["rabbits", "oryctolagus"]),
]
SPECIES_FALLBACK = "Other"
SPECIES_AXIS = [s for s, _ in SPECIES_RULES] + [SPECIES_FALLBACK]
# A paper tagged with several species goes to the FIRST rule that matches; the full
# MeSH string is kept in the cell detail so nothing is lost.

# --- model axis -------------------------------------------------------------------
# Order notes:
#   ZSF1 precedes SHR and ZDF -- its full name contains "Zucker" and "spontaneously
#     hypertensive", so it would otherwise be captured by them.
#   HFD+L-NAME precedes everything diet-related; bare HFD sits LAST before the
#     fallbacks so it only fires when no second hit was found.
#   DOCA-salt precedes CKD -- the DOCA model routinely includes uninephrectomy.
#   IR precedes MI -- reperfusion studies also say "myocardial infarction".
MODEL_RULES = [
    ("ZSF1",                  {"any": ["zsf1", "zsf-1"]}),
    ("ZDF",                   {"any": ["zdf", "zucker diabetic fatty"]}),
    ("Dahl salt-sensitive",   {"any": ["dahl"]}),
    ("SHR",                   {"any": ["shr", "shrs", "spontaneously hypertensive"]}),
    ("db/db",                 {"any": ["db/db", "dbdb", "db/db mouse", "db/db mice"]}),
    ("HFD+L-NAME",            {"all": [["high-fat", "high fat", "hfd", "western diet"],
                                       ["l-name", "nitro-l-arginine", "lname"]]}),
    ("DOCA-salt",             {"any": ["doca", "deoxycorticosterone", "desoxycorticosterone"]}),
    ("Ang II",                {"any": ["angiotensin ii", "angiotensin-ii", "ang ii",
                                       "angii", "ang-ii"]}),
    ("TAC/pressure-overload", {"any": ["transverse aortic constriction", "aortic constriction",
                                       "aortic banding", "pressure overload", "tac",
                                       "(tac)", "aortic ligation"]}),
    ("IR",                    {"any": ["ischemia-reperfusion", "ischaemia-reperfusion",
                                       "ischemia/reperfusion", "ischaemia/reperfusion",
                                       "i/r injury", "transient occlusion",
                                       "transient ligation", "reperfusion"]}),
    ("MI",                    {"any": ["myocardial infarction", "coronary ligation",
                                       "coronary artery ligation", "lad ligation",
                                       "permanent ligation", "permanent occlusion",
                                       "left anterior descending"]}),
    ("isoproterenol",         {"any": ["isoproterenol", "isoprenaline", "(iso)"]}),
    ("STZ",                   {"any": ["streptozotocin", "streptozocin", "stz"]}),
    ("doxorubicin",           {"any": ["doxorubicin", "adriamycin"]}),
    ("pacing",                {"any": ["tachypacing", "tachy-pacing", "rapid pacing",
                                       "rapid ventricular pacing", "pacing-induced",
                                       "ventricular pacing", "atrial pacing"]}),
    ("CKD",                   {"any": ["nephrectomy", "uninephrectomy", "5/6 subtotal",
                                       "subtotal nephrectomy", "renal mass reduction",
                                       "chronic kidney disease", "adenine diet",
                                       "renal ablation"]}),
    ("aged",                  {"any": ["aged", "aging", "ageing", "senescent",
                                       "senescence", "old mice", "old rats"]}),
    # bare diet with no second hit -- last, so any co-occurring driver wins first
    ("HFD",                   {"any": ["high-fat", "high fat", "hfd", "western diet",
                                       "high-fat high-salt"]}),
]
MODEL_FALLBACK_GENETIC = "other-genetic"        # model_type == genetic_strain
MODEL_FALLBACK_OTHER = "other-combination"
GENETIC_HINTS = ["knockout", "knock-out", "transgenic", "ob/ob", "zucker", "-/-",
                 "null mice", "mutant", "knockin", "knock-in", "knockdown"]
MODEL_AXIS = [m for m, _ in MODEL_RULES] + [MODEL_FALLBACK_GENETIC, MODEL_FALLBACK_OTHER]

# --- heart-failure phenotype each model is normally taken to represent -------------
HF_CLASS = {
    "ZSF1": "hfpef", "Dahl salt-sensitive": "hfpef", "db/db": "hfpef", "ZDF": "hfpef",
    "SHR": "hfpef", "HFD": "hfpef", "HFD+L-NAME": "hfpef", "DOCA-salt": "hfpef",
    "aged": "hfpef", "CKD": "hfpef",
    "MI": "hfref", "IR": "hfref", "isoproterenol": "hfref", "pacing": "hfref",
    "doxorubicin": "hfref",
    "TAC/pressure-overload": "mixed", "Ang II": "mixed", "STZ": "mixed",
    "other-genetic": "unclassified", "other-combination": "unclassified",
}

# --- outcome axis: the extraction enum, used as-is --------------------------------
OUTCOME_AXIS = ["cardiac", "vascular", "pulmonary", "skeletal_muscle", "metabolic",
                "renal", "molecular", "histological", "behavioral_exercise", "other"]

# --- species x model pairs that cannot exist --------------------------------------
# Rendered as a fourth cell state; never counted as a gap, never used in adjacency.
NOT_APPLICABLE = {
    ("Mouse", "ZSF1"),                  # rat strain
    ("Mouse", "Dahl salt-sensitive"),   # rat strain
    ("Mouse", "ZDF"),                   # rat strain
    ("Mouse", "SHR"),                   # rat strain
    ("Rat", "db/db"),                   # mouse Lepr mutation
}

# --- cell status thresholds -------------------------------------------------------
# An empty cell is only "no_evidence_found" when enough papers sat at that
# species x model pair to have reported the outcome. Below this it is "unscreened".
MIN_SCREENED = 3
