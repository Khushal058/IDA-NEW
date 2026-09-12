"""
generate_dataset.py
====================
Regenerates combined_dataset.csv (and the two source .xlsx files) so that:
  - Gender includes Other (1=Male, 2=Female, 3=Other)
  - Education / Occupation / Income match Dr. Ruchita Dixit's demographic
    sheet exactly (7 / 11 / 7 categories respectively)
  - ALL 23 signs & symptoms use the full 4-level scale (0=Absent, 1=Mild,
    2=Moderate, 3=Severe)
  - Every patient gets the FULL Lab Report: all 9 CBC parameters and all
    15 Peripheral Smear (PS) findings (24 lab columns total, prefixed
    cbc_/ps_). The Iron Profile has been removed entirely — it is no
    longer generated, stored, or used in any calculation.
  - Outcome (IDA Positive/Negative) is computed with the SAME
    classify_lab_verdict() logic web_app.py / gui_app.py use to confirm
    IDA from a real patient's lab report (the field lists and function
    below are copied verbatim from web_app.py, since this codebase keeps
    each script self-contained rather than sharing a module) — so the
    label the AI model is trained on is always in sync with the app's
    lab-confirmation logic.
      Outcome = 1  <=>  CBC criterion met OR PS criterion met (simple,
      deterministic rule — no "X/5" or "X/10" fractional scoring):
        CBC: Hb < 13 (male) / < 12 (female) g/dL AND MCV < 80 fL AND MCH < 27 pg
        PS:  RBC size = Microcytic AND RBC staining = Hypochromic
      Either one alone is sufficient — they are independent diagnostic
      pathways to the same Outcome, per Dr. Dixit's simplified rule.

Per Mam's brief (case-control training with both symptoms AND lab
reports; AI learns symptom weightage from many combined patients; final
model uses symptoms only): all 24 lab values are generated and saved
here, but train_model.py explicitly excludes every cbc_/ps_ column
from the model's input features, so the trained model itself only sees
symptoms + demographics. The full lab data stays in the dataset purely
as the (case/control) source used to derive Outcome, and for reference.

Symptom severity is sampled with different probability tables depending
on Outcome, so the AI model has a genuine, learnable signal from symptoms
alone. CBC/PS parameters other than the 5 CORE ones are correlated with
Outcome probabilistically (not forced) — mirrors real patients, where not
every finding lines up perfectly, and keeps the panel a genuinely
learnable, non-trivial signal rather than a deterministic rule.

Run this ONCE before train_model.py whenever the symptom/demographic/lab
coding changes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PER_CLASS = 10000

# ---------------------------------------------------------------------------
# Symptom severity probability tables: P(0=Absent, 1=Mild, 2=Moderate, 3=Severe)
# ---------------------------------------------------------------------------
TIER_A = ["Paleness","Fatigue","BrittleNails","Dizziness","SOB","PoorSleep",
          "Irritability","HairLoss","DrySkin","Appetite","ColdIntolerance","Headache"]
TIER_B = ["Tongue","EarNoise","Tachycardia","Infections","Bruising","RLS","Dysphagia"]
TIER_C = ["Pica","BlueSclera","AngularStomatitis"]   # Spleen removed — clinical exam finding, not a self-reported symptom (per Dr. Dixit)

ALL_SYMPTOMS = TIER_A + TIER_B + TIER_C
assert len(ALL_SYMPTOMS) == 22

PROBS_POS = {}
PROBS_NEG = {}
for s in TIER_A:
    PROBS_POS[s] = [0.05, 0.15, 0.35, 0.45]
    PROBS_NEG[s] = [0.82, 0.14, 0.03, 0.01]
for s in TIER_B:
    PROBS_POS[s] = [0.25, 0.30, 0.25, 0.20]
    PROBS_NEG[s] = [0.88, 0.10, 0.015, 0.005]
for s in TIER_C:
    PROBS_POS[s] = [0.45, 0.30, 0.15, 0.10]
    PROBS_NEG[s] = [0.94, 0.045, 0.010, 0.005]

_SYMPTOM_WEIGHTS = np.array([3.0]*len(TIER_A) + [1.5]*len(TIER_B) + [1.0]*len(TIER_C))
_SYMPTOM_WEIGHTS = _SYMPTOM_WEIGHTS / _SYMPTOM_WEIGHTS.sum()

def sample_symptoms(n, positive):
    out = {s: np.zeros(n, dtype=int) for s in ALL_SYMPTOMS}
    n_elevated = rng.integers(2, 19, size=n) if positive else rng.integers(0, 4, size=n)
    for i in range(n):
        k = int(min(n_elevated[i], len(ALL_SYMPTOMS)))
        elevated = set(rng.choice(ALL_SYMPTOMS, size=k, replace=False, p=_SYMPTOM_WEIGHTS)) if k > 0 else set()
        for s in ALL_SYMPTOMS:
            if s in elevated:
                sev = rng.choice([2, 3], p=[0.35, 0.65]) if positive else rng.choice([1, 2], p=[0.7, 0.3])
            else:
                sev = rng.choice([0, 1], p=[0.75, 0.25]) if positive else rng.choice([0, 1], p=[0.94, 0.06])
            out[s][i] = sev
    return out


def sample_demographics(n, positive):
    age = np.clip(rng.normal(38 if positive else 34, 15, n).round(), 12, 85).astype(int)
    if positive:
        gender = rng.choice([1, 2, 3], size=n, p=[0.29, 0.70, 0.01])
    else:
        gender = rng.choice([1, 2, 3], size=n, p=[0.54, 0.45, 0.01])
    address = rng.choice([1, 2], size=n, p=[0.60, 0.40] if positive else [0.45, 0.55])
    edu_pos_w = np.array([0.22, 0.22, 0.20, 0.16, 0.11, 0.07, 0.02])
    edu_neg_w = np.array([0.08, 0.12, 0.15, 0.18, 0.20, 0.19, 0.08])
    education = rng.choice(np.arange(1, 8), size=n, p=(edu_pos_w if positive else edu_neg_w))
    occ_pos_w = np.array([0.08, 0.22, 0.05, 0.08, 0.05, 0.09, 0.13, 0.13, 0.12, 0.03, 0.02])
    occ_neg_w = np.array([0.10, 0.14, 0.10, 0.13, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.02])
    occupation = rng.choice(np.arange(1, 12), size=n, p=(occ_pos_w if positive else occ_neg_w))
    inc_pos_w = np.array([0.02, 0.05, 0.08, 0.15, 0.20, 0.25, 0.25])
    inc_neg_w = np.array([0.10, 0.15, 0.20, 0.20, 0.15, 0.12, 0.08])
    income = rng.choice(np.arange(1, 8), size=n, p=(inc_pos_w if positive else inc_neg_w))
    return age, gender, address, education, occupation, income


# ===========================================================================
# Lab Report field definitions + classify_lab_verdict — copied verbatim from
# web_app.py so this script stays self-contained (this codebase duplicates
# these per-file rather than sharing a module; keep all 3 copies identical
# if you ever change the cut-offs).
# ===========================================================================
LAB_CBC_FIELDS = [
    ("hb",             "Hemoglobin (Hb)",       "g/dL",     12.0, 0.0, 25.0, 0.1, "<", 12.0),
    ("rbc_count",      "RBC Count",             "×10⁶/µL",  3.8,  0.0, 10.0, 0.1, "<", 3.8),
    ("hematocrit",     "Haematocrit (PCV)",     "%",        36.0, 0.0, 60.0, 0.5, "<", 36.0),
    ("mcv",            "MCV",                   "fL",       80.0, 0.0, 130.0,0.5, "<", 80.0),
    ("mch",            "MCH",                   "pg",       27.0, 0.0, 45.0, 0.5, "<", 27.0),
    ("mchc",           "MCHC",                  "g/dL",     32.0, 0.0, 40.0, 0.5, "<", 32.0),
    ("rdwcv",          "RDW-CV",                "%",        11.5, 0.0, 30.0, 0.1, ">", 14.5),
    ("wbc_count",      "Total WBC Count",       "/µL",      4000.0, 0.0, 30000.0, 100.0, None, None),
    ("platelet_count", "Platelet Count",        "lakh/µL",  1.5,  0.0, 10.0, 0.1, None, None),
]
LAB_PS_FIELDS = [
    ("rbc_size",            "RBC size",                   ["Microcytic","Normocytic","Macrocytic"], {"Microcytic"}),
    ("rbc_staining",        "RBC staining",               ["Hypochromic","Normochromic"], {"Hypochromic"}),
    ("anisocytosis",        "Anisocytosis",                ["Present","Absent"], {"Present"}),
    ("poikilocytosis",      "Poikilocytosis",              ["Present","Absent"], {"Present"}),
    ("pencil_cells",        "Pencil cells / Elliptocytes", ["Present","Absent"], {"Present"}),
    ("target_cells",        "Target cells",                 ["Present","Absent"], {"Absent"}),
    ("teardrop_cells",      "Teardrop cells",               ["Present","Absent"], {"Absent"}),
    ("schistocytes",        "Schistocytes",                 ["Present","Absent"], {"Absent"}),
    ("rouleaux",            "Rouleaux formation",           ["Present","Absent"], {"Absent"}),
    ("polychromasia",       "Polychromasia",                ["Present","Absent"], {"Absent"}),
    ("nucleated_rbc",       "Nucleated RBCs",               ["Present","Absent"], {"Absent"}),
    ("wbc_morphology",      "WBC morphology",               ["Normal","Abnormal"], {"Normal"}),
    ("platelet_morphology", "Platelet morphology",          ["Normal","Abnormal"], {"Normal"}),
    ("platelet_number",     "Platelet number",              ["Adequate","Increased","Decreased"], {"Adequate","Increased"}),
    ("hemoparasites",       "Hemoparasites",                ["Seen","Not seen"], {"Not seen"}),
]
def classify_lab_verdict(lab_values: dict, gender: int = None):
    """Verbatim copy of web_app.py's classify_lab_verdict — simple,
    deterministic CBC-OR-PS rule, no fractional "X/5"/"X/10" scoring."""
    hb, mcv, mch = lab_values.get("hb"), lab_values.get("mcv"), lab_values.get("mch")
    rbc_size, rbc_staining = lab_values.get("rbc_size"), lab_values.get("rbc_staining")

    hb_cutoff = 13.0 if gender == 1 else 12.0  # Male=13, Female/Other/unknown=12

    cbc_entered = hb is not None and mcv is not None and mch is not None
    cbc_met = cbc_entered and (hb < hb_cutoff and mcv < 80.0 and mch < 27.0)

    ps_entered = rbc_size is not None and rbc_staining is not None
    ps_met = ps_entered and (rbc_size == "Microcytic" and rbc_staining == "Hypochromic")

    if not cbc_entered and not ps_entered:
        verdict = "Lab values not yet entered"
    elif cbc_met or ps_met:
        verdict = "Laboratory findings SUGGESTIVE of Iron Deficiency Anaemia"
    elif cbc_entered and ps_entered:
        verdict = "Laboratory findings NOT suggestive of Iron Deficiency Anaemia"
    else:
        missing = "Peripheral Smear (RBC size + staining)" if not ps_entered else "CBC (Hb, MCV, MCH)"
        verdict = f"Not suggestive so far — {missing} not yet entered"

    return verdict, {"cbc_met": cbc_met, "ps_met": ps_met, "cbc_entered": cbc_entered, "ps_entered": ps_entered}


# ---------------------------------------------------------------------------
# Full Lab Report generation (24 parameters: 9 CBC + 15 PS — Iron Profile
# removed).
# Simple rule (guaranteed exactly): Outcome = 1 <=> CBC criterion met OR
# PS criterion met (classify_lab_verdict's cbc_met/ps_met — see there).
# Every other parameter (remaining CBC + remaining 13 PS findings) is
# correlated with Outcome probabilistically, not forced.
# ---------------------------------------------------------------------------
def _sample_numeric_side(op, cutoff, mn, mx, suggestive, margin_frac=0.35):
    span = (mx - mn) * margin_frac
    if op == "<":
        lo, hi = (max(mn, cutoff - span), cutoff * 0.98) if suggestive else (cutoff * 1.02, min(mx, cutoff + span * 1.4))
    else:
        lo, hi = (cutoff * 1.02, min(mx, cutoff + span)) if suggestive else (max(mn, cutoff - span * 1.4), cutoff * 0.98)
    lo, hi = min(lo, hi), max(lo, hi)
    return rng.uniform(lo, hi)


def sample_full_labs(n, positive, gender_arr):
    cbc_defs = {k: (unit, default, mn, mx, step, op, cut) for k, label, unit, default, mn, mx, step, op, cut in LAB_CBC_FIELDS}
    ps_defs = {k: (options, suggestive_vals) for k, label, options, suggestive_vals in LAB_PS_FIELDS}

    cbc = {k: np.zeros(n) for k in cbc_defs}
    ps = {k: [None] * n for k, *_r in LAB_PS_FIELDS}

    CBC_CORR_PROB = 0.55 if positive else 0.10   # non-criterion CBC params
    PS_CORR_PROB = 0.55 if positive else 0.10    # non-criterion PS params

    for i in range(n):
        hb_cutoff = 13.0 if gender_arr[i] == 1 else 12.0  # Male=13, Female/Other=12

        if positive:
            # Which pathway(s) this positive patient satisfies — CBC-only,
            # PS-only, or both — so the AI sees varied real-world-like
            # patterns rather than every positive patient looking identical.
            pathway = rng.choice(["cbc", "ps", "both"], p=[0.45, 0.45, 0.10])
            cbc_suggestive = pathway in ("cbc", "both")
            ps_suggestive = pathway in ("ps", "both")
        else:
            cbc_suggestive = False
            ps_suggestive = False

        # --- CBC criterion: Hb (gender-specific cutoff), MCV, MCH ---
        _, _, mn, mx, *_ = cbc_defs["hb"]
        cbc["hb"][i] = _sample_numeric_side("<", hb_cutoff, mn, mx, cbc_suggestive)
        _, _, mn, mx, *_ = cbc_defs["mcv"]
        cbc["mcv"][i] = _sample_numeric_side("<", 80.0, mn, mx, cbc_suggestive)
        _, _, mn, mx, *_ = cbc_defs["mch"]
        cbc["mch"][i] = _sample_numeric_side("<", 27.0, mn, mx, cbc_suggestive)

        # --- PS criterion: RBC size = Microcytic AND RBC staining = Hypochromic ---
        if ps_suggestive:
            ps["rbc_size"][i] = "Microcytic"
            ps["rbc_staining"][i] = "Hypochromic"
        else:
            ps["rbc_size"][i] = rng.choice([o for o in ps_defs["rbc_size"][0] if o != "Microcytic"])
            ps["rbc_staining"][i] = rng.choice([o for o in ps_defs["rbc_staining"][0] if o != "Hypochromic"])

        # --- Remaining CBC parameters (correlated, not forced) ---
        for key, (unit, default, mn, mx, step, op, cut) in cbc_defs.items():
            if key in ("hb", "mcv", "mch"):
                continue
            if op is None:  # WBC / Platelet Count: reference-only, no cut-off
                shift = default * 0.15 if (key == "platelet_count" and positive) else 0.0
                cbc[key][i] = max(mn, min(mx, rng.normal(default + shift, default * 0.12)))
            else:
                suggestive = rng.random() < CBC_CORR_PROB
                cbc[key][i] = _sample_numeric_side(op, cut, mn, mx, suggestive)

        # --- Remaining Peripheral Smear findings (correlated, not forced) ---
        # Anisocytosis / Poikilocytosis / Pencil cells get a higher
        # correlation with the PS criterion specifically — per this
        # hospital's reporting convention, they're only written up as
        # "Present" alongside a microcytic-hypochromic smear, so tying
        # their probability to ps_suggestive (rather than a fully
        # independent draw) mirrors real report patterns.
        for key, label, options, suggestive_vals in LAB_PS_FIELDS:
            if key in ("rbc_size", "rbc_staining"):
                continue
            corr_prob = (0.85 if ps_suggestive else 0.05) if key in ("anisocytosis", "poikilocytosis", "pencil_cells") else PS_CORR_PROB
            sv = list(suggestive_vals)
            if rng.random() < corr_prob:
                ps[key][i] = rng.choice(sv)
            else:
                other = [o for o in options if o not in suggestive_vals] or options
                ps[key][i] = rng.choice(other)

    for k in cbc:
        cbc[k] = np.round(cbc[k], 2)
    return cbc, ps


def build(n, positive):
    age, gender, address, education, occupation, income = sample_demographics(n, positive)
    symptoms = sample_symptoms(n, positive)
    cbc, ps = sample_full_labs(n, positive, gender)

    df = pd.DataFrame({
        "RecordID": np.arange(n),
        "Age": age, "Gender": gender, "Address": address,
        "Education": education, "Occupation": occupation, "Income": income,
        **symptoms,
        **{f"cbc_{k}": v for k, v in cbc.items()},
        **{f"ps_{k}": v for k, v in ps.items()},
    })

    # Outcome computed with the SAME classify_lab_verdict() the apps use.
    outcomes = np.zeros(n, dtype=int)
    for i in range(n):
        lab_values = {k: cbc[k][i] for k in cbc}
        lab_values.update({k: ps[k][i] for k in ps})
        _, detail = classify_lab_verdict(lab_values, gender=gender[i])
        outcomes[i] = 1 if (detail["cbc_met"] or detail["ps_met"]) else 0
    df["Outcome"] = outcomes
    return df


df_pos = build(N_PER_CLASS, True)
df_neg = build(N_PER_CLASS, False)
df_pos["RecordID"] = np.arange(1, N_PER_CLASS + 1)
df_neg["RecordID"] = np.arange(N_PER_CLASS + 1, 2 * N_PER_CLASS + 1)

print("Positive-set Outcome value counts (should be ~all 1, by construction):")
print(df_pos["Outcome"].value_counts())
print("Negative-set Outcome value counts (should be ~all 0, by construction):")
print(df_neg["Outcome"].value_counts())

df_pos = df_pos[df_pos["Outcome"] == 1].reset_index(drop=True)
df_neg = df_neg[df_neg["Outcome"] == 0].reset_index(drop=True)
print(f"After dropping any boundary mismatches: {len(df_pos)} positive, {len(df_neg)} negative records.")

df_pos.to_excel("10K_individual_with_Iron_deficiency.xlsx", index=False)
df_neg.to_excel("10K_individual_without_Iron_deficiency.xlsx", index=False)

combined = pd.concat([df_pos, df_neg], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.to_csv("combined_dataset.csv", index=False)

print("Saved combined_dataset.csv with", len(combined), "records.")
print(combined["Outcome"].value_counts())
